"""
Unit tests for pre-training innovations (预训练创新特性单元测试):
1. GNS-Adaptive Batch Size Scheduling (自适应 Batch Size 调度)
2. Semantic Fingerprint Blocking (语义指纹阻断防泄露)
3. LG-Opt: Loss-Gradient Decoupled Rescaling (Loss 偏导梯度自适应重缩放)
"""

import os
import json
import tempfile
import shutil
import unittest
import torch
import torch.nn as nn


# ==============================================================================
# Helper: Simple model for gradient-based tests
# ==============================================================================

class SimpleLinearModel(nn.Module):
    """Minimal model for gradient computation tests."""
    def __init__(self, in_features=32, out_features=16):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 64)
        self.linear2 = nn.Linear(64, out_features)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


# ==============================================================================
# 1. GNS Estimator Tests (自适应 Batch Size 调度测试)
# ==============================================================================

class TestGradientNoiseScaleEstimator(unittest.TestCase):
    """Tests for the GNS-Adaptive Batch Size Scheduling feature."""

    def setUp(self):
        from utils.gns_estimator import GradientNoiseScaleEstimator
        self.GNSEstimator = GradientNoiseScaleEstimator

    def test_initialization_defaults(self):
        """Test default initialization parameters."""
        gns = self.GNSEstimator()
        self.assertEqual(gns.current_grad_accum_steps, 4)
        self.assertEqual(gns.max_grad_accum_steps, 16)
        self.assertEqual(gns.check_interval, 100)
        self.assertIsNone(gns.gns_ema)
        self.assertEqual(gns.latest_gns, 0.0)

    def test_initialization_custom(self):
        """Test custom initialization parameters."""
        gns = self.GNSEstimator(
            initial_grad_accum_steps=2,
            max_grad_accum_steps=8,
            check_interval=50,
        )
        self.assertEqual(gns.current_grad_accum_steps, 2)
        self.assertEqual(gns.max_grad_accum_steps, 8)
        self.assertEqual(gns.check_interval, 50)

    def test_record_micro_batch_grad(self):
        """Test that micro-batch gradient norms are recorded correctly."""
        gns = self.GNSEstimator()
        model = SimpleLinearModel()

        # Generate a gradient via backward
        x = torch.randn(4, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()

        gns.record_micro_batch_grad(model)
        self.assertEqual(len(gns._micro_batch_grad_norms_sq), 1)
        self.assertGreater(gns._micro_batch_grad_norms_sq[0], 0.0)

    def test_record_accumulated_grad(self):
        """Test that accumulated gradient norm is recorded correctly."""
        gns = self.GNSEstimator()
        model = SimpleLinearModel()

        x = torch.randn(4, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()

        gns.record_accumulated_grad(model)
        self.assertGreater(gns._accumulated_grad_norm_sq, 0.0)

    def test_compute_gns_with_insufficient_data(self):
        """GNS should return 0.0 when there are fewer than 2 micro-batches."""
        gns = self.GNSEstimator()
        # No micro-batch recorded
        self.assertEqual(gns.compute_gns(), 0.0)

        # Only 1 micro-batch
        gns._micro_batch_grad_norms_sq = [1.0]
        self.assertEqual(gns.compute_gns(), 0.0)

    def test_compute_gns_with_data(self):
        """GNS should return a positive value with sufficient micro-batch data."""
        gns = self.GNSEstimator()
        model = SimpleLinearModel()

        # Simulate multiple micro-batches with different gradients
        for i in range(4):
            model.zero_grad()
            x = torch.randn(4, 32) * (i + 1)  # Different scales
            out = model(x)
            loss = out.sum()
            loss.backward()
            gns.record_micro_batch_grad(model)

        gns.record_accumulated_grad(model)
        gns_value = gns.compute_gns()
        self.assertGreater(gns_value, 0.0)

    def test_step_resets_accumulators(self):
        """Calling step() should reset per-step accumulators."""
        gns = self.GNSEstimator()
        gns._micro_batch_grad_norms_sq = [1.0, 2.0, 3.0]
        gns._accumulated_grad_norm_sq = 5.0

        gns.step(current_step=1)

        self.assertEqual(len(gns._micro_batch_grad_norms_sq), 0)
        self.assertEqual(gns._accumulated_grad_norm_sq, 0.0)

    def test_step_does_not_increase_before_check_interval(self):
        """grad_accum_steps should not change before check_interval."""
        gns = self.GNSEstimator(
            initial_grad_accum_steps=4,
            check_interval=100,
        )
        # Simulate a step before the check interval
        gns._micro_batch_grad_norms_sq = [1.0, 2.0, 3.0, 4.0]
        gns._accumulated_grad_norm_sq = 10.0

        result = gns.step(current_step=50)
        self.assertEqual(result, 4)  # Should not change

    def test_step_caps_at_max(self):
        """grad_accum_steps should never exceed max_grad_accum_steps."""
        gns = self.GNSEstimator(
            initial_grad_accum_steps=8,
            max_grad_accum_steps=16,
            check_interval=1,
            growth_factor=4.0,  # Would push to 32
        )
        # Force GNS to be very high relative to EMA
        gns.gns_ema = 1.0
        gns._micro_batch_grad_norms_sq = [1.0, 100.0]  # High variance
        gns._accumulated_grad_norm_sq = 1000.0

        result = gns.step(current_step=1)
        self.assertLessEqual(result, 16)

    def test_get_metrics(self):
        """get_metrics should return a dict with expected keys."""
        gns = self.GNSEstimator()
        metrics = gns.get_metrics()

        self.assertIn("pretrain/gns", metrics)
        self.assertIn("pretrain/gns_ema", metrics)
        self.assertIn("pretrain/grad_accum_steps", metrics)
        # Should be numeric values
        for v in metrics.values():
            self.assertIsInstance(v, (int, float))


# ==============================================================================
# 2. Benchmark Leakage Blocker Tests (语义指纹阻断防泄露测试)
# ==============================================================================

class TestBenchmarkLeakageBlocker(unittest.TestCase):
    """Tests for the Semantic Fingerprint Blocking feature."""

    def setUp(self):
        from utils.leakage_blocker import BenchmarkLeakageBlocker
        self.BlockerClass = BenchmarkLeakageBlocker

        # Create temporary eval data directory with sample benchmark data
        self.temp_dir = tempfile.mkdtemp()
        self.eval_dir = os.path.join(self.temp_dir, "eval")
        os.makedirs(self.eval_dir)

        # Write a sample MMLU-style benchmark file
        mmlu_data = [
            {"question": "What is the capital of France? The answer is Paris, which is known for the Eiffel Tower."},
            {"question": "Who discovered penicillin? Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital."},
            {"question": "What is photosynthesis? It is the process by which green plants convert sunlight into chemical energy."},
        ]
        with open(os.path.join(self.eval_dir, "mmlu.jsonl"), "w") as f:
            for item in mmlu_data:
                f.write(json.dumps(item) + "\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_initialization_with_data(self):
        """Blocker should build fingerprint set from eval data."""
        blocker = self.BlockerClass(eval_data_dir=self.eval_dir)
        self.assertGreater(len(blocker.fingerprint_set), 0)

    def test_initialization_missing_directory(self):
        """Blocker should handle missing eval directory gracefully."""
        blocker = self.BlockerClass(eval_data_dir="/nonexistent/path")
        self.assertEqual(len(blocker.fingerprint_set), 0)

    def test_initialization_empty_directory(self):
        """Blocker should handle empty eval directory gracefully."""
        empty_dir = os.path.join(self.temp_dir, "empty_eval")
        os.makedirs(empty_dir)
        blocker = self.BlockerClass(eval_data_dir=empty_dir)
        self.assertEqual(len(blocker.fingerprint_set), 0)

    def test_extract_ngrams_basic(self):
        """N-gram extraction should produce correct character-level n-grams."""
        blocker = self.BlockerClass(eval_data_dir="/nonexistent", ngram_size=5)
        ngrams = blocker._extract_ngrams("hello world")
        self.assertIn("hello", ngrams)
        self.assertIn("ello ", ngrams)
        self.assertIn("world", ngrams)
        self.assertEqual(len(ngrams), len("hello world") - 5 + 1)

    def test_extract_ngrams_normalization(self):
        """N-gram extraction should normalize text (lowercase, whitespace collapse)."""
        blocker = self.BlockerClass(eval_data_dir="/nonexistent", ngram_size=5)
        ngrams1 = blocker._extract_ngrams("Hello World")
        ngrams2 = blocker._extract_ngrams("hello  world")
        # Both should normalize to "hello world"
        self.assertEqual(ngrams1, ngrams2)

    def test_extract_ngrams_short_text(self):
        """N-gram extraction should return empty set for text shorter than n."""
        blocker = self.BlockerClass(eval_data_dir="/nonexistent", ngram_size=13)
        ngrams = blocker._extract_ngrams("short")
        self.assertEqual(len(ngrams), 0)

    def test_check_and_mask_no_tokenizer(self):
        """check_and_mask should be a no-op without a tokenizer."""
        blocker = self.BlockerClass(eval_data_dir=self.eval_dir, tokenizer=None)
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 10))

        new_input_ids, new_labels = blocker.check_and_mask(input_ids, labels)
        self.assertTrue(torch.equal(input_ids, new_input_ids))
        self.assertTrue(torch.equal(labels, new_labels))

    def test_check_and_mask_no_fingerprints(self):
        """check_and_mask should be a no-op without fingerprints."""
        blocker = self.BlockerClass(eval_data_dir="/nonexistent")
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 10))

        new_input_ids, new_labels = blocker.check_and_mask(input_ids, labels)
        self.assertTrue(torch.equal(input_ids, new_input_ids))
        self.assertTrue(torch.equal(labels, new_labels))

    def test_check_and_mask_does_not_modify_input_ids(self):
        """check_and_mask should never modify input_ids, only labels."""
        blocker = self.BlockerClass(eval_data_dir=self.eval_dir)
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 10))
        original_input_ids = input_ids.clone()

        new_input_ids, _ = blocker.check_and_mask(input_ids, labels)
        self.assertTrue(torch.equal(original_input_ids, new_input_ids))

    def test_check_and_mask_labels_not_modified_inplace(self):
        """check_and_mask should clone labels, not modify the original."""
        blocker = self.BlockerClass(eval_data_dir=self.eval_dir)
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 10))
        original_labels = labels.clone()

        # Even if masking occurs, original should be unchanged
        _, new_labels = blocker.check_and_mask(input_ids, labels)
        self.assertTrue(torch.equal(original_labels, labels))

    def test_check_and_mask_contaminated_sample(self):
        """Contaminated samples should have labels set to -100."""
        blocker = self.BlockerClass(
            eval_data_dir=self.eval_dir,
            ngram_size=13,
            contamination_threshold=0.1,
        )

        # Create a mock tokenizer that returns a known contaminated text
        class MockTokenizer:
            def decode(self, token_ids):
                return "What is the capital of France? The answer is Paris, which is known for the Eiffel Tower."

        blocker.tokenizer = MockTokenizer()

        input_ids = torch.randint(0, 100, (1, 20))
        labels = torch.ones(1, 20, dtype=torch.long)

        _, new_labels = blocker.check_and_mask(input_ids, labels)
        # All labels should be -100 (masked)
        self.assertTrue((new_labels == -100).all())

    def test_check_and_mask_clean_sample(self):
        """Clean samples should have labels unchanged."""
        blocker = self.BlockerClass(
            eval_data_dir=self.eval_dir,
            ngram_size=13,
            contamination_threshold=0.1,
        )

        class MockTokenizer:
            def decode(self, token_ids):
                return "This is completely unrelated training text about machine learning architectures and optimization techniques."

        blocker.tokenizer = MockTokenizer()

        input_ids = torch.randint(0, 100, (1, 20))
        labels = torch.ones(1, 20, dtype=torch.long)

        _, new_labels = blocker.check_and_mask(input_ids, labels)
        # Labels should remain unchanged
        self.assertTrue(torch.equal(new_labels, labels))

    def test_get_stats(self):
        """get_stats should return proper statistics dict."""
        blocker = self.BlockerClass(eval_data_dir=self.eval_dir)
        stats = blocker.get_stats()

        self.assertIn("pretrain/leakage_checked", stats)
        self.assertIn("pretrain/leakage_blocked", stats)
        self.assertIn("pretrain/leakage_block_rate", stats)
        self.assertIn("pretrain/fingerprint_count", stats)
        self.assertEqual(stats["pretrain/leakage_checked"], 0)
        self.assertEqual(stats["pretrain/leakage_blocked"], 0)

    def test_multiple_benchmark_fields(self):
        """Blocker should handle different field names (text, prompt, input)."""
        mixed_dir = os.path.join(self.temp_dir, "mixed_eval")
        os.makedirs(mixed_dir)

        mixed_data = [
            {"text": "Some evaluation benchmark text that is long enough for 13-grams to work properly"},
            {"prompt": "Another evaluation prompt text that is long enough for 13-grams to work properly"},
            {"input": "An input field used in some evaluation benchmark formats with enough characters"},
        ]
        with open(os.path.join(mixed_dir, "mixed.jsonl"), "w") as f:
            for item in mixed_data:
                f.write(json.dumps(item) + "\n")

        blocker = self.BlockerClass(eval_data_dir=mixed_dir)
        self.assertGreater(len(blocker.fingerprint_set), 0)


# ==============================================================================
# 3. LG-Opt Loss-Gradient Rescaler Tests (Loss 偏导梯度自适应重缩放测试)
# ==============================================================================

class TestLossGradientRescaler(unittest.TestCase):
    """Tests for the LG-Opt Loss-Gradient Decoupled Rescaling feature."""

    def setUp(self):
        from utils.lg_opt import LossGradientRescaler
        self.RescalerClass = LossGradientRescaler

    def test_initialization_defaults(self):
        """Test default initialization parameters."""
        rescaler = self.RescalerClass()
        self.assertEqual(rescaler.ema_alpha, 0.99)
        self.assertEqual(rescaler.high_deviation_threshold, 0.5)
        self.assertEqual(rescaler.low_deviation_threshold, 0.02)
        self.assertEqual(rescaler.high_deviation_scale, 0.1)
        self.assertEqual(rescaler.low_deviation_scale, 0.8)
        self.assertIsNone(rescaler.ema_loss)
        self.assertEqual(rescaler.step_count, 0)

    def test_update_ema_first_step(self):
        """First EMA update should set ema_loss to the current loss."""
        rescaler = self.RescalerClass()
        rescaler.update_ema(2.5)
        self.assertEqual(rescaler.ema_loss, 2.5)

    def test_update_ema_subsequent_steps(self):
        """EMA should smoothly track the loss over time."""
        rescaler = self.RescalerClass(ema_alpha=0.9)
        rescaler.update_ema(1.0)
        rescaler.update_ema(2.0)
        # EMA = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        self.assertAlmostEqual(rescaler.ema_loss, 1.1, places=5)

    def test_compute_scale_factor_during_warmup(self):
        """During warmup, scale factor should always be 1.0."""
        rescaler = self.RescalerClass(warmup_steps=10)
        rescaler.ema_loss = 1.0
        rescaler.step_count = 5  # Still in warmup

        scale = rescaler.compute_scale_factor(current_loss=5.0)
        self.assertEqual(scale, 1.0)

    def test_compute_scale_factor_high_deviation(self):
        """High deviation (dirty data) should return low scale factor."""
        rescaler = self.RescalerClass(
            high_deviation_threshold=0.5,
            high_deviation_scale=0.1,
            warmup_steps=0,
        )
        rescaler.ema_loss = 1.0
        rescaler.step_count = 10

        # Loss of 2.0 → delta = |2.0 - 1.0| / 1.0 = 1.0 > 0.5
        scale = rescaler.compute_scale_factor(current_loss=2.0)
        self.assertEqual(scale, 0.1)

    def test_compute_scale_factor_low_deviation(self):
        """Low deviation (redundant data) should return mild penalty scale."""
        rescaler = self.RescalerClass(
            low_deviation_threshold=0.02,
            low_deviation_scale=0.8,
            warmup_steps=0,
        )
        rescaler.ema_loss = 1.0
        rescaler.step_count = 10

        # Loss of 1.01 → delta = |1.01 - 1.0| / 1.0 = 0.01 < 0.02
        scale = rescaler.compute_scale_factor(current_loss=1.01)
        self.assertEqual(scale, 0.8)

    def test_compute_scale_factor_normal_range(self):
        """Normal deviation should return scale factor of 1.0."""
        rescaler = self.RescalerClass(
            high_deviation_threshold=0.5,
            low_deviation_threshold=0.02,
            warmup_steps=0,
        )
        rescaler.ema_loss = 1.0
        rescaler.step_count = 10

        # Loss of 1.2 → delta = |1.2 - 1.0| / 1.0 = 0.2 (between 0.02 and 0.5)
        scale = rescaler.compute_scale_factor(current_loss=1.2)
        self.assertEqual(scale, 1.0)

    def test_compute_scale_factor_zero_ema(self):
        """Should handle near-zero EMA gracefully (no division by zero)."""
        rescaler = self.RescalerClass(warmup_steps=0)
        rescaler.ema_loss = 0.0
        rescaler.step_count = 10

        scale = rescaler.compute_scale_factor(current_loss=1.0)
        self.assertEqual(scale, 1.0)  # Should not crash, returns 1.0

    def test_rescale_gradients_high_deviation(self):
        """Gradient rescaling should reduce gradient magnitudes for outlier batches."""
        rescaler = self.RescalerClass(
            high_deviation_threshold=0.5,
            high_deviation_scale=0.1,
            warmup_steps=0,
        )
        rescaler.ema_loss = 1.0
        rescaler.step_count = 10

        model = SimpleLinearModel()
        x = torch.randn(4, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Record original gradient norm
        original_grad_norm = sum(
            p.grad.float().pow(2).sum().item()
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        # Apply rescaling with outlier loss
        scale = rescaler.rescale_gradients(model, current_loss=2.0)
        self.assertEqual(scale, 0.1)

        # Check gradient norm was reduced
        new_grad_norm = sum(
            p.grad.float().pow(2).sum().item()
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        self.assertAlmostEqual(new_grad_norm, original_grad_norm * 0.1, places=3)

    def test_rescale_gradients_normal_range(self):
        """Gradients should not be modified in normal loss range."""
        rescaler = self.RescalerClass(warmup_steps=0)
        rescaler.ema_loss = 1.0
        rescaler.step_count = 10

        model = SimpleLinearModel()
        x = torch.randn(4, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Record original gradient
        original_grads = {
            name: p.grad.clone()
            for name, p in model.named_parameters() if p.grad is not None
        }

        # Apply rescaling with normal loss
        scale = rescaler.rescale_gradients(model, current_loss=1.1)
        self.assertEqual(scale, 1.0)

        # Check gradients are unchanged
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.allclose(p.grad, original_grads[name]))

    def test_rescale_gradients_updates_ema(self):
        """rescale_gradients should update the EMA after computing scale factor."""
        rescaler = self.RescalerClass(warmup_steps=0, ema_alpha=0.9)
        rescaler.ema_loss = 1.0
        rescaler.step_count = 10

        model = SimpleLinearModel()
        x = torch.randn(4, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()

        rescaler.rescale_gradients(model, current_loss=1.2)

        # EMA should have been updated: 0.9 * 1.0 + 0.1 * 1.2 = 1.02
        self.assertAlmostEqual(rescaler.ema_loss, 1.02, places=5)

    def test_get_metrics(self):
        """get_metrics should return a dict with expected keys."""
        rescaler = self.RescalerClass()
        metrics = rescaler.get_metrics()

        self.assertIn("pretrain/lg_opt_scale", metrics)
        self.assertIn("pretrain/lg_opt_delta", metrics)
        self.assertIn("pretrain/lg_opt_ema_loss", metrics)
        for v in metrics.values():
            self.assertIsInstance(v, (int, float))

    def test_latest_delta_tracking(self):
        """latest_delta should track the most recent deviation."""
        rescaler = self.RescalerClass(warmup_steps=0)
        rescaler.ema_loss = 1.0
        rescaler.step_count = 10

        rescaler.compute_scale_factor(current_loss=1.3)
        self.assertAlmostEqual(rescaler.latest_delta, 0.3, places=5)

    def test_latest_scale_factor_tracking(self):
        """latest_scale_factor should track the most recent scale."""
        rescaler = self.RescalerClass(
            warmup_steps=0,
            high_deviation_threshold=0.5,
            high_deviation_scale=0.1,
        )
        rescaler.ema_loss = 1.0
        rescaler.step_count = 10

        rescaler.compute_scale_factor(current_loss=2.0)
        self.assertEqual(rescaler.latest_scale_factor, 0.1)

        rescaler.compute_scale_factor(current_loss=1.1)
        self.assertEqual(rescaler.latest_scale_factor, 1.0)


# ==============================================================================
# 4. Integration-level Tests (集成测试)
# ==============================================================================

class TestPretrainInnovationIntegration(unittest.TestCase):
    """Integration tests verifying all three innovations work together."""

    def test_all_imports(self):
        """All three innovation modules should be importable."""
        from utils.gns_estimator import GradientNoiseScaleEstimator
        from utils.leakage_blocker import BenchmarkLeakageBlocker
        from utils.lg_opt import LossGradientRescaler

        # Instantiate all three
        gns = GradientNoiseScaleEstimator()
        blocker = BenchmarkLeakageBlocker(eval_data_dir="/nonexistent")
        lg = LossGradientRescaler()

        self.assertIsNotNone(gns)
        self.assertIsNotNone(blocker)
        self.assertIsNotNone(lg)

    def test_gns_and_lgopt_combined_training_step(self):
        """Simulate a training step with both GNS and LG-Opt active."""
        from utils.gns_estimator import GradientNoiseScaleEstimator
        from utils.lg_opt import LossGradientRescaler

        gns = GradientNoiseScaleEstimator(initial_grad_accum_steps=2)
        lg = LossGradientRescaler(warmup_steps=0)
        lg.ema_loss = 1.0
        lg.step_count = 10

        model = SimpleLinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Simulate 2 micro-batch accumulation
        for _ in range(2):
            optimizer.zero_grad()
            x = torch.randn(4, 32)
            out = model(x)
            loss = out.sum()
            loss.backward()
            gns.record_micro_batch_grad(model)

        gns.record_accumulated_grad(model)

        # LG-Opt rescale
        scale = lg.rescale_gradients(model, current_loss=1.1)
        self.assertEqual(scale, 1.0)

        # GNS step
        new_accum = gns.step(current_step=1)
        self.assertGreaterEqual(new_accum, 2)

        # Optimizer step should still work
        optimizer.step()

    def test_leakage_blocker_with_mock_batch(self):
        """Leakage blocker should work with tensor batches in training loop style."""
        from utils.leakage_blocker import BenchmarkLeakageBlocker

        temp_dir = tempfile.mkdtemp()
        eval_dir = os.path.join(temp_dir, "eval")
        os.makedirs(eval_dir)

        # Write benchmark data
        with open(os.path.join(eval_dir, "test_bench.jsonl"), "w") as f:
            f.write(json.dumps({"question": "A very specific benchmark question that should be detected by the blocker system"}) + "\n")

        try:
            class MockTokenizer:
                def decode(self, ids):
                    return "Completely unrelated text about neural network architectures"

            blocker = BenchmarkLeakageBlocker(
                eval_data_dir=eval_dir,
                tokenizer=MockTokenizer(),
            )

            # Clean data should pass through
            input_ids = torch.randint(0, 100, (4, 32))
            labels = torch.ones(4, 32, dtype=torch.long)

            _, new_labels = blocker.check_and_mask(input_ids, labels)
            self.assertTrue(torch.equal(new_labels, labels))
        finally:
            shutil.rmtree(temp_dir)

    def test_metrics_dict_keys_no_collision(self):
        """All three innovations should have non-overlapping metric keys."""
        from utils.gns_estimator import GradientNoiseScaleEstimator
        from utils.leakage_blocker import BenchmarkLeakageBlocker
        from utils.lg_opt import LossGradientRescaler

        gns = GradientNoiseScaleEstimator()
        blocker = BenchmarkLeakageBlocker(eval_data_dir="/nonexistent")
        lg = LossGradientRescaler()

        gns_keys = set(gns.get_metrics().keys())
        blocker_keys = set(blocker.get_stats().keys())
        lg_keys = set(lg.get_metrics().keys())

        # No key collisions between modules
        self.assertEqual(len(gns_keys & blocker_keys), 0, "GNS and Blocker have overlapping metric keys")
        self.assertEqual(len(gns_keys & lg_keys), 0, "GNS and LG-Opt have overlapping metric keys")
        self.assertEqual(len(blocker_keys & lg_keys), 0, "Blocker and LG-Opt have overlapping metric keys")


if __name__ == "__main__":
    unittest.main()
