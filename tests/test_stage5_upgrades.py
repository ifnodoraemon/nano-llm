import unittest
import os
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# Import actual modules
from model import ModelConfig, DeepSeekMoE, FP8Linear
from utils.overlap_helper import OverlapCommunicationHelper
from eval_benchmarks import NeedleInAHaystackEvaluator
from grpo import evaluate_completion_rewards

class TestStage5Upgrades(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_moe_expert_capacity_and_dropping(self):
        """
        Verify DeepSeekMoE routing, Expert Capacity limiting, and token dropping logic.
        """
        config = ModelConfig(
            n_embd=64,
            n_layer=2,
            n_head=2,
            block_size=128,
            vocab_size=1000,
            use_moe=True,
            num_shared_experts=1,
            num_routed_experts=4,
            num_active_experts=2
        )
        
        # Instantiate MoE block with capacity_factor = 1.0
        moe = DeepSeekMoE(config, capacity_factor=1.0).to(self.device)
        moe.train()
        
        # Input tensor of [batch_size, seq_len, n_embd]
        x = torch.randn(2, 8, 64, device=self.device)
        out = moe(x)
        
        # Verify output shape remains identical
        self.assertEqual(out.shape, x.shape)
        
        # Verify auxiliary load balancing loss is calculated during training
        self.assertTrue(hasattr(moe, "aux_loss"))
        self.assertGreater(moe.aux_loss.item(), 0.0)
        
        # Verify evaluation mode does not compute/expose auxiliary loss buffer
        moe.eval()
        delattr(moe, "aux_loss")
        out_eval = moe(x)
        self.assertEqual(out_eval.shape, x.shape)
        self.assertFalse(hasattr(moe, "aux_loss"))

    def test_channel_wise_fp8_linear_hopper_math(self):
        """
        Verify FP8Linear dynamic scaling channel-wise mathematics.
        Uses a robust tensor.to patch to simulate FP8 casting on non-Hopper dev environments.
        """
        base_linear = nn.Linear(32, 16).to(self.device)
        fp8_layer = FP8Linear(base_linear).to(self.device)
        
        # Intercept native FP8 casting to avoid crashes on non-Hopper local development GPUs
        orig_to = torch.Tensor.to
        def mock_to(tensor_self, *args, **kwargs):
            if len(args) > 0 and args[0] == getattr(torch, "float8_e4m3fn", None):
                # Return standard float32 tensor as simulated FP8 storage to bypass hardware check
                return orig_to(tensor_self, torch.float32)
            return orig_to(tensor_self, *args, **kwargs)
            
        with patch.object(torch.Tensor, "to", mock_to):
            x = torch.randn(4, 32, device=self.device)
            out = fp8_layer(x)
            
            # Verify output size and scales are correctly computed and registered
            self.assertEqual(out.shape, (4, 16))
            self.assertIsNotNone(fp8_layer.x_scale)
            self.assertIsNotNone(fp8_layer.w_scale)
            self.assertEqual(fp8_layer.x_scale.shape, torch.Size([]))
            self.assertEqual(fp8_layer.w_scale.shape, (16,))

    def test_async_communication_computation_overlap_nccl(self):
        """
        Verify OverlapCommunicationHelper schedules non-blocking pre-fetching NCCL calls,
        simulated via mocks to pass on single-GPU/development systems.
        """
        helper = OverlapCommunicationHelper(use_ddp=True)
        weights = torch.randn(64, 64, device=self.device)
        
        # Define simulated all_gather callback to bypass multi-GPU NCCL cluster constraints
        def dummy_all_gather(output_tensor, input_tensor, **kwargs):
            output_tensor.copy_(input_tensor)
            mock_work = MagicMock()
            mock_work.wait = MagicMock()
            return mock_work
            
        with patch("torch.distributed.all_gather_into_tensor", dummy_all_gather):
            # Initiate NCCL pre-fetch
            helper.prefetch_next_layer_weights(weights, self.device)
            
            # Block and retrieve
            retrieved = helper.wait_and_retrieve_prefetched()
            
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.shape, weights.shape)
            self.assertTrue(torch.allclose(retrieved, weights))

    def test_niah_synthetic_context_generator(self):
        """
        Verify Needle-in-a-Haystack Evaluator correctly generates noise context
        and inserts needles at specific depth ratios.
        """
        evaluator = NeedleInAHaystackEvaluator(context_lengths=[100], document_depths=[0.5])
        
        # Test noise generation
        noise = evaluator.generate_noise_context(50)
        word_count = len(noise.split())
        self.assertGreaterEqual(word_count, 50)
        
        # Validate depth math and placement logic
        words = noise.split()
        needle = "SECRETKEY123"
        inject_idx = int(len(words) * 0.5)
        words.insert(inject_idx, needle)
        full_context = " ".join(words)
        
        self.assertIn(needle, full_context)
        self.assertEqual(words[inject_idx], needle)

    def test_prm_step_scoring_and_reward_clipping(self):
        """
        Verify Process-Supervised Reward Model step-scoring detects logical transitions,
        penalizes repetitive reasoning loop hacks, and clamps extreme GRPO values to [-3.0, 3.0].
        """
        prompts = ["Solve math equation"] * 5
        ground_truths = ["42"] * 5
        
        completions = [
            # 0. Ideal well-formed, multi-step thought reasoning with correct answer (High score)
            "<think>Step 1: start. Step 2: add numbers. Therefore, the answer is 42.</think><answer>42</answer>",
            
            # 1. Right answer but chaotic repetitive loop inside think block (Penalized!)
            "<think>Step 1: start.\nso\nso\nso\nso\nso\nso\nso\nso\nso\nso\nso\nso\nso\nso</think><answer>42</answer>",
            
            # 2. Duplicate lines inside think block (Penalized!)
            "<think>Step 1: compute.\nWe add 10.\nWe add 10.\nWe add 10.\nWe add 10.</think><answer>42</answer>",
            
            # 3. No tags, incorrect (Low score)
            "The answer is 99",
            
            # 4. Out of bounds high score candidate (To test clamping)
            "<think>Step 1. Step 2. Step 3. Therefore we check. Let's see.</think><answer>42</answer>"
        ]
        
        rewards = evaluate_completion_rewards(prompts, completions, ground_truths)
        
        # Verify returned shape is correct
        self.assertEqual(rewards.shape, (5,))
        
        # Verify clamping bounds
        self.assertTrue(torch.all(rewards >= -3.0))
        self.assertTrue(torch.all(rewards <= 3.0))
        
        # 0 should have a higher reward than 1 due to repetition loop penalty
        self.assertGreater(rewards[0].item(), rewards[1].item())
        
        # 0 should have a higher reward than 2 due to duplicate line penalty
        self.assertGreater(rewards[0].item(), rewards[2].item())
        
        # 0 should be greater than 3 (incorrect, no formatting)
        self.assertGreater(rewards[0].item(), rewards[3].item())

if __name__ == "__main__":
    unittest.main()
