"""
Smoke test: runs a minimal end-to-end pipeline to verify all components work.
Uses the "tiny" model config (4L/8H/512E, no MLA/MoE) for fast execution.
"""

import os
import sys
import json
import tempfile
import unittest
import torch
from contextlib import redirect_stdout, redirect_stderr
import io


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestSmokePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_01_model_instantiation(self):
        """Verify model creation with default config."""
        from model import ModelConfig, Transformer

        config = ModelConfig(
            block_size=128,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            use_mla=False,
            use_moe=False,
        )
        model = Transformer(config).to(self.device)
        self.assertEqual(model.config.n_layer, 2)
        self.assertEqual(model.config.n_embd, 128)

    def test_02_forward_pass(self):
        """Verify a single forward pass works."""
        from model import ModelConfig, Transformer

        config = ModelConfig(
            block_size=128,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            use_mla=False,
            use_moe=False,
        )
        model = Transformer(config).to(self.device)
        tokens = torch.randint(0, 1000, (2, 64), device=self.device)
        logits, loss, aux_loss = model(tokens, targets=tokens)
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)

    def test_03_mla_forward_pass(self):
        """Verify MLA mode forward pass."""
        from model import ModelConfig, Transformer

        config = ModelConfig(
            block_size=128,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            use_mla=True,
            kv_comp_dim=32,
            use_moe=False,
        )
        model = Transformer(config).to(self.device)
        tokens = torch.randint(0, 1000, (2, 64), device=self.device)
        logits, loss, aux_loss = model(tokens, targets=tokens)
        self.assertIsNotNone(loss)

    def test_04_moe_forward_pass(self):
        """Verify MoE mode forward pass with aux_loss."""
        from model import ModelConfig, Transformer

        config = ModelConfig(
            block_size=128,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            use_mla=False,
            use_moe=True,
            num_routed_experts=4,
            num_active_experts=2,
        )
        model = Transformer(config).to(self.device).train()
        tokens = torch.randint(0, 1000, (2, 64), device=self.device)
        logits, loss, aux_loss = model(tokens, targets=tokens)
        self.assertIsNotNone(loss)

    def test_05_export_dora(self):
        """Verify export runs without error."""
        from model import ModelConfig, Transformer

        config = ModelConfig(
            block_size=128,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            use_mla=False,
            use_moe=False,
        )
        model = Transformer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
            }, ckpt_path)

            from export_dora import export
            export(checkpoint_path=ckpt_path, dest_dir=os.path.join(tmpdir, "export"))

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "export", "config.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "export", "model.safetensors")))

            # Verify model_type is nano-lm
            with open(os.path.join(tmpdir, "export", "config.json")) as f:
                export_cfg = json.load(f)
            self.assertEqual(export_cfg["model_type"], "nano-lm")

    def test_06_config_system(self):
        """Verify config system loads correctly."""
        from config import load_config, MODEL_PRESETS, detect_hardware

        cfg = load_config(mode="dev", model_size="tiny")
        self.assertEqual(cfg.mode, "dev")
        self.assertEqual(cfg.model_size, "tiny")

        hw = detect_hardware()
        self.assertIsNotNone(hw.gpu_name)

        self.assertIn("16B-equivalent", MODEL_PRESETS)
        self.assertIn("1B-equivalent", MODEL_PRESETS)

    def test_07_tokenizer_loader(self):
        """Verify tokenizer loader works."""
        # Should fall back to AutoTokenizer when no custom tokenizer exists
        from utils.tokenizer_loader import load_tokenizer
        tokenizer = load_tokenizer(fallback_model_name="gpt2")
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(tokenizer.pad_token_id)

    def test_08_checkpoint_utils(self):
        """Verify checkpoint utility functions."""
        from utils.checkpoint_utils import translate_fp8_keys, load_checkpoint_with_fp8_translation
        from model import ModelConfig

        # Test FP8 key translation
        state_dict = {
            "base_linear.weight": torch.randn(64, 64),
            "base_linear.bias": torch.randn(64),
            "w_scale": torch.tensor(1.0),
            "normal.weight": torch.randn(64, 64),
        }
        translated = translate_fp8_keys(state_dict)
        self.assertIn("weight", translated)
        self.assertIn("bias", translated)
        self.assertNotIn("w_scale", translated)
        self.assertIn("normal.weight", translated)
        self.assertEqual(translated["weight"].dtype, torch.float16)

    def test_09_deepseek_config_presets(self):
        """Verify all DeepSeek config presets are valid."""
        from model import get_deepseek_config

        for size in ["tiny", "small", "medium", "1B-equivalent", "3B-equivalent", "7B-equivalent", "16B-equivalent"]:
            config = get_deepseek_config(size)
            self.assertIsNotNone(config)
            self.assertEqual(config.vocab_size, 32000)

    def test_10_nano_lm_config(self):
        """Verify NanoLMConfig is HF-compatible."""
        from configuration_nano_lm import NanoLMConfig

        config = NanoLMConfig(
            n_layer=12,
            n_head=16,
            n_embd=1024,
            use_mla=True,
            use_moe=True,
        )
        self.assertEqual(config.model_type, "nano-lm")
        self.assertGreater(config.intermediate_size, 0)
        self.assertEqual(config.head_dim, 64)


if __name__ == "__main__":
    unittest.main()
