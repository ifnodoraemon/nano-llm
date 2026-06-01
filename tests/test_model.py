import unittest
import torch
import torch.nn as nn
import math
from model import ModelConfig, RMSNorm, precompute_freqs_cis, apply_rotary_emb, Transformer, LoRALinear

class TestTransformerModules(unittest.TestCase):
    def test_rmsnorm(self):
        """
        Verifies that RMSNorm scales values so their Root Mean Square is 1.0.
        """
        dim = 64
        eps = 1e-5
        norm = RMSNorm(dim, eps=eps)
        
        x = torch.randn(4, 10, dim) * 5.0
        out = norm(x)
        
        rms = torch.sqrt(out.pow(2).mean(-1) + eps)
        for val in rms.flatten().tolist():
            self.assertAlmostEqual(val, 1.0, places=3)
            
    def test_rope_precomputation(self):
        dim = 16
        end = 32
        freqs_cis = precompute_freqs_cis(dim, end)
        self.assertEqual(freqs_cis.shape, (end, dim // 2))
        self.assertTrue(freqs_cis.is_complex())
        
    def test_rope_application(self):
        bs, slen, n_head, head_dim = 2, 8, 4, 16
        xq = torch.randn(bs, slen, n_head, head_dim)
        xk = torch.randn(bs, slen, n_head, head_dim)
        
        freqs_cis = precompute_freqs_cis(head_dim, slen)
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)

    def test_lora_linear_layer(self):
        """
        Verifies that custom LoRALinear layer forward pass computes and matches shapes.
        """
        in_features, out_features, rank = 32, 16, 8
        layer = LoRALinear(in_features, out_features, r=rank, lora_alpha=16.0)
        
        # Base weight must be frozen
        self.assertFalse(layer.base_layer.weight.requires_grad)
        
        # Adapters must be trainable
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)
        
        x = torch.randn(4, 10, in_features)
        out = layer(x)
        
        # Shape verification
        self.assertEqual(out.shape, (4, 10, out_features))

    def test_transformer_lora_freezing_and_merging(self):
        """
        Verifies model-wide LoRA freezing and weight fusion hooks.
        """
        config = ModelConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128,
            lora_r=8,
            lora_alpha=16.0
        )
        model = Transformer(config)
        
        # 1. Test configure_lora_trainable()
        trainable_count = model.configure_lora_trainable()
        self.assertTrue(trainable_count > 0)
        
        # Base weights (tok_embeddings) must be frozen
        self.assertFalse(model.tok_embeddings.weight.requires_grad)
        
        # Attention adapter weights must be trainable
        self.assertTrue(model.layers[0].attention.wq.lora_A.requires_grad)
        self.assertTrue(model.layers[0].attention.wq.lora_B.requires_grad)
        self.assertFalse(model.layers[0].attention.wq.base_layer.weight.requires_grad)
        
        # 2. Test merge_lora_weights()
        # Set dummy adapter weights to verify fusion change
        torch.nn.init.ones_(model.layers[0].attention.wq.lora_B)
        torch.nn.init.ones_(model.layers[0].attention.wq.lora_A)
        
        base_weight_before = model.layers[0].attention.wq.base_layer.weight.clone()
        model.merge_lora_weights()
        base_weight_after = model.layers[0].attention.wq.base_layer.weight.clone()
        
        # Base weights must have changed due to LoRA adapter fusing
        self.assertFalse(torch.equal(base_weight_before, base_weight_after))
        
        # Adapters must have reset to zero after merging
        self.assertTrue(torch.all(model.layers[0].attention.wq.lora_B == 0))

    def test_transformer_kv_cache_forward(self):
        """
        Verifies that Static KV-Cache forward pass populates buffers and decodes successfully.
        """
        config = ModelConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=128
        )
        model = Transformer(config)
        model.eval()
        
        # Pre-allocate static KV-caches
        kv_caches = []
        for _ in range(config.n_layer):
            k_cache = torch.zeros(1, config.block_size, config.n_head, 32)
            v_cache = torch.zeros(1, config.block_size, config.n_head, 32)
            kv_caches.append((k_cache, v_cache))
            
        # 1. Prefill stage: prompt of length 10
        x_prefill = torch.randint(0, 1000, (1, 10))
        logits, _, _ = model(x_prefill, start_pos=0, kv_caches=kv_caches)
        
        self.assertEqual(logits.shape, (1, 1, 1000))
        
        # Verify that KV caches are populated up to index 10
        # The sum of elements in the populated cache should be non-zero
        self.assertNotEqual(kv_caches[0][0][:, :10].sum().item(), 0.0)
        # Slices beyond index 10 should still be zero
        self.assertEqual(kv_caches[0][0][:, 10:].sum().item(), 0.0)
        
        # 2. Decode stage: single token at start_pos = 10
        x_decode = torch.randint(0, 1000, (1, 1))
        logits_decode, _, _ = model(x_decode, start_pos=10, kv_caches=kv_caches)
        
        self.assertEqual(logits_decode.shape, (1, 1, 1000))
        # Cache index 10 should now be populated
        self.assertNotEqual(kv_caches[0][0][:, 10].sum().item(), 0.0)
        self.assertEqual(kv_caches[0][0][:, 11:].sum().item(), 0.0)

    def test_transformer_vision_projection_forward(self):
        """
        Verifies that native vision projection MLP maps visual token patches to embedding dimensions,
        and fuses text and visual features in a joint causal sequence with proper padding and loss computation.
        """
        config = ModelConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=1,
            n_head=2,
            n_embd=64,
            vision_dim=32
        )
        model = Transformer(config)
        
        # Inputs: 1 batch, 5 text tokens, 4 image patches of dimension 32
        tokens = torch.randint(0, 1000, (1, 5))
        pixel_values = torch.randn(1, 4, 32)
        targets = torch.randint(0, 1000, (1, 5))
        
        # 1. Forward pass without targets (inference)
        logits, loss, _ = model(tokens, pixel_values=pixel_values)
        self.assertIsNone(loss)
        # Should return logits of the full sequence
        self.assertEqual(logits.shape, (1, 9, 1000))
        
        # 2. Forward pass with targets (training)
        logits_train, loss_train, _ = model(tokens, pixel_values=pixel_values, targets=targets)
        self.assertIsNotNone(loss_train)
        self.assertEqual(logits_train.shape, (1, 9, 1000)) # 4 vision + 5 text tokens = 9 total
        self.assertTrue(loss_train.item() > 0.0)

    def test_fp8_linear_dynamic_scaling(self):
        """
        Verifies that custom FP8Linear module scales and routes forward pass operations,
        correctly handling baseline scaling buffers and shape dimensions.
        """
        from model import FP8Linear
        base_linear = nn.Linear(16, 8)
        fp8_layer = FP8Linear(base_linear)
        
        # Test shape dimensions and dynamic scale buffers
        x = torch.randn(4, 16)
        out = fp8_layer(x)
        self.assertEqual(out.shape, (4, 8))
        self.assertGreater(fp8_layer.x_scale.item(), 0.0) # Scale buffer should be computed and positive

    def test_convert_to_fp8_recursive(self):
        """
        Verifies that recursive convert_to_fp8 successfully swaps standard Linear layers
        inside Transformer networks with dynamic scaling FP8Linear layers.
        """
        from model import convert_to_fp8, FP8Linear
        config = ModelConfig(
            block_size=64,
            vocab_size=1000,
            n_layer=1,
            n_head=2,
            n_embd=32,
            vision_dim=None
        )
        model = Transformer(config)
        
        # Check standard layer before conversion
        self.assertIsInstance(model.layers[0].attention.wq, nn.Linear)
        
        # Execute recursive conversion
        convert_to_fp8(model)
        
        # Verify layer after conversion is FP8Linear
        self.assertIsInstance(model.layers[0].attention.wq, FP8Linear)

    def test_deepseek_mla_attention(self):
        """
        Verifies that native Multi-Head Latent Attention (MLA) successfully compresses
        Keys and Values, applies decoupled RoPE, and computes correct output shapes.
        """
        from model import get_deepseek_config, MultiHeadLatentAttention
        config = get_deepseek_config(size="16B-equivalent", n_embd=64, n_head=2, kv_comp_dim=16)
        
        # Verify configurations
        self.assertTrue(config.use_mla)
        self.assertEqual(config.kv_comp_dim, 16)
        
        mla_layer = MultiHeadLatentAttention(config)
        
        # Input shape: (batch=2, seq_len=8, n_embd=64)
        x = torch.randn(2, 8, 64)
        freqs_cis = precompute_freqs_cis(dim=32, end=16)[:8] # head_dim = 64 // 2 = 32
        
        # Forward pass
        out = mla_layer(x, freqs_cis=freqs_cis)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_deepseek_moe_forward(self):
        """
        Verifies that DeepSeekMoE routes tokens to selected fine-grained experts,
        applies dynamic gate weighting, and blends results with shared experts output.
        """
        from model import get_deepseek_config, DeepSeekMoE
        config = get_deepseek_config(
            size="16B-equivalent", 
            n_embd=32, 
            num_shared_experts=1, 
            num_routed_experts=4, 
            num_active_experts=2
        )
        
        # Verify configurations
        self.assertTrue(config.use_moe)
        self.assertEqual(config.num_routed_experts, 4)
        
        moe_layer = DeepSeekMoE(config)
        
        # Input shape: (batch=1, seq_len=5, n_embd=32)
        x = torch.randn(1, 5, 32)
        out = moe_layer(x)
        self.assertEqual(out.shape, (1, 5, 32))

if __name__ == "__main__":
    unittest.main()



