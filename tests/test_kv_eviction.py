import unittest
import torch
from utils.kv_eviction import H2OKVCacheEvictor, StreamingLLMEvictor

class TestKVEviction(unittest.TestCase):
    def test_streaming_llm_eviction(self):
        evictor = StreamingLLMEvictor(num_sinks=4, recent_window=10)
        self.assertEqual(evictor.max_cache_size, 14)
        
        # Mock KV cache: [Batch=1, Num_heads=2, Seq_len=20, Head_dim=8]
        k_cache = torch.randn(1, 2, 20, 8)
        v_cache = torch.randn(1, 2, 20, 8)
        
        comp_k, comp_v = evictor.evict_kv_cache(k_cache, v_cache)
        
        # Verify shape has compressed to max_cache_size (14)
        self.assertEqual(comp_k.shape, (1, 2, 14, 8))
        self.assertEqual(comp_v.shape, (1, 2, 14, 8))
        
        # Verify sinks (first 4 elements) are preserved identically
        self.assertTrue(torch.equal(comp_k[:, :, :4, :], k_cache[:, :, :4, :]))
        # Verify sliding window (last 10 elements) are preserved identically from original suffix
        self.assertTrue(torch.equal(comp_k[:, :, 4:, :], k_cache[:, :, 10:, :]))

    def test_h2o_eviction(self):
        evictor = H2OKVCacheEvictor(max_cache_size=16, num_sinks=4, recent_window=8)
        self.assertEqual(evictor.heavy_hitters_budget, 4)
        
        # Mock KV cache and attention scores
        # k_cache shape: [1, 2, 24, 8]
        # attn_scores shape: [1, 2, 24]
        k_cache = torch.randn(1, 2, 24, 8)
        v_cache = torch.randn(1, 2, 24, 8)
        
        # Set specific middle indices to have very high attention scores
        attn_scores = torch.zeros(1, 2, 24)
        attn_scores[:, :, 6] = 999.0
        attn_scores[:, :, 7] = 999.0
        attn_scores[:, :, 8] = 999.0
        attn_scores[:, :, 9] = 999.0
        
        comp_k, comp_v = evictor.evict_kv_cache(k_cache, v_cache, attn_scores)
        
        # Verify compressed size is 16
        self.assertEqual(comp_k.shape, (1, 2, 16, 8))
        self.assertEqual(comp_v.shape, (1, 2, 16, 8))
        
        # Verify sinks (first 4) are preserved
        self.assertTrue(torch.equal(comp_k[:, :, :4, :], k_cache[:, :, :4, :]))

if __name__ == "__main__":
    unittest.main()
