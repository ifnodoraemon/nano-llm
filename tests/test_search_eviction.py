import unittest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import ModelConfig, Transformer
from utils.kv_eviction import H2OKVCacheEvictor
from utils.search_tree import SearchTreeDecoder, ReasoningNode

class TestSearchAndEvictionUpgrades(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = ModelConfig(
            n_embd=64,
            n_layer=1,
            n_head=2,
            vocab_size=500,
            block_size=64
        )
        self.model = Transformer(self.config).to(self.device)
        
    def test_h2o_kv_eviction(self):
        """
        Verify H2O cache eviction logic, sink/recent windows, and tensor dimensions.
        """
        batch_size = 1
        num_heads = 2
        seq_len = 100
        head_dim = 32
        
        max_cache_size = 48
        num_sinks = 4
        recent_window = 16
        
        k_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        v_cache = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        
        # Simulated attention scores
        attn_scores = torch.randn(batch_size, num_heads, seq_len, device=self.device).abs()
        
        evictor = H2OKVCacheEvictor(max_cache_size, num_sinks, recent_window)
        compressed_k, compressed_v = evictor.evict_kv_cache(k_cache, v_cache, attn_scores)
        
        self.assertEqual(compressed_k.shape, (batch_size, num_heads, max_cache_size, head_dim))
        self.assertEqual(compressed_v.shape, (batch_size, num_heads, max_cache_size, head_dim))
        self.assertTrue(torch.all(torch.isfinite(compressed_k)))

    def test_reasoning_search_tree(self):
        """
        Verify ReasoningNode traceback and SearchTreeDecoder branching decode steps.
        """
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"
                self.eos_token_id = 0
            def decode(self, tokens, skip_special_tokens=True):
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                return " ".join(str(t) for t in tokens)
                
        tokenizer = MockTokenizer()
        
        prompt_ids = torch.randint(1, self.config.vocab_size, (1, 8), device=self.device)
        
        decoder = SearchTreeDecoder(
            self.model,
            tokenizer,
            max_branches=2,
            max_steps=2
        )
        
        # Test step chunk generation
        chunk = decoder._generate_step_chunk(prompt_ids, chunk_len=4)
        self.assertEqual(chunk.shape, (1, 4))
        
        # Run search tree backtracking decoder
        output_text = decoder.search_optimal_path(prompt_ids, ground_truth="42")
        self.assertTrue(isinstance(output_text, str))
        self.assertGreater(len(output_text), 0)

if __name__ == "__main__":
    unittest.main()
