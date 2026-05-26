import unittest
import torch
from utils.rag_retriever import ChunkProcessor, DenseRetriever, SparseRetriever, HybridRetriever
from utils.kv_eviction import H2OKVCacheEvictor, StreamingLLMEvictor

class TestRagKvEviction(unittest.TestCase):
    def test_chunk_processor(self):
        processor = ChunkProcessor(chunk_size=10, chunk_overlap=2)
        text = "This is a very long sentence. It has many words. We want to chunk it correctly."
        chunks = processor.split_text(text)
        
        self.assertGreater(len(chunks), 1)
        self.assertIn("This is a very long sentence.", chunks[0])

    def test_dense_retriever_pytorch(self):
        retriever = DenseRetriever(embed_dim=32)
        chunks = [
            "Attention is all you need for deep learning.",
            "Recurrent neural networks process sequential data.",
            "Convolutional neural networks are great for computer vision."
        ]
        retriever.fit(chunks)
        
        # Query semantic search
        results = retriever.retrieve("deep learning neural", top_k=2)
        self.assertEqual(len(results), 2)
        # Verify first match is chunk 0 or 1 (contains neural/learning)
        first_match_idx = results[0][0]
        self.assertIn(first_match_idx, [0, 1])

    def test_sparse_retriever_bm25(self):
        retriever = SparseRetriever()
        chunks = [
            "Attention is all you need for deep learning.",
            "Recurrent neural networks process sequential data.",
            "Convolutional networks are great for computer vision."
        ]
        retriever.fit(chunks)
        
        results = retriever.retrieve("computer vision", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 2) # Matches the third document containing "computer vision"

    def test_hybrid_retriever_rrf(self):
        retriever = HybridRetriever(rrf_k=60)
        chunks = [
            "Attention is all you need for deep learning.",
            "Recurrent neural networks process sequential data.",
            "Convolutional networks are great for computer vision."
        ]
        retriever.fit(chunks)
        
        results = retriever.retrieve("deep learning", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], chunks[0])

    def test_h2o_kv_eviction(self):
        # max_cache_size = 8, sinks = 2, recent = 2, heavy hitters budget = 8 - 2 - 2 = 4
        evictor = H2OKVCacheEvictor(max_cache_size=8, num_sinks=2, recent_window=2)
        
        # k_cache, v_cache shape: [Batch=1, Heads=1, Seq_len=12, Head_dim=4]
        k_cache = torch.randn(1, 1, 12, 4)
        v_cache = torch.randn(1, 1, 12, 4)
        
        # Attention scores sum across positions
        attn_scores = torch.zeros(1, 1, 12)
        # Make tokens 4 and 5 heavy hitters
        attn_scores[0, 0, 4] = 10.0
        attn_scores[0, 0, 5] = 10.0
        
        comp_k, comp_v = evictor.evict_kv_cache(k_cache, v_cache, attn_scores)
        self.assertEqual(comp_k.shape, (1, 1, 8, 4))
        self.assertEqual(comp_v.shape, (1, 1, 8, 4))

if __name__ == "__main__":
    unittest.main()
