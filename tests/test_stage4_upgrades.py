import unittest
import os
import json
import torch
import shutil
import time
from utils.dist_helper import autotune_nccl
from utils.checkpoint_saver import BackgroundCheckpointSaver, ElasticRestoreManager
from deduplicate import MinHashDeduplicator, HighFidelityQualityFilter
from utils.paged_attention import PagedCacheManager, PagedAttentionKernel

class TestStage4Upgrades(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./tests/test_outputs"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_nccl_autotune(self):
        # Run autotuning
        autotune_nccl()
        # Verify default vars are injected
        self.assertEqual(os.environ.get("NCCL_DEBUG"), "INFO")
        self.assertEqual(os.environ.get("NCCL_BUFFSIZE"), "4194304")

    def test_background_checkpoint_and_restore(self):
        # Create a simple mock model and optimizer
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
            def forward(self, x):
                return self.linear(x)
                
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        saver = BackgroundCheckpointSaver()
        restore_mgr = ElasticRestoreManager(self.test_dir)
        
        # Verify no checkpoints initially
        has_ckpt, _, _ = restore_mgr.auto_detect_checkpoint()
        self.assertFalse(has_ckpt)
        
        # Save checkpoint asynchronously
        success = saver.save_checkpoint(
            model=model,
            optimizer=optimizer,
            lr_scheduler=None,
            config=None,
            step=42,
            epoch=1,
            loss=0.314,
            out_dir=self.test_dir
        )
        self.assertTrue(success)
        
        # Wait up to 1 second for the background thread to finish writing to disk
        for _ in range(20):
            if os.path.exists(restore_mgr.filepath) and os.path.exists(restore_mgr.manifestpath):
                break
            time.sleep(0.05)
            
        # Verify auto-detect finds it now
        has_ckpt, step, epoch = restore_mgr.auto_detect_checkpoint()
        self.assertTrue(has_ckpt)
        self.assertEqual(step, 42)
        self.assertEqual(epoch, 1)
        
        # Test hot restoration
        restored_step, restored_epoch, restored_loss = restore_mgr.restore_training_state(model, optimizer)
        self.assertEqual(restored_step, 42)
        self.assertEqual(restored_epoch, 1)
        self.assertAlmostEqual(restored_loss, 0.314)

    def test_minhash_lsh_and_quality_filter(self):
        dedup = MinHashDeduplicator(num_hashes=64, shingle_size=3, num_bands=8, rows_per_band=8)
        q_filter = HighFidelityQualityFilter(min_words=5, max_words=50000)
        
        # 1. Test quality filter
        good_text = "The quick brown fox jumps over the lazy dog. A wonderful sunny day in San Francisco."
        bad_short = "Short."
        bad_symbol = "##### $$$$ @@@@ **** &&&&"
        
        self.assertTrue(q_filter.is_high_quality(good_text)[0])
        self.assertFalse(q_filter.is_high_quality(bad_short)[0])
        self.assertFalse(q_filter.is_high_quality(bad_symbol)[0])
        
        # 2. Test MinHash signature & LSH
        sig1 = dedup.compute_signature(dedup.get_shingles(good_text))
        similar_text = "The quick brown fox jumped over that lazy dog! Beautiful sunny day in San Francisco."
        sig2 = dedup.compute_signature(dedup.get_shingles(similar_text))
        
        similarity = dedup.estimate_jaccard(sig1, sig2)
        self.assertTrue(similarity > 0.6) # Should be highly similar
        
        # LSH bucket check: similar texts should share at least one band bucket
        buckets1 = dedup.get_lsh_buckets(sig1)
        buckets2 = dedup.get_lsh_buckets(sig2)
        
        shared_buckets = set(buckets1).intersection(set(buckets2))
        self.assertTrue(len(shared_buckets) > 0) # Candidate generation success

    def test_paged_attention_allocations(self):
        n_kv_heads = 4
        head_dim = 64
        block_size = 16
        
        cm = PagedCacheManager(num_blocks=10, block_size=block_size, num_heads=n_kv_heads, head_dim=head_dim, device="cpu")
        
        # Allocate 2 blocks for sequence 0
        allocated = cm.allocate_blocks(seq_id=0, num_blocks_needed=2)
        self.assertEqual(len(allocated), 2)
        self.assertEqual(len(cm.free_block_ids), 8)
        self.assertIn(0, cm.page_tables)
        self.assertEqual(len(cm.page_tables[0]), 2)
        
        # Write to slot
        k_token = torch.randn(n_kv_heads, head_dim, dtype=torch.float16)
        v_token = torch.randn(n_kv_heads, head_dim, dtype=torch.float16)
        cm.write_to_cache(seq_id=0, logical_pos=0, k_token=k_token, v_token=v_token)
        
        # Ensure free slots count updated on the block
        p_block_id = cm.page_tables[0][0]
        self.assertEqual(cm.block_pool[p_block_id].free_slots, block_size - 1)
        
        # Free sequence and verify block pool recovery
        cm.free_sequence(seq_id=0)
        self.assertNotIn(0, cm.page_tables)
        self.assertEqual(len(cm.free_block_ids), 10)

if __name__ == "__main__":
    unittest.main()
