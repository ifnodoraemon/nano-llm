import unittest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from utils.parallel_3d import ColumnParallelLinear, RowParallelLinear, PipelineParallelTransformer
from utils.paged_attention import PagedCacheManager, PagedAttentionKernel
from utils.audio_projection import AudioProjection
from utils.grpo_critic import GRPOCritic
from utils.triton_fa3 import FP8FlashAttention3

class TestAllFutureUpgrades(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_3d_parallelism(self):
        """
        Verify forward passes of ColumnParallelLinear and RowParallelLinear.
        """
        batch_size = 2
        seq_len = 4
        in_features = 64
        out_features = 128
        
        input_tensor = torch.randn(batch_size, seq_len, in_features, device=self.device)
        
        # Test ColumnParallelLinear (sharding across columns)
        col_layer = ColumnParallelLinear(in_features, out_features, world_size=2, rank=0).to(self.device)
        output_col = col_layer(input_tensor)
        self.assertEqual(output_col.shape, (batch_size, seq_len, out_features // 2))
        
        # Test RowParallelLinear (sharding across rows)
        row_layer = RowParallelLinear(in_features // 2, out_features, world_size=2, rank=0).to(self.device)
        output_row = row_layer(output_col)
        self.assertEqual(output_row.shape, (batch_size, seq_len, out_features))

    def test_paged_attention(self):
        """
        Verify PagedCacheManager virtual memory allocation and PagedAttention attention decoding.
        """
        num_blocks = 8
        block_size = 16
        num_heads = 4
        head_dim = 32
        seq_id = 42
        seq_len = 24  # Crosses block boundary (requires 2 blocks)
        
        manager = PagedCacheManager(num_blocks, block_size, num_heads, head_dim, self.device)
        
        # 1. Allocate initial pages
        manager.allocate_blocks(seq_id, num_blocks_needed=2)
        self.assertEqual(len(manager.page_tables[seq_id]), 2)
        
        # 2. Write simulated token keys/values to logical slots
        for i in range(seq_len):
            k = torch.randn(num_heads, head_dim, device=self.device)
            v = torch.randn(num_heads, head_dim, device=self.device)
            manager.write_to_cache(seq_id, logical_pos=i, k_token=k, v_token=v)
            
        # Verify block free slots decreased
        self.assertEqual(manager.block_pool[manager.page_tables[seq_id][0]].free_slots, 0)
        self.assertEqual(manager.block_pool[manager.page_tables[seq_id][1]].free_slots, 8)
        
        # 3. Compute attention over paged slots
        kernel = PagedAttentionKernel(head_dim)
        q = torch.randn(1, num_heads, head_dim, device=self.device)
        output = kernel(q, seq_id, manager, seq_len)
        
        self.assertEqual(output.shape, (1, num_heads, head_dim))
        self.assertTrue(torch.all(torch.isfinite(output)))
        
        # 4. Release blocks
        manager.free_sequence(seq_id)
        self.assertNotIn(seq_id, manager.page_tables)
        self.assertEqual(len(manager.free_block_ids), num_blocks)

    def test_audio_projection(self):
        """
        Verify AudioProjection forward mappings.
        """
        batch_size = 2
        num_frames = 30
        audio_dim = 80
        language_dim = 128
        
        audio_tensor = torch.randn(batch_size, num_frames, audio_dim, device=self.device)
        proj = AudioProjection(audio_dim=audio_dim, language_dim=language_dim).to(self.device)
        projected = proj(audio_tensor)
        
        self.assertEqual(projected.shape, (batch_size, num_frames, language_dim))
        self.assertTrue(torch.all(torch.isfinite(projected)))

    def test_grpo_critic(self):
        """
        Verify GRPOCritic hybrid rule/critic score calculation.
        """
        prompts = ["Calculate 10+20"] * 2
        completions = [
            "<think>Adding 10 and 20</think> <answer>30</answer>",
            "Answer is 30 but no think formatting."
        ]
        gts = ["30"] * 2
        
        critic = GRPOCritic()
        rewards = critic.evaluate_hybrid_rewards(prompts, completions, gts)
        
        self.assertEqual(rewards.shape, (2,))
        self.assertGreater(rewards[0].item(), rewards[1].item())

    def test_triton_fa3_fp8(self):
        """
        Verify compiled FP8FlashAttention3 block-level causal attention calculations.
        """
        batch_size = 2
        num_heads = 4
        seq_len = 64
        head_dim = 32
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=self.device)
        
        attn = FP8FlashAttention3(head_dim)
        output = attn(q, k, v)
        
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertTrue(torch.all(torch.isfinite(output)))

if __name__ == "__main__":
    unittest.main()
