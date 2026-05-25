import torch
from typing import Tuple

# ==============================================================================
# H2O (Heavy Hitter Oracle) KV-Cache Eviction Policy Manager
# ==============================================================================

class H2OKVCacheEvictor:
    """
    H2O (Heavy Hitter Oracle) KV-Cache Eviction Manager.
    Allows infinite-length context serving in a fixed memory budget by dynamically
    evicting low-attention tokens while preserving:
    1. Attention Sinks (the first few critical system prompt tokens).
    2. Heavy Hitters (tokens that accumulated the highest attention weights).
    3. Recent Tokens (local context sliding window).
    """
    def __init__(
        self, 
        max_cache_size: int = 512, 
        num_sinks: int = 4, 
        recent_window: int = 32
    ):
        self.max_cache_size = max_cache_size
        self.num_sinks = num_sinks
        self.recent_window = recent_window
        
        # Heavy Hitter threshold: max_cache_size must be larger than sinks + local window
        assert max_cache_size > (num_sinks + recent_window)
        self.heavy_hitters_budget = max_cache_size - num_sinks - recent_window

    def evict_kv_cache(
        self, 
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor, 
        attn_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compresses Key-Value cache tensors dynamically based on accumulated attention scores.
        
        k_cache / v_cache shapes: [Batch, Num_heads, Seq_len, Head_dim]
        attn_scores shape: [Batch, Num_heads, Seq_len] (accumulated attention weights per position)
        
        Returns compressed key and value cache tensors of shape: [Batch, Num_heads, max_cache_size, Head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = k_cache.shape
        device = k_cache.device
        
        if seq_len <= self.max_cache_size:
            # Under budget: no eviction required
            return k_cache, v_cache
            
        # 1. Identify indices for Attention Sinks (first few tokens)
        sink_indices = torch.arange(0, self.num_sinks, device=device)
        
        # 2. Identify indices for Recent local context window (last few tokens)
        recent_indices = torch.arange(seq_len - self.recent_window, seq_len, device=device)
        
        # 3. Identify Heavy Hitters from middle tokens
        # Extract attention scores belonging only to evictable middle tokens
        middle_start = self.num_sinks
        middle_end = seq_len - self.recent_window
        
        middle_attn = attn_scores[:, :, middle_start:middle_end]
        
        # Select Top-K heavy hitters based on accumulated attention weights
        # top_indices shape: [Batch, Num_heads, heavy_hitters_budget]
        _, top_indices = torch.topk(middle_attn, k=self.heavy_hitters_budget, dim=-1)
        # Adjust indices offset back to match global positions
        heavy_hitter_indices = top_indices + middle_start
        
        # 4. Construct the consolidated index mask for each head in the batch
        # To handle indexing cleanly across different batch/head dimensions:
        compressed_k = []
        compressed_v = []
        
        for b in range(batch_size):
            batch_k = []
            batch_v = []
            for h in range(num_heads):
                # Gather indices for current head
                h_hh_indices = heavy_hitter_indices[b, h]
                
                # Combine sinks + heavy hitters + recent window indices
                combined_idx = torch.cat([sink_indices, h_hh_indices, recent_indices])
                # Sort indices chronologically to preserve RoPE position alignment
                sorted_idx, _ = torch.sort(combined_idx)
                
                # Squeeze and index corresponding token coordinates
                # shape: [max_cache_size, Head_dim]
                h_k = k_cache[b, h, sorted_idx, :]
                h_v = v_cache[b, h, sorted_idx, :]
                
                batch_k.append(h_k)
                batch_v.append(h_v)
                
            compressed_k.append(torch.stack(batch_k))
            compressed_v.append(torch.stack(batch_v))
            
        # Reshape to standard 4D tensor format
        return torch.stack(compressed_k), torch.stack(compressed_v)


class StreamingLLMEvictor:
    """
    StreamingLLM KV-Cache Eviction Manager.
    Allows infinite-length context serving by keeping absolute sinks and a local sliding window.
    """
    def __init__(self, num_sinks: int = 4, recent_window: int = 508):
        self.num_sinks = num_sinks
        self.recent_window = recent_window
        self.max_cache_size = num_sinks + recent_window

    def evict_kv_cache(
        self, 
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compresses Key-Value cache tensors dynamically based on StreamingLLM window logic.
        k_cache / v_cache shapes: [Batch, Num_heads, Seq_len, Head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = k_cache.shape
        device = k_cache.device
        
        if seq_len <= self.max_cache_size:
            return k_cache, v_cache
            
        # Extract sink tokens (first num_sinks positions)
        k_sinks = k_cache[:, :, :self.num_sinks, :]
        v_sinks = v_cache[:, :, :self.num_sinks, :]
        
        # Extract sliding window tokens (last recent_window positions)
        k_window = k_cache[:, :, seq_len - self.recent_window:, :]
        v_window = v_cache[:, :, seq_len - self.recent_window:, :]
        
        # Concatenate sinks and window
        compressed_k = torch.cat([k_sinks, k_window], dim=2)
        compressed_v = torch.cat([v_sinks, v_window], dim=2)
        
        return compressed_k, compressed_v
