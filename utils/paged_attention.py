import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# ==============================================================================
# PagedAttention Virtual Page KV-Cache Management Suite (vLLM Style)
# ==============================================================================

class PhysicalBlock:
    """
    Represents a single physical memory block (page) containing Key-Value slots.
    """
    def __init__(self, block_id: int, block_size: int, num_heads: int, head_dim: int, device: torch.device):
        self.block_id = block_id
        self.block_size = block_size  # Number of token slots per block (e.g. 16)
        
        # Pre-allocate physical tensor buffers to prevent VRAM allocations during decoding
        # Shape: [Block_size, Num_heads, Head_dim]
        self.k_buffer = torch.zeros(block_size, num_heads, head_dim, dtype=torch.float16, device=device)
        self.v_buffer = torch.zeros(block_size, num_heads, head_dim, dtype=torch.float16, device=device)
        self.free_slots = block_size


class PagedCacheManager:
    """
    Paged Cache Manager that dynamically tracks physical VRAM blocks and maps
    logical sequence token positions to physical blocks using Page Tables.
    """
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int, device: torch.device):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # Instantiate pool of pre-allocated physical blocks
        self.block_pool = [
            PhysicalBlock(i, block_size, num_heads, head_dim, device) 
            for i in range(num_blocks)
        ]
        self.free_block_ids = list(range(num_blocks))
        
        # Page table mapping: sequence_id -> list of allocated physical block IDs
        self.page_tables: Dict[int, List[int]] = {}

    def allocate_blocks(self, seq_id: int, num_blocks_needed: int) -> List[int]:
        """Allocates free physical blocks to a sequence logical page address."""
        if len(self.free_block_ids) < num_blocks_needed:
            raise RuntimeError("Out-Of-Memory: No free physical KV blocks available in PagedCache pool!")
            
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_block_ids.pop(0)
            allocated.append(block_id)
            
        if seq_id not in self.page_tables:
            self.page_tables[seq_id] = []
        self.page_tables[seq_id].extend(allocated)
        return allocated

    def free_sequence(self, seq_id: int):
        """Releases physical blocks mapped to sequence back to free pool."""
        if seq_id in self.page_tables:
            block_ids = self.page_tables[seq_id]
            for bid in block_ids:
                # Reset free slots
                self.block_pool[bid].free_slots = self.block_size
                self.free_block_ids.append(bid)
            del self.page_tables[seq_id]

    def write_to_cache(self, seq_id: int, logical_pos: int, k_token: torch.Tensor, v_token: torch.Tensor):
        """
        Writes a single token's Key and Value projections into the mapped physical slot.
        k_token / v_token shapes: [Num_heads, Head_dim]
        """
        page_table = self.page_tables[seq_id]
        
        # Find which physical block and slot index corresponds to logical position
        block_idx = logical_pos // self.block_size
        slot_idx = logical_pos % self.block_size
        
        # Allocate new page block dynamically on-the-fly if boundary crossed
        if block_idx >= len(page_table):
            self.allocate_blocks(seq_id, num_blocks_needed=1)
            page_table = self.page_tables[seq_id]
            
        physical_block_id = page_table[block_idx]
        block = self.block_pool[physical_block_id]
        
        # Direct writing (zero-copy) into pre-allocated memory slices
        block.k_buffer[slot_idx] = k_token.to(dtype=torch.float16)
        block.v_buffer[slot_idx] = v_token.to(dtype=torch.float16)
        block.free_slots -= 1


# ==============================================================================
# PagedAttention Attention Calculation Kernel
# ==============================================================================

class PagedAttentionKernel(nn.Module):
    """
    Computes standard scaled dot product attention queries directly over sharded
    physical block allocations mapped inside the PagedCacheManager.
    """
    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = 1.0 / (head_dim ** 0.5)

    def forward(
        self, 
        q: torch.Tensor, 
        seq_id: int, 
        cache_manager: PagedCacheManager, 
        seq_len: int
    ) -> torch.Tensor:
        """
        q shape: [1, Num_heads, Head_dim] (decoding single query token)
        Returns weighted value outputs: [1, Num_heads, Head_dim]
        """
        num_heads = q.size(1)
        head_dim = q.size(2)
        
        page_table = cache_manager.page_tables[seq_id]
        
        # Step 1. Gather scattered KV states from physical blocks to a temporary stack
        # Temporarily rebuild continuous keys/values only for active query length seq_len
        # In low-level vLLM, this is calculated directly in-place inside CUDA/Triton kernels!
        gathered_k = torch.zeros(seq_len, num_heads, head_dim, dtype=torch.float16, device=q.device)
        gathered_v = torch.zeros(seq_len, num_heads, head_dim, dtype=torch.float16, device=q.device)
        
        for i in range(seq_len):
            block_idx = i // cache_manager.block_size
            slot_idx = i % cache_manager.block_size
            
            p_block_id = page_table[block_idx]
            block = cache_manager.block_pool[p_block_id]
            
            gathered_k[i] = block.k_buffer[slot_idx]
            gathered_v[i] = block.v_buffer[slot_idx]
            
        # Reshape gathered keys/values to calculate dot products
        # gathered_k: [SeqLen, Num_heads, Head_dim] -> [Num_heads, SeqLen, Head_dim]
        gathered_k = gathered_k.transpose(0, 1)
        gathered_v = gathered_v.transpose(0, 1)
        
        # q: [1, Num_heads, Head_dim] -> [Num_heads, 1, Head_dim]
        q = q.squeeze(0).unsqueeze(1)
        
        # Step 2. Calculate Attention weights
        # scores shape: [Num_heads, 1, SeqLen]
        scores = torch.bmm(q, gathered_k.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # Step 3. Compute weighted output
        # out shape: [Num_heads, 1, Head_dim]
        out = torch.bmm(attn_weights, gathered_v)
        
        # Return reshaped output: [1, Num_heads, Head_dim]
        return out.transpose(0, 1)
