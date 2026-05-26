import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Dict

_EP_GROUP = None
_EP_SIZE = 1
_EP_RANK = 0

def init_ep_process_group(ep_size: int = 1):
    """
    Initializes Expert Parallelism (EP) process groups.
    Assumes standard 3D parallel topology order: TP -> PP -> DP/EP.
    Ranks are grouped to partition MoE experts.
    """
    global _EP_GROUP, _EP_SIZE, _EP_RANK
    _EP_SIZE = ep_size
    
    if not dist.is_initialized() or ep_size <= 1:
        _EP_SIZE = 1
        _EP_RANK = 0
        return
        
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    _EP_RANK = rank % ep_size
    
    # Create EP subgroups
    num_ep_groups = world_size // ep_size
    for i in range(num_ep_groups):
        ranks = list(range(i * ep_size, (i + 1) * ep_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _EP_GROUP = group

def get_ep_group():
    return _EP_GROUP

def get_ep_size():
    return _EP_SIZE

def get_ep_rank():
    return _EP_RANK


class ExpertParallelRouter(nn.Module):
    """
    Expert Parallelism Router.
    Routes tokens dynamically across different GPU devices using all-to-all communication.
    Supports a fully functional local fallback for non-distributed training/testing.
    """
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.ep_size = get_ep_size()
        self.ep_rank = get_ep_rank()
        
        # Calculate local experts slice
        assert num_experts % self.ep_size == 0, f"Experts ({num_experts}) must be divisible by EP size ({self.ep_size})"
        self.experts_per_rank = num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.experts_per_rank
        self.local_expert_end = self.local_expert_start + self.experts_per_rank

    def is_local_expert(self, expert_idx: int) -> bool:
        """Returns True if the expert belongs to the current GPU rank."""
        return self.local_expert_start <= expert_idx < self.local_expert_end

    def get_local_expert_index(self, expert_idx: int) -> int:
        """Maps global expert index to local expert slice index [0, experts_per_rank)."""
        return expert_idx - self.local_expert_start

    def forward(
        self, 
        tokens: torch.Tensor, 
        gate_weights: torch.Tensor, 
        expert_indices: torch.Tensor,
        local_expert_runner: nn.ModuleList
    ) -> torch.Tensor:
        """
        Routes tokens to their target experts across GPUs via all-to-all,
        runs local experts, and returns the gathered outputs back to original sequence order.
        
        tokens: [Num_tokens, d_model]
        gate_weights: [Num_tokens, Top_K] (weights for routing)
        expert_indices: [Num_tokens, Top_K] (global expert indices for routing)
        local_expert_runner: nn.ModuleList of local experts (length experts_per_rank)
        """
        num_tokens, top_k = expert_indices.shape
        d_model = tokens.size(-1)
        device = tokens.device
        
        # Non-distributed fallback
        if self.ep_size <= 1 or not dist.is_initialized():
            output = torch.zeros_like(tokens)
            # Process everything locally
            for k in range(top_k):
                indices = expert_indices[:, k]
                weights = gate_weights[:, k].unsqueeze(-1) # [Num_tokens, 1]
                
                # Execute each expert locally on routed tokens
                for exp_idx in range(self.num_experts):
                    mask = (indices == exp_idx)
                    if mask.any():
                        exp_tokens = tokens[mask]
                        # Run local expert
                        exp_out = local_expert_runner[exp_idx](exp_tokens)
                        output[mask] += exp_out * weights[mask]
            return output

        # 1. Distributed EP Token Routing using simulated/real all-to-all collectives
        # Determine target rank for each token's routing choices
        # E.g. expert_idx // experts_per_rank -> rank
        target_ranks = expert_indices // self.experts_per_rank # [Num_tokens, Top_K]
        
        # Prepare tokens to send to each rank
        send_tokens_list = [[] for _ in range(self.ep_size)]
        send_weights_list = [[] for _ in range(self.ep_size)]
        send_exp_indices_list = [[] for _ in range(self.ep_size)]
        
        # Keep track of original coordinates to reconstruct final sequence
        send_orig_coords = [[] for _ in range(self.ep_size)]
        
        for t in range(num_tokens):
            for k in range(top_k):
                r = target_ranks[t, k].item()
                send_tokens_list[r].append(tokens[t])
                send_weights_list[r].append(gate_weights[t, k])
                send_exp_indices_list[r].append(expert_indices[t, k])
                send_orig_coords[r].append((t, k))
                
        # Stack inputs for communication
        send_counts = [len(send_tokens_list[r]) for r in range(self.ep_size)]
        
        # Comm step 1: Exchange send counts to compute receive counts
        recv_counts = [0] * self.ep_size
        # All-to-all send/recv counts
        dist.all_to_all_single(
            torch.tensor(recv_counts, device=device),
            torch.tensor(send_counts, device=device),
            group=get_ep_group()
        )
        
        # Comm step 2: Exchange tokens, weights, global expert indices, and original coordinates
        flat_send_tokens = torch.cat([torch.stack(lst) if lst else torch.empty(0, d_model, device=device) for lst in send_tokens_list])
        flat_send_weights = torch.cat([torch.stack(lst) if lst else torch.empty(0, device=device) for lst in send_weights_list])
        flat_send_exp_indices = torch.cat([torch.stack(lst) if lst else torch.empty(0, dtype=torch.long, device=device) for lst in send_exp_indices_list])
        
        # Setup receive buffers
        total_recv = sum(recv_counts)
        flat_recv_tokens = torch.empty(total_recv, d_model, device=device)
        flat_recv_weights = torch.empty(total_recv, device=device)
        flat_recv_exp_indices = torch.empty(total_recv, dtype=torch.long, device=device)
        
        # Execute all-to-all communication for token payload
        dist.all_to_all_single(flat_recv_tokens, flat_send_tokens, group=get_ep_group())
        dist.all_to_all_single(flat_recv_weights, flat_send_weights, group=get_ep_group())
        dist.all_to_all_single(flat_recv_exp_indices, flat_send_exp_indices, group=get_ep_group())
        
        # 2. Local Computation
        # Compute outputs for received tokens using local experts
        flat_recv_outputs = torch.zeros_like(flat_recv_tokens)
        
        for exp_idx in range(self.local_expert_start, self.local_expert_end):
            local_idx = self.get_local_expert_index(exp_idx)
            mask = (flat_recv_exp_indices == exp_idx)
            if mask.any():
                local_tokens = flat_recv_tokens[mask]
                local_out = local_expert_runner[local_idx](local_tokens)
                flat_recv_outputs[mask] = local_out
                
        # 3. Distributed Return Routing (all-to-all return)
        flat_send_outputs = torch.empty(total_recv, d_model, device=device)
        # Setup send/recv counts for return (which are inverse of forward routing)
        dist.all_to_all_single(flat_send_outputs, flat_recv_outputs, group=get_ep_group())
        
        # Reconstruct output sequence
        output_tokens = torch.zeros_like(tokens)
        
        # Place returned tokens into original sequence using tracking coordinates
        idx = 0
        for r in range(self.ep_size):
            count = send_counts[r]
            if count > 0:
                recv_chunk = flat_send_outputs[idx:idx + count]
                weights_chunk = flat_send_weights[idx:idx + count]
                coords = send_orig_coords[r]
                for i, (t_coord, k_coord) in enumerate(coords):
                    output_tokens[t_coord] += recv_chunk[i] * weights_chunk[i]
                idx += count
                
        return output_tokens
