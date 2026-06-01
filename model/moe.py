"""DeepSeekMoE: Mixture of Shared & Routed Experts Block."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


class DeepSeekMoE(nn.Module):
    def __init__(self, config: ModelConfig, capacity_factor: float = 1.2):
        super().__init__()
        from model import SwiGLU_MLP  # lazy import to avoid circular dependency

        self.num_shared = config.num_shared_experts
        self.num_routed = config.num_routed_experts
        self.num_active = config.num_active_experts
        self.capacity_factor = capacity_factor

        # 1. Shared Experts (static SwiGLU_MLP modules)
        self.shared_experts = nn.ModuleList([SwiGLU_MLP(config) for _ in range(self.num_shared)])

        # 2. Routed Experts (partitioned if EP is active)
        from utils.expert_parallel import ExpertParallelRouter, get_ep_size
        self.ep_size = get_ep_size()

        self.experts_per_rank = self.num_routed // self.ep_size
        self.routed_experts = nn.ModuleList([SwiGLU_MLP(config) for _ in range(self.experts_per_rank)])

        # 3. Router Gate (maps hidden state to routing logits)
        self.router = nn.Linear(config.n_embd, self.num_routed, bias=False)

        if self.ep_size > 1:
            self.router_parallel = ExpertParallelRouter(self.num_routed)

        # Placeholder for future DeepEP integration
        self.deepep = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        flat_x = x.view(-1, d_model) # Shape: (total_tokens, d_model)
        num_tokens = flat_x.size(0)

        # 1. Process Shared Experts (Always active)
        shared_out = torch.zeros_like(flat_x)
        for expert in self.shared_experts:
            shared_out += expert(flat_x)

        # 2. Routed Experts (Gated execution)
        router_logits = self.router(flat_x) # Shape: (total_tokens, num_routed)
        gate_probs = F.softmax(router_logits, dim=-1)

        # Select Top-K experts per token
        topk_weights, topk_indices = torch.topk(gate_probs, self.num_active, dim=-1)

        # Normalize weights to sum to 1.0 per token
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Calculate load balancing auxiliary loss (Auxiliary loss) if in training
        if self.training:
            # Fraction of tokens dispatched to each expert
            dispatched = torch.zeros(self.num_routed, device=x.device)
            for k in range(self.num_active):
                indices = topk_indices[:, k]
                dispatched.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
            fi = dispatched / num_tokens
            pi = gate_probs.mean(dim=0)
            aux_loss = self.num_routed * torch.sum(fi * pi)
            # Store aux loss as buffer so parent block can access it
            self.aux_loss = aux_loss

        # Run gated execution
        if self.ep_size > 1:
            routed_out = self.router_parallel(flat_x, topk_weights, topk_indices, self.routed_experts)
        else:
            expert_capacity = int((num_tokens / self.num_routed) * self.capacity_factor)
            expert_capacity = max(4, expert_capacity) # Ensure a baseline minimum capacity

            routed_out = torch.zeros_like(flat_x)

            # Batched dispatch: collect all routing entries, sort by expert, then slice
            token_idx_flat = torch.arange(num_tokens, device=x.device).repeat_interleave(self.num_active)
            expert_idx_flat = topk_indices.reshape(-1)      # (num_tokens * num_active,)
            weight_flat = topk_weights.reshape(-1)

            # Sort by expert index to group tokens heading to the same expert
            sorted_expert_idx, sort_order = torch.sort(expert_idx_flat)
            sorted_token_idx = token_idx_flat[sort_order]
            sorted_weights = weight_flat[sort_order]

            # Count tokens per expert and compute offsets into the sorted arrays
            expert_counts = torch.bincount(sorted_expert_idx, minlength=self.num_routed)
            offsets = torch.cat([
                torch.zeros(1, device=x.device, dtype=torch.long),
                torch.cumsum(expert_counts, dim=0),
            ])

            for expert_idx in range(self.num_routed):
                start = offsets[expert_idx].item()
                end = offsets[expert_idx + 1].item()
                if start == end:
                    continue

                t_idx = sorted_token_idx[start:end]
                w = sorted_weights[start:end]

                if t_idx.numel() > expert_capacity:
                    _, top_k = torch.topk(w, expert_capacity)
                    t_idx = t_idx[top_k]
                    w = w[top_k]

                if t_idx.numel() > 0:
                    expert_tokens = flat_x[t_idx]
                    expert_out = self.routed_experts[expert_idx](expert_tokens)
                    routed_out[t_idx] += expert_out * w.unsqueeze(-1)

        # Blended combination of Shared and Routed experts output
        output = shared_out + routed_out
        return output.view(bsz, seqlen, d_model)
