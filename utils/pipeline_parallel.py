import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Any, Callable, Dict, Optional

_PP_GROUP = None
_PP_SIZE = 1
_PP_RANK = 0

def init_pp_process_group(pp_size: int = 1):
    """
    Initializes Pipeline Parallelism (PP) process groups.
    Assumes standard 3D parallel topology order: TP -> PP -> DP.
    Ranks are grouped such that adjacent pipeline stages are in the same PP group.
    """
    global _PP_GROUP, _PP_SIZE, _PP_RANK
    _PP_SIZE = pp_size
    
    if not dist.is_initialized() or pp_size <= 1:
        _PP_SIZE = 1
        _PP_RANK = 0
        return
        
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # E.g. 8 GPUs, TP=4, PP=2, DP=1
    # TP group 0: Ranks [0,1,2,3] -> stage 0
    # TP group 1: Ranks [4,5,6,7] -> stage 1
    # PP groups: Group 0 contains [0, 4], Group 1 [1, 5], Group 2 [2, 6], Group 3 [3, 7]
    tp_size = world_size // pp_size # simple fallback if DP=1
    
    _PP_RANK = rank // tp_size # PP stage index
    
    # Create PP subgroups using Gloo backend
    for i in range(tp_size):
        ranks = [i + j * tp_size for j in range(pp_size)]
        group = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            _PP_GROUP = group

def get_pp_group():
    return _PP_GROUP

def get_pp_size():
    return _PP_SIZE

def get_pp_rank():
    return _PP_RANK


class PipelineStage(nn.Module):
    """
    PipelineStage wraps a contiguous subset of model layers.
    - Stage 0 contains Embedding + first slice of layers.
    - Intermediate stages contain middle layers.
    - Final stage contains last layers + LayerNorm + Head.
    """
    def __init__(self, layers: nn.ModuleList, embedding: Optional[nn.Module] = None, head: Optional[nn.Module] = None, freqs_cis: Optional[torch.Tensor] = None):
        super().__init__()
        self.layers = layers
        self.embedding = embedding
        self.head = head
        self.pp_rank = get_pp_rank()
        self.pp_size = get_pp_size()
        self.freqs_cis = freqs_cis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pp_rank == 0 and self.embedding is not None:
            x = self.embedding(x)
            
        freqs_cis = None
        if self.freqs_cis is not None:
            self.freqs_cis = self.freqs_cis.to(x.device)
            seqlen = x.size(1)
            freqs_cis = self.freqs_cis[:seqlen]
            
        for layer in self.layers:
            # Debugging print
            print(f"[DEBUG PP Stage {self.pp_rank}] layer: {type(layer)}, has_attn: {hasattr(layer, 'attention')}, freqs_cis_none: {freqs_cis is None}", flush=True)
            if hasattr(layer, "attention"):
                x = layer(x, freqs_cis)
            else:
                x = layer(x)
            
        if self.pp_rank == self.pp_size - 1 and self.head is not None:
            x = self.head(x)
            
        return x


class OneFOneBScheduler:
    """
    1F1B (One Forward One Backward) Pipeline Parallel Scheduler.
    Schedules micro-batches dynamically to minimize memory footprint and pipeline bubble.
    """
    def __init__(self, stage: PipelineStage, num_microbatches: int, d_model: int):
        self.stage = stage
        self.num_microbatches = num_microbatches
        self.d_model = d_model
        self.pp_rank = get_pp_rank()
        self.pp_size = get_pp_size()
        self.pp_group_ranks = []
        self.send_handles = []
        if dist.is_initialized() and self.pp_size > 1:
            self.pp_group_ranks = dist.get_process_group_ranks(get_pp_group())
        
    def _recv_forward(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Receives activation tensor from the previous stage."""
        if self.pp_rank == 0:
            return None
        # Allocate on CPU with torch.bfloat16 explicitly for Gloo P2P transfer
        x_cpu = torch.empty(shape, device="cpu", dtype=torch.bfloat16)
        if dist.is_initialized() and self.pp_size > 1:
            # Receive from previous stage's global rank in the Gloo PP subgroup
            src_global = self.pp_group_ranks[self.pp_rank - 1]
            work = dist.irecv(x_cpu, src=src_global, group=get_pp_group())
            work.wait()
        x = x_cpu.to(device)
        x.requires_grad = True
        return x
        
    def _send_forward(self, x: torch.Tensor):
        """Sends activation tensor to the next stage."""
        if self.pp_rank == self.pp_size - 1:
            return
        if dist.is_initialized() and self.pp_size > 1:
            # Copy to CPU and cast to torch.bfloat16 explicitly
            x_cpu = x.detach().cpu().to(torch.bfloat16)
            # Send to next stage's global rank in the Gloo PP subgroup
            dst_global = self.pp_group_ranks[self.pp_rank + 1]
            work = dist.isend(x_cpu, dst=dst_global, group=get_pp_group())
            self.send_handles.append(work)
            
    def _recv_backward(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Receives gradient tensor from the next stage."""
        if self.pp_rank == self.pp_size - 1:
            return None
        # Allocate on CPU with torch.bfloat16 explicitly for Gloo P2P transfer
        dy_cpu = torch.empty(shape, device="cpu", dtype=torch.bfloat16)
        if dist.is_initialized() and self.pp_size > 1:
            # Receive from next stage's global rank in the Gloo PP subgroup
            src_global = self.pp_group_ranks[self.pp_rank + 1]
            work = dist.irecv(dy_cpu, src=src_global, group=get_pp_group())
            work.wait()
        return dy_cpu.to(device)
        
    def _send_backward(self, dy: torch.Tensor):
        """Sends gradient tensor to the previous stage."""
        if self.pp_rank == 0:
            return
        if dist.is_initialized() and self.pp_size > 1:
            # Copy to CPU and cast to torch.bfloat16 explicitly
            dy_cpu = dy.detach().cpu().to(torch.bfloat16)
            # Send to previous stage's global rank in the Gloo PP subgroup
            dst_global = self.pp_group_ranks[self.pp_rank - 1]
            work = dist.isend(dy_cpu, dst=dst_global, group=get_pp_group())
            self.send_handles.append(work)

    def run_1f1b(
        self, 
        micro_batches: List[torch.Tensor], 
        targets: Optional[List[torch.Tensor]] = None,
        loss_fn: Optional[Callable] = None,
        device: torch.device = torch.device("cpu")
    ) -> List[torch.Tensor]:
        """
        Executes the 1F1B schedule.
        Returns a list of calculated loss tensors (only valid on the final stage).
        """
        num_mbs = len(micro_batches)
        assert num_mbs == self.num_microbatches
        
        # Buffer to keep track of forward activations for backward pass
        # Key: micro-batch index, Value: (input_tensor, output_tensor)
        fwd_activations: Dict[int, tuple] = {}
        losses = []
        
        # Simple non-distributed fallback (Sequential)
        if self.pp_size <= 1 or not dist.is_initialized():
            for i in range(num_mbs):
                x = micro_batches[i].to(device)
                y = self.stage(x)
                if targets is not None and loss_fn is not None:
                    loss = loss_fn(y, targets[i].to(device))
                    losses.append(loss)
                    loss.backward()
                else:
                    # In mock mode, we just backward with a dummy gradient if no targets provided
                    dy = torch.ones_like(y)
                    y.backward(dy)
            return losses

        # Distribute shape detection
        # Assume batch dimensions match
        if self.pp_rank == 0:
            batch_shape = (micro_batches[0].size(0), micro_batches[0].size(1), self.d_model)
        else:
            batch_shape = (micro_batches[0].size(0), micro_batches[0].size(1), self.d_model)
            
        # Startup phase: Warmup steps to fill the pipeline
        warmup_steps = min(num_mbs, self.pp_size - self.pp_rank - 1)
        
        fwd_idx = 0
        bwd_idx = 0
        
        # Perform warmup forward passes
        for _ in range(warmup_steps):
            # Recv activation from previous stage
            if self.pp_rank == 0:
                x = micro_batches[fwd_idx].to(device)
            else:
                x = self._recv_forward(batch_shape, device)
                
            # Process local layer
            y = self.stage(x)
            fwd_activations[fwd_idx] = (x, y)
            
            # Send to next stage
            self._send_forward(y)
            fwd_idx += 1
            
        # Steady-state phase: alternate 1 forward and 1 backward
        while bwd_idx < num_mbs:
            # If we still have micro-batches to process in forward:
            if fwd_idx < num_mbs:
                # 1. Forward Step
                if self.pp_rank == 0:
                    x = micro_batches[fwd_idx].to(device)
                else:
                    x = self._recv_forward(batch_shape, device)
                    
                y = self.stage(x)
                fwd_activations[fwd_idx] = (x, y)
                self._send_forward(y)
                fwd_idx += 1
                
            # 2. Backward Step (1F1B)
            # Recv gradient from next stage
            if self.pp_rank == self.pp_size - 1:
                # Final stage calculates loss and starts backprop
                x, y = fwd_activations[bwd_idx]
                if targets is not None and loss_fn is not None:
                    loss = loss_fn(y, targets[bwd_idx].to(device))
                    losses.append(loss)
                    loss.backward()
                else:
                    loss = y.mean()
                    losses.append(loss)
                    loss.backward()
                    
                if self.pp_rank > 0:
                    # Send grad to previous stage
                    self._send_backward(x.grad)
            else:
                dy = self._recv_backward(batch_shape, device)
                x, y = fwd_activations[bwd_idx]
                
                # Backpropagate gradient
                y.backward(dy)
                
                if self.pp_rank > 0:
                    self._send_backward(x.grad)
                    
            bwd_idx += 1
            
        # Wait for all asynchronous sends to complete
        for work in self.send_handles:
            if work is not None:
                work.wait()
        self.send_handles.clear()
        
        return losses
