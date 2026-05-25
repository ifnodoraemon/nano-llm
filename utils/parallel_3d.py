import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional

# ==============================================================================
# 1. ColumnParallelLinear (Megatron-LM Style Column-Sharded Projections)
# ==============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer. The weight matrix is sharded along the column dimension (out_features).
    
    Megatron Column Sharding:
    Weight: [In_features, Out_features / World_size]
    Forward pass: Y_i = X * W_i
    Communication: No communication required in forward pass, but backward pass requires All-Reduce.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        world_size: int = 1,
        rank: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Calculate local column dimension
        self.local_out_features = out_features // world_size
        
        # Instantiate local parameter partition
        self.weight = nn.Parameter(torch.empty(self.in_features, self.local_out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # standard Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Step 1. Local matrix multiplication: Y_i = X * W_i
        # input_: [Batch, SeqLen, In_features] -> out: [Batch, SeqLen, Out_features / World_size]
        output_parallel = torch.matmul(input_, self.weight)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
            
        # In a real distributed Megatron cluster, we would handle communication.
        # If distributed is active and configured:
        # dist.all_reduce(grad) in backward is implicitly handled by PyTorch autograd hooks!
        return output_parallel


# ==============================================================================
# 2. RowParallelLinear (Megatron-LM Style Row-Sharded Projections)
# ==============================================================================

class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer. The weight matrix is sharded along the row dimension (in_features).
    
    Megatron Row Sharding:
    Weight: [In_features / World_size, Out_features]
    Forward pass: Y = sum_i( X_i * W_i )
    Communication: Forward pass requires All-Reduce across columns, backward requires no communication.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        world_size: int = 1,
        rank: int = 0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Calculate local row dimension
        self.local_in_features = in_features // world_size
        
        # Instantiate local parameter partition
        self.weight = nn.Parameter(torch.empty(self.local_in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Only Rank 0 has the bias to prevent summing bias multiple times during All-Reduce!
            if self.rank != 0:
                with torch.no_grad():
                    self.bias.zero_()

    def forward(self, input_parallel: torch.Tensor) -> torch.Tensor:
        # input_parallel: [Batch, SeqLen, In_features / World_size]
        # output_parallel: [Batch, SeqLen, Out_features]
        output_parallel = torch.matmul(input_parallel, self.weight)
        
        # Communication: All-Reduce sum to collect results from all column shards
        if self.world_size > 1 and dist.is_initialized():
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
            
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
            
        return output_parallel

import math

# ==============================================================================
# 3. PipelineParallelTransformer (Sequential Layer-Sharded Orchestration)
# ==============================================================================

class PipelineParallelTransformer(nn.Module):
    """
    Pipeline-Parallel Layer Orchestrator. Shards transformer blocks sequentially
    across different GPU device ranks in the pipeline.
    
    Pipeline layout:
    Rank 0: Embedding + Layers[0..k]
    Rank 1: Layers[k..2k]
    Rank N: Layers[nk..] + RMSNorm + Head
    """
    def __init__(
        self, 
        layers: nn.ModuleList, 
        embedding: nn.Module, 
        head: nn.Module,
        pipeline_ranks: List[int],
        current_rank: int
    ):
        super().__init__()
        self.embedding = embedding
        self.head = head
        self.pipeline_ranks = pipeline_ranks
        self.current_rank = current_rank
        self.num_stages = len(pipeline_ranks)
        
        # Divide layers evenly across pipeline stages
        num_layers = len(layers)
        layers_per_stage = num_layers // self.num_stages
        
        # Calculate local layer slicing index bounds
        self.start_layer = self.current_rank * layers_per_stage
        self.end_layer = self.start_layer + layers_per_stage if self.current_rank < self.num_stages - 1 else num_layers
        
        # Keep only the layers belonging to the current device rank to conserve HBM
        self.local_layers = nn.ModuleList([layers[i] for i in range(self.start_layer, self.end_layer)])
        logger_name = f"Pipeline Stage Rank {self.current_rank}"
        print(f"[{logger_name}] Active layers mapping: {self.start_layer} to {self.end_layer-1}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1. Input Stage: Rank 0 processes Embeddings
        if self.current_rank == 0:
            x = self.embedding(x)
            
        # Step 2. Layer Processing Stage: Run local slice of layers
        for layer in self.local_layers:
            x = layer(x)
            
        # Step 3. Pipeline Communication: Stream activations to next rank
        if self.num_stages > 1 and dist.is_initialized():
            next_rank = (self.current_rank + 1) % self.num_stages
            prev_rank = (self.current_rank - 1) % self.num_stages
            
            # Forward pass communication: Send activations to next rank
            if self.current_rank < self.num_stages - 1:
                # Send shape/tensor to next stage rank asynchronously
                dist.send(tensor=x, dst=self.pipeline_ranks[next_rank])
                # Return empty/dummy since active processing shifts to next stage
                return x
            else:
                # Final stage rank receives activations and completes head projections
                incoming = torch.empty_like(x)
                dist.recv(tensor=incoming, src=self.pipeline_ranks[prev_rank])
                x = incoming
                x = self.head(x)
                return x
                
        # Non-distributed fallback
        if self.current_rank == self.num_stages - 1:
            x = self.head(x)
        return x
