import math
import torch
import torch.nn as nn
import torch.distributed as dist

# Global variables for 3D parallel groups
_TP_GROUP = None
_TP_SIZE = 1
_TP_RANK = 0

def init_tp_process_group(tp_size: int = 1):
    """
    Initializes Tensor Parallelism (TP) process groups.
    Assumes standard 3D parallel topology order: TP -> PP -> DP.
    E.g. for 8 GPUs, TP=4, PP=2, DP=1:
      Ranks [0,1,2,3] are TP group 0 (PP stage 0)
      Ranks [4,5,6,7] are TP group 1 (PP stage 1)
    """
    global _TP_GROUP, _TP_SIZE, _TP_RANK
    _TP_SIZE = tp_size
    
    if not dist.is_initialized() or tp_size <= 1:
        _TP_SIZE = 1
        _TP_RANK = 0
        return
        
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    _TP_RANK = rank % tp_size
    
    # Create TP subgroups
    num_tp_groups = world_size // tp_size
    for i in range(num_tp_groups):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TP_GROUP = group

def get_tp_group():
    return _TP_GROUP

def get_tp_size():
    return _TP_SIZE

def get_tp_rank():
    return _TP_RANK

# ==============================================================================
# Megatron-style Autograd Communication Functions
# ==============================================================================

class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if get_tp_size() > 1 and dist.is_initialized():
            dist.all_reduce(grad_output, group=get_tp_group())
        return grad_output

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        if get_tp_size() > 1 and dist.is_initialized():
            dist.all_reduce(input_, group=get_tp_group())
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# ==============================================================================
# ColumnParallelLinear & RowParallelLinear Layers
# ==============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer. The weight matrix is sharded along the column dimension (out_features).
    Weight: [In_features, Out_features / TP_size]
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_method=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = get_tp_size()
        
        # Calculate local partition size
        self.local_out_features = out_features // self.tp_size
        
        # Local parameter partition
        self.weight = nn.Parameter(torch.empty(self.in_features, self.local_out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method=None):
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Step 1: Copy input to model parallel region (all-reduce grads in backward)
        input_parallel = _CopyToModelParallelRegion.apply(input_)
        
        # Step 2: Local matrix multiplication
        output_parallel = torch.matmul(input_parallel, self.weight)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
            
        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer. The weight matrix is sharded along the row dimension (in_features).
    Weight: [In_features / TP_size, Out_features]
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_method=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = get_tp_size()
        
        # Calculate local partition size
        self.local_in_features = in_features // self.tp_size
        
        # Local parameter partition
        self.weight = nn.Parameter(torch.empty(self.local_in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method=None):
        if init_method is not None:
            init_method(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
        if self.bias is not None:
            # Only rank 0 of the TP group has bias, to avoid multiplying the bias during all-reduce sum!
            if get_tp_rank() != 0:
                with torch.no_grad():
                    self.bias.zero_()
            # Initialize bias
            else:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_parallel: torch.Tensor) -> torch.Tensor:
        # Step 1: Local matrix multiplication
        output_parallel = torch.matmul(input_parallel, self.weight)
        
        # Step 2: Reduce from model parallel region (all-reduce sum in forward, pass-through grads in backward)
        output_ = _ReduceFromModelParallelRegion.apply(output_parallel)
        
        if self.bias is not None:
            output_ = output_ + self.bias
            
        return output_
