import logging

logger = logging.getLogger(__name__)

# Theoretical Peak BF16/FP16 performance for common GPUs (in FLOPS):
# - H800/H100 SXM5: approx 2000e12 (with Tensor Cores). 
# - H800/H100 PCIe: approx 700e12.
# - Without sparsity optimizations, dense peak is approx 200 TFLOPS (200e12) or higher depending on SXM configuration.
# We default to 200 TFLOPS (200e12) as a highly reliable baseline for dense BF16 training on H800.
PEAK_H800_FLOPS = 200e12 

def estimate_step_flops(
    n_parameters: int, 
    batch_size: int, 
    seq_len: int, 
    n_layer: int,
    n_embd: int,
    n_head: int,
    use_activation_checkpointing: bool = False
) -> float:
    """
    Estimates the mathematical FLOPs required for a single training step.
    
    A standard Transformer training step (forward + backward pass) requires:
    FLOPs = 6 * N * tokens_per_step
    Where N is the total number of parameters, and tokens_per_step = batch_size * seq_len.
    
    If activation checkpointing is used, we re-run the forward pass once during backward,
    adding another 2 * N * tokens_per_step.
    
    We also add the attention overhead which is quadratic:
    Attention FLOPs = 12 * L * H * S^2 * D
    Where L is n_layer, H is n_head, S is seq_len, D is head_dimension (n_embd / n_head).
    """
    tokens_per_step = batch_size * seq_len
    
    # 1) Base parameter FLOPs
    multiplier = 8.0 if use_activation_checkpointing else 6.0
    param_flops = multiplier * n_parameters * tokens_per_step
    
    # 2) Quadratic Attention FLOPs
    head_dim = n_embd // n_head
    attention_flops = 12.0 * n_layer * n_head * (seq_len ** 2) * head_dim * batch_size
    
    total_flops = param_flops + attention_flops
    return total_flops


def calculate_mfu(
    step_flops: float, 
    elapsed_time: float, 
    grad_accum_steps: int, 
    world_size: int,
    peak_flops: float = PEAK_H800_FLOPS
) -> float:
    """
    Calculates the Model FLOPs Utilization (MFU) percentage.
    
    MFU = (Achieved FLOPs / sec per GPU) / Peak theoretical FLOPs
    """
    if elapsed_time <= 0:
        return 0.0
        
    # Total FLOPs processed across all GPUs during the accumulation step
    total_step_flops = step_flops * grad_accum_steps * world_size
    
    # Achieved FLOPs per second per GPU
    achieved_flops_per_gpu = (total_step_flops / elapsed_time) / world_size
    
    # MFU ratio
    mfu_ratio = achieved_flops_per_gpu / peak_flops
    return mfu_ratio * 100
