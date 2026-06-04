import logging
import os
import subprocess
import torch

logger = logging.getLogger(__name__)

# Cache for peak FLOPS to avoid NVML/subprocess queries on every training step
_cached_peak_flops = None

def get_gpu_peak_flops() -> float:
    """
    Dynamically detects the GPU type, count of SMs, and max clock frequency to
    calculate the theoretical peak dense BF16/FP16 FLOPS directly from the hardware.
    
    If NVML or CUDA is not available, it uses name-based lookup or falls back to
    a standard H800 peak dense value (495 TFLOPS).
    """
    global _cached_peak_flops
    if _cached_peak_flops is not None:
        return _cached_peak_flops

    # Default fallback: H800 SXM5 Dense BF16 Peak (495 TFLOPS)
    default_flops = 495e12

    if not torch.cuda.is_available():
        _cached_peak_flops = default_flops
        return _cached_peak_flops

    try:
        device_id = torch.cuda.current_device()
    except Exception:
        device_id = 0

    try:
        props = torch.cuda.get_device_properties(device_id)
        gpu_name = props.name
        major = props.major
        minor = props.minor
        sm_count = props.multi_processor_count
    except Exception as e:
        logger.warning(f"Failed to query PyTorch CUDA device properties: {e}")
        _cached_peak_flops = default_flops
        return _cached_peak_flops

    max_clock_mhz = None

    # 1. Try to use pynvml (NVIDIA Management Library)
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            # Map PyTorch GPU to NVML GPU using device UUID
            uuid = f"GPU-{props.uuid}"
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        except Exception:
            # Fallback to index matching
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
        max_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        pynvml.nvmlShutdown()
    except Exception:
        # 2. Try to run nvidia-smi as fallback
        try:
            cmd = "nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            clocks = [int(val.strip()) for val in result.stdout.strip().split("\n") if val.strip().isdigit()]
            if clocks:
                if device_id < len(clocks):
                    max_clock_mhz = clocks[device_id]
                else:
                    max_clock_mhz = clocks[0]
        except Exception:
            pass

    # 3. Determine FLOPS per SM per cycle based on Compute Capability for FP16/BF16 dense Tensor Cores
    flops_per_sm_cycle = None
    if major == 9:
        if minor == 0:  # Hopper (H100, H800) -> 2048 FP16/BF16 dense FLOPS per SM per cycle
            flops_per_sm_cycle = 2048
        elif minor == 8: # Blackwell (B100/B200) -> 4096 dense FLOPS
            flops_per_sm_cycle = 4096
    elif major == 8:
        if minor == 0:  # Ampere datacenter (A100, A800) -> 2048 dense FLOPS
            flops_per_sm_cycle = 2048
        elif minor in (6, 9):  # Ampere/Ada Lovelace consumer (RTX 3090, 4090, L4, L40) -> 512 dense FLOPS
            flops_per_sm_cycle = 512
    elif major == 7:
        if minor == 0:  # Volta (V100) -> 512 dense FLOPS
            flops_per_sm_cycle = 512
        elif minor == 5:  # Turing (T4) -> 256 dense FLOPS
            flops_per_sm_cycle = 256

    # 4. Compute peak FLOPS or fall back to dictionary lookup
    if flops_per_sm_cycle is not None and max_clock_mhz is not None:
        calculated_flops = sm_count * (max_clock_mhz * 1e6) * flops_per_sm_cycle
        if 1e10 < calculated_flops < 1e16:
            _cached_peak_flops = float(calculated_flops)
            logger.info(
                f"📟 GPU Peak FLOPS auto-detected from hardware: {gpu_name} ({sm_count} SMs @ {max_clock_mhz} MHz) -> "
                f"{_cached_peak_flops / 1e12:.2f} TFLOPS (dense BF16/FP16)"
            )
            return _cached_peak_flops

    # Fallback: Dictionary Lookup based on GPU Name
    name_lower = gpu_name.lower()
    known_gpus = {
        "h100-sxm": 495e12,
        "h100-pcie": 378e12,
        "h800": 495e12,
        "a100": 312e12,
        "a800": 312e12,
        "a30": 165e12,
        "v100": 125e12,
        "l40": 90e12,
        "l4": 60e12,
        "4090": 165e12,
        "3090": 71e12,
        "3080": 58e12,
        "a6000": 76e12,
        "6000": 91e12,
    }

    for key, val in known_gpus.items():
        if key in name_lower:
            _cached_peak_flops = val
            logger.info(f"📟 GPU Peak FLOPS lookup matched: '{gpu_name}' -> {val / 1e12:.1f} TFLOPS")
            return _cached_peak_flops

    # Ultimate fallback based on compute capability general classes
    if major == 9:
        _cached_peak_flops = 495e12
    elif major == 8:
        _cached_peak_flops = 312e12
    elif major == 7:
        _cached_peak_flops = 125e12
    else:
        _cached_peak_flops = default_flops

    logger.info(f"📟 GPU Peak FLOPS fallback: {gpu_name} (CC {major}.{minor}) -> {_cached_peak_flops / 1e12:.1f} TFLOPS")
    return _cached_peak_flops


# Keep for backward compatibility
PEAK_H800_FLOPS = 495e12


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
    peak_flops: float = None
) -> float:
    """
    Calculates the Model FLOPs Utilization (MFU) percentage.
    
    MFU = (Achieved FLOPs / sec per GPU) / Peak theoretical FLOPs
    """
    if peak_flops is None:
        peak_flops = get_gpu_peak_flops()

    if elapsed_time <= 0:
        return 0.0
        
    # Total FLOPs processed across all GPUs during the accumulation step
    total_step_flops = step_flops * grad_accum_steps * world_size
    
    # Achieved FLOPs per second per GPU
    achieved_flops_per_gpu = (total_step_flops / elapsed_time) / world_size
    
    # MFU ratio
    mfu_ratio = achieved_flops_per_gpu / peak_flops
    return mfu_ratio * 100
