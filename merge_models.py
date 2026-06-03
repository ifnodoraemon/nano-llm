import os
import argparse
import logging
import torch
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Spherical Linear Interpolation (SLERP) for smooth weight blending."""
    low_norm = low / (torch.norm(low) + 1e-8)
    high_norm = high / (torch.norm(high) + 1e-8)
    
    # Calculate cosine of angle between vectors
    dot = torch.dot(low_norm.flatten(), high_norm.flatten())
    omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
    so = torch.sin(omega)
    
    if so < 1e-6:
        # Fallback to linear interpolation if angle is extremely small
        return (1.0 - val) * low + val * high
        
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res

def ties_merge(base_weight: torch.Tensor, active_weights: List[torch.Tensor], fraction: float = 0.2) -> torch.Tensor:
    """TIES-Merging (Pruning bottom magnitude, sign consensus, and averaging)."""
    # 1. Compute weight deltas relative to base
    deltas = [w - base_weight for w in active_weights]
    
    # 2. Trim: keep top fraction by magnitude and mask the rest to 0
    trimmed_deltas = []
    for d in deltas:
        d_flat = d.flatten()
        k = int(d_flat.numel() * fraction)
        if k > 0:
            threshold = torch.topk(d_flat.abs(), k).values[-1]
            mask = d.abs() >= threshold
            trimmed_deltas.append(d * mask)
        else:
            trimmed_deltas.append(torch.zeros_like(d))
            
    # 3. Sign Agreement: determine parameter signs that agree across deltas
    sign_sum = torch.zeros_like(base_weight)
    for td in trimmed_deltas:
        sign_sum += td.sign()
        
    majority_sign = sign_sum.sign()
    
    # 4. Elective Sum: sum up deltas that match the majority sign, and average
    total_delta = torch.zeros_like(base_weight)
    count = torch.zeros_like(base_weight)
    for td in trimmed_deltas:
        matching_sign_mask = (td.sign() == majority_sign) & (majority_sign != 0)
        total_delta += td * matching_sign_mask
        count += matching_sign_mask.float()
        
    # Avoid division by zero
    count = torch.clamp(count, min=1.0)
    final_delta = total_delta / count
    
    return base_weight + final_delta

def dare_merge(base_weight: torch.Tensor, active_weights: List[torch.Tensor], drop_rate: float = 0.2) -> torch.Tensor:
    """DARE (Drop and Rescale) Merge."""
    merged_delta = torch.zeros_like(base_weight)
    for w in active_weights:
        delta = w - base_weight
        # Randomly drop delta values
        mask = torch.rand_like(delta) > drop_rate
        dropped_delta = delta * mask
        # Rescale the remaining deltas
        rescaled_delta = dropped_delta / (1.0 - drop_rate)
        merged_delta += rescaled_delta
        
    return base_weight + (merged_delta / len(active_weights))

def merge_checkpoints(
    base_path: str,
    models_to_merge: List[str],
    output_path: str,
    method: str = "ties",
    slerp_val: float = 0.5,
    ties_fraction: float = 0.2,
    dare_drop_rate: float = 0.2
):
    logger.info(f"Loading base model checkpoint from: {base_path}")
    from utils.checkpoint_utils import load_checkpoint_with_fp8_translation
    base_config, base_state = load_checkpoint_with_fp8_translation(base_path, map_location="cpu")
    
    active_states = []
    for path in models_to_merge:
        logger.info(f"Loading checkpoints to merge: {path}")
        _, state = load_checkpoint_with_fp8_translation(path, map_location="cpu")
        active_states.append(state)
        
    merged_state = {}
    
    # Merge parameter keys present in base state
    keys = list(base_state.keys())
    for key in keys:
        base_val = base_state[key]
        
        # Verify if key is present in all models to merge
        if not all(key in state for state in active_states):
            logger.warning(f"Key {key} not present in all checkpoints. Skipping merge for this key.")
            merged_state[key] = base_val
            continue
            
        active_vals = [state[key] for state in active_states]
        
        # Only merge floating point tensors (weights), keep long/integer/non-tensors intact
        if not isinstance(base_val, torch.Tensor) or not torch.is_floating_point(base_val):
            merged_state[key] = base_val
            continue
            
        if method == "linear":
            # Simple average
            avg_val = torch.stack(active_vals).mean(dim=0)
            merged_state[key] = avg_val
        elif method == "slerp":
            if len(active_vals) != 1:
                raise ValueError("SLERP merge method currently supports exactly one model to merge with base.")
            merged_state[key] = slerp(slerp_val, base_val, active_vals[0])
        elif method == "ties":
            merged_state[key] = ties_merge(base_val, active_vals, fraction=ties_fraction)
        elif method == "dare":
            merged_state[key] = dare_merge(base_val, active_vals, drop_rate=dare_drop_rate)
        else:
            raise ValueError(f"Unknown merge method: {method}")
            
    logger.info(f"Saving merged checkpoint to: {output_path} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "model_state_dict": merged_state,
        "config": base_config
    }, output_path)
    logger.info("Model merge complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-llm: Model Merging Toolkit (SLERP / TIES / DARE)")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model .pt checkpoint")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of model checkpoints to merge with base")
    parser.add_argument("--output_path", type=str, required=True, help="Target path to save merged checkpoint")
    parser.add_argument("--method", type=str, choices=["linear", "slerp", "ties", "dare"], default="ties", help="Merging algorithm")
    parser.add_argument("--slerp_val", type=float, default=0.5, help="Interpolation factor for SLERP (default: 0.5)")
    parser.add_argument("--ties_fraction", type=float, default=0.2, help="Fraction of weight deltas to keep in TIES (default: 0.2)")
    parser.add_argument("--dare_drop_rate", type=float, default=0.2, help="Dropout rate for DARE (default: 0.2)")
    args = parser.parse_args()
    
    model_paths = [p.strip() for p in args.models.split(",") if p.strip()]
    merge_checkpoints(
        base_path=args.base_model,
        models_to_merge=model_paths,
        output_path=args.output_path,
        method=args.method,
        slerp_val=args.slerp_val,
        ties_fraction=args.ties_fraction,
        dare_drop_rate=args.dare_drop_rate
    )
