import argparse
import logging
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Pure PyTorch Asymmetric RTN Quantization Math
# ==============================================================================

def quantize_asymmetric_weight(
    weight: torch.Tensor, 
    bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies asymmetric Round-To-Nearest (RTN) weight-only quantization per-channel.
    
    Returns:
        quantized_weight: Integer tensor sharded within range [0, 2^bits - 1]
        scale: Scaling tensor (float)
        zero_point: Zero-point offset tensor (float/int)
    """
    # Weight shape: (out_features, in_features)
    # Quantize along out_features (per-channel, dim=0)
    q_max = (2 ** bits) - 1
    
    # 1. Fetch channel-wise min & max bounds
    w_min = weight.min(dim=-1, keepdim=True)[0]
    w_max = weight.max(dim=-1, keepdim=True)[0]
    
    # Force min/max ranges to contain 0
    w_min = torch.clamp(w_min, max=0.0)
    w_max = torch.clamp(w_max, min=0.0)
    
    # 2. Compute scale: s = (max - min) / (2^b - 1)
    # Avoid zero division with eps
    scale = (w_max - w_min) / q_max
    scale = torch.clamp(scale, min=1e-8)
    
    # 3. Compute zero point: z = round(-min / s)
    zero_point = torch.round(-w_min / scale)
    zero_point = torch.clamp(zero_point, 0.0, q_max)
    
    # 4. Quantize: W_q = clamp(round(W / s) + z, 0, q_max)
    # Broadcast scale and zero-point along channels
    quantized_weight = torch.round(weight / scale) + zero_point
    quantized_weight = torch.clamp(quantized_weight, 0.0, q_max).to(torch.uint8)
    
    return quantized_weight, scale, zero_point


def dequantize_weight(
    quantized_weight: torch.Tensor, 
    scale: torch.Tensor, 
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantizes weight back to float:
    W_dequant = (W_q - z) * s
    """
    return (quantized_weight.float() - zero_point) * scale

# ==============================================================================
# Model-Wide Quantization Pipeline
# ==============================================================================

def quantize_model_checkpoint(
    checkpoint_path: str, 
    output_path: str, 
    bits: int = 8
):
    """
    Loads SFT/DPO checkpoint, parses linear weight layers, 
    applies RTN compression, and saves a simulated quantized checkpoint.
    """
    logger.info(f"Loading checkpoint state from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    
    # Linear layer weight keys that represent projections (wq, wk, wv, wo, w1, w2, w3, output)
    target_layer_keys = [
        "attention.wq.base_layer.weight",
        "attention.wk.base_layer.weight",
        "attention.wv.base_layer.weight",
        "attention.wo.base_layer.weight",
        "attention.wq.weight",
        "attention.wk.weight",
        "attention.wv.weight",
        "attention.wo.weight",
        "feed_forward.w1.base_layer.weight",
        "feed_forward.w2.base_layer.weight",
        "feed_forward.w3.base_layer.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
        "output.weight"
    ]
    
    quantized_state_dict = {}
    
    total_original_bytes = 0
    total_quantized_bytes = 0
    
    logger.info(f"Starting RTN {bits}-bit quantization of linear layers...")
    
    for key, val in state_dict.items():
        # Check if the parameter is a target linear weight
        is_target_linear = any(key.endswith(suffix) for suffix in target_layer_keys)
        
        # We only quantize weight parameters, not biases or RMSNorm scales (standard best practice)
        if is_target_linear and val.ndim == 2:
            original_size_bytes = val.numel() * 2 # FP16/BF16 is 2 bytes
            
            # Apply our custom RTN quantization
            q_w, scale, zp = quantize_asymmetric_weight(val, bits=bits)
            
            # Dequantize to simulate post-training quantization error
            dequant_w = dequantize_weight(q_w, scale, zp).to(val.dtype)
            quantized_state_dict[key] = dequant_w
            
            # Estimated bytes: integer weight (bits/8 bytes) + scale/zp overhead (negligible)
            quantized_size_bytes = val.numel() * (bits / 8)
            
            total_original_bytes += original_size_bytes
            total_quantized_bytes += quantized_size_bytes
            
            logger.info(
                f"Layer: {key[:35]}... | Shape: {list(val.shape)} | "
                f"Original: {original_size_bytes/(1024**2):.1f}MB -> Quantized: {quantized_size_bytes/(1024**2):.1f}MB"
            )
        else:
            # Keep embeddings and norms in high precision (standard practice)
            quantized_state_dict[key] = val
            total_original_bytes += val.numel() * 2
            total_quantized_bytes += val.numel() * 2
            
    # Calculate compression metrics
    compression_ratio = total_original_bytes / total_quantized_bytes
    logger.info("=======================================================================")
    logger.info("📊 Quantization Performance Summary:")
    logger.info(f"💾 Original Checkpoint size: {total_original_bytes / (1024**2):.2f} MB")
    logger.info(f"💾 Quantized Checkpoint size (est): {total_quantized_bytes / (1024**2):.2f} MB")
    logger.info(f"🏎️  Compression Ratio: {compression_ratio:.2f}x")
    logger.info("=======================================================================")
    
    # Save checkpoint
    checkpoint["model_state_dict"] = quantized_state_dict
    checkpoint["quantized_bits"] = bits
    
    logger.info(f"Writing quantized model checkpoint to: {output_path}...")
    torch.save(checkpoint, output_path)
    logger.info("Quantization successfully completed!")

# ==============================================================================
# CLI Orchestrator
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-llm: Round-to-Nearest Weight Quantization Suite")
    parser.add_argument("--src", type=str, required=True, help="Path to SFT or DPO model checkpoint .pt file")
    parser.add_argument("--dest", type=str, required=True, help="Output target path to write quantized checkpoint")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8, help="Quantization bits (4-bit or 8-bit)")
    args = parser.parse_args()
    
    quantize_model_checkpoint(args.src, args.dest, bits=args.bits)
