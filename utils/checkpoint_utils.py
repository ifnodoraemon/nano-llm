"""
Shared checkpoint loading with FP8 key translation.
"""

import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def translate_fp8_keys(state_dict: dict) -> dict:
    """Convert FP8 linear layer keys to standard floating-point keys."""
    has_fp8 = any("base_linear" in k for k in state_dict.keys())
    if not has_fp8:
        return state_dict

    logger.info("Native FP8 weights detected. Translating FP8 keys to standard format...")
    clean = {}
    for k, v in state_dict.items():
        if "base_linear.weight" in k:
            clean[k.replace("base_linear.weight", "weight")] = v.half()
        elif "base_linear.bias" in k:
            clean[k.replace("base_linear.bias", "bias")] = v.half()
        elif "w_scale" in k or "x_scale" in k or "y_scale" in k or "scale" in k:
            continue
        else:
            clean[k] = v.half() if isinstance(v, torch.Tensor) else v
    return clean


def load_checkpoint_with_fp8_translation(
    checkpoint_path: str,
    map_location: str = "cpu",
    config_key: str = "config",
    state_keys: Optional[Tuple[str, ...]] = None,
) -> Tuple[dict, dict]:
    """
    Load checkpoint and translate FP8 keys.

    Tries multiple state dict key names in order.

    Returns:
        (model_config, state_dict) tuple.
    """
    if state_keys is None:
        state_keys = ("model_state_dict", "model")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model_config = checkpoint[config_key]

    state_dict = None
    for key in state_keys:
        state_dict = checkpoint.get(key)
        if state_dict is not None:
            break

    if state_dict is None:
        tried = ", ".join(state_keys)
        raise KeyError(f"Could not locate model state dictionary in checkpoint. Tried keys: {tried}")

    state_dict = translate_fp8_keys(state_dict)
    return model_config, state_dict
