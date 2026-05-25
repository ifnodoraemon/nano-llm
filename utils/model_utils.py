import torch
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def configure_logging(level=logging.INFO):
    """Configures clean console output formatting."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level
    )

def count_parameters(model) -> Dict[str, int]:
    """
    Calculates total and trainable parameters inside a model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "percentage_trainable": 100 * trainable_params / total_params if total_params > 0 else 0
    }
