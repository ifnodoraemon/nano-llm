"""Shared training utilities: cosine LR scheduler, optimizer weight-decay grouping,
and dataset validation.

Eliminates duplication of get_learning_rate() and configure_optimizers()
across pretrain/train/align training scripts.
"""

import math
import logging
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def validate_dataset(dataset: Dataset, tokenizer=None, name: str = "dataset") -> None:
    """Validate a dataset before training: check it's non-empty and log key stats."""
    if len(dataset) == 0:
        raise ValueError(f"Dataset '{name}' is empty — training cannot proceed.")

    logger.info(f"Dataset '{name}': {len(dataset)} samples loaded")

    if tokenizer is not None and len(dataset) > 0:
        sample = dataset[0]
        if "prompt_ids" in sample:
            tok_len = len(sample["prompt_ids"])
        elif "input_ids" in sample:
            tok_len = sample["input_ids"].numel() if hasattr(sample["input_ids"], "numel") else len(sample["input_ids"])
        else:
            tok_len = None
        if tok_len is not None:
            logger.info(f"  First sample token length: {tok_len}")


def assert_grad_accum_safe(dataloader: DataLoader, grad_accum_steps: int, batch_size: int) -> None:
    """Assert that gradient accumulation will actually trigger optimizer steps.

    If len(dataloader) < grad_accum_steps, the modulo check
    (batch_idx + 1) % grad_accum_steps == 0 never fires and training silently fails.
    """
    if len(dataloader) < grad_accum_steps:
        raise AssertionError(
            f"grad_accum_steps ({grad_accum_steps}) exceeds dataloader length ({len(dataloader)}). "
            f"Optimizer will never step. Reduce grad_accum_steps or increase dataset/batch_size."
        )


def get_cosine_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, decay_steps: int) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step > decay_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizers(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple = (0.9, 0.95),
    device_type: str = "cuda",
) -> torch.optim.AdamW:
    """Create AdamW optimizer with weight decay applied only to 2D+ parameters."""
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    logger.info(f"Optimizer: decay params={len(decay_params)} ({num_decay:,}), nodecay params={len(nodecay_params)} ({num_nodecay:,})")

    fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == "cuda"
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"Using fused AdamW: {use_fused}")

    return optimizer
