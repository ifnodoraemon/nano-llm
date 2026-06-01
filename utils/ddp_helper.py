"""Distributed Data Parallel (DDP) initialization helper.

Eliminates the DDP init boilerplate duplicated across pretrain/train/align/grpo.
"""

import os
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def init_ddp() -> dict:
    """Initialize DDP process group and return distributed context.

    Returns a dict with keys:
        ddp: bool — whether DDP is active
        rank: int — global rank
        local_rank: int — local rank on this node
        world_size: int — total number of processes
        device: str — CUDA device string or "cpu"
        is_master: bool — True if this is the master process (rank 0)
    """
    ddp_active = "WORLD_SIZE" in os.environ
    if ddp_active:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        is_master = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_master = True
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "ddp": ddp_active,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_master": is_master,
    }
