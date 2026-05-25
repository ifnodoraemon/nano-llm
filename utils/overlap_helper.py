import os
import torch
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class OverlapCommunicationHelper:
    """
    High-performance 4D Parallelism Communication-Computation Overlapping Helper.
    Spawns asynchronous non-blocking pre-fetching operations using PyTorch distributed
    all_gather_into_tensor. Optimized exclusively for Hopper H800 clusters.
    """
    def __init__(self, use_ddp: bool = True):
        self.use_ddp = use_ddp
        self._async_work: Optional[Any] = None
        self._prefetched_param: Optional[torch.Tensor] = None

    def prefetch_next_layer_weights(self, next_layer_weight: torch.Tensor, device: torch.device):
        """
        Asynchronously schedules pre-fetching of parameters of the next layer
        overlapping with active matrix multiplications on GPU via NCCL.
        """
        # Allocate flat buffer and issue asynchronous non-blocking NCCL all_gather
        flat_buffer = torch.empty_like(next_layer_weight, device=device)
        work = torch.distributed.all_gather_into_tensor(
            flat_buffer, 
            next_layer_weight, 
            async_op=True
        )
        self._async_work = work
        self._prefetched_param = flat_buffer
        logger.debug("📡 [Overlap Helper] Asynchronous NCCL prefetch task successfully enqueued!")

    def wait_and_retrieve_prefetched(self) -> Optional[torch.Tensor]:
        """
        Blocks until the pre-fetching NCCL operation completes and returns the parameter.
        """
        if self._async_work is not None:
            self._async_work.wait() # Block until NCCL transfer completes
            
        param = self._prefetched_param
        self._async_work = None
        self._prefetched_param = None
        return param
