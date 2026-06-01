import os
import json
import torch
import threading
import logging
import copy
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    elif isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    else:
        return obj

class BackgroundCheckpointSaver:
    """
    Non-blocking, thread-safe asynchronous checkpoint writer.
    Copies model weights to host CPU memory on the main thread (extremely fast, <100ms),
    and delegates heavy disk serialization (torch.save) to a background thread to prevent
    training iteration jitter.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._active_thread: Optional[threading.Thread] = None

    def _async_save_worker(self, data: Dict[str, Any], filepath: str, manifestpath: str, manifest_data: Dict[str, Any]):
        try:
            # 1. Write the main checkpoint (.pt file) in background thread
            torch.save(data, filepath)
            
            # 2. Write the manifest metadata
            with open(manifestpath, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
                
            logger.info(f"💾 [Asynchronous Checkpoint] Successfully saved checkpoint to {filepath} and updated manifest.")
        except Exception as e:
            logger.error(f"❌ [Asynchronous Checkpoint] Failed to save checkpoint: {e}")
        finally:
            with self._lock:
                self._active_thread = None

    def save_checkpoint(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any],
        config: Any,
        step: int, 
        epoch: int,
        loss: float,
        out_dir: str
    ) -> bool:
        """
        Enqueues a new non-blocking asynchronous checkpoint write.
        Returns True if successfully queued, False if a previous save is still active.
        """
        with self._lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                logger.warning("⚠️ [Asynchronous Checkpoint] Skipping save request: Previous saving thread is still running.")
                return False
                
        # 1. Perform CPU parameter clone on main thread (fast and thread-safe)
        logger.info(f"⚡ [Asynchronous Checkpoint] Deep-copying state dicts for step {step} on main thread...")
        
        # Extract base state dict (unwrap DDP if needed)
        raw_model = model.module if hasattr(model, "module") else model
        model_state_cpu = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
        optimizer_state_cpu = _to_cpu(optimizer.state_dict())
        
        scheduler_state = None
        if lr_scheduler is not None:
            scheduler_state = _to_cpu(lr_scheduler.state_dict())
            
        checkpoint_data = {
            "model_state_dict": model_state_cpu,
            "optimizer_state_dict": optimizer_state_cpu,
            "scheduler_state_dict": scheduler_state,
            "config": config,
            "step": step,
            "epoch": epoch,
            "loss": loss
        }
        
        import time
        manifest_data = {
            "latest_step": step,
            "latest_epoch": epoch,
            "latest_loss": loss,
            "timestamp": float(time.time())
        }
        
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, "checkpoint_elastic.pt")
        manifestpath = os.path.join(out_dir, "training_manifest.json")
        
        # 2. Launch background thread for heavy disk IO
        save_thread = threading.Thread(
            target=self._async_save_worker,
            args=(checkpoint_data, filepath, manifestpath, manifest_data)
        )
        save_thread.daemon = True
        
        with self._lock:
            self._active_thread = save_thread
            save_thread.start()
            
        logger.info("🚀 [Asynchronous Checkpoint] Main thread resumed immediately!")
        return True


class ElasticRestoreManager:
    """
    Manages elastic fault-tolerant hot-restoring. Automatically detects valid checkpoint 
    manifests and loads the training states in a single line, allowing automatic recovery
    from node drops or silent hardware fail.
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.filepath = os.path.join(out_dir, "checkpoint_elastic.pt")
        self.manifestpath = os.path.join(out_dir, "training_manifest.json")

    def auto_detect_checkpoint(self) -> Tuple[bool, int, int]:
        """
        Returns (exists_valid_checkpoint, latest_step, latest_epoch).
        """
        if os.path.exists(self.filepath) and os.path.exists(self.manifestpath):
            try:
                with open(self.manifestpath, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                return True, manifest.get("latest_step", 0), manifest.get("latest_epoch", 0)
            except Exception:
                pass
        return False, 0, 0

    def restore_training_state(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None
    ) -> Tuple[int, int, float]:
        """
        Loads the saved training state into model and optimizer.
        Returns (restored_step, restored_epoch, restored_loss).
        """
        logger.info(f"🔄 [Elastic Recovery] Attemping training state recovery from {self.filepath}...")
        checkpoint = torch.load(self.filepath, map_location="cpu", weights_only=False)
        
        # Load weights
        raw_model = model.module if hasattr(model, "module") else model
        state_dict = checkpoint["model_state_dict"]
        # Ensure older checkpoints without _is_scale_initialized are compatible
        for k in list(state_dict.keys()):
            if "x_scale" in k:
                init_key = k.replace("x_scale", "_is_scale_initialized")
                if init_key not in state_dict:
                    state_dict[init_key] = torch.tensor(True)
        raw_model.load_state_dict(state_dict)
        
        # Load optimizer
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler
        if lr_scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        step = checkpoint.get("step", 0)
        epoch = checkpoint.get("epoch", 0)
        loss = checkpoint.get("loss", 0.0)
        
        border = "=" * 60
        logger.info(border)
        logger.info(f"🎉 [Elastic Recovery] SUCCESSFUL training state self-healed!")
        logger.info(f"  🔹 Resuming from Step: {step}")
        logger.info(f"  🔹 Resuming from Epoch: {epoch}")
        logger.info(f"  🔹 Latest training Loss: {loss:.4f}")
        logger.info(border)
        
        return step, epoch, loss
