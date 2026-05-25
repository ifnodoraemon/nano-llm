import os
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TrainingTelemetry:
    """
    Tracks LLM training statistics (loss, gradient norm) and provides early-warning
    safeguards. Automatically triggers checkpoint weight rollback and learning rate
    decay when severe divergence or gradient spikes (loss spikes) are detected.
    """
    def __init__(
        self,
        ema_alpha: float = 0.95,
        spike_threshold: float = 5.0,
        loss_spike_factor: float = 1.5,
        min_steps_before_checking: int = 10
    ):
        """
        Args:
            ema_alpha: Smoothing factor for Exponential Moving Average (EMA).
            spike_threshold: Threshold factor for gradient norm spike (e.g. 5x of EMA).
            loss_spike_factor: Threshold factor for sudden loss increase (e.g. 1.5x of EMA).
            min_steps_before_checking: Warmup steps before actively triggering self-healing.
        """
        self.ema_alpha = ema_alpha
        self.spike_threshold = spike_threshold
        self.loss_spike_factor = loss_spike_factor
        self.min_steps_before_checking = min_steps_before_checking
        
        self.loss_ema = None
        self.grad_norm_ema = None
        self.steps_tracked = 0
        
    def update(self, loss: float, grad_norm: float):
        """
        Updates EMA tracking statistics with the current training step metrics.
        """
        self.steps_tracked += 1
        
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = self.ema_alpha * self.loss_ema + (1.0 - self.ema_alpha) * loss
            
        if self.grad_norm_ema is None:
            self.grad_norm_ema = grad_norm
        else:
            self.grad_norm_ema = self.ema_alpha * self.grad_norm_ema + (1.0 - self.ema_alpha) * grad_norm

    def check_anomaly(self, loss: float, grad_norm: float) -> Dict[str, Any]:
        """
        Checks if the current step's loss or gradient norm indicates an anomaly.
        """
        result = {"is_anomaly": False, "reason": None}
        
        # Check for NaN / Inf first
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            result["is_anomaly"] = True
            result["reason"] = "NaN or Inf loss detected"
            return result
            
        if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
            result["is_anomaly"] = True
            result["reason"] = "NaN or Inf gradient norm detected"
            return result
            
        # Check against EMA baseline after warm-up steps
        if self.steps_tracked > self.min_steps_before_checking:
            # Gradient Norm spike check
            if self.grad_norm_ema is not None and grad_norm > self.spike_threshold * self.grad_norm_ema:
                result["is_anomaly"] = True
                result["reason"] = f"Gradient norm spike: current={grad_norm:.4f}, EMA={self.grad_norm_ema:.4f} (> {self.spike_threshold}x)"
                return result
                
            # Loss spike check
            if self.loss_ema is not None and loss > self.loss_spike_factor * self.loss_ema:
                result["is_anomaly"] = True
                result["reason"] = f"Sudden loss spike: current={loss:.4f}, EMA={self.loss_ema:.4f} (> {self.loss_spike_factor}x)"
                return result
                
        return result

    def check_and_rollback(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: float,
        grad_norm: float,
        checkpoint_path: str,
        current_lr: float,
        lr_decay_factor: float = 0.5
    ) -> Dict[str, Any]:
        """
        Verifies step metrics and dynamically rolls back model parameters & reduces learning rate
        if an anomaly or loss spike is detected.
        
        Returns:
            Dict containing rollback status, new learning rate, and log message.
        """
        self.update(loss, grad_norm)
        check = self.check_anomaly(loss, grad_norm)
        
        if not check["is_anomaly"]:
            return {"rolled_back": False, "new_lr": current_lr}
            
        logger.warning(f"⚠️ TRAINING STABILITY ALARM TRIGGERED! Reason: {check['reason']}")
        
        # Attempt checkpoint rollback
        rollback_success = False
        if os.path.exists(checkpoint_path):
            try:
                logger.info(f"Attempting weight rollback to last stable checkpoint: '{checkpoint_path}'...")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Extract state dict
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                    
                # Load parameters into active model
                # Unwrap DDP if model is wrapped
                raw_model = model.module if hasattr(model, "module") else model
                raw_model.load_state_dict(state_dict)
                rollback_success = True
                logger.info("Successfully rolled back active model parameters.")
            except Exception as e:
                logger.error(f"Failed to load rollback checkpoint: {e}")
        else:
            logger.warning(f"No stable checkpoint found at '{checkpoint_path}' for rollback. Continuing with current parameters.")
            
        # Steer learning rate downwards
        new_lr = current_lr * lr_decay_factor
        logger.info(f"Steering learning rate down: {current_lr:.6e} -> {new_lr:.6e} (factor = {lr_decay_factor})")
        
        for g in optimizer.param_groups:
            g["lr"] = new_lr
            
        # Reset EMA tracking metrics to let the system stabilize at new LR
        self.loss_ema = None
        self.grad_norm_ema = None
        
        return {
            "rolled_back": True,
            "rollback_success": rollback_success,
            "new_lr": new_lr,
            "reason": check["reason"]
        }
