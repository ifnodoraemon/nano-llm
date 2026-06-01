"""
Unified experiment tracking: WandB with TensorBoard fallback.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified logging interface for WandB (primary) with TensorBoard fallback."""

    def __init__(
        self,
        project: str = "nano-llm",
        config: Optional[Dict[str, Any]] = None,
        mode: str = "auto",  # "wandb", "tensorboard", "auto", "offline"
        log_dir: str = "./logs",
    ):
        self.project = project
        self.log_dir = log_dir
        self.mode = mode
        self.backend = None
        self.writer = None
        self._step = 0

        os.makedirs(log_dir, exist_ok=True)

        if mode == "offline" or mode == "disabled":
            return

        # Try WandB first
        if mode in ("wandb", "auto"):
            try:
                import wandb
                wandb_mode = "offline" if mode == "offline" else "online"
                wandb.init(
                    project=project,
                    config=config or {},
                    dir=log_dir,
                    mode=wandb_mode,
                )
                self.backend = "wandb"
                logger.info(f"WandB initialized (project={project})")
                return
            except Exception as e:
                if mode == "wandb":
                    raise
                logger.warning(f"WandB init failed ({e}), falling back to TensorBoard")

        # Fallback to TensorBoard
        if mode in ("tensorboard", "auto"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
                self.backend = "tensorboard"
                logger.info(f"TensorBoard initialized (log_dir={log_dir})")
                return
            except Exception as e:
                if mode == "tensorboard":
                    raise
                logger.warning(f"TensorBoard init failed ({e}), logging disabled")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics dict at current step."""
        if step is not None:
            self._step = step

        if self.backend == "wandb":
            import wandb
            wandb.log(metrics, step=self._step)
        elif self.backend == "tensorboard" and self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, self._step)
                elif isinstance(v, dict):
                    self.writer.add_scalars(k, v, self._step)

        self._step += 1

    def log_config(self, config: Dict[str, Any]):
        if self.backend == "wandb":
            import wandb
            wandb.config.update(config)

    def log_model(self, model_path: str):
        if self.backend == "wandb":
            import wandb
            wandb.save(model_path)

    def finish(self):
        if self.backend == "wandb":
            import wandb
            wandb.finish()
        elif self.backend == "tensorboard" and self.writer is not None:
            self.writer.close()

    @property
    def step(self) -> int:
        return self._step
