"""
LG-Opt: Loss-Gradient Decoupled Rescaling (Loss 偏导梯度自适应重缩放)

Dynamically scales gradients based on how much the current batch loss deviates
from the EMA historical loss. Outlier loss batches (likely dirty/corrupt data)
get their gradients suppressed, while redundant data batches get a mild penalty.

核心原理:
    1. 维护训练 loss 的指数移动平均 (EMA)
    2. backward() 后, optimizer.step() 前, 计算偏差: delta = |loss - ema| / ema
    3. delta > 0.5 (异常高 loss, 可能脏数据): 梯度缩放 0.1
    4. delta < 0.02 (loss 几乎不变, 冗余数据): 梯度缩放 0.8
    5. 其他情况: 不缩放 (factor = 1.0)
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LossGradientRescaler:
    """Rescales gradients based on loss deviation from historical EMA.

    Provides automatic suppression of outlier gradients from dirty data
    and mild penalty for redundant data, improving training robustness.

    Args:
        ema_alpha: Smoothing factor for the loss EMA. Higher = slower adaptation.
            Default 0.99 provides stable tracking with ~100-step memory.
        high_deviation_threshold: If delta > this, treat as outlier (dirty data).
        low_deviation_threshold: If delta < this, treat as redundant data.
        high_deviation_scale: Gradient scaling factor for outlier batches.
        low_deviation_scale: Gradient scaling factor for redundant batches.
        warmup_steps: Number of initial steps before applying rescaling,
            to allow the EMA to stabilize.
    """

    def __init__(
        self,
        ema_alpha: float = 0.99,
        high_deviation_threshold: float = 0.5,
        low_deviation_threshold: float = 0.02,
        high_deviation_scale: float = 0.1,
        low_deviation_scale: float = 0.8,
        warmup_steps: int = 10,
    ):
        self.ema_alpha = ema_alpha
        self.high_deviation_threshold = high_deviation_threshold
        self.low_deviation_threshold = low_deviation_threshold
        self.high_deviation_scale = high_deviation_scale
        self.low_deviation_scale = low_deviation_scale
        self.warmup_steps = warmup_steps

        # EMA state (EMA 状态)
        self.ema_loss: Optional[float] = None
        self.step_count: int = 0
        self.latest_scale_factor: float = 1.0
        self.latest_delta: float = 0.0

    def update_ema(self, current_loss: float) -> None:
        """Update the EMA of training loss.

        更新训练 loss 的指数移动平均.

        Args:
            current_loss: The loss value from the current training step.
        """
        if self.ema_loss is None:
            self.ema_loss = current_loss
        else:
            self.ema_loss = self.ema_alpha * self.ema_loss + (1 - self.ema_alpha) * current_loss
        self.step_count += 1

    def compute_scale_factor(self, current_loss: float) -> float:
        """Compute the gradient scaling factor based on loss deviation.

        根据 loss 偏差计算梯度缩放系数.

        Args:
            current_loss: The loss value from the current training step.

        Returns:
            Scale factor to apply to all gradients (0.1, 0.8, or 1.0).
        """
        # During warmup, no rescaling (预热阶段不缩放)
        if self.step_count < self.warmup_steps or self.ema_loss is None:
            self.latest_scale_factor = 1.0
            self.latest_delta = 0.0
            return 1.0

        # Compute relative deviation (计算相对偏差)
        if abs(self.ema_loss) < 1e-8:
            # EMA is essentially zero, skip rescaling
            self.latest_scale_factor = 1.0
            self.latest_delta = 0.0
            return 1.0

        delta = abs(current_loss - self.ema_loss) / abs(self.ema_loss)
        self.latest_delta = delta

        if delta > self.high_deviation_threshold:
            # Abnormally high loss — likely dirty/corrupt data (异常高 loss, 可能脏数据)
            self.latest_scale_factor = self.high_deviation_scale
        elif delta < self.low_deviation_threshold:
            # Loss barely changing — redundant data (loss 几乎不变, 冗余数据)
            self.latest_scale_factor = self.low_deviation_scale
        else:
            # Normal range — no scaling (正常范围, 不缩放)
            self.latest_scale_factor = 1.0

        return self.latest_scale_factor

    def rescale_gradients(self, model: nn.Module, current_loss: float) -> float:
        """Compute scale factor and apply it to all model gradients in-place.

        Should be called after backward() but before optimizer.step().

        backward() 之后, optimizer.step() 之前调用.
        计算缩放系数并原地应用到所有模型梯度.

        Args:
            model: The model whose gradients to rescale.
            current_loss: The loss value for the current step.

        Returns:
            The scale factor that was applied.
        """
        scale_factor = self.compute_scale_factor(current_loss)

        # Apply in-place gradient scaling (原地梯度缩放)
        if scale_factor != 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale_factor)

            if self.step_count % 100 == 0 or scale_factor == self.high_deviation_scale:
                logger.info(
                    f"🔧 LG-Opt: scale={scale_factor:.2f}, "
                    f"delta={self.latest_delta:.4f}, "
                    f"loss={current_loss:.4f}, ema_loss={self.ema_loss:.4f}"
                )

        # Update EMA after computing scale factor (计算缩放后更新 EMA)
        self.update_ema(current_loss)

        return scale_factor

    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics for experiment tracking.

        返回指标用于实验追踪.
        """
        return {
            "pretrain/lg_opt_scale": self.latest_scale_factor,
            "pretrain/lg_opt_delta": self.latest_delta,
            "pretrain/lg_opt_ema_loss": self.ema_loss if self.ema_loss is not None else 0.0,
        }
