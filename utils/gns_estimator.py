"""
GNS-Adaptive Batch Size Scheduling (自适应 Batch Size 调度)

Estimates the Gradient Noise Scale (GNS) online during training and dynamically
adjusts `grad_accum_steps`. In early training, uses smaller batch (faster exploration);
in later training, automatically increases batch size as GNS grows.

核心原理:
    GNS = ||G_big||^2 / (||G_small - G_big||^2 / (B - 1))
    其中 G_small 是单个 micro-batch 的梯度, G_big 是完整累积 batch 的梯度,
    B 是当前 batch size. GNS 越大说明梯度噪声越小, 应该增大 batch size.

Reference: McCandlish et al., "An Empirical Model of Large-Batch Training" (2018)
"""

import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientNoiseScaleEstimator:
    """Estimates GNS online and recommends grad_accum_steps adjustments.

    Uses the squared gradient norms from individual micro-batches (G_small) and
    the full accumulated gradient (G_big) to compute the gradient noise scale.

    Args:
        initial_grad_accum_steps: Starting gradient accumulation step count.
        max_grad_accum_steps: Upper bound cap for dynamic adjustment.
        check_interval: How often (in optimizer steps) to evaluate GNS and adjust.
        growth_factor: Multiplicative factor when increasing grad_accum_steps.
        gns_growth_threshold: GNS must exceed this multiple of the running average
            to trigger a batch size increase.
        ema_alpha: Smoothing factor for exponential moving average of GNS.
    """

    def __init__(
        self,
        initial_grad_accum_steps: int = 4,
        max_grad_accum_steps: int = 16,
        check_interval: int = 100,
        growth_factor: float = 2.0,
        gns_growth_threshold: float = 1.5,
        ema_alpha: float = 0.95,
    ):
        self.current_grad_accum_steps = initial_grad_accum_steps
        self.max_grad_accum_steps = max_grad_accum_steps
        self.check_interval = check_interval
        self.growth_factor = growth_factor
        self.gns_growth_threshold = gns_growth_threshold
        self.ema_alpha = ema_alpha

        # Running EMA of GNS for trend detection (GNS 趋势追踪)
        self.gns_ema: Optional[float] = None
        # Latest raw GNS value for logging
        self.latest_gns: float = 0.0

        # Per-step accumulators for micro-batch gradient norms (每步梯度范数累积器)
        self._micro_batch_grad_norms_sq: list = []
        self._accumulated_grad_norm_sq: float = 0.0

    def record_micro_batch_grad(self, model: nn.Module) -> None:
        """Record the squared gradient norm from the current micro-batch.

        Should be called after each micro-batch backward() but before
        zeroing gradients. We snapshot the current gradient state.

        每个 micro-batch backward() 后调用, 记录当前梯度快照的范数平方.
        """
        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += p.grad.detach().float().pow(2).sum().item()
        self._micro_batch_grad_norms_sq.append(grad_norm_sq)

    def record_accumulated_grad(self, model: nn.Module) -> None:
        """Record the squared gradient norm of the fully accumulated gradient.

        Should be called after all micro-batches have been accumulated (before
        optimizer.step()).

        所有 micro-batch 累积完成后调用, 记录完整梯度的范数平方.
        """
        grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm_sq += p.grad.detach().float().pow(2).sum().item()
        self._accumulated_grad_norm_sq = grad_norm_sq

    def compute_gns(self) -> float:
        """Compute the Gradient Noise Scale from recorded gradient norms.

        GNS = ||G_big||^2 / (||G_small - G_big||^2 / (B - 1))

        In practice, we approximate: the denominator uses the variance of
        micro-batch gradient norms as a proxy for ||G_small - G_big||^2.

        Returns:
            The estimated GNS value, or 0.0 if insufficient data.
        """
        B = len(self._micro_batch_grad_norms_sq)
        if B < 2:
            return 0.0

        g_big_sq = self._accumulated_grad_norm_sq

        # Compute variance of micro-batch gradient norms as noise proxy
        # 使用 micro-batch 梯度范数的方差作为噪声的代理估计
        mean_g_small_sq = sum(self._micro_batch_grad_norms_sq) / B
        variance = sum(
            (g - mean_g_small_sq) ** 2 for g in self._micro_batch_grad_norms_sq
        ) / max(1, B - 1)

        # Avoid division by zero (防止除零)
        if variance < 1e-12:
            return float("inf")

        gns = g_big_sq / variance
        return gns

    def step(self, current_step: int) -> int:
        """Evaluate GNS and potentially adjust grad_accum_steps.

        Should be called once per optimizer step. Only modifies
        grad_accum_steps every `check_interval` steps.

        每个优化器 step 后调用. 每 check_interval 步评估一次 GNS 并可能调整 batch size.

        Args:
            current_step: The current training step number.

        Returns:
            The (possibly updated) grad_accum_steps value.
        """
        # Compute and cache latest GNS
        gns = self.compute_gns()
        self.latest_gns = gns

        # Reset per-step accumulators (清空单步累积器)
        self._micro_batch_grad_norms_sq = []
        self._accumulated_grad_norm_sq = 0.0

        # Update EMA (更新指数移动平均)
        if gns > 0 and gns != float("inf"):
            if self.gns_ema is None:
                self.gns_ema = gns
            else:
                self.gns_ema = self.ema_alpha * self.gns_ema + (1 - self.ema_alpha) * gns

        # Only check for adjustment at the specified interval (仅在指定间隔检查)
        if current_step > 0 and current_step % self.check_interval == 0:
            if self.gns_ema is not None and self.gns_ema > 0:
                # If current GNS significantly exceeds EMA, increase batch size
                # GNS 显著超过 EMA 时, 增大 batch size
                if gns > self.gns_growth_threshold * self.gns_ema:
                    new_steps = min(
                        int(self.current_grad_accum_steps * self.growth_factor),
                        self.max_grad_accum_steps,
                    )
                    if new_steps > self.current_grad_accum_steps:
                        logger.info(
                            f"📈 GNS adaptive: increasing grad_accum_steps "
                            f"{self.current_grad_accum_steps} → {new_steps} "
                            f"(GNS={gns:.2f}, EMA={self.gns_ema:.2f})"
                        )
                        self.current_grad_accum_steps = new_steps

        return self.current_grad_accum_steps

    def get_metrics(self) -> Dict[str, Any]:
        """Return current GNS metrics for experiment tracking.

        返回当前 GNS 指标用于实验追踪.
        """
        return {
            "pretrain/gns": self.latest_gns if self.latest_gns != float("inf") else 0.0,
            "pretrain/gns_ema": self.gns_ema if self.gns_ema is not None else 0.0,
            "pretrain/grad_accum_steps": self.current_grad_accum_steps,
        }
