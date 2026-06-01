"""
Centralized configuration system for nano-llm.
Supports hardware auto-detection, preset profiles, and dev/prod mode switching.

Usage:
    from config import load_config, ConfigPreset

    # Load with hardware auto-detection
    cfg = load_config("prod")

    # Use preset
    cfg = load_config("dev", model_size="16B-equivalent")
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Models
# =============================================================================


@dataclass
class ModelPreset:
    """Pre-configured model architecture preset."""

    name: str
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 4096
    vocab_size: int = 32000
    use_mla: bool = False
    use_moe: bool = False
    kv_comp_dim: int = 128
    num_shared_experts: int = 1
    num_routed_experts: int = 8
    num_active_experts: int = 2


@dataclass
class HardwareProfile:
    """Auto-detected or manually specified hardware configuration."""

    num_gpus: int = 1
    gpu_name: str = "unknown"
    gpu_memory_gb: float = 0
    recommended_batch_size: int = 4
    recommended_grad_accum: int = 4
    supports_fp8: bool = False
    supports_bf16: bool = False


@dataclass
class TrainingConfig:
    """Unified training configuration."""

    # General
    mode: Literal["dev", "prod"] = "dev"
    seed: int = 42

    # Hardware
    num_gpus: int = 1
    master_port: int = 29500

    # Model preset
    model_size: str = "tiny"
    model_preset: Optional[ModelPreset] = None

    # Training hyperparameters
    batch_size: int = 4
    grad_accum_steps: int = 4
    max_steps: int = 200
    epochs: int = 3
    max_lr: float = 2e-5
    min_lr: float = 2e-6
    warmup_steps: int = 100
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    max_length: int = 4096

    # Distributed training
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1

    # Precision
    use_fp8: bool = False
    use_triton: bool = False
    use_triton_mla: bool = False
    use_lora: bool = False
    use_checkpoint: bool = False

    # Data
    data_dir: str = "./data"
    sft_data: str = "./data/train_sft_premium.jsonl"
    dpo_data: str = "./data/train_dpo_premium.jsonl"
    grpo_data: str = "./data/train_grpo_premium.jsonl"

    # Model
    block_size: int = 1024

    # Output
    output_dir: str = "./outputs"
    log_dir: str = "./logs"

    # Checkpoint
    sft_checkpoint_path: str = "./outputs/checkpoint_sft.pt"
    dpo_checkpoint_path: str = "./outputs_dpo/checkpoint_dpo.pt"

    # DPO specific
    dpo_beta: float = 0.1
    max_prompt_length: int = 2048

    # GRPO specific
    group_size: int = 4
    max_prompt_len: int = 512
    max_gen_len: int = 256
    grpo_beta: float = 0.04
    clip_eps: float = 0.2

    def __post_init__(self):
        # Dynamically align checkpoint paths based on output_dir
        self.sft_checkpoint_path = os.path.join(self.output_dir, "checkpoint_sft.pt")
        self.dpo_checkpoint_path = os.path.join(self.output_dir.replace("outputs", "outputs_dpo"), "checkpoint_dpo.pt")


# =============================================================================
# Model Presets
# =============================================================================

MODEL_PRESETS: dict[str, ModelPreset] = {
    "tiny": ModelPreset(
        name="tiny",
        n_layer=4,
        n_head=8,
        n_embd=512,
        block_size=1024,
        vocab_size=32000,
    ),
    "small": ModelPreset(
        name="small",
        n_layer=12,
        n_head=16,
        n_embd=1024,
        block_size=2048,
        vocab_size=32000,
    ),
    "medium": ModelPreset(
        name="medium",
        n_layer=24,
        n_head=16,
        n_embd=2048,
        block_size=4096,
        vocab_size=32000,
    ),
    "16B-equivalent": ModelPreset(
        name="16B-equivalent",
        n_layer=24,
        n_head=32,
        n_embd=2048,
        block_size=4096,
        vocab_size=32000,
        use_mla=True,
        use_moe=True,
        kv_comp_dim=128,
        num_shared_experts=1,
        num_routed_experts=8,
        num_active_experts=2,
    ),
    "1B-equivalent": ModelPreset(
        name="1B-equivalent",
        n_layer=16,
        n_head=16,
        n_embd=1536,
        block_size=4096,
        vocab_size=32000,
        use_mla=True,
        use_moe=True,
        kv_comp_dim=128,
        num_shared_experts=1,
        num_routed_experts=8,
        num_active_experts=2,
    ),
    "3B-equivalent": ModelPreset(
        name="3B-equivalent",
        n_layer=20,
        n_head=24,
        n_embd=2048,
        block_size=4096,
        vocab_size=32000,
        use_mla=True,
        use_moe=True,
        kv_comp_dim=128,
        num_shared_experts=1,
        num_routed_experts=8,
        num_active_experts=2,
    ),
    "7B-equivalent": ModelPreset(
        name="7B-equivalent",
        n_layer=28,
        n_head=32,
        n_embd=2560,
        block_size=4096,
        vocab_size=32000,
        use_mla=True,
        use_moe=True,
        kv_comp_dim=128,
        num_shared_experts=1,
        num_routed_experts=8,
        num_active_experts=2,
    ),
    "1.5B-dense": ModelPreset(
        name="1.5B-dense",
        n_layer=24,
        n_head=32,
        n_embd=2048,
        block_size=4096,
        vocab_size=32000,
    ),
    "2.7B-dense": ModelPreset(
        name="2.7B-dense",
        n_layer=32,
        n_head=32,
        n_embd=2560,
        block_size=4096,
        vocab_size=32000,
    ),
}


# =============================================================================
# Hardware Auto-Detection
# =============================================================================


def detect_hardware() -> HardwareProfile:
    """Auto-detect available hardware and return recommended settings."""
    try:
        import torch
    except ImportError:
        return HardwareProfile(num_gpus=0, gpu_name="no-torch")

    if not torch.cuda.is_available():
        return HardwareProfile(num_gpus=0, gpu_name="cpu-only")

    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    gpu_memory_gb = props.total_memory / (1024**3)

    # Determine capabilities
    supports_bf16 = torch.cuda.is_bf16_supported()
    supports_fp8 = props.major >= 9  # Hopper+

    # Heuristic batch size based on GPU memory
    if gpu_memory_gb >= 80:
        recommended_batch_size = 8
        recommended_grad_accum = 4
    elif gpu_memory_gb >= 40:
        recommended_batch_size = 4
        recommended_grad_accum = 4
    elif gpu_memory_gb >= 24:
        recommended_batch_size = 2
        recommended_grad_accum = 8
    else:
        recommended_batch_size = 1
        recommended_grad_accum = 16

    return HardwareProfile(
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        recommended_batch_size=recommended_batch_size,
        recommended_grad_accum=recommended_grad_accum,
        supports_fp8=supports_fp8,
        supports_bf16=supports_bf16,
    )


# =============================================================================
# Config Loading
# =============================================================================


def load_config(
    mode: Literal["dev", "prod"] = "dev",
    model_size: Optional[str] = None,
    num_gpus: Optional[int] = None,
    **overrides,
) -> TrainingConfig:
    """
    Load a unified training configuration.

    Args:
        mode: "dev" for fast iteration, "prod" for full-scale training.
        model_size: Key from MODEL_PRESETS (default: "tiny" for dev, "1.5B-dense" for prod).
        num_gpus: Override detected GPU count.
        **overrides: Any TrainingConfig field can be overridden.
    """
    hw = detect_hardware()

    if model_size is None:
        model_size = "tiny" if mode == "dev" else "1.5B-dense"

    if model_size not in MODEL_PRESETS:
        raise ValueError(f"Unknown model size '{model_size}'. Available: {list(MODEL_PRESETS.keys())}")

    preset = MODEL_PRESETS[model_size]
    actual_gpus = num_gpus if num_gpus is not None else max(hw.num_gpus, 1)

    # Dev mode: smaller steps, faster iteration
    if mode == "dev":
        cfg = TrainingConfig(
            mode="dev",
            model_size=model_size,
            model_preset=preset,
            num_gpus=actual_gpus,
            master_port=29500,
            batch_size=min(hw.recommended_batch_size, 2),
            grad_accum_steps=2,
            max_steps=10,
            epochs=1,
            max_lr=1e-4,
            min_lr=1e-5,
            warmup_steps=5,
            max_length=512,
            use_fp8=False,
            use_triton=False,
            use_triton_mla=False,
            use_lora=False,
            output_dir="./outputs_dev",
            log_dir="./logs",
        )
    else:
        cfg = TrainingConfig(
            mode="prod",
            model_size=model_size,
            model_preset=preset,
            num_gpus=actual_gpus,
            master_port=29500,
            batch_size=hw.recommended_batch_size,
            grad_accum_steps=hw.recommended_grad_accum,
            max_steps=5000,
            epochs=3,
            max_lr=6e-4,
            min_lr=6e-5,
            warmup_steps=100,
            use_fp8=hw.supports_fp8,
            use_triton=False,
            use_triton_mla=False,
            use_lora=False,
            output_dir="./outputs",
            log_dir="./logs",
        )

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logger.warning(f"Ignoring unknown config override: {key}={value}")

    return cfg


def config_to_model_config(cfg: TrainingConfig, **kwargs) -> "ModelConfig":
    """Convert a TrainingConfig to a ModelConfig for use with model.py.

    Extra kwargs are forwarded to ModelConfig to allow overrides (e.g., vision_dim=None).
    """
    from model import ModelConfig

    # Extract common config/parallelism overrides from kwargs to avoid duplicate keyword argument errors
    block_size = kwargs.pop("block_size", cfg.block_size)
    tp_size = kwargs.pop("tp_size", cfg.tp_size)
    pp_size = kwargs.pop("pp_size", cfg.pp_size)
    ep_size = kwargs.pop("ep_size", cfg.ep_size)
    use_triton_mla = kwargs.pop("use_triton_mla", cfg.use_triton_mla)
    use_triton = kwargs.pop("use_triton", cfg.use_triton)
    use_checkpoint = kwargs.pop("use_checkpoint", cfg.use_checkpoint)

    preset = cfg.model_preset
    if preset is None:
        preset = MODEL_PRESETS.get(cfg.model_size, MODEL_PRESETS["tiny"])

    base = ModelConfig(
        block_size=block_size or preset.block_size,
        vocab_size=preset.vocab_size,
        n_layer=preset.n_layer,
        n_head=preset.n_head,
        n_embd=preset.n_embd,
        use_mla=preset.use_mla,
        use_moe=preset.use_moe,
        kv_comp_dim=preset.kv_comp_dim,
        num_shared_experts=preset.num_shared_experts,
        num_routed_experts=preset.num_routed_experts,
        num_active_experts=preset.num_active_experts,
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        use_triton_mla=use_triton_mla,
        use_triton=use_triton,
        use_checkpoint=use_checkpoint,
        **kwargs,
    )
    return base


def print_config(cfg: TrainingConfig) -> None:
    """Pretty-print a configuration for logging."""
    hw = detect_hardware()
    logger.info("=" * 60)
    logger.info(f"  Mode: {cfg.mode} | Model: {cfg.model_size} | GPUs: {cfg.num_gpus}")
    logger.info(f"  GPU: {hw.gpu_name} ({hw.gpu_memory_gb:.1f} GB) | FP8: {hw.supports_fp8} | BF16: {hw.supports_bf16}")
    logger.info(f"  Batch: {cfg.batch_size} x {cfg.grad_accum_steps} accum | LR: {cfg.max_lr}")
    logger.info(f"  Steps: {cfg.max_steps} | Epochs: {cfg.epochs} | Max Length: {cfg.max_length}")
    if cfg.model_preset:
        p = cfg.model_preset
        logger.info(f"  Arch: {p.n_layer}L-{p.n_head}H-{p.n_embd}E | MLA: {p.use_mla} | MoE: {p.use_moe}")
    logger.info("=" * 60)
