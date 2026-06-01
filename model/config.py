from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    block_size: int = 4096      # Context window length
    vocab_size: int = 32000     # Token vocabulary size
    n_layer: int = 32           # Number of transformer layers
    n_head: int = 32            # Number of attention heads
    n_kv_head: Optional[int] = None # Number of KV heads
    n_embd: int = 4096          # Embedding dimension size
    multiple_of: int = 256      # MLP hidden dimension rounding multiple
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5      # RMSNorm epsilon
    lora_r: int = 0             # LoRA rank (0 to disable)
    lora_alpha: float = 32.0    # LoRA alpha multiplier
    lora_dropout: float = 0.05  # LoRA dropout fraction
    vision_dim: Optional[int] = 1152 # Vision feature dim (SigLIP/ViT; None to disable VLM)
    use_checkpoint: bool = False # Enable activation checkpointing for Billion-scale VRAM savings
    use_mla: bool = True        # Enable DeepSeek Multi-Head Latent Attention (MLA)
    kv_comp_dim: int = 128      # MLA KV compressed latent dimension
    use_moe: bool = True        # Enable DeepSeekMoE Mixture of Experts
    num_shared_experts: int = 1 # Number of shared static experts
    num_routed_experts: int = 8 # Number of total routed experts
    num_active_experts: int = 2 # Number of active routed experts per token
    rope_scaling: float = 1.0   # NTK-Aware RoPE scaling factor for 1M long-context extrapolation
    attn_scale_multiplier: float = 1.0 # Attention logits scaling multiplier to sharpen attention maps
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    use_triton_mla: bool = False
    use_triton: bool = False
    aux_loss_weight: float = 0.01  # Weight for MoE load balancing auxiliary loss
