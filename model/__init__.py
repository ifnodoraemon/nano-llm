import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from model.config import ModelConfig
from model.mla import MultiHeadLatentAttention
from model.moe import DeepSeekMoE


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, use_triton: bool = False) -> torch.Tensor:
        if use_triton:
            from utils.triton_kernels import triton_rms_norm
            return triton_rms_norm(x, self.weight, self.eps)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ==============================================================================
# Custom LoRA (Low-Rank Adaptation) Layer From Scratch
# ==============================================================================

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        bias: bool = False
    ):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        dropout_x = self.lora_dropout(x.to(self.lora_A.dtype))
        lora_out = (dropout_x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out.type_as(base_out)


# ==============================================================================
# Native Vision-Language Projection Connector (LLaVA-style)
# ==============================================================================

class VisionProjection(nn.Module):
    """
    2-Layer MLP Connector mapping visual embeddings to model's embedding dimension:
    h = Linear2(SiLU(Linear1(vision_features)))
    """
    def __init__(self, vision_dim: int, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(vision_dim, n_embd)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


# ==============================================================================
# Rotary Position Embeddings (RoPE)
# ==============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scaling_factor: float = 1.0) -> torch.Tensor:
    # NTK-Aware RoPE Scaling: scale base theta dynamically based on scaling factor
    # This prevents high-frequency loss during long-context extrapolation up to 1M tokens
    if scaling_factor > 1.0:
        theta = theta * (scaling_factor ** (dim / (dim - 2)))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==============================================================================
# Model Sub-modules
# ==============================================================================

class SwiGLU_MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        hidden_dim = int(2 * (config.n_embd * 4) / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        from utils.tensor_parallel import ColumnParallelLinear, RowParallelLinear, get_tp_size
        tp_active = get_tp_size() > 1

        def get_linear(in_dim, out_dim, is_col=True):
            if tp_active:
                if is_col:
                    return ColumnParallelLinear(in_dim, out_dim, bias=False)
                else:
                    return RowParallelLinear(in_dim, out_dim, bias=False)
            if config.lora_r > 0:
                return LoRALinear(in_dim, out_dim, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias=False)
            return nn.Linear(in_dim, out_dim, bias=False)

        self.w1 = get_linear(config.n_embd, hidden_dim, is_col=True)
        self.w3 = get_linear(config.n_embd, hidden_dim, is_col=True)
        self.w2 = get_linear(hidden_dim, config.n_embd, is_col=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.use_triton:
            from utils.triton_kernels import triton_swiglu
            gate = self.w1(x)
            up = self.w3(x)
            return self.w2(triton_swiglu(gate, up))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        from utils.tensor_parallel import ColumnParallelLinear, RowParallelLinear, get_tp_size
        tp_size = get_tp_size()

        self.n_heads = config.n_head // tp_size
        self.n_kv_heads = (config.n_kv_head if config.n_kv_head is not None else config.n_head) // tp_size
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_heads // self.n_kv_heads

        def get_linear(in_dim, out_dim, is_col=True):
            if tp_size > 1:
                if is_col:
                    return ColumnParallelLinear(in_dim, out_dim, bias=False)
                else:
                    return RowParallelLinear(in_dim, out_dim, bias=False)
            if config.lora_r > 0:
                return LoRALinear(in_dim, out_dim, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias=False)
            return nn.Linear(in_dim, out_dim, bias=False)

        self.wq = get_linear(config.n_embd, config.n_head * self.head_dim, is_col=True)
        self.wk = get_linear(config.n_embd, (config.n_kv_head if config.n_kv_head is not None else config.n_head) * self.head_dim, is_col=True)
        self.wv = get_linear(config.n_embd, (config.n_kv_head if config.n_kv_head is not None else config.n_head) * self.head_dim, is_col=True)
        self.wo = get_linear(config.n_head * self.head_dim, config.n_embd, is_col=False)
        self.attn_scale_multiplier = config.attn_scale_multiplier

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        bs, slen, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if kv_cache is not None and start_pos is not None:
            cache_k, cache_v = kv_cache
            cache_k[:, start_pos : start_pos + seqlen] = xk
            cache_v[:, start_pos : start_pos + seqlen] = xv

            xk = cache_k[:, : start_pos + seqlen]
            xv = cache_v[:, : start_pos + seqlen]

        xk = self.repeat_kv(xk, self.n_rep)
        xv = self.repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Multiply standard scaling by attn_scale_multiplier to sharpen/soften attention maps
        custom_scale = (1.0 / math.sqrt(self.head_dim)) * self.attn_scale_multiplier

        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=True if mask is None and start_pos is None else False,
            scale=custom_scale
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.use_mla:
            self.attention = MultiHeadLatentAttention(config)
        else:
            self.attention = CausalSelfAttention(config)

        if config.use_moe:
            self.feed_forward = DeepSeekMoE(config)
        else:
            self.feed_forward = SwiGLU_MLP(config)

        self.attention_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        kv_cache = None,  # Tensor (MLA latent) or Tuple[Tensor, Tensor] (standard KV cache)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        use_triton = getattr(self.feed_forward.config if hasattr(self.feed_forward, 'config') else self.attention.config, 'use_triton', False)
        h = x + self.attention(self.attention_norm(x, use_triton=use_triton), freqs_cis, mask, start_pos, kv_cache)
        out = h + self.feed_forward(self.ffn_norm(h, use_triton=use_triton))
        aux_loss = getattr(self.feed_forward, 'aux_loss', None) if hasattr(self.feed_forward, 'aux_loss') else None
        return out, aux_loss


# ==============================================================================
# Full Native Multimodal Vision-Language Model Assembly
# ==============================================================================

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)

        # SigLIP/ViT Multimodal Projection layer
        if config.vision_dim is not None:
            self.vision_projection = VisionProjection(vision_dim=config.vision_dim, n_embd=config.n_embd)
        else:
            self.vision_projection = None

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            dim=config.n_embd // config.n_head,
            end=config.block_size * 2,
            scaling_factor=config.rope_scaling
        )

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.weight.requires_grad:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and module.bias.requires_grad:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_lora_trainable(self) -> int:
        for param in self.parameters():
            param.requires_grad = False

        lora_params_count = 0
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                lora_params_count += param.numel()

        return lora_params_count

    def merge_lora_weights(self):
        if self.config.lora_r == 0:
            return

        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                with torch.no_grad():
                    delta_weight = (module.lora_B @ module.lora_A) * module.scaling
                    module.base_layer.weight.data += delta_weight.to(module.base_layer.weight.dtype)
                    torch.nn.init.zeros_(module.lora_B)

    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None, # Shape: (batch, num_patches, vision_dim)
        targets: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        kv_caches: Optional[List] = None,  # List of Tensor (MLA) or Tuple[Tensor, Tensor] (standard)
        return_all_logits: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        _bsz, seqlen = tokens.shape

        # 1. Embed text tokens
        h = self.tok_embeddings(tokens)

        # 2. Extract and project visual embeddings if present
        if pixel_values is not None and self.vision_projection is not None:
            # h_vision shape: (batch, num_patches, n_embd)
            h_vision = self.vision_projection(pixel_values)

            # Fuse visual and text tokens: [vision_tokens, text_tokens]
            # This is a standard unified multimodal causal sequence (LLaVA-style)
            h = torch.cat([h_vision, h], dim=1)

        total_seqlen = h.size(1)
        pos = start_pos if start_pos is not None else 0
        needed_end = pos + total_seqlen
        if needed_end > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(
                dim=self.config.n_embd // self.config.n_head,
                end=needed_end * 2,
                scaling_factor=self.config.rope_scaling
            )

        self.freqs_cis = self.freqs_cis.to(tokens.device)
        freqs_cis = self.freqs_cis[pos : pos + total_seqlen]

        # 3. Autoregressive multi-modal self-attention layers
        total_aux_loss = torch.tensor(0.0, device=h.device)
        for idx, layer in enumerate(self.layers):
            layer_cache = kv_caches[idx] if kv_caches is not None else None

            if self.training and self.config.use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                def custom_layer_forward(hidden_states, rope_freqs):
                    out, _ = layer(hidden_states, rope_freqs, mask=mask, start_pos=start_pos, kv_cache=layer_cache)
                    return out
                h = checkpoint(custom_layer_forward, h, freqs_cis, use_reentrant=False)
            else:
                h, aux_loss = layer(h, freqs_cis, mask=mask, start_pos=start_pos, kv_cache=layer_cache)
                if aux_loss is not None:
                    total_aux_loss = total_aux_loss + aux_loss

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)

            # If visual tokens were prepended, we must pad SFT targets with -100 (mask out)
            # so the model is not trained to generate/predict visual pixels!
            if pixel_values is not None:
                num_patches = pixel_values.size(1)
                targets_padded = torch.cat([
                    torch.full((_bsz, num_patches), -100, dtype=torch.long, device=targets.device),
                    targets
                ], dim=1)
            else:
                targets_padded = targets

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets_padded.view(-1),
                ignore_index=-100
            )
            # Add MoE load balancing auxiliary loss
            if total_aux_loss.item() > 0:
                loss = loss + self.config.aux_loss_weight * total_aux_loss
        else:
            if return_all_logits or (start_pos is None and kv_caches is None):
                logits = self.output(h)
            else:
                logits = self.output(h[:, [-1], :])
            loss = None

        return logits, loss, total_aux_loss.detach() if total_aux_loss.item() > 0 else None


# ==============================================================================
# Native FP8 (Float8) Mixed-Precision Layer wrappers (Hopper H800 Optimized)
# ==============================================================================

class FP8Linear(nn.Module):
    """
    Linear layer that performs forward pass in FP8 (e4m3) with Channel-wise dynamic scale factors.
    Optimized exclusively for Hopper H800 (Compute Capability >= 9.0).
    """
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.base_linear = base_linear
        # Register default tensors to ensure state_dict serializability and avoid loader errors
        self.register_buffer("x_scale", torch.tensor(1.0))
        self.register_buffer("w_scale", torch.ones(base_linear.out_features))
        self.register_buffer("_is_scale_initialized", torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Dynamic scaling factor calculation (Float8 max is 448.0)
        # Activation is scaled tensor-wide (scalar); weights are scaled channel-wise (per output channel)
        with torch.no_grad():
            x_max = x.abs().max()
            w_max = self.base_linear.weight.abs().max(dim=1).values

            curr_x_scale = torch.tensor(448.0 / (x_max.item() + 1e-5), device=x.device)
            curr_w_scale = 448.0 / (w_max + 1e-5)

            if not self._is_scale_initialized.item():
                self.x_scale.copy_(curr_x_scale)
                self.w_scale.copy_(curr_w_scale)
                self._is_scale_initialized.fill_(True)
            else:
                self.x_scale.copy_(0.9 * self.x_scale + 0.1 * curr_x_scale)
                self.w_scale.copy_(0.9 * self.w_scale + 0.1 * curr_w_scale)

        # 2. Scale & Cast to Float8 e4m3
        x_fp8 = (x * self.x_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        w_fp8 = (self.base_linear.weight * self.w_scale.unsqueeze(1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        # 3. Compute MatMul and scale back channel-wise
        out = torch.matmul(x_fp8.to(torch.float32), w_fp8.to(torch.float32).t())
        out = out / (self.x_scale * self.w_scale.unsqueeze(0))

        if self.base_linear.bias is not None:
            out = out + self.base_linear.bias.type_as(out)
        return out.type_as(x)

def convert_to_fp8(model: nn.Module):
    """
    Recursively replaces all standard Linear layers inside attention and MLP blocks
    with dynamic scaling FP8Linear layers.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name != "output":
            setattr(model, name, FP8Linear(child))
        elif isinstance(child, LoRALinear):
            child.base_layer = FP8Linear(child.base_layer)
        else:
            convert_to_fp8(child)

def get_preset_config(size: str = "7B", **kwargs) -> ModelConfig:
    """
    Returns pre-configured ModelConfig presets for standard Billion-scale architectures:
    Presets include: '1.5B', '3B', and '7B' LLaMA layouts.
    """
    presets = {
        "1.5B": {
            "n_layer": 28,
            "n_head": 16,
            "n_embd": 2048,
            "block_size": 2048,
            "vocab_size": 32000
        },
        "3B": {
            "n_layer": 32,
            "n_head": 32,
            "n_embd": 3072,
            "block_size": 4096,
            "vocab_size": 32000
        },
        "7B": {
            "n_layer": 32,
            "n_head": 32,
            "n_embd": 4096,
            "block_size": 4096,
            "vocab_size": 32000
        }
    }

    if size not in presets:
        raise ValueError(f"Unknown preset size: '{size}'. Available sizes: {list(presets.keys())}")

    config_dict = presets[size].copy()
    config_dict.update(kwargs)
    return ModelConfig(**config_dict)


def get_deepseek_config(size: str = "16B-equivalent", **kwargs) -> ModelConfig:
    """
    Returns pre-configured ModelConfig presets for DeepSeek MLA + MoE architectures.

    Available sizes:
      - "tiny": 4L/8H/512E, no MLA/MoE (for fast smoke testing)
      - "small": 12L/16H/1024E, standard architecture
      - "medium": 24L/16H/2048E, standard architecture
      - "1B-equivalent": 16L/16H/1536E, MLA + 8-expert MoE
      - "3B-equivalent": 20L/24H/2048E, MLA + 8-expert MoE
      - "7B-equivalent": 28L/32H/2560E, MLA + 8-expert MoE
      - "16B-equivalent": 24L/32H/2048E, MLA + 8-expert MoE (deeper)
    """
    presets = {
        "tiny": {
            "n_layer": 4,
            "n_head": 8,
            "n_embd": 512,
            "block_size": 1024,
            "vocab_size": 32000,
            "use_mla": False,
            "use_moe": False,
        },
        "small": {
            "n_layer": 12,
            "n_head": 16,
            "n_embd": 1024,
            "block_size": 2048,
            "vocab_size": 32000,
            "use_mla": False,
            "use_moe": False,
        },
        "medium": {
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 2048,
            "block_size": 4096,
            "vocab_size": 32000,
            "use_mla": False,
            "use_moe": False,
        },
        "1B-equivalent": {
            "n_layer": 16,
            "n_head": 16,
            "n_embd": 1536,
            "block_size": 4096,
            "vocab_size": 32000,
            "use_mla": True,
            "kv_comp_dim": 128,
            "use_moe": True,
            "num_shared_experts": 1,
            "num_routed_experts": 8,
            "num_active_experts": 2,
        },
        "3B-equivalent": {
            "n_layer": 20,
            "n_head": 24,
            "n_embd": 2048,
            "block_size": 4096,
            "vocab_size": 32000,
            "use_mla": True,
            "kv_comp_dim": 128,
            "use_moe": True,
            "num_shared_experts": 1,
            "num_routed_experts": 8,
            "num_active_experts": 2,
        },
        "7B-equivalent": {
            "n_layer": 28,
            "n_head": 32,
            "n_embd": 2560,
            "block_size": 4096,
            "vocab_size": 32000,
            "use_mla": True,
            "kv_comp_dim": 128,
            "use_moe": True,
            "num_shared_experts": 1,
            "num_routed_experts": 8,
            "num_active_experts": 2,
        },
        "16B-equivalent": {
            "n_layer": 24,
            "n_head": 32,
            "n_embd": 2048,
            "block_size": 4096,
            "vocab_size": 32000,
            "use_mla": True,
            "kv_comp_dim": 128,
            "use_moe": True,
            "num_shared_experts": 1,
            "num_routed_experts": 8,
            "num_active_experts": 2,
        },
    }

    if size not in presets:
        raise ValueError(f"Unknown DeepSeek size: '{size}'. Available sizes: {list(presets.keys())}")

    config_dict = presets[size].copy()
    config_dict.update(kwargs)
    return ModelConfig(**config_dict)
