"""DeepSeek Multi-Head Latent Attention (MLA) with KV latent compression."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from model.config import ModelConfig


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        from utils.tensor_parallel import ColumnParallelLinear, RowParallelLinear, get_tp_size
        tp_size = get_tp_size()

        self.n_heads = config.n_head // tp_size
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_comp_dim = config.kv_comp_dim

        def get_linear(in_dim, out_dim, is_col=True):
            if tp_size > 1:
                if is_col:
                    return ColumnParallelLinear(in_dim, out_dim, bias=False)
                else:
                    return RowParallelLinear(in_dim, out_dim, bias=False)
            if config.lora_r > 0:
                from model import LoRALinear
                return LoRALinear(in_dim, out_dim, r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias=False)
            return nn.Linear(in_dim, out_dim, bias=False)

        # 1. KV Latent Compression Layers (kv_down_proj remains replicated)
        self.kv_down_proj = get_linear(config.n_embd, self.kv_comp_dim, is_col=True)
        self.kv_up_proj = get_linear(self.kv_comp_dim, config.n_head * self.head_dim * 2, is_col=True)
        self.wq = get_linear(config.n_embd, config.n_head * self.head_dim, is_col=True)
        self.wo = get_linear(config.n_head * self.head_dim, config.n_embd, is_col=False)
        self.attn_scale_multiplier = config.attn_scale_multiplier

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        kv_cache = None,  # Tensor for latent cache, Tuple for standard cache
    ) -> torch.Tensor:
        from model import apply_rotary_emb  # lazy import to avoid circular dependency

        bsz, seqlen, _ = x.shape

        # 1. Query projection
        xq = self.wq(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)

        # 2. Key and Value Latent Compression
        latent_kv = self.kv_down_proj(x)  # Shape: (bsz, seqlen, kv_comp_dim)

        # 3. MLA latent caching: store compressed latent, decompress on-the-fly
        if kv_cache is not None and start_pos is not None:
            if isinstance(kv_cache, torch.Tensor):
                # Latent-level cache: store and retrieve compressed KV latent
                kv_cache[:, start_pos : start_pos + seqlen] = latent_kv
                cached_latent = kv_cache[:, : start_pos + seqlen]
                decompressed_kv = self.kv_up_proj(cached_latent)
            else:
                # Standard full K/V head cache (fallback for non-MLA inference)
                decompressed_kv = self.kv_up_proj(latent_kv)
                k_decomp, v_decomp = torch.chunk(decompressed_kv, 2, dim=-1)
                xk = k_decomp.view(bsz, seqlen, self.n_heads, self.head_dim)
                xv = v_decomp.view(bsz, seqlen, self.n_heads, self.head_dim)
                xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
                cache_k, cache_v = kv_cache
                cache_k[:, start_pos : start_pos + seqlen] = xk
                cache_v[:, start_pos : start_pos + seqlen] = xv
                xk = cache_k[:, : start_pos + seqlen]
                xv = cache_v[:, : start_pos + seqlen]

                if self.config.use_triton_mla:
                    from utils.triton_kernels import triton_mla_flash_attn
                    scale = 1.0 / math.sqrt(self.head_dim)
                    output = triton_mla_flash_attn(xq, xk, xv, scale, self.attn_scale_multiplier)
                    output = output.view(bsz, seqlen, -1)
                else:
                    xq_t = xq.transpose(1, 2)
                    xk_t = xk.transpose(1, 2)
                    xv_t = xv.transpose(1, 2)
                    custom_scale = (1.0 / math.sqrt(self.head_dim)) * self.attn_scale_multiplier
                    output = F.scaled_dot_product_attention(
                        xq_t, xk_t, xv_t,
                        attn_mask=mask,
                        dropout_p=0.0,
                        is_causal=True if mask is None and start_pos is None else False,
                        scale=custom_scale,
                    )
                    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                return self.wo(output)
        else:
            decompressed_kv = self.kv_up_proj(latent_kv)

        # For latent-level cache or no cache
        kv_seqlen = decompressed_kv.size(1)
        k_decomp, v_decomp = torch.chunk(decompressed_kv, 2, dim=-1)
        xk = k_decomp.view(bsz, kv_seqlen, self.n_heads, self.head_dim)
        xv = v_decomp.view(bsz, kv_seqlen, self.n_heads, self.head_dim)

        # 4. Decoupled RoPE injection
        # Apply RoPE to query
        from model import precompute_freqs_cis
        def apply_rotary_emb_single(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
            tensor_ = torch.view_as_complex(tensor.float().reshape(*tensor.shape[:-1], -1, 2))
            ndim = tensor_.ndim
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(tensor_.shape)]
            freqs_reshaped = freqs.view(*shape)
            out = torch.view_as_real(tensor_ * freqs_reshaped).flatten(3)
            return out.type_as(tensor)

        xq = apply_rotary_emb_single(xq, freqs_cis)

        # Apply RoPE to key (sliced 0 to kv_seqlen)
        if not hasattr(self, 'freqs_cis') or self.freqs_cis.shape[0] < kv_seqlen:
            self.freqs_cis = precompute_freqs_cis(
                dim=self.head_dim,
                end=max(kv_seqlen * 2, 4096),
                scaling_factor=self.config.rope_scaling
            ).to(xq.device)
        freqs_cis_k = self.freqs_cis[:kv_seqlen]
        xk = apply_rotary_emb_single(xk, freqs_cis_k)

        if self.config.use_triton_mla:
            from utils.triton_kernels import triton_mla_flash_attn
            scale = 1.0 / math.sqrt(self.head_dim)
            output = triton_mla_flash_attn(xq, xk, xv, scale, self.attn_scale_multiplier)
            output = output.view(bsz, seqlen, -1)
        else:
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

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
