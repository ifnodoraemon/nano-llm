"""
HuggingFace-compatible configuration for nano-lm models with MLA + MoE.
"""

from transformers import PretrainedConfig


class NanoLMConfig(PretrainedConfig):
    model_type = "nano-lm"

    def __init__(
        self,
        vocab_size: int = 32000,
        n_layer: int = 24,
        n_head: int = 32,
        n_embd: int = 2048,
        n_kv_head: int = None,
        block_size: int = 4096,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = None,
        norm_eps: float = 1e-5,
        use_mla: bool = True,
        kv_comp_dim: int = 128,
        use_moe: bool = True,
        num_shared_experts: int = 1,
        num_routed_experts: int = 8,
        num_active_experts: int = 2,
        rope_scaling: float = 1.0,
        attn_scale_multiplier: float = 1.0,
        use_triton_mla: bool = False,
        use_triton: bool = False,
        bos_token_id: int = 10000,
        eos_token_id: int = 10001,
        pad_token_id: int = 10002,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.use_mla = use_mla
        self.kv_comp_dim = kv_comp_dim
        self.use_moe = use_moe
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.num_active_experts = num_active_experts
        self.rope_scaling = rope_scaling
        self.attn_scale_multiplier = attn_scale_multiplier
        self.use_triton_mla = use_triton_mla
        self.use_triton = use_triton

        # Compute hidden_dim (intermediate_size) for SwiGLU
        hidden_dim = int(2 * (n_embd * 4) / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        self.intermediate_size = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.head_dim = n_embd // n_head
