"""
HuggingFace-compatible model class for nano-lm with MLA + MoE.
Supports AutoModelForCausalLM.from_pretrained() and model.generate().
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from configuration_nano_lm import NanoLMConfig


class NanoLMPreTrainedModel(PreTrainedModel):
    config_class = NanoLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _keys_to_ignore_on_load_missing = ["tok_embeddings.weight"]

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


class NanoLMForCausalLM(NanoLMPreTrainedModel):
    """HuggingFace-compatible nano-lm model for causal language modeling."""

    def __init__(self, config: NanoLMConfig):
        super().__init__(config)
        from model import ModelConfig, Transformer

        model_config = ModelConfig(
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            n_embd=config.n_embd,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,
            use_mla=config.use_mla,
            kv_comp_dim=config.kv_comp_dim,
            use_moe=config.use_moe,
            num_shared_experts=config.num_shared_experts,
            num_routed_experts=config.num_routed_experts,
            num_active_experts=config.num_active_experts,
            rope_scaling=config.rope_scaling,
            attn_scale_multiplier=config.attn_scale_multiplier,
        )
        self.model = Transformer(model_config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            tokens = inputs_embeds  # Fallback
        else:
            tokens = input_ids

        logits, loss, aux_loss = self.model(tokens, targets=labels)

        if not return_dict:
            return (loss, logits) if loss is not None else logits

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        position_ids = None
        if past_key_values is not None:
            position_ids = torch.tensor([past_key_values[0][0].shape[1]], dtype=torch.long, device=input_ids.device)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values  # Simplified for now


# Register model so AutoModelForCausalLM works
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("nano-lm", NanoLMConfig)
AutoModelForCausalLM.register(NanoLMConfig, NanoLMForCausalLM)
