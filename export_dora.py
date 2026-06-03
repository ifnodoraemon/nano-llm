import os
import sys
import json
import torch
import shutil
from safetensors.torch import save_file


def export(checkpoint_path: str = None, dest_dir: str = None):
    src_pt = checkpoint_path
    if not src_pt:
        if os.path.exists("./outputs/checkpoint_pretrain.pt"):
            src_pt = "./outputs/checkpoint_pretrain.pt"
        else:
            src_pt = "./outputs_grpo/checkpoint_grpo_epoch_1.pt"
    dest_dir = dest_dir or "./outputs/dora"
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Loading checkpoint from: {src_pt}...")
    checkpoint = torch.load(src_pt, map_location="cpu", weights_only=False)

    nano_state = checkpoint.get("model_state_dict", checkpoint.get("model", {}))
    # Strip torch.compile prefix '_orig_mod.' from keys
    nano_state = {k.replace("_orig_mod.", ""): v for k, v in nano_state.items()}
    nano_config = checkpoint["config"]
    use_mla = getattr(nano_config, 'use_mla', False)
    use_moe = getattr(nano_config, 'use_moe', False)

    print(f"Architecture: MLA={use_mla}, MoE={use_moe}")
    print("Performing weight key mapping to nano-lm format...")
    hf_state = {}

    # Core embeddings and lm_head
    mapping = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }

    for i in range(nano_config.n_layer):
        mapping[f"model.layers.{i}.input_layernorm.weight"] = f"layers.{i}.attention_norm.weight"
        mapping[f"model.layers.{i}.post_attention_layernorm.weight"] = f"layers.{i}.ffn_norm.weight"
        mapping[f"model.layers.{i}.self_attn.o_proj.weight"] = f"layers.{i}.attention.wo.weight"

        if use_mla:
            # MLA: q_proj, kv_down_proj, kv_up_proj
            mapping[f"model.layers.{i}.self_attn.q_proj.weight"] = f"layers.{i}.attention.wq.weight"
            mapping[f"model.layers.{i}.self_attn.kv_down_proj.weight"] = f"layers.{i}.attention.kv_down_proj.weight"
            mapping[f"model.layers.{i}.self_attn.kv_up_proj.weight"] = f"layers.{i}.attention.kv_up_proj.weight"
        else:
            # Standard: q_proj, k_proj, v_proj
            mapping[f"model.layers.{i}.self_attn.q_proj.weight"] = f"layers.{i}.attention.wq.weight"
            mapping[f"model.layers.{i}.self_attn.k_proj.weight"] = f"layers.{i}.attention.wk.weight"
            mapping[f"model.layers.{i}.self_attn.v_proj.weight"] = f"layers.{i}.attention.wv.weight"

        if use_moe:
            # MoE: router + shared_experts + routed_experts
            mapping[f"model.layers.{i}.mlp.router.weight"] = f"layers.{i}.feed_forward.router.weight"
            for j in range(getattr(nano_config, 'num_shared_experts', 1)):
                for proj_name in ["w1", "w2", "w3"]:
                    mapping[f"model.layers.{i}.mlp.shared_experts.{j}.{proj_name}.weight"] = (
                        f"layers.{i}.feed_forward.shared_experts.{j}.{proj_name}.weight"
                    )
            for j in range(getattr(nano_config, 'num_routed_experts', 8)):
                for proj_name in ["w1", "w2", "w3"]:
                    mapping[f"model.layers.{i}.mlp.routed_experts.{j}.{proj_name}.weight"] = (
                        f"layers.{i}.feed_forward.routed_experts.{j}.{proj_name}.weight"
                    )
        else:
            # Standard: gate, up, down
            mapping[f"model.layers.{i}.mlp.gate_proj.weight"] = f"layers.{i}.feed_forward.w1.weight"
            mapping[f"model.layers.{i}.mlp.down_proj.weight"] = f"layers.{i}.feed_forward.w2.weight"
            mapping[f"model.layers.{i}.mlp.up_proj.weight"] = f"layers.{i}.feed_forward.w3.weight"

    for hf_key, nano_key in mapping.items():
        if nano_key in nano_state:
            hf_state[hf_key] = nano_state[nano_key].half()
        else:
            print(f"Warning: missing key in checkpoint: {nano_key}")

    # Save safetensors
    safetensors_path = os.path.join(dest_dir, "model.safetensors")
    print(f"Saving safetensors weights to: {safetensors_path}...")
    save_file(hf_state, safetensors_path)

    # Compute intermediate_size
    hidden_dim = int(2 * (nano_config.n_embd * 4) / 3)
    if getattr(nano_config, 'ffn_dim_multiplier', None) is not None:
        hidden_dim = int(nano_config.ffn_dim_multiplier * hidden_dim)
    multiple_of = getattr(nano_config, 'multiple_of', 256)
    intermediate_size = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    # Save config.json with nano-lm model_type
    config_data = {
        "architectures": ["NanoLMForCausalLM"],
        "model_type": "nano-lm",
        "auto_map": {
            "AutoConfig": "configuration_nano_lm.NanoLMConfig",
            "AutoModelForCausalLM": "modeling_nano_lm.NanoLMForCausalLM",
        },
        "bos_token_id": 10000,
        "eos_token_id": 10001,
        "pad_token_id": 10002,
        "hidden_size": nano_config.n_embd,
        "head_dim": nano_config.n_embd // nano_config.n_head,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": nano_config.block_size,
        "num_attention_heads": nano_config.n_head,
        "num_hidden_layers": nano_config.n_layer,
        "num_key_value_heads": nano_config.n_head if getattr(nano_config, 'n_kv_head', None) is None else nano_config.n_kv_head,
        "rms_norm_eps": getattr(nano_config, 'norm_eps', 1e-5),
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "use_cache": True,
        "vocab_size": nano_config.vocab_size,
        "use_mla": use_mla,
        "kv_comp_dim": getattr(nano_config, 'kv_comp_dim', 128),
        "use_moe": use_moe,
        "num_shared_experts": getattr(nano_config, 'num_shared_experts', 1),
        "num_routed_experts": getattr(nano_config, 'num_routed_experts', 8),
        "num_active_experts": getattr(nano_config, 'num_active_experts', 2),
    }

    config_path = os.path.join(dest_dir, "config.json")
    print(f"Saving nano-lm configuration to: {config_path}...")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # Save generation_config.json
    gen_config_data = {
        "bos_token_id": 10000,
        "eos_token_id": 10001,
        "pad_token_id": 10002,
        "max_length": nano_config.block_size,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }
    gen_config_path = os.path.join(dest_dir, "generation_config.json")
    print(f"Saving generation configuration to: {gen_config_path}...")
    with open(gen_config_path, "w", encoding="utf-8") as f:
        json.dump(gen_config_data, f, indent=2)

    # Copy tokenizer and modeling files
    print("Packaging tokenizer and model files...")
    shutil.copy("./data/custom_tokenizer.json", os.path.join(dest_dir, "custom_tokenizer.json"))
    shutil.copy("./tokenizer.py", os.path.join(dest_dir, "tokenizer.py"))
    shutil.copy("./configuration_nano_lm.py", os.path.join(dest_dir, "configuration_nano_lm.py"))
    shutil.copy("./modeling_nano_lm.py", os.path.join(dest_dir, "modeling_nano_lm.py"))

    # Save tokenizer_config.json with auto_map for both tokenizer and model
    tok_config_data = {
        "auto_map": {
            "AutoTokenizer": ["tokenizer.CustomBPETokenizer", None],
        },
        "tokenizer_class": "CustomBPETokenizer",
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>",
        "model_max_length": nano_config.block_size,
    }
    tok_config_path = os.path.join(dest_dir, "tokenizer_config.json")
    print(f"Saving tokenizer configuration to: {tok_config_path}...")
    with open(tok_config_path, "w", encoding="utf-8") as f:
        json.dump(tok_config_data, f, indent=2)

    # Generate model card
    _generate_model_card(dest_dir, nano_config, use_mla, use_moe)

    print("Conversion completed. Model exported to nano-lm format.")


def _generate_model_card(dest_dir, nano_config, use_mla, use_moe):
    arch_desc = []
    if use_mla:
        arch_desc.append("Multi-Head Latent Attention (MLA)")
    else:
        arch_desc.append("Multi-Head Attention (MHA)")
    if use_moe:
        arch_desc.append(f"Mixture of Experts ({getattr(nano_config, 'num_routed_experts', 8)} experts, {getattr(nano_config, 'num_active_experts', 2)} active)")
    else:
        arch_desc.append("SwiGLU MLP")

    card = f"""---
language: en
tags:
- nano-llm
- causal-lm
- pytorch
license: mit
---

# nano-lm Model

## Architecture
- **Layers**: {nano_config.n_layer}
- **Heads**: {nano_config.n_head}
- **Embedding Dimension**: {nano_config.n_embd}
- **Context Length**: {nano_config.block_size}
- **Vocab Size**: {nano_config.vocab_size}
- **Architecture**: {', '.join(arch_desc)}

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("path/to/model", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("path/to/model", trust_remote_code=True)
```
"""
    card_path = os.path.join(dest_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)


if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    _logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Export nano-lm checkpoint to Safetensors")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to source .pt checkpoint")
    parser.add_argument("--dest_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload exported model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default=None, help="Hugging Face repo ID (username/repo_name)")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--quantize", action="store_true", help="Generate int4/int8 quantized versions after export")
    args = parser.parse_args()

    dest = args.dest_dir or "./outputs/dora"
    export(checkpoint_path=args.checkpoint_path, dest_dir=dest)

    # Optional: generate quantized versions
    if args.quantize:
        src_ckpt = args.checkpoint_path
        if not src_ckpt:
            if os.path.exists("./outputs/checkpoint_pretrain.pt"):
                src_ckpt = "./outputs/checkpoint_pretrain.pt"
            else:
                src_ckpt = "./outputs_grpo/checkpoint_grpo_epoch_1.pt"
        for bits in [8, 4]:
            _logger.info(f"Generating int{bits} quantized model...")
            try:
                from quantize import quantize_model_checkpoint
                quant_dest = os.path.join(dest, f"model_int{bits}.pt")
                quantize_model_checkpoint(src_ckpt, quant_dest, bits=bits)
                _logger.info(f"int{bits} quantized model saved to {quant_dest}")
            except Exception as e:
                _logger.warning(f"int{bits} quantization failed ({e}) — continuing without int{bits} model")

    # Optional: push to Hugging Face Hub
    if args.push_to_hub:
        repo_id = args.repo_id or os.environ.get("HF_REPO_ID")
        if not repo_id:
            _logger.error("--repo_id is required when --push_to_hub is set")
            sys.exit(1)
        _logger.info(f"Publishing model to Hugging Face Hub: {repo_id}...")
        try:
            from utils.upload_hub import upload_to_huggingface
            upload_to_huggingface(folder_path=dest, repo_id=repo_id, token=args.hf_token)
        except Exception as e:
            _logger.warning(f"Hugging Face upload failed ({e}) — export completed but model not published")
