import os
import argparse
import logging
import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoConfig

from model import ModelConfig, Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Mapping Dictionary Definitions (HF <-> nano-llm)
# ==============================================================================

def get_keys_mapping(n_layers: int, lora_active: bool = False) -> Dict[str, str]:
    """
    Returns a dictionary mapping HuggingFace model weight names to nano-llm weight names.
    """
    mapping = {}
    
    # 1. Embeddings & final output norm/head mapping
    mapping["model.embed_tokens.weight"] = "tok_embeddings.weight"
    mapping["model.norm.weight"] = "norm.weight"
    mapping["lm_head.weight"] = "output.weight"
    
    # 2. Block layer mapping
    for i in range(n_layers):
        base_hf = f"model.layers.{i}"
        base_nano = f"layers.{i}"
        
        # Norms
        mapping[f"{base_hf}.input_layernorm.weight"] = f"{base_nano}.attention_norm.weight"
        mapping[f"{base_hf}.post_attention_layernorm.weight"] = f"{base_nano}.ffn_norm.weight"
        
        # Attention
        # If LoRA is enabled, the base Linear weights reside inside base_layer.weight
        suffix = ".base_layer.weight" if lora_active else ".weight"
        mapping[f"{base_hf}.self_attn.q_proj.weight"] = f"{base_nano}.attention.wq{suffix}"
        mapping[f"{base_hf}.self_attn.k_proj.weight"] = f"{base_nano}.attention.wk{suffix}"
        mapping[f"{base_hf}.self_attn.v_proj.weight"] = f"{base_nano}.attention.wv{suffix}"
        mapping[f"{base_hf}.self_attn.o_proj.weight"] = f"{base_nano}.attention.wo{suffix}"
        
        # MLP
        mapping[f"{base_hf}.mlp.gate_proj.weight"] = f"{base_nano}.feed_forward.w1{suffix}"
        mapping[f"{base_hf}.mlp.down_proj.weight"] = f"{base_nano}.feed_forward.w2{suffix}"
        mapping[f"{base_hf}.mlp.up_proj.weight"] = f"{base_nano}.feed_forward.w3{suffix}"
        
    return mapping


# ==============================================================================
# Bidirectional Conversion Actions
# ==============================================================================

def convert_hf_to_nano(hf_model_path: str, output_pt_path: str):
    """Loads a HuggingFace checkpoint and outputs a nano-llm .pt state dict."""
    logger.info(f"Loading HuggingFace model configuration from: {hf_model_path}")
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    
    # Map HuggingFace configuration parameters to ModelConfig
    n_embd = getattr(hf_config, "hidden_size", 4096)
    n_head = getattr(hf_config, "num_attention_heads", 32)
    n_kv_head = getattr(hf_config, "num_key_value_heads", n_head)
    n_layer = getattr(hf_config, "num_hidden_layers", 32)
    vocab_size = getattr(hf_config, "vocab_size", 32000)
    block_size = getattr(hf_config, "max_position_embeddings", 4096)
    
    nano_config = ModelConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        lora_r=0  # Disable LoRA during conversion (weights map directly to standard base)
    )
    
    logger.info(f"Loading HF model weights (CPU fallback)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cpu"
    )
    hf_state = hf_model.state_dict()
    
    logger.info("Performing weight key translation...")
    mapping = get_keys_mapping(n_layer, lora_active=False)
    nano_state = {}
    
    for hf_key, nano_key in mapping.items():
        if hf_key in hf_state:
            nano_state[nano_key] = hf_state[hf_key]
        else:
            logger.warning(f"Key missing in HF checkpoint: {hf_key}")
            
    # Write checkpoint
    logger.info(f"Saving translated weights to {output_pt_path}...")
    torch.save({
        "model_state_dict": nano_state,
        "config": nano_config
    }, output_pt_path)
    logger.info("Conversion (HF -> nano-llm) complete!")


def convert_nano_to_hf(nano_pt_path: str, hf_config_ref_path: str, output_hf_dir: str):
    """Loads a nano-llm .pt file and exports a HuggingFace standard checkpoint directory."""
    logger.info(f"Loading nano-llm checkpoint from: {nano_pt_path}")
    checkpoint = torch.load(nano_pt_path, map_location="cpu", weights_only=False)
    nano_state = checkpoint["model_state_dict"]
    nano_config = checkpoint["config"]
    
    # 1. Fuse LoRA weights if lora_r > 0
    if nano_config.lora_r > 0:
        logger.info(f"Active LoRA adapters detected (r={nano_config.lora_r}). Fusing adapter layers before export...")
        # Instantiate model to handle weight merging automatically
        temp_model = Transformer(nano_config)
        temp_model.load_state_dict(nano_state)
        temp_model.merge_lora_weights()
        
        # Fetch fused standard states
        nano_state = {k: v for k, v in temp_model.state_dict().items() if "lora_" not in k}
        
    # 2. Map weight names back to HuggingFace format
    logger.info("Translating keys to HuggingFace standard...")
    mapping = get_keys_mapping(nano_config.n_layer, lora_active=False)
    
    # Invert mapping to go from nano-llm -> HuggingFace
    reverse_mapping = {nano_key: hf_key for hf_key, nano_key in mapping.items()}
    
    hf_state = {}
    for nano_key, hf_key in reverse_mapping.items():
        # Standardize LoRA paths if temp_model converted them
        clean_key = nano_key.replace(".base_layer.weight", ".weight")
        if clean_key in nano_state:
            hf_state[hf_key] = nano_state[clean_key]
        else:
            logger.warning(f"Key missing in nano-llm checkpoint: {nano_key}")
            
    # 3. Save standard Hugging Face model
    logger.info(f"Instantiating reference model from: {hf_config_ref_path}...")
    hf_config = AutoConfig.from_pretrained(hf_config_ref_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    
    # Copy weights directly
    hf_model.load_state_dict(hf_state, strict=False)
    
    logger.info(f"Writing HuggingFace model checkpoint folder to: {output_hf_dir}...")
    os.makedirs(output_hf_dir, exist_ok=True)
    hf_model.save_pretrained(output_hf_dir, safe_serialization=True)
    
    # Copy reference tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_config_ref_path)
    tokenizer.save_pretrained(output_hf_dir)
    
    logger.info("Conversion (nano-llm -> HF) succeeded!")

# ==============================================================================
# CLI Orchestrator
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-llm: Bidirectional Weight Converter Suite")
    parser.add_argument("--mode", type=str, choices=["hf2nano", "nano2hf"], required=True, help="Conversion direction")
    parser.add_argument("--src", type=str, required=True, help="Path to source model folder or checkpoint file")
    parser.add_argument("--dest", type=str, required=True, help="Target path/folder to save converted checkpoint")
    parser.add_argument("--ref_hf", type=str, default="qwen/Qwen2.5-7B", help="Reference ModelScope/HF path for configuration generation")
    args = parser.parse_args()
    
    if args.mode == "hf2nano":
        convert_hf_to_nano(args.src, args.dest)
    elif args.mode == "nano2hf":
        convert_nano_to_hf(args.src, args.ref_hf, args.dest)
