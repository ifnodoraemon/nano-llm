import os
import argparse
import logging
import torch
import torch.nn as nn
from model import ModelConfig, Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ONNXTransformerWrapper(nn.Module):
    """
    ONNX-compatible wrapper for the Transformer model.
    Accepts flat list of KV caches as a single parameter to comply with PyTorch tracing.
    """
    def __init__(self, model: Transformer):
        super().__init__()
        self.model = model
        self.config = model.config
        self.use_mla = getattr(self.config, "use_mla", False)

    def forward(self, tokens: torch.Tensor, start_pos: torch.Tensor, flat_kv_caches: list):
        # Reconstruct kv_caches from flat list
        kv_caches = []
        if self.use_mla:
            # Each layer has 1 latent cache tensor
            for i in range(self.config.n_layer):
                kv_caches.append(flat_kv_caches[i])
        else:
            # Each layer has K and V cache tensors as a tuple
            for i in range(self.config.n_layer):
                kv_caches.append((flat_kv_caches[2 * i], flat_kv_caches[2 * i + 1]))

        pos = int(start_pos.item())
        
        logits, _, _ = self.model(
            tokens=tokens,
            start_pos=pos,
            kv_caches=kv_caches
        )
        
        # Flatten updated kv_caches for output
        flat_updated_caches = []
        for i in range(self.config.n_layer):
            if self.use_mla:
                flat_updated_caches.append(kv_caches[i])
            else:
                flat_updated_caches.append(kv_caches[i][0])
                flat_updated_caches.append(kv_caches[i][1])
                
        return logits, flat_updated_caches

def export_to_onnx(checkpoint_path: str, output_path: str, opset_version: int = 17):
    logger.info(f"Loading nano-llm checkpoint from: {checkpoint_path}")
    from utils.checkpoint_utils import load_checkpoint_with_fp8_translation
    config, state_dict = load_checkpoint_with_fp8_translation(checkpoint_path, map_location="cpu")
    
    logger.info(f"Instantiating model from config: {config}")
    model = Transformer(config)
    model.load_state_dict(state_dict)
    model.eval()
    
    wrapper = ONNXTransformerWrapper(model)
    
    # Prepare dummy inputs
    batch_size = 1
    seq_len = 1
    tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)
    start_pos = torch.tensor(0, dtype=torch.long)
    
    # Initialize flat dummy KV caches
    flat_kv_caches = []
    n_kv_heads = config.n_kv_head if config.n_kv_head is not None else config.n_head
    head_dim = config.n_embd // config.n_head
    max_block_size = config.block_size
    use_mla = getattr(config, "use_mla", False)
    
    for _ in range(config.n_layer):
        if use_mla:
            kv_comp_dim = getattr(config, "kv_comp_dim", 128)
            latent_cache = torch.zeros(batch_size, max_block_size, kv_comp_dim, dtype=torch.float32)
            flat_kv_caches.append(latent_cache)
        else:
            k_cache = torch.zeros(batch_size, max_block_size, n_kv_heads, head_dim, dtype=torch.float32)
            v_cache = torch.zeros(batch_size, max_block_size, n_kv_heads, head_dim, dtype=torch.float32)
            flat_kv_caches.append(k_cache)
            flat_kv_caches.append(v_cache)
            
    # Input names
    input_names = ["tokens", "start_pos"]
    for i in range(config.n_layer):
        if use_mla:
            input_names.append(f"past_key_value_layer_{i}")
        else:
            input_names.append(f"past_key_layer_{i}")
            input_names.append(f"past_value_layer_{i}")
            
    # Output names
    output_names = ["logits"]
    for i in range(config.n_layer):
        if use_mla:
            output_names.append(f"present_key_value_layer_{i}")
        else:
            output_names.append(f"present_key_layer_{i}")
            output_names.append(f"present_value_layer_{i}")
            
    logger.info(f"Exporting model to ONNX format at: {output_path} ...")
    
    # We trace the model using torch.onnx.export
    # Disable tracing warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    
    torch.onnx.export(
        wrapper,
        (tokens, start_pos, flat_kv_caches),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamo=False
    )
    
    logger.info("ONNX export completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-llm: Native ONNX Exporter")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to nano-llm .pt checkpoint file")
    parser.add_argument("--output_path", type=str, required=True, help="Target path to save the .onnx file")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX Opset version (default: 17)")
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint_path, args.output_path, args.opset_version)
