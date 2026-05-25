import sys
import time
import argparse
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# High-Performance Autoregressive Generation with Static KV-Cache
# ==============================================================================

@torch.no_grad()
def generate_with_static_cache(
    model: Transformer,
    tokenizer,
    prompt: str,
    pixel_values: torch.Tensor = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cuda"
) -> str:
    """
    Generates text using pre-allocated static KV-cache buffers to avoid quadratic attention.
    
    1) Stage 1 (Prefill): Feed the full prompt at start_pos = 0 to populate the cache.
    2) Stage 2 (Decode): Feed only the single newly generated token at start_pos = current_len.
    """
    model.eval()
    
    # Format prompt matching standard ChatML SFT template
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # --- Pre-allocate contiguous static KV cache buffers ---
    n_kv_heads = model.config.n_kv_head if model.config.n_kv_head is not None else model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head
    
    kv_caches = []
    for _ in range(model.config.n_layer):
        # Shape: (batch_size=1, max_block_size, n_kv_heads, head_dim)
        k_cache = torch.zeros(1, model.config.block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(1, model.config.block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        kv_caches.append((k_cache, v_cache))
        
    logger.info(f"Pre-allocated static KV-Cache buffers for {model.config.n_layer} layers. Memory allocated successfully.")
    
    # --------------------------------------------------------------------------
    # Stage 1: Prefill (process full prompt)
    # --------------------------------------------------------------------------
    logger.info(f"--- Stage 1: Prefill (ingesting {prompt_len} prompt tokens) ---")
    start_time = time.time()
    
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device) # (1, prompt_len)
    
    # Forward pass on the full prompt to populate the cache up to prompt_len
    # logits shape: (1, 1, vocab_size) - only output final token logits
    if pixel_values is not None:
        pixel_values_batch = pixel_values.unsqueeze(0).to(device)
    else:
        pixel_values_batch = None
        
    logits, _ = model(x, pixel_values=pixel_values_batch, start_pos=0, kv_caches=kv_caches)
    logits = logits[:, -1, :] # Select final step logits
    
    ttft = time.time() - start_time
    logger.info(f"⏱️ Time to First Token (TTFT): {ttft*1000:.2f} ms")
    
    # Sample first assistant token
    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
    print("\n🤖 Assistant: ", end="", flush=True)
    word = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=False)
    print(word, end="", flush=True)
    
    tokens_generated = 1
    
    # --------------------------------------------------------------------------
    # Stage 2: Decode (autoregressive token-by-token loop)
    # --------------------------------------------------------------------------
    # We only feed the single newly sampled token (shape: 1, 1) rather than the growing history!
    curr_pos = prompt_len
    if pixel_values is not None:
        curr_pos += pixel_values.size(0)
    
    decode_start_time = time.time();
    
    for _ in range(max_new_tokens - 1):
        if curr_pos >= model.config.block_size - 1:
            logger.warning("Context block limit reached. Terminating generation.")
            break
            
        # Feed ONLY the single new token!
        x_step = next_token
        
        # Forward pass: start_pos specifies where to insert into static KV cache
        logits, _ = model(x_step, start_pos=curr_pos, kv_caches=kv_caches)
        logits = logits[:, -1, :]
        
        # Sample next token
        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        curr_pos += 1
        tokens_generated += 1
        
        # Stream print
        word = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=False)
        print(word, end="", flush=True)
        
        # Terminate if EOS reached
        if next_token.item() in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]:
            break
            
    print() # Final newline
    
    decode_elapsed = time.time() - decode_start_time
    decode_throughput = (tokens_generated - 1) / decode_elapsed if decode_elapsed > 0 else 0
    total_elapsed = time.time() - start_time
    
    logger.info(
        f"\n⚡ Generated {tokens_generated} tokens in {total_elapsed:.2f}s | "
        f"Decode speed: {decode_throughput:.2f} tokens/second"
    )
    return


@torch.no_grad()
def generate_with_speculative_decoding(
    model: Transformer,
    tokenizer,
    prompt: str,
    pixel_values: torch.Tensor = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    device: str = "cuda"
) -> str:
    """
    Generates text using Speculative Decoding (投机采样) to accelerate decoding throughput.
    1. Instantiates a 2-layer draft model sharing embedding weights.
    2. Draft model greedily generates K draft tokens.
    3. Target model verifies the K draft tokens in a single forward pass.
    4. Accepts matched prefix, replaces mismatch with target prediction, and repeats.
    """
    import copy
    from model import ModelConfig
    
    model.eval()
    
    # 1. Instantiate the Draft Model (shared config but only 2 layers)
    draft_config = copy.deepcopy(model.config)
    draft_config.n_layer = min(2, model.config.n_layer)
    draft_model = Transformer(draft_config).to(device)
    
    # Share embedding and output layer weights to align token semantics
    draft_model.tok_embeddings.weight.data.copy_(model.tok_embeddings.weight.data)
    draft_model.output.weight.data.copy_(model.output.weight.data)
    draft_model.eval()
    
    # Format prompt matching ChatML
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # --- Pre-allocate contiguous static KV cache buffers for BOTH target and draft models ---
    n_kv_heads = model.config.n_kv_head if model.config.n_kv_head is not None else model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head
    
    target_caches = []
    draft_caches = []
    
    for _ in range(model.config.n_layer):
        k_cache = torch.zeros(1, model.config.block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(1, model.config.block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        target_caches.append((k_cache, v_cache))
        
    for _ in range(draft_config.n_layer):
        k_cache = torch.zeros(1, model.config.block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(1, model.config.block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        draft_caches.append((k_cache, v_cache))
        
    logger.info("Pre-allocated static KV-Cache buffers for Speculative Decoding (Target & Draft).")
    
    # --------------------------------------------------------------------------
    # Stage 1: Prefill BOTH models
    # --------------------------------------------------------------------------
    logger.info(f"Prefilling target & draft models with {prompt_len} tokens...")
    start_time = time.time()
    
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    if pixel_values is not None:
        pixel_values_batch = pixel_values.unsqueeze(0).to(device)
    else:
        pixel_values_batch = None
        
    # Prefill target model
    target_logits, _ = model(x, pixel_values=pixel_values_batch, start_pos=0, kv_caches=target_caches)
    target_logits = target_logits[:, -1, :]
    
    # Prefill draft model
    draft_logits, _ = draft_model(x, pixel_values=pixel_values_batch, start_pos=0, kv_caches=draft_caches)
    draft_logits = draft_logits[:, -1, :]
    
    ttft = time.time() - start_time
    logger.info(f"⏱️ Prefill complete | Time to First Token (TTFT): {ttft*1000:.2f} ms")
    
    # Sample first token
    next_token = torch.argmax(target_logits, dim=-1, keepdim=True)
    
    print("\n🚀 [Speculative serving] Assistant: ", end="", flush=True)
    word = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=False)
    print(word, end="", flush=True)
    
    tokens_generated = 1
    
    # --------------------------------------------------------------------------
    # Stage 2: Speculative decoding loop
    # --------------------------------------------------------------------------
    curr_pos = prompt_len
    if pixel_values is not None:
        curr_pos += pixel_values.size(0)
        
    decode_start_time = time.time()
    
    K = 4 # Speculative lookahead window size
    
    while tokens_generated < max_new_tokens:
        if curr_pos + K >= model.config.block_size - 1:
            logger.warning("Context window bounds reached. Exiting decoding.")
            break
            
        # 1. Draft model generates K tokens greedily (using its own KV-cache)
        draft_tokens = []
        temp_pos = curr_pos
        
        # Populate draft KV cache at curr_pos with the last confirmed token
        draft_model(next_token, start_pos=temp_pos, kv_caches=draft_caches)
        temp_pos += 1
        
        draft_input = next_token
        for _ in range(K):
            d_logits, _ = draft_model(draft_input, start_pos=temp_pos, kv_caches=draft_caches)
            d_tok = torch.argmax(d_logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(d_tok)
            draft_input = d_tok
            temp_pos += 1
            
        # 2. Target model verifies K tokens
        target_predictions = []
        target_input = next_token
        
        t_logits, _ = model(target_input, start_pos=curr_pos, kv_caches=target_caches)
        t_pred = torch.argmax(t_logits[:, -1, :], dim=-1, keepdim=True)
        target_predictions.append(t_pred)
        
        for i in range(K - 1):
            t_logits, _ = model(draft_tokens[i], start_pos=curr_pos + 1 + i, kv_caches=target_caches)
            t_pred = torch.argmax(t_logits[:, -1, :], dim=-1, keepdim=True)
            target_predictions.append(t_pred)
            
        # 3. Compare draft tokens vs target predictions
        accepted_tokens = 0
        for i in range(K):
            if draft_tokens[i].item() == target_predictions[i].item():
                accepted_tokens += 1
                next_token = draft_tokens[i]
                # Print accepted token
                word = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=False)
                print(word, end="", flush=True)
                tokens_generated += 1
            else:
                # Mismatch! The target prediction is the ground truth
                next_token = target_predictions[i]
                word = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=False)
                print(word, end="", flush=True)
                tokens_generated += 1
                break
                
        # Update current positions
        curr_pos += (accepted_tokens + 1)
        
        # Terminate if EOS reached
        if next_token.item() in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]:
            break
            
    print() # Final newline
    
    decode_elapsed = time.time() - decode_start_time
    decode_throughput = (tokens_generated - 1) / decode_elapsed if decode_elapsed > 0 else 0
    total_elapsed = time.time() - start_time
    
    logger.info(
        f"\n⚡ Generated {tokens_generated} tokens in {total_elapsed:.2f}s | "
        f"Decode speed (with Speculative Decoding): {decode_throughput:.2f} tokens/second"
    )
    return


# ==============================================================================
# Serving Endpoint Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="nano-llm: Autoregressive KV-Cached Serving Server")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved .pt model checkpoint file")
    parser.add_argument("--prompt", type=str, default="Tell me a joke about computer programming.", help="Prompt content to trigger")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation diversity temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K tail filter limits")
    parser.add_argument("--image", type=str, default=None, help="Optional image path or URL for multimodal VLM serving")
    parser.add_argument("--speculative", type=bool, default=False, help="Enable Speculative Decoding acceleration with 2-layer draft model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading custom checkpoint state from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_config = checkpoint["config"]
    
    # Instantiate Model architecture
    model = Transformer(model_config).to(device)
    # Load parameters
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B") 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        
    # Process image if provided
    if args.image is not None:
        from utils.vision_helper import extract_image_features
        pixel_values = extract_image_features(
            image_path_or_url=args.image,
            vision_dim=model_config.vision_dim or 1152,
            num_patches=16,
            device=device
        )
    else:
        pixel_values = None
        
    if args.speculative:
        logger.info(f"🚀 Running high-performance SPECULATIVE DECODING autoregressive serving on {device}...")
        generate_with_speculative_decoding(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
    else:
        logger.info(f"🚀 Running high-performance static KV-cached autoregressive serving on {device}...")
        generate_with_static_cache(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

if __name__ == "__main__":
    main()
