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
    logits, _ = model(x, start_pos=0, kv_caches=kv_caches)
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
    
    decode_start_time = time.time()
    
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
        
    logger.info(f"🚀 Running high-performance static KV-cached autoregressive serving on {device}...")
    generate_with_static_cache(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )

if __name__ == "__main__":
    main()
