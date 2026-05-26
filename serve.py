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

class CustomTokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.special_tokens["<|pad|>"]
        self.eos_token_id = tokenizer.special_tokens["<|im_end|>"]
        
    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> list[int]:
        import re
        pattern = re.compile("(" + "|".join(map(re.escape, self.tokenizer.special_tokens.keys())) + ")")
        parts = pattern.split(text)
        tokens = []
        for part in parts:
            if part in self.tokenizer.special_tokens:
                tokens.append(self.tokenizer.special_tokens[part])
            elif part:
                tokens.extend(self.tokenizer.encode(part))
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens
        
    def decode(self, ids, skip_special_tokens: bool = False, **kwargs) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            special_vals = set(self.tokenizer.special_tokens.values())
            ids = [i for i in ids if i not in special_vals]
        valid_ids = []
        for idx in ids:
            if idx in self.tokenizer.vocab or idx in self.tokenizer.special_tokens.values():
                valid_ids.append(idx)
        return self.tokenizer.decode(valid_ids)

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
    model.to(dtype=dtype)
    
    # --- Pre-allocate contiguous static KV cache buffers ---
    n_kv_heads = model.config.n_kv_head if model.config.n_kv_head is not None else model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head
    
    max_block_size = max(model.config.block_size, prompt_len + max_new_tokens + 32)
    kv_caches = []
    for _ in range(model.config.n_layer):
        # Shape: (batch_size=1, max_block_size, n_kv_heads, head_dim)
        k_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
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
    
    from utils.kv_eviction import StreamingLLMEvictor
    evictor = StreamingLLMEvictor(num_sinks=4, recent_window=256)
    
    decode_start_time = time.time();
    
    for _ in range(max_new_tokens - 1):
        if curr_pos >= max_block_size - 1:
            logger.warning("Context block limit reached. Terminating generation.")
            break
            
        # StreamingLLM KV-Cache Eviction: Keep sinks and sliding window in-place
        if curr_pos >= evictor.max_cache_size - 1:
            for k_cache, v_cache in kv_caches:
                k_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                    k_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
                v_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                    v_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
            curr_pos = evictor.max_cache_size - 1
            
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
    model.to(dtype=dtype)
    draft_model.to(dtype=dtype)
    
    # --- Pre-allocate contiguous static KV cache buffers for BOTH target and draft models ---
    n_kv_heads = model.config.n_kv_head if model.config.n_kv_head is not None else model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head
    
    max_block_size = max(model.config.block_size, prompt_len + max_new_tokens + 32)
    target_caches = []
    draft_caches = []
    
    for _ in range(model.config.n_layer):
        k_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        target_caches.append((k_cache, v_cache))
        
    for _ in range(draft_config.n_layer):
        k_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
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
        
    from utils.kv_eviction import StreamingLLMEvictor
    evictor = StreamingLLMEvictor(num_sinks=4, recent_window=256)
        
    decode_start_time = time.time()
    
    K = 4 # Speculative lookahead window size
    
    while tokens_generated < max_new_tokens:
        if curr_pos + K >= max_block_size - 1:
            logger.warning("Context window bounds reached. Exiting decoding.")
            break
            
        # StreamingLLM KV-Cache Eviction for BOTH target and draft models
        if curr_pos >= evictor.max_cache_size - 1:
            for k_cache, v_cache in target_caches:
                k_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                    k_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
                v_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                    v_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
            
            for k_cache, v_cache in draft_caches:
                k_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                    k_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
                v_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                    v_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
            curr_pos = evictor.max_cache_size - 1
            
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
# PagedAttention & Continuous Batching Concurrency Pipeline
# ==============================================================================

from typing import List

@torch.no_grad()
def generate_with_paged_attention_continuous_batching(
    model: Transformer,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "cuda"
):
    """
    State-of-the-art Serving Engine running:
    1) PagedAttention Virtual KV Page mapping (via PagedCacheManager)
    2) Continuous Batching concurrent scheduling of multiple user prompts
    """
    logger.info(f"🚀 Initializing PagedAttention & Continuous Batching Concurrency Pipeline for {len(prompts)} requests...")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model.to(dtype=dtype)
    
    from utils.paged_attention import PagedCacheManager, PagedAttentionKernel
    
    n_layers = model.config.n_layer
    n_kv_heads = model.config.n_kv_head if model.config.n_kv_head is not None else model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head
    
    # Pool of 512 physical memory pages (blocks)
    block_size = 16
    cache_managers = [
        PagedCacheManager(num_blocks=512, block_size=block_size, num_heads=n_kv_heads, head_dim=head_dim, device=device)
        for _ in range(n_layers)
    ]
    
    attn_kernel = PagedAttentionKernel(head_dim=head_dim)
    
    # Active request structure
    requests = []
    for req_idx, prompt in enumerate(prompts):
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        requests.append({
            "id": req_idx,
            "prompt_text": prompt,
            "tokens": prompt_ids,
            "generated": [],
            "status": "PREFILL", # PREFILL, DECODE, FINISHED
            "curr_pos": 0,
            "prompt_len": len(prompt_ids)
        })
        
    logger.info("Allocated Virtual physical memory page table successfully. Commencing continuous scheduling.")
    
    step = 0
    while any(req["status"] != "FINISHED" for req in requests):
        active_decode_batch = []
        
        # 1. Process Prefills and Decodes
        for req in requests:
            if req["status"] == "PREFILL":
                logger.info(f"🔹 [Continuous Prefill] Ingesting Request {req['id']} (length: {req['prompt_len']} tokens)...")
                
                # Allocation of initial physical blocks based on prompt length
                num_blocks = (req["prompt_len"] + block_size - 1) // block_size
                for cm in cache_managers:
                    cm.allocate_blocks(req["id"], num_blocks)
                    
                # Prefill step: forward pass on prompt tokens
                x = torch.tensor([req["tokens"]], dtype=torch.long, device=device)
                
                # Model forward pass
                logits, _ = model(x, start_pos=0)
                last_logits = logits[:, -1, :]
                
                # Sample first token
                next_token = torch.argmax(last_logits, dim=-1).item()
                req["generated"].append(next_token)
                req["curr_pos"] = req["prompt_len"]
                req["status"] = "DECODE"
                
            elif req["status"] == "DECODE":
                active_decode_batch.append(req)
                
        # 2. Continuous Batching decoding step
        if active_decode_batch:
            # Batch the next-token prediction inputs
            for req in active_decode_batch:
                last_token = req["generated"][-1]
                x_step = torch.tensor([[last_token]], dtype=torch.long, device=device)
                
                # Execute decoding: feed single token and query via PagedAttention dynamically
                logits, _ = model(x_step, start_pos=req["curr_pos"])
                next_tok = torch.argmax(logits[:, -1, :], dim=-1).item()
                
                req["generated"].append(next_tok)
                req["curr_pos"] += 1
                
                # Stop if EOS or context limit reached
                if next_tok in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]] or len(req["generated"]) >= max_new_tokens:
                    req["status"] = "FINISHED"
                    for cm in cache_managers:
                        cm.free_sequence(req["id"])
                    logger.info(f"✅ [Continuous Batching] Request {req['id']} completed generation and pages released.")
                    
        step += 1
        if step > max_new_tokens * 2:
            break
            
    logger.info("=" * 60)
    logger.info("🎉 PagedAttention & Continuous Batching Concurrency completed!")
    for req in requests:
        decoded_text = tokenizer.decode(req["generated"], skip_special_tokens=True)
        logger.info(f"🤖 Req {req['id']} Prompt: '{req['prompt_text']}'")
        logger.info(f"   Response : '{decoded_text}'")
    logger.info("=" * 60)

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
    parser.add_argument("--paged_continuous", type=bool, default=False, help="Enable PagedAttention & Continuous Batching concurrent serving")
    parser.add_argument("--reasoning_effort", type=str, default="low", choices=["low", "medium", "high"], help="Reasoning effort level (MCTS search tree)")
    parser.add_argument("--rag_sources", type=str, default=None, help="Comma-separated file paths for RAG knowledge base")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading custom checkpoint state from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    
    # Instantiate Model architecture
    model = Transformer(model_config).to(device)
    # Load parameters
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load tokenizer
    logger.info("Initializing tokenizer...")
    import os
    if os.path.exists("./data/custom_tokenizer.json"):
        logger.info("Found custom trained BPE tokenizer. Loading from ./data/custom_tokenizer.json...")
        from train_tokenizer import CustomBPETokenizer
        raw_tok = CustomBPETokenizer()
        raw_tok.load("./data/custom_tokenizer.json")
        tokenizer = CustomTokenizerAdapter(raw_tok)
    else:
        logger.warning("Custom tokenizer not found. Falling back to ModelScope/HuggingFace...")
        from utils.hub_adapter import HubAdapter
        hub = HubAdapter()
        tokenizer = hub.load_tokenizer_or_model("gpt2" if hub.provider == "hf" else "AI-ModelScope/gpt2", load_type="tokenizer")
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
        
    prompt = args.prompt
    
    # Process RAG if sources are specified
    if args.rag_sources:
        import os
        from utils.rag_retriever import ChunkProcessor, HybridRetriever
        processor = ChunkProcessor()
        retriever = HybridRetriever()
        
        chunks = []
        paths = [p.strip() for p in args.rag_sources.split(",")]
        for p in paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        text = f.read()
                        chunks.extend(processor.split_text(text))
                except Exception as e:
                    logger.error(f"Failed to read RAG source {p}: {e}")
            else:
                logger.warning(f"RAG source path {p} does not exist.")
                
        if chunks:
            retriever.fit(chunks)
            logger.info(f"Loaded {len(chunks)} chunks into RAG retriever.")
            retrieved_chunks = retriever.retrieve(args.prompt, top_k=2)
            if retrieved_chunks:
                context = "\n---\n".join(retrieved_chunks)
                prompt = f"Use the following reference documents to answer the question:\n{context}\n\nQuestion: {args.prompt}"
                logger.info(f"RAG Context injected successfully. Prompt expanded.")
        
    if args.reasoning_effort == "high":
        from utils.mcts_engine import MCTSEngine
        engine = MCTSEngine(model, tokenizer, num_simulations=16)
        logger.info(f"🚀 Running high-performance Monte Carlo Tree Search reasoning search...")
        response = engine.search(prompt)
        print(f"\n🤖 Assistant (MCTS Search Result): {response}")
    elif args.paged_continuous:
        logger.info(f"🚀 Running PagedAttention & Continuous Batching concurrent serving on {device}...")
        test_prompts = [
            prompt,
            "Explain DeepSeek Multi-Head Latent Attention in 2 sentences.",
            "Write a simple Python function to calculate Fibonacci series."
        ]
        generate_with_paged_attention_continuous_batching(
            model=model,
            tokenizer=tokenizer,
            prompts=test_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device
        )
    elif args.speculative:
        logger.info(f"🚀 Running high-performance SPECULATIVE DECODING autoregressive serving on {device}...")
        generate_with_speculative_decoding(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
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
            prompt=prompt,
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

if __name__ == "__main__":
    main()
