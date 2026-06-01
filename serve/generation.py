"""Tokenization adapter and static KV-cache generation."""

import time
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CustomTokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.special_tokens["<|pad|>"]
        self.eos_token_id = tokenizer.special_tokens["<|im_end|>"]

    def __len__(self) -> int:
        return len(self.tokenizer.vocab) + len(self.tokenizer.special_tokens)

    def __call__(self, text: str, **kwargs) -> dict:
        input_ids = self.encode(text, **kwargs)
        if kwargs.get("return_tensors") == "pt":
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor([input_ids])
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

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


@torch.no_grad()
def generate_with_static_cache(
    model,
    tokenizer,
    prompt: str,
    pixel_values: torch.Tensor = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cuda",
) -> str:
    """Generate text using pre-allocated static KV-cache buffers."""
    model.eval()

    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model.to(dtype=dtype)

    n_kv_heads = model.config.n_kv_head if model.config.n_kv_head is not None else model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head
    use_mla = getattr(model.config, 'use_mla', False)

    max_block_size = max(model.config.block_size, prompt_len + max_new_tokens + 32)
    kv_caches = []
    for _ in range(model.config.n_layer):
        if use_mla:
            kv_comp_dim = getattr(model.config, 'kv_comp_dim', 128)
            latent_cache = torch.zeros(1, max_block_size, kv_comp_dim, device=device, dtype=dtype)
            kv_caches.append(latent_cache)
        else:
            k_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
            v_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
            kv_caches.append((k_cache, v_cache))

    if use_mla:
        mla_mem = max_block_size * model.config.n_layer * model.config.kv_comp_dim * 2
        standard_mem = max_block_size * model.config.n_layer * n_kv_heads * head_dim * 2 * 2
        logger.info(f"MLA latent KV-cache allocated ({mla_mem / 1024 / 1024:.1f} MB vs {standard_mem / 1024 / 1024:.1f} MB standard, "
                     f"{standard_mem / mla_mem:.1f}x compression)")
    else:
        logger.info(f"Pre-allocated static KV-Cache buffers for {model.config.n_layer} layers.")

    logger.info(f"--- Stage 1: Prefill (ingesting {prompt_len} prompt tokens) ---")
    start_time = time.time()

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    pixel_values_batch = pixel_values.unsqueeze(0).to(device) if pixel_values is not None else None

    logits, _, _ = model(x, pixel_values=pixel_values_batch, start_pos=0, kv_caches=kv_caches)
    logits = logits[:, -1, :]

    ttft = time.time() - start_time
    logger.info(f"Time to First Token (TTFT): {ttft*1000:.2f} ms")

    if temperature == 0.0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    print("\nAssistant: ", end="", flush=True)
    word = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=False)
    print(word, end="", flush=True)

    tokens_generated = 1
    curr_pos = prompt_len
    if pixel_values is not None:
        curr_pos += pixel_values.size(0)

    from utils.kv_eviction import StreamingLLMEvictor
    evictor = StreamingLLMEvictor(num_sinks=4, recent_window=256)
    decode_start_time = time.time()

    for _ in range(max_new_tokens - 1):
        if curr_pos >= max_block_size - 1:
            logger.warning("Context block limit reached. Terminating generation.")
            break

        if curr_pos >= evictor.max_cache_size - 1:
            for k_cache, v_cache in kv_caches:
                k_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                    k_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
                v_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                    v_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
            curr_pos = evictor.max_cache_size - 1

        logits, _, _ = model(next_token, start_pos=curr_pos, kv_caches=kv_caches)
        logits = logits[:, -1, :]

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

        word = tokenizer.decode(next_token[0].tolist(), skip_special_tokens=False)
        print(word, end="", flush=True)

        if next_token.item() in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]:
            break

    print()
    decode_elapsed = time.time() - decode_start_time
    decode_throughput = (tokens_generated - 1) / decode_elapsed if decode_elapsed > 0 else 0
    total_elapsed = time.time() - start_time
    logger.info(
        f"\nGenerated {tokens_generated} tokens in {total_elapsed:.2f}s | "
        f"Decode speed: {decode_throughput:.2f} tokens/second"
    )
