"""Speculative decoding with 2-layer draft model for accelerated inference."""

import copy
import time
import logging
import torch
import torch.nn.functional as F
from model import Transformer

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_with_speculative_decoding(
    model,
    tokenizer,
    prompt: str,
    pixel_values: torch.Tensor = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    device: str = "cuda",
) -> str:
    """Speculative decoding: draft model generates K tokens, target model verifies."""
    from model import ModelConfig

    model.eval()

    draft_config = copy.deepcopy(model.config)
    draft_config.n_layer = min(2, model.config.n_layer)
    draft_model = Transformer(draft_config).to(device)
    draft_model.tok_embeddings.weight.data.copy_(model.tok_embeddings.weight.data)
    draft_model.output.weight.data.copy_(model.output.weight.data)
    draft_model.eval()

    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model.to(dtype=dtype)
    draft_model.to(dtype=dtype)

    n_kv_heads = model.config.n_kv_head if model.config.n_kv_head is not None else model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head

    max_block_size = max(model.config.block_size, prompt_len + max_new_tokens + 32)
    use_mla = getattr(model.config, 'use_mla', False)
    target_caches = []
    draft_caches = []
    for _ in range(model.config.n_layer):
        if use_mla:
            kv_comp_dim = getattr(model.config, 'kv_comp_dim', 128)
            latent_cache = torch.zeros(1, max_block_size, kv_comp_dim, device=device, dtype=dtype)
            target_caches.append(latent_cache)
        else:
            k_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
            v_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
            target_caches.append((k_cache, v_cache))
    for _ in range(draft_config.n_layer):
        if use_mla:
            kv_comp_dim = getattr(draft_config, 'kv_comp_dim', 128)
            latent_cache = torch.zeros(1, max_block_size, kv_comp_dim, device=device, dtype=dtype)
            draft_caches.append(latent_cache)
        else:
            k_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
            v_cache = torch.zeros(1, max_block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
            draft_caches.append((k_cache, v_cache))

    logger.info("Prefilling target & draft models...")
    start_time = time.time()

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    pixel_values_batch = pixel_values.unsqueeze(0).to(device) if pixel_values is not None else None

    target_logits, _, _ = model(x, pixel_values=pixel_values_batch, start_pos=0, kv_caches=target_caches)
    target_logits = target_logits[:, -1, :]
    draft_logits, _, _ = draft_model(x, pixel_values=pixel_values_batch, start_pos=0, kv_caches=draft_caches)

    ttft = time.time() - start_time
    logger.info(f"Prefill complete | TTFT: {ttft*1000:.2f} ms")

    next_token = torch.argmax(target_logits, dim=-1, keepdim=True)
    generated_ids = [next_token.item()]

    print("\n[Speculative] Assistant: ", end="", flush=True)
    word = tokenizer.decode(next_token[0].tolist())
    print(word, end="", flush=True)

    tokens_generated = 1
    curr_pos = prompt_len
    if pixel_values is not None:
        curr_pos += pixel_values.size(0)

    from utils.kv_eviction import StreamingLLMEvictor
    evictor = StreamingLLMEvictor(num_sinks=4, recent_window=256)
    decode_start_time = time.time()
    K = 4

    while tokens_generated < max_new_tokens:
        if curr_pos + K >= max_block_size - 1:
            break

        if curr_pos >= evictor.max_cache_size - 1:
            for caches in [target_caches, draft_caches]:
                for idx, cache_item in enumerate(caches):
                    if isinstance(cache_item, tuple):
                        k_cache, v_cache = cache_item
                        k_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                            k_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
                        v_cache[:, evictor.num_sinks : evictor.max_cache_size - 1, :, :] = \
                            v_cache[:, curr_pos - evictor.recent_window + 1 : curr_pos, :, :].clone()
                    else:
                        cache_item[:, evictor.num_sinks : evictor.max_cache_size - 1, :] = \
                            cache_item[:, curr_pos - evictor.recent_window + 1 : curr_pos, :].clone()
            curr_pos = evictor.max_cache_size - 1

        draft_tokens = []
        temp_pos = curr_pos
        draft_model(next_token, start_pos=temp_pos, kv_caches=draft_caches)
        temp_pos += 1
        draft_input = next_token
        for _ in range(K):
            d_logits, _, _ = draft_model(draft_input, start_pos=temp_pos, kv_caches=draft_caches)
            d_tok = torch.argmax(d_logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(d_tok)
            draft_input = d_tok
            temp_pos += 1

        target_predictions = []
        t_logits, _, _ = model(next_token, start_pos=curr_pos, kv_caches=target_caches)
        target_predictions.append(torch.argmax(t_logits[:, -1, :], dim=-1, keepdim=True))
        for i in range(K - 1):
            t_logits, _, _ = model(draft_tokens[i], start_pos=curr_pos + 1 + i, kv_caches=target_caches)
            target_predictions.append(torch.argmax(t_logits[:, -1, :], dim=-1, keepdim=True))

        accepted = 0
        for i in range(K):
            if draft_tokens[i].item() == target_predictions[i].item():
                accepted += 1
                next_token = draft_tokens[i]
                word = tokenizer.decode(next_token[0].tolist())
                print(word, end="", flush=True)
                generated_ids.append(next_token.item())
                tokens_generated += 1
            else:
                next_token = target_predictions[i]
                word = tokenizer.decode(next_token[0].tolist())
                print(word, end="", flush=True)
                generated_ids.append(next_token.item())
                tokens_generated += 1
                break

        curr_pos += accepted + 1

        eos_id = getattr(tokenizer, 'eos_token_id', None)
        im_end_ids = tokenizer.encode("<|im_end|>")
        stop_ids = set()
        if eos_id is not None:
            stop_ids.add(eos_id)
        if im_end_ids:
            stop_ids.add(im_end_ids[0])
        if next_token.item() in stop_ids:
            break

    print()
    decode_elapsed = time.time() - decode_start_time
    decode_throughput = (tokens_generated - 1) / decode_elapsed if decode_elapsed > 0 else 0
    total_elapsed = time.time() - start_time
    logger.info(
        f"\nGenerated {tokens_generated} tokens in {total_elapsed:.2f}s | "
        f"Decode speed (Speculative): {decode_throughput:.2f} tok/s"
    )
    return tokenizer.decode(generated_ids)
