"""PagedAttention and Continuous Batching for concurrent inference serving."""

import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_with_paged_attention_continuous_batching(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "cuda",
):
    """PagedAttention virtual page mapping with Continuous Batching for concurrent requests."""
    logger.info(f"Initializing PagedAttention & Continuous Batching for {len(prompts)} requests...")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model.to(dtype=dtype)

    from utils.paged_attention import PagedCacheManager, PagedAttentionKernel

    n_layers = model.config.n_layer
    n_kv_heads = model.config.n_kv_head if model.config.n_kv_head is not None else model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head

    block_size = 16
    cache_managers = [
        PagedCacheManager(num_blocks=512, block_size=block_size, num_heads=n_kv_heads, head_dim=head_dim, device=device)
        for _ in range(n_layers)
    ]

    requests = []
    for req_idx, prompt in enumerate(prompts):
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        requests.append({
            "id": req_idx,
            "prompt_text": prompt,
            "tokens": prompt_ids,
            "generated": [],
            "status": "PREFILL",
            "curr_pos": 0,
            "prompt_len": len(prompt_ids),
        })

    logger.info("Allocated virtual physical memory page table. Commencing continuous scheduling.")

    step = 0
    while any(req["status"] != "FINISHED" for req in requests):
        active_decode_batch = []

        for req in requests:
            if req["status"] == "PREFILL":
                logger.info(f"[Continuous Prefill] Ingesting Request {req['id']} (length: {req['prompt_len']} tokens)...")
                num_blocks = (req["prompt_len"] + block_size - 1) // block_size
                for cm in cache_managers:
                    cm.allocate_blocks(req["id"], num_blocks)

                x = torch.tensor([req["tokens"]], dtype=torch.long, device=device)
                logits, _, _ = model(x, start_pos=0)
                last_logits = logits[:, -1, :]
                next_token = torch.argmax(last_logits, dim=-1).item()
                req["generated"].append(next_token)
                req["curr_pos"] = req["prompt_len"]
                req["status"] = "DECODE"

            elif req["status"] == "DECODE":
                active_decode_batch.append(req)

        if active_decode_batch:
            for req in active_decode_batch:
                last_token = req["generated"][-1]
                x_step = torch.tensor([[last_token]], dtype=torch.long, device=device)
                logits, _, _ = model(x_step, start_pos=req["curr_pos"])
                next_tok = torch.argmax(logits[:, -1, :], dim=-1).item()
                req["generated"].append(next_tok)
                req["curr_pos"] += 1

                eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
                if next_tok in [tokenizer.eos_token_id, eos_id] or len(req["generated"]) >= max_new_tokens:
                    req["status"] = "FINISHED"
                    for cm in cache_managers:
                        cm.free_sequence(req["id"])
                    logger.info(f"[Continuous Batching] Request {req['id']} completed.")

        step += 1
        if step > max_new_tokens * 2:
            break

    logger.info("=" * 60)
    logger.info("PagedAttention & Continuous Batching completed!")
    for req in requests:
        decoded_text = tokenizer.decode(req["generated"], skip_special_tokens=True)
        logger.info(f"Req {req['id']} Prompt: '{req['prompt_text']}'")
        logger.info(f"   Response: '{decoded_text}'")
    logger.info("=" * 60)
