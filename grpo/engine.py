"""GRPO generation engine with KV-cache and action logprob computation."""

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_completions(
    model,
    tokenizer=None,
    prompt_ids: torch.Tensor = None,
    pixel_values: torch.Tensor = None,
    max_gen_len: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    use_mcts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate completions with KV-cache for GRPO rollouts."""
    # Handle positional args flexibility if tokenizer is omitted:
    # e.g., generate_completions(model, prompt_ids, ...)
    if prompt_ids is None:
        if isinstance(tokenizer, torch.Tensor):
            prompt_ids = tokenizer
            tokenizer = None
        else:
            raise ValueError("prompt_ids is required for generate_completions")
            
    model.eval()
    batch_size, prompt_len = prompt_ids.shape
    device = prompt_ids.device

    if use_mcts:
        from utils.mcts_engine import MCTSEngine
        assert tokenizer is not None, "tokenizer is required for MCTS search rollouts"
        max_total_len = prompt_len + max_gen_len
        full_seqs = torch.zeros(batch_size, max_total_len, dtype=torch.long, device=device)
        full_seqs[:, :prompt_len] = prompt_ids
        gen_mask = torch.zeros_like(full_seqs, dtype=torch.float32)
        gen_mask[:, prompt_len:] = 1.0
        
        for b in range(batch_size):
            p_text = tokenizer.decode(prompt_ids[b].tolist(), skip_special_tokens=True)
            engine = MCTSEngine(model, tokenizer, num_simulations=4)
            response_text = engine.search(p_text)
            r_ids = tokenizer.encode(response_text, add_special_tokens=False)
            r_ids = r_ids[:max_gen_len]
            full_seqs[b, prompt_len:prompt_len + len(r_ids)] = torch.tensor(r_ids, dtype=torch.long, device=device)
            gen_mask[b, prompt_len + len(r_ids):] = 0.0
            
        return full_seqs, gen_mask

    use_mla = getattr(model.config, 'use_mla', False)
    kv_comp_dim = getattr(model.config, 'kv_comp_dim', 128)
    n_kv_heads = model.config.n_kv_head if model.config.n_kv_head is not None else model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head

    full_seqs = torch.zeros(batch_size, prompt_len + max_gen_len, dtype=torch.long, device=device)
    full_seqs[:, :prompt_len] = prompt_ids
    gen_mask = torch.zeros_like(full_seqs, dtype=torch.float32)
    gen_mask[:, prompt_len:] = 1.0

    max_total_len = prompt_len + max_gen_len
    model_dtype = next(model.parameters()).dtype
    kv_caches = []
    for _ in range(model.config.n_layer):
        if use_mla:
            latent_cache = torch.zeros(batch_size, max_total_len, kv_comp_dim, device=device, dtype=model_dtype)
            kv_caches.append(latent_cache)
        else:
            k_cache = torch.zeros(batch_size, max_total_len, n_kv_heads, head_dim, device=device, dtype=model_dtype)
            v_cache = torch.zeros(batch_size, max_total_len, n_kv_heads, head_dim, device=device, dtype=model_dtype)
            kv_caches.append((k_cache, v_cache))

    # Prefill
    logits, _, _ = model(full_seqs[:, :prompt_len], pixel_values=pixel_values, start_pos=0, kv_caches=kv_caches)

    for i in range(max_gen_len):
        curr_pos = prompt_len + i
        last_token = full_seqs[:, curr_pos - 1:curr_pos]
        logits, _, _ = model(last_token, pixel_values=pixel_values, start_pos=curr_pos - 1, kv_caches=kv_caches)

        next_logits = logits[:, -1, :]
        if temperature > 0.0:
            next_logits = next_logits / temperature
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = -float('inf')
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        full_seqs[:, curr_pos] = next_token.squeeze(-1)

    return full_seqs, gen_mask


def compute_action_logprobs(logits: torch.Tensor, seqs: torch.Tensor, gen_mask: torch.Tensor) -> torch.Tensor:
    """Compute action log-probabilities for GRPO advantage-weighted updates."""
    # Shift: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_seqs = seqs[:, 1:].contiguous()
    shift_mask = gen_mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log-prob of the actual token
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_seqs.unsqueeze(-1)).squeeze(-1)
    # Zero out prompt/non-generated positions
    token_log_probs = token_log_probs * shift_mask

    return token_log_probs.sum(dim=-1)
