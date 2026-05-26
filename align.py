import os
import sys
import time
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from model import ModelConfig, Transformer
from data import DPODataset
from utils.model_utils import configure_logging, count_parameters

logger = logging.getLogger(__name__)

# ==============================================================================
# Pure PyTorch DPO Log-Probability Math
# ==============================================================================

def compute_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the log-probability of generated tokens in an autoregressive causal setting.
    Masked prompt tokens (labeled -100) are ignored.
    
    Mathematically:
    log_prob = sum_{t in generation} log Softmax(logits_t)[labels_t]
    
    In PyTorch: negative cross-entropy is equivalent to positive log-likelihood!
    """
    # Autoregressive shift: predict next token (shift logits right by 1, labels left by 1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate negative log-likelihood per token
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    nll = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    nll = nll.view(shift_labels.size())  # Reshape to (batch_size, seq_len - 1)
    
    # Log-probability is negative NLL summed across the sequence
    log_probs = -nll.sum(dim=-1)
    return log_probs


def compute_dpo_loss(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    reference_chosen_logprobs: torch.Tensor,
    reference_rejected_logprobs: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Computes DPO Loss and implicit rewards.
    DPO Loss = -E[log_sigmoid(beta * (log_ratio_policy - log_ratio_reference))]
    """
    # Policy log-ratio (chosen / rejected)
    policy_log_ratio = policy_chosen_logprobs - policy_rejected_logprobs
    # Reference log-ratio
    reference_log_ratio = reference_chosen_logprobs - reference_rejected_logprobs
    
    # Standard DPO objective scale
    dpo_logits = beta * (policy_log_ratio - reference_log_ratio)
    loss = -F.logsigmoid(dpo_logits).mean()
    
    # Calculate implicit rewards for logging purposes (detached from gradient computation)
    chosen_rewards = beta * (policy_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = beta * (policy_rejected_logprobs - reference_rejected_logprobs).detach()
    
    # Accuracy measures how often the policy prefers the chosen response over the rejected one
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss, chosen_rewards.mean(), rejected_rewards.mean(), accuracy

# ==============================================================================
# Main Orchestrated DPO DDP Alignment Script
# ==============================================================================

def train():
    parser = argparse.ArgumentParser(description="nano-llm: Pure PyTorch Multi-GPU DPO Alignment Engine")
    parser.add_argument("--sft_checkpoint_path", type=str, required=True, help="Path to SFT trained checkpoint .pt file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to DPO pairwise train.jsonl file")
    parser.add_argument("--max_length", type=int, default=4096, help="Combined prompt + response limit")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Prompt token limit")
    parser.add_argument("--epochs", type=int, default=1, help="Training epoch limit")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--beta", type=float, default=0.1, help="KL regularization multiplier beta")
    parser.add_argument("--max_lr", type=float, default=5e-6, help="Peak learning rate (DPO requires lower LRs than SFT)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay multiplier")
    parser.add_argument("--output_dir", type=str, default="./outputs_dpo", help="Output saving directory")
    parser.add_argument("--seed", type=int, default=42, help="Random initialization seed")
    args = parser.parse_args()

    # 1. Initialize PyTorch DDP
    ddp = "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        is_master = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_master = True

    # Setup master logging
    log_level = logging.INFO if is_master else logging.WARNING
    configure_logging(level=log_level)
    
    logger.info(f"--- DPO DDP Node initialized | Rank: {ddp_rank} | World Size: {ddp_world_size} ---")
    
    torch.manual_seed(args.seed + ddp_rank)
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Tokenizer & Dataset Loader
    logger.info("Initializing tokenizer...")
    from serve import CustomTokenizerAdapter
    if os.path.exists("./data/custom_tokenizer.json"):
        logger.info("Found custom trained BPE tokenizer. Loading from ./data/custom_tokenizer.json...")
        from train_tokenizer import CustomBPETokenizer
        raw_tok = CustomBPETokenizer()
        raw_tok.load("./data/custom_tokenizer.json")
        tokenizer = CustomTokenizerAdapter(raw_tok)
    else:
        logger.warning("Custom tokenizer not found. Falling back to AutoTokenizer...")
        from utils.hub_adapter import HubAdapter
        hub = HubAdapter()
        tokenizer = hub.load_tokenizer_or_model("Qwen/Qwen2.5-7B" if hub.provider == "hf" else "qwen/Qwen2.5-7B", load_type="tokenizer", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    dataset = DPODataset(
        data_path=args.data_path, 
        tokenizer=tokenizer, 
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length
    )
    
    sampler = DistributedSampler(dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True) if ddp else None
    
    # Custom DPO collator (since chosen & rejected are padded separately in PyTorch loaders)
    def dpo_collator(samples):
        batch = {}
        for key in ["chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"]:
            tensor_list = [item[key] for item in samples]
            # Pad batch entries to the longest in current batch to conserve HBM
            max_len = max(x.size(0) for x in tensor_list)
            pad_val = tokenizer.pad_token_id if "input_ids" in key else -100
            
            batch[key] = torch.stack([
                torch.cat([x, torch.tensor([pad_val] * (max_len - x.size(0)), dtype=torch.long)])
                for x in tensor_list
            ])
        return batch

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        shuffle=(sampler is None),
        collate_fn=dpo_collator,
        pin_memory=True
    )

    # 3. Load Policy and Reference Models
    logger.info(f"Loading base checkpoint config from {args.sft_checkpoint_path}...")
    checkpoint = torch.load(args.sft_checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]
    
    logger.info("Initializing POLICY Model...")
    policy_model = Transformer(model_config).to(device)
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    
    logger.info("Initializing REFERENCE Model (Gradients disabled)...")
    reference_model = Transformer(model_config).to(device)
    reference_model.load_state_dict(checkpoint["model_state_dict"])
    reference_model.eval()
    
    # Disable gradients on Reference model to save massive GPU cycles
    for param in reference_model.parameters():
        param.requires_grad = False

    # Wrap Policy model in DDP
    if ddp:
        policy_model = DDP(policy_model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)

    # 4. Optimizer Setup
    # Filter weight decay parameters
    param_dict = {pn: p for pn, p in policy_model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.max_lr, betas=(0.9, 0.95))

    # 5. DPO Training Loop
    total_steps = len(dataloader) * args.epochs
    step_counter = 0
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    logger.info("🚀 Starting DPO alignment loop...")
    
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
            
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract chosen & rejected batch tensors
            c_input_ids = batch["chosen_input_ids"].to(device, non_blocking=True)
            c_labels = batch["chosen_labels"].to(device, non_blocking=True)
            r_input_ids = batch["rejected_input_ids"].to(device, non_blocking=True)
            r_labels = batch["rejected_labels"].to(device, non_blocking=True)
            
            # --- Autocast forward pass for Policy Model ---
            with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                # Chosen forward pass
                policy_chosen_logits, _ = policy_model(c_input_ids)
                policy_chosen_logprobs = compute_logprobs(policy_chosen_logits, c_labels)
                
                # Rejected forward pass
                policy_rejected_logits, _ = policy_model(r_input_ids)
                policy_rejected_logprobs = compute_logprobs(policy_rejected_logits, r_labels)
                
            # --- Autocast forward pass for Reference Model (no gradient tracking) ---
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                    ref_chosen_logits, _ = reference_model(c_input_ids)
                    ref_chosen_logprobs = compute_logprobs(ref_chosen_logits, c_labels)
                    
                    ref_rejected_logits, _ = reference_model(r_input_ids)
                    ref_rejected_logprobs = compute_logprobs(ref_rejected_logits, r_labels)
                    
            # --- Compute Raw DPO Loss ---
            loss, reward_c, reward_r, accuracy = compute_dpo_loss(
                policy_chosen_logprobs=policy_chosen_logprobs,
                policy_rejected_logprobs=policy_rejected_logprobs,
                reference_chosen_logprobs=ref_chosen_logprobs,
                reference_rejected_logprobs=ref_rejected_logprobs,
                beta=args.beta
            )
            
            # Scale loss matching gradient accumulation
            loss = loss / args.grad_accum_steps
            
            # DDP backward synchronization only triggers at accumulation end step
            if ddp:
                policy_model.require_backward_grad_sync = (batch_idx + 1) % args.grad_accum_steps == 0
                
            # Execute backward
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_counter += 1
                
                if is_master and step_counter % 5 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Step {step_counter}/{total_steps//args.grad_accum_steps} | "
                        f"DPO Loss: {loss.item()*args.grad_accum_steps:.4f} | "
                        f"Reward Chosen: {reward_c.item():.4f} | "
                        f"Reward Rej: {reward_r.item():.4f} | "
                        f"Accuracy: {accuracy.item()*100:.1f}% | "
                        f"Time: {elapsed:.2f}s"
                    )
                    start_time = time.time()

    # Save final DPO checkpoint
    if is_master:
        final_path = os.path.join(args.output_dir, "checkpoint_dpo.pt")
        logger.info(f"Saving final DPO model checkpoint to {final_path}...")
        raw_model = policy_model.module if ddp else policy_model
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "config": model_config,
            "step": step_counter
        }, final_path)
        
    if ddp:
        dist.destroy_process_group()
    logger.info("nano-llm: DPO alignment completed successfully!")

if __name__ == "__main__":
    train()
