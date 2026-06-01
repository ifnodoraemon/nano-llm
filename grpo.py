import os
import sys
import time
import json
import argparse
import logging
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Any
from transformers import AutoTokenizer

from model import ModelConfig, Transformer
from utils.model_utils import configure_logging, count_parameters
from utils.sandbox_executor import SandboxCodeExecutor
from utils.ddp_helper import init_ddp
from utils.tokenizer_loader import load_tokenizer
from utils.checkpoint_utils import load_checkpoint_with_fp8_translation
from utils.training_utils import validate_dataset, assert_grad_accum_safe

from grpo.dataset import GRPODataset, grpo_collate_fn
from grpo.rewards import extract_answer, evaluate_completion_rewards, GRPORewardScaler, AdaptiveKLTuner
from grpo.engine import generate_completions, compute_action_logprobs

logger = logging.getLogger(__name__)

# ==============================================================================
# 5. Distributed GRPO Engine Loop
# ==============================================================================

def train():
    parser = argparse.ArgumentParser(description="nano-llm: Pure PyTorch Multi-GPU GRPO Engine")
    parser.add_argument("--sft_checkpoint_path", type=str, required=True, help="Path to base SFT model checkpoint .pt")
    parser.add_argument("--data_path", type=str, required=True, help="Path to prompt seed train.jsonl file")
    parser.add_argument("--group_size", type=int, default=4, help="Group size G (completions per prompt)")
    parser.add_argument("--max_prompt_len", type=int, default=512, help="Prompt limit")
    parser.add_argument("--max_gen_len", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--epochs", type=int, default=1, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size of prompts per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--beta", type=float, default=0.04, help="KL regularization penalty scaling")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip range parameter")
    parser.add_argument("--max_lr", type=float, default=1e-6, help="Max learning rate")
    parser.add_argument("--output_dir", type=str, default="./outputs_grpo", help="Model saving directory")
    parser.add_argument("--use_mcts", type=str, default="False", help="Enable Monte Carlo Tree Search rollouts in GRPO")
    args = parser.parse_args()

    # 1. Initialize PyTorch DDP
    dist_info = init_ddp()
    use_ddp = dist_info["ddp"]
    ddp_rank = dist_info["rank"]
    ddp_local_rank = dist_info["local_rank"]
    ddp_world_size = dist_info["world_size"]
    device = dist_info["device"]
    is_master = dist_info["is_master"]
    
    # Dist environment auto-tuning
    from utils.dist_helper import autotune_nccl
    autotune_nccl()

    from utils.checkpoint_saver import BackgroundCheckpointSaver, ElasticRestoreManager
    saver = BackgroundCheckpointSaver()
    restore_mgr = ElasticRestoreManager(args.output_dir)

    from utils.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(
        project="nano-llm",
        config=vars(args),
        mode="offline" if is_master else "disabled",
        log_dir="./logs",
    )

    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
        configure_logging(logging.INFO)
        logger.info("Initializing Group Relative Policy Optimization (GRPO)...")
        # 0. Initialize hardware telemetry monitor
        from utils.system_monitor import SystemMonitor
        monitor = SystemMonitor()
    else:
        configure_logging(logging.ERROR)
        monitor = None

    # Load Tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = load_tokenizer(fallback_model_name="gpt2")
    
    # Load base model checkpoint
    if is_master:
        logger.info(f"Loading SFT model from checkpoint: {args.sft_checkpoint_path}")
    config, state_dict = load_checkpoint_with_fp8_translation(args.sft_checkpoint_path, map_location="cpu")
    
    # Instantiate models
    # Policy model (Trainable)
    policy_model = Transformer(config).to(device)
    policy_model.load_state_dict(state_dict)
    
    # Reference model (Frozen baseline anchor)
    ref_model = Transformer(config).to(device)
    ref_model.load_state_dict(state_dict)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    if use_ddp:
        policy_model = DDP(policy_model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
        
    # Optimizer Fused AdamW
    optim_groups = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.max_lr, weight_decay=0.01, fused=True)

    # Auto-detect checkpoint to self-heal and hot-restore on failure
    has_ckpt, restored_step, restored_epoch = restore_mgr.auto_detect_checkpoint()
    if has_ckpt:
        restored_step, restored_epoch, _ = restore_mgr.restore_training_state(policy_model, optimizer)

    # Loader setup
    dataset = GRPODataset(args.data_path, tokenizer, max_prompt_length=args.max_prompt_len)
    validate_dataset(dataset, tokenizer=tokenizer, name="GRPO training dataset")

    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        collate_fn=lambda b: grpo_collate_fn(b, pad_token_id=tokenizer.pad_token_id), 
        shuffle=(sampler is None)
    )
    assert_grad_accum_safe(loader, args.grad_accum_steps, args.batch_size)

    # High-end alignment controllers & mutation engine initialization
    reward_scaler = GRPORewardScaler()
    kl_tuner = AdaptiveKLTuner(target_kl=0.2)
    
    from utils.evol_instruct import EvolInstructEngine
    import random
    evol_engine = EvolInstructEngine()
    current_beta = args.beta

    # Training epochs loop
    step_idx = 0
    start_epoch = 0
    if has_ckpt:
        step_idx = restored_step
        start_epoch = restored_epoch
    if is_master:
        tracker.log_config({
            "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
            "batch_size": args.batch_size, "group_size": args.group_size,
            "max_prompt_len": args.max_prompt_len, "max_gen_len": args.max_gen_len,
            "grad_accum_steps": args.grad_accum_steps, "epochs": args.epochs,
            "beta": args.beta, "clip_eps": args.clip_eps, "max_lr": args.max_lr,
        })
    try:
        for epoch in range(start_epoch, args.epochs):
            if sampler:
                sampler.set_epoch(epoch)

            policy_model.train()
            for batch in loader:
                optimizer.zero_grad()

                prompt_texts = batch["prompt_texts"]
                prompt_ids = batch["prompt_ids"].to(device)
                attention_masks = batch["attention_masks"].to(device)
                pixel_values = batch["pixel_values"]
                ground_truths = batch["ground_truths"]

                micro_batch_size = len(prompt_texts)

                # Accumulator tensors for step logs
                total_loss = 0.0
                total_reward = 0.0
                total_kl = 0.0

                # Process prompt-by-prompt to manage micro-step groups cleanly
                for p_idx in range(micro_batch_size):
                    p_text = prompt_texts[p_idx]
                    p_ids = prompt_ids[p_idx].unsqueeze(0).to(device)  # Shape [1, prompt_len]
                    gt = ground_truths[p_idx]

                    # Apply online Evol-Instruct prompt mutation with 50% probability
                    if random.random() < 0.5:
                        p_text = evol_engine.mutate(p_text)

                    # Replicate prompt ids G times (Group size replication)
                    # G_prompt_ids shape: [G, prompt_len]
                    G_prompt_ids = p_ids.repeat(args.group_size, 1)

                    # Handle multimodal pixel values replication
                    if pixel_values is not None:
                        p_val = pixel_values[p_idx].unsqueeze(0).repeat(args.group_size, 1, 1).to(device)
                    else:
                        p_val = None

                    # 1. Rollout: Sample G completions from current policy model
                    # full_seqs shape: [G, prompt_len + max_gen_len]
                    full_seqs, gen_mask = generate_completions(
                        model=policy_model.module if use_ddp else policy_model, 
                        prompt_ids=G_prompt_ids,
                        pixel_values=p_val,
                        max_gen_len=args.max_gen_len,
                        temperature=0.8,
                        top_p=0.9,
                        use_mcts=args.use_mcts.lower() == "true",
                        tokenizer=tokenizer
                    )

                    # Decode completions to text for reward evaluations
                    completions_text = []
                    for g_i in range(args.group_size):
                        gen_tokens = full_seqs[g_i, G_prompt_ids.shape[1]:]
                        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        completions_text.append(gen_text)

                    # 2. Score completions via our rule-based verifiers
                    rewards = evaluate_completion_rewards([p_text] * args.group_size, completions_text, [gt] * args.group_size)
                    rewards = rewards.to(device)
                    total_reward += rewards.mean().item()

                    # 3. Advantage calculation (Normalizing rewards using GRPORewardScaler)
                    scaled_rewards = reward_scaler.update_and_scale(rewards)
                    r_mean = scaled_rewards.mean()
                    r_std = scaled_rewards.std()
                    # If std is zero (all samples scored same), set std to 1.0 to avoid NaNs
                    if r_std < 1e-6:
                        r_std = 1.0
                    advantages = (scaled_rewards - r_mean) / (r_std + 1e-8)

                    # 4. Compute Log Probabilities under Old/Reference Model & Current Policy Model
                    # In GRPO, we sample using policy_model and compute active forward logits
                    policy_model.train()
                    policy_logits, _, _ = policy_model(full_seqs, pixel_values=p_val)
                    policy_logprobs = compute_action_logprobs(policy_logits, full_seqs, gen_mask)

                    with torch.no_grad():
                        ref_logits, _, _ = ref_model(full_seqs, pixel_values=p_val)
                        ref_logprobs = compute_action_logprobs(ref_logits, full_seqs, gen_mask)

                    # In first micro-step, old_policy logprobs are equivalent to policy logprobs (surrogate starts at 1.0)
                    old_policy_logprobs = policy_logprobs.detach().clone()

                    # 5. Calculate surrogate GRPO loss
                    # Log ratios
                    log_ratio = policy_logprobs - old_policy_logprobs
                    ratio = torch.exp(log_ratio)

                    # Clipped surrogate objective
                    surrogate1 = ratio * advantages
                    surrogate2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
                    surrogate_loss = -torch.min(surrogate1, surrogate2).mean()

                    # KL Divergence Regularization penalty
                    # KL = exp(ref_logprob - policy_logprob) - (ref_logprob - policy_logprob) - 1
                    kl_div = torch.exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1
                    kl_loss = kl_div.mean()
                    total_kl += kl_loss.item()

                    # Grand GRPO Loss
                    loss = surrogate_loss + current_beta * kl_loss
                    loss = loss / (micro_batch_size * args.grad_accum_steps)
                    total_loss += loss.item()

                    # Backpropagation
                    loss.backward()

                # Gradient clipping and optimizer step
                if (step_idx + 1) % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Dynamically tune current_beta using the Adaptive KL controller
                    mean_kl = total_kl / micro_batch_size
                    current_beta = kl_tuner.tune_beta(mean_kl, current_beta)

                    if is_master:
                        tracker.log({
                            "grpo/loss": total_loss * args.grad_accum_steps,
                            "grpo/mean_reward": total_reward / micro_batch_size,
                            "grpo/kl_div": mean_kl,
                            "grpo/current_beta": current_beta,
                        }, step=step_idx)
                        telemetry_str = monitor.get_formatted_telemetry()
                        logger.info(
                            f"Step {step_idx + 1} | "
                            f"Loss: {total_loss * args.grad_accum_steps:.4f} | "
                            f"Mean Reward: {total_reward / micro_batch_size:.2f} | "
                            f"Mean KL: {total_kl / micro_batch_size:.4f} | "
                            f"Adaptive Beta: {current_beta:.4f} | "
                            f"{telemetry_str}"
                        )

                        # Print detailed ASCII telemetry cockpit every 50 steps
                        if (step_idx + 1) % 50 == 0:
                            monitor.print_dashboard()

                        # Save system telemetry to outputs directory
                        import json
                        os.makedirs("outputs", exist_ok=True)
                        with open("outputs/system_telemetry.json", "w") as f:
                            json.dump(monitor.get_telemetry_report(), f, indent=2)

                        # Non-blocking asynchronous checkpoint and manifest update every 50 steps
                        if (step_idx + 1) % 50 == 0:
                            saver.save_checkpoint(
                                model=policy_model,
                                optimizer=optimizer,
                                lr_scheduler=None,
                                config=config,
                                step=step_idx + 1,
                                epoch=epoch,
                                loss=total_loss * args.grad_accum_steps,
                                out_dir=args.output_dir
                            )

                step_idx += 1

            # Epoch Checkpoint
            if is_master:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                logger.info(f"Epoch {epoch + 1} training complete. Saving policy model...")
                checkpoint_out = {
                    "config": config,
                    "model": policy_model.module.state_dict() if use_ddp else policy_model.state_dict()
                }
                main_path = os.path.join(args.output_dir, f"checkpoint_grpo_epoch_{epoch+1}.pt")
                timestamped_path = os.path.join(args.output_dir, f"checkpoint_grpo_epoch_{epoch+1}_{timestamp}.pt")
                logger.info(f"💾 Saving GRPO checkpoint to {main_path} and {timestamped_path}")
                torch.save(checkpoint_out, main_path)
                torch.save(checkpoint_out, timestamped_path)

        if use_ddp:
            dist.destroy_process_group()
        if is_master:
            logger.info("Group Relative Policy Optimization (GRPO) training finished successfully!")

    finally:
        if is_master:
            tracker.finish()
if __name__ == "__main__":
    train()
