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
from transformers import AutoTokenizer

from model import ModelConfig, Transformer
from utils.model_utils import configure_logging, count_parameters
from utils.sandbox_executor import SandboxCodeExecutor

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Multimodal Dynamic Dataset for GRPO (Loads reasoning prompts & visual patches)
# ==============================================================================

class GRPODataset(Dataset):
    """
    Dataset of prompts, optional images, and expected answers/ground truths for GRPO training.
    Format of input file: JSONL containing {"prompt": "...", "image": "...", "ground_truth": "..."}
    """
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_prompt_length: int = 1024, vision_dim: int = 1152, num_patches: int = 16):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.vision_dim = vision_dim
        self.num_patches = num_patches
        
        self.prompts = []
        self.images = []
        self.ground_truths = []
        
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        self.prompts.append(data["prompt"])
                        self.images.append(data.get("image", None))
                        self.ground_truths.append(data.get("ground_truth", ""))
                    except Exception as e:
                        logger.warning(f"Error parsing line: {e}")
        else:
            # Fallback mock dataset for demonstration and local testing
            logger.warning(f"Data path {data_path} not found. Loading mock mathematical reasoning dataset.")
            mock_prompts = [
                ("Solve: 12 + 15 * 2. Please reason step-by-step and wrap your final number in <answer></answer> tags.", None, "42"),
                ("What is the square of 9? Please think step-by-step and wrap your final number in <answer></answer> tags.", None, "81"),
                ("Compute: (25 - 5) / 4. Please show your thinking process and wrap your final answer in <answer></answer>.", None, "5"),
                ("What is 100 divided by 4? Reason and output final number inside <answer></answer> tags.", None, "25")
            ]
            for p, img, gt in mock_prompts:
                self.prompts.append(p)
                self.images.append(img)
                self.ground_truths.append(gt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        image_path = self.images[idx]
        ground_truth = self.ground_truths[idx]
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Process image features if image exists (simulate or extract)
        if image_path is not None:
            pixel_values = torch.randn(self.num_patches, self.vision_dim)
        else:
            pixel_values = None
            
        return {
            "prompt_text": prompt,
            "prompt_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": pixel_values,
            "ground_truth": ground_truth
        }

def grpo_collate_fn(batch):
    """
    Collate function to pad variable-length prompt token sequences and stack vision patches.
    """
    prompt_texts = [item["prompt_text"] for item in batch]
    prompt_ids_list = [item["prompt_ids"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]
    
    # Pad input IDs
    padded_prompt_ids = nn.utils.rnn.pad_sequence(
        prompt_ids_list,
        batch_first=True,
        padding_value=0  # Assumes 0 is pad_token_id
    )
    
    # Attention mask
    attention_masks = (padded_prompt_ids != 0).long()
    
    # Batch pixel values
    pixel_values_list = []
    has_images = False
    for item in batch:
        p_val = item["pixel_values"]
        if p_val is not None:
            has_images = True
            pixel_values_list.append(p_val)
        else:
            # Create a zero placeholder to maintain batch dimension shape
            pixel_values_list.append(torch.zeros(16, 1152))
            
    batch_pixel_values = torch.stack(pixel_values_list) if has_images else None
    
    return {
        "prompt_texts": prompt_texts,
        "prompt_ids": padded_prompt_ids,
        "attention_masks": attention_masks,
        "pixel_values": batch_pixel_values,
        "ground_truths": ground_truths
    }

# ==============================================================================
# 2. Rule-Based Reward Evaluators (DeepSeek-R1-Zero Style)
# ==============================================================================

def extract_answer(completion_text: str) -> str:
    """
    Extracts content inside <answer>...</answer> tags.
    """
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, completion_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def evaluate_completion_rewards(
    prompts: list[str], 
    completions: list[str], 
    ground_truths: list[str]
) -> torch.Tensor:
    """
    Calculates rewards for a group of completions.
    1. Format Reward: checks if output contains <think>...</think> and <answer>...</answer>
    2. Math Reward: checks if the extracted answer matches the ground truth
    3. Code Sandbox Reward: extracts and executes generated python code in a sandbox,
       adding rewards for successful compilation and correct logic, or penalizing errors.
    4. Length Regularization: penalizes infinite repetitive generations
    """
    rewards = []
    sandbox = SandboxCodeExecutor(timeout=2.0)
    
    for prompt, completion, gt in zip(prompts, completions, ground_truths):
        reward = 0.0
        
        # A. Format Reward: Reward correct thinking and answer tags
        has_think = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        
        if has_think and has_answer:
            reward += 1.0  # Proper reasoning structure
        elif has_think or has_answer:
            reward += 0.3  # Partial formatting
            
        # B. Math Accuracy Reward: Check if answer matches ground truth
        extracted = extract_answer(completion)
        if gt and extracted:
            if extracted == gt:
                reward += 2.0  # Correct answer match
            elif extracted in gt or gt in extracted:
                reward += 0.5  # Substring partial match
                
        # C. Code Sandbox Reward: If completion contains code, execute and verify it
        extracted_code = sandbox.extract_code(completion)
        if extracted_code:
            # Execute in sandbox
            res = sandbox.execute_and_verify(extracted_code)
            if res["success"]:
                reward += 1.0  # Code successfully compiled and exited 0!
                
                # Check if code output matches ground truth (for coding questions)
                stdout = res["stdout"].strip()
                if gt and stdout == gt.strip():
                    reward += 1.5  # Code executed and matched the ground truth answer!
                elif gt and (gt.strip() in stdout or stdout in gt.strip()):
                    reward += 0.5  # Substring match on execution output
            else:
                # Syntax error, runtime error, security exception, or timeout
                reward -= 0.5  # Code failed execution or violated security bounds
                
        # D. Repetitive/Length Penalty: Penalize excessively verbose or empty generations
        comp_len = len(completion)
        if comp_len > 1500 or comp_len < 10:
            reward -= 0.5
            
        rewards.append(reward)
        
    return torch.tensor(rewards, dtype=torch.float32)


# ==============================================================================
# 2.5. High-End Alignment Controllers (Reward Scaling & Adaptive KL)
# ==============================================================================

import math

class GRPORewardScaler:
    """
    Tracks running statistics of rewards in a GRPO training loop
    and normalizes them to mean 0, variance 1.
    """
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update_and_scale(self, rewards: torch.Tensor) -> torch.Tensor:
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item()
        if torch.isnan(torch.tensor(batch_var)) or batch_var == 0:
            batch_var = 1.0
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1.0 - self.momentum) * batch_var
            
        self.count += 1
        
        # Z-score normalization
        normalized = (rewards - self.mean) / (math.sqrt(self.var) + 1e-8)
        return normalized


class AdaptiveKLTuner:
    """
    Dynamically scales the KL penalty coefficient beta based on target KL divergence bounds.
    """
    def __init__(self, target_kl: float = 0.2, beta_min: float = 0.005, beta_max: float = 0.5):
        self.target_kl = target_kl
        self.beta_min = beta_min
        self.beta_max = beta_max

    def tune_beta(self, current_kl: float, current_beta: float) -> float:
        if current_kl > self.target_kl * 1.5:
            # Policy is diverging too much from reference model -> increase beta
            new_beta = min(self.beta_max, current_beta * 1.2)
        elif current_kl < self.target_kl * 0.5:
            # Policy is too close -> decrease beta to encourage exploration
            new_beta = max(self.beta_min, current_beta / 1.2)
        else:
            new_beta = current_beta
        return new_beta


# ==============================================================================
# 3. Autoregressive Generation Helper with Temperature & Top-p
# ==============================================================================

@torch.no_grad()
def generate_completions(
    model: Transformer, 
    prompt_ids: torch.Tensor, 
    pixel_values: torch.Tensor = None,
    max_gen_len: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates tokens autoregressively from prompts and returns the full 
    token sequences (prompt + generation) alongside generation masks.
    """
    model.eval()
    batch_size, prompt_len = prompt_ids.shape
    device = prompt_ids.device
    
    # Pre-allocate full sequence tensor
    full_seqs = torch.zeros(batch_size, prompt_len + max_gen_len, dtype=torch.long, device=device)
    full_seqs[:, :prompt_len] = prompt_ids
    
    # Generation mask: 0 for prompt tokens, 1 for generated tokens
    gen_mask = torch.zeros_like(full_seqs, dtype=torch.float32)
    gen_mask[:, prompt_len:] = 1.0
    
    # Autoregressive generation loop
    for i in range(max_gen_len):
        curr_pos = prompt_len + i
        # Slice active window
        logits = model(full_seqs[:, :curr_pos], pixel_values=pixel_values)
        # Next-token logits
        next_logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature > 0.0:
            next_logits = next_logits / temperature
            
            # Top-p (nucleus) filtering
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter remove mask back
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = -float('inf')
            
            # Sample from filtered distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy search
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            
        full_seqs[:, curr_pos] = next_token.squeeze(-1)
        
    return full_seqs, gen_mask

# ==============================================================================
# 4. Pure PyTorch Log-Probability Math for GRPO
# ==============================================================================

def compute_action_logprobs(logits: torch.Tensor, seqs: torch.Tensor, gen_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes standard log-probabilities of generated tokens (actions) under a policy.
    Does NOT calculate gradients on prompt tokens (where gen_mask == 0).
    """
    # Shift logits right by 1 and targets left by 1 for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = seqs[:, 1:].contiguous()
    shift_mask = gen_mask[:, 1:].contiguous()
    
    # Calculate negative cross-entropy per position
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    nll = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_targets.view(-1)
    )
    nll = nll.view(shift_targets.size())  # Reshape back to [batch_size, seq_len - 1]
    
    # Nullify NLL values outside the generation mask (prompt tokens / padded tokens)
    masked_nll = nll * shift_mask
    
    # Logprob is the sum of negative NLL across generation steps
    log_probs = -masked_nll.sum(dim=-1)
    return log_probs

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
    args = parser.parse_args()

    # DDP Distributed environment
    ddp_rank = int(os.environ.get("RANK", -1))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = ddp_rank != -1
    
    if use_ddp:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
        is_master = ddp_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True

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
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Fallback BPE tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model checkpoint
    if is_master:
        logger.info(f"Loading base SFT model from checkpoint: {args.sft_checkpoint_path}")
    checkpoint = torch.load(args.sft_checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    # Instantiate models
    # Policy model (Trainable)
    policy_model = Transformer(config).to(device)
    model_state = checkpoint.get("model_state_dict", checkpoint.get("model", None))
    if model_state is None:
        raise KeyError("Could not locate model state dictionary in SFT checkpoint!")
    policy_model.load_state_dict(model_state)
    
    # Reference model (Frozen baseline anchor)
    ref_model = Transformer(config).to(device)
    ref_model.load_state_dict(model_state)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    if use_ddp:
        policy_model = DDP(policy_model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
        
    # Optimizer Fused AdamW
    optim_groups = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.max_lr, weight_decay=0.01, fused=True)

    # Loader setup
    dataset = GRPODataset(args.data_path, tokenizer, max_prompt_length=args.max_prompt_len)
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        collate_fn=grpo_collate_fn, 
        shuffle=(sampler is None)
    )

    # High-end alignment controllers & mutation engine initialization
    reward_scaler = GRPORewardScaler()
    kl_tuner = AdaptiveKLTuner(target_kl=0.2)
    
    from utils.evol_instruct import EvolInstructEngine
    import random
    evol_engine = EvolInstructEngine()
    current_beta = args.beta

    # Training epochs loop
    step_idx = 0
    for epoch in range(args.epochs):
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
                    policy_model.module if use_ddp else policy_model, 
                    G_prompt_ids,
                    pixel_values=p_val,
                    max_gen_len=args.max_gen_len,
                    temperature=0.8,
                    top_p=0.9
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
                policy_logits = policy_model(full_seqs, pixel_values=p_val)
                policy_logprobs = compute_action_logprobs(policy_logits, full_seqs, gen_mask)
                
                with torch.no_grad():
                    ref_logits = ref_model(full_seqs, pixel_values=p_val)
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
            
            step_idx += 1
            
        # Epoch Checkpoint
        if is_master:
            logger.info(f"Epoch {epoch + 1} training complete. Saving policy model...")
            checkpoint_out = {
                "config": config,
                "model": policy_model.module.state_dict() if use_ddp else policy_model.state_dict()
            }
            torch.save(checkpoint_out, os.path.join(args.output_dir, f"checkpoint_grpo_epoch_{epoch+1}.pt"))

    if use_ddp:
        dist.destroy_process_group()
    if is_master:
        logger.info("Group Relative Policy Optimization (GRPO) training finished successfully!")

if __name__ == "__main__":
    train()
