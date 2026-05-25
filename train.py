import os
import sys
import math
import time
import argparse
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from model import ModelConfig, Transformer
from data import SFTDataset, SequencePackingCollator
from utils.model_utils import configure_logging, count_parameters
from utils.profiler import estimate_step_flops, calculate_mfu

logger = logging.getLogger(__name__)

# ==============================================================================
# Learning Rate Scheduler with Cosine Decay
# ==============================================================================

def get_learning_rate(step: int, max_lr: float, min_lr: float, warmup_steps: int, decay_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step > decay_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ==============================================================================
# Optimizer Weight Decay Segregation
# ==============================================================================

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
    logger.info(f"Decaying weight parameters: {len(decay_params)} blocks with {num_decay_params:,} parameters")
    logger.info(f"Non-decaying weight parameters: {len(nodecay_params)} blocks with {num_nodecay_params:,} parameters")
    
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"Using fused AdamW optimizer: {use_fused}")
    
    return optimizer


# ==============================================================================
# Main Orchestrated SFT DDP Training Script
# ==============================================================================

def train():
    parser = argparse.ArgumentParser(description="nano-llm: High-Performance Distributed SFT Engine")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B", help="HF base model identifier")
    parser.add_argument("--data_path", type=str, required=True, help="Path to SFT train.jsonl file")
    parser.add_argument("--max_length", type=int, default=4096, help="Sequence context length")
    parser.add_argument("--epochs", type=int, default=3, help="Training epoch limit")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_lr", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=2e-6, help="Minimum decay learning rate floor")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR linear warmup step size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay multiplier")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping norm threshold")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output saving directory")
    parser.add_argument("--save_steps", type=int, default=200, help="Step intervals to save model weights")
    parser.add_argument("--seed", type=int, default=42, help="Random initialization seed")
    
    # PEFT custom LoRA parameters
    parser.add_argument("--use_lora", type=bool, default=False, help="Train only custom LoRA adapters from scratch")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank dimension size")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha scaling multiplier")
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
    
    logger.info(f"--- DDP Node initialized | Rank: {ddp_rank} | Local Rank: {ddp_local_rank} | World Size: {ddp_world_size} ---")
    
    torch.manual_seed(args.seed + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + ddp_rank)
        
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Tokenizer & Dataset Loader
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    dataset = SFTDataset(data_path=args.data_path, tokenizer=tokenizer, max_length=args.max_length)
    collator = SequencePackingCollator(pad_token_id=tokenizer.pad_token_id, max_length=args.max_length)

    sampler = DistributedSampler(dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True) if ddp else None
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        shuffle=(sampler is None),
        collate_fn=collator,
        pin_memory=True
    )

    # 3. Load Modern LLaMA Model with custom LoRA configurations
    logger.info("Assembling custom LLaMA Transformer architecture...")
    model_config = ModelConfig(
        block_size=args.max_length,
        vocab_size=len(tokenizer),
        n_layer=28,   
        n_head=16,
        n_embd=2048,
        lora_r=args.lora_r if args.use_lora else 0,
        lora_alpha=args.lora_alpha
    )
    
    model = Transformer(model_config).to(device)
    
    # If use_lora is active, freeze base layer parameters before creating optimizer
    if args.use_lora:
        logger.info(f"Toggling parameter freezing: FREEZING base layers, unfreezing custom LoRA (r={args.lora_r}) adapters...")
        trainable_params = model.configure_lora_trainable()
        logger.info(f"LoRA config unfreezing succeeded! Trainable parameters: {trainable_params:,}")
    
    param_report = count_parameters(model)
    logger.info(f"Model Parameters: {param_report['total_params']:,} total | {param_report['trainable_params']:,} trainable ({param_report['percentage_trainable']:.2f}% active)")

    # Wrap model in standard DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)

    # 4. Optimizer & Scaling Setup
    optimizer = configure_optimizers(
        model=model,
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        betas=(0.9, 0.95),
        device_type="cuda" if "cuda" in device else "cpu"
    )
    
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and not torch.cuda.is_bf16_supported())
    
    # Pre-estimate mathematical FLOPs per step for MFU profiling
    step_flops = estimate_step_flops(
        n_parameters=param_report["total_params"],
        batch_size=args.batch_size,
        seq_len=args.max_length,
        n_layer=model_config.n_layer,
        n_embd=model_config.n_embd,
        n_head=model_config.n_head
    )
    
    # 5. Training Epoch Loop
    total_steps = len(dataloader) * args.epochs
    decay_steps = total_steps
    step_counter = 0
    
    logger.info(f"🚀 Launching SFT DDP loop: total steps={total_steps} | micro-batch={args.batch_size} | accum steps={args.grad_accum_steps}")
    
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
            
        model.train()
        epoch_loss = 0.0
        
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                logits, loss = model(input_ids, targets=labels)
                loss = loss / args.grad_accum_steps
                
            epoch_loss += loss.item() * args.grad_accum_steps
            
            if ddp:
                model.require_backward_grad_sync = (batch_idx + 1) % args.grad_accum_steps == 0
                
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                lr = get_learning_rate(
                    step=step_counter,
                    max_lr=args.max_lr,
                    min_lr=args.min_lr,
                    warmup_steps=args.warmup_steps,
                    decay_steps=decay_steps
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                    
                if args.clip_grad > 0:
                    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                        
                if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad(set_to_none=True)
                step_counter += 1
                
                # Rank 0 Master logging with MFU (Model FLOPs Utilization) tracking!
                if is_master and step_counter % 10 == 0:
                    elapsed = time.time() - start_time
                    tokens_processed = batch["input_ids"].numel() * args.grad_accum_steps * ddp_world_size
                    throughput = tokens_processed / elapsed
                    
                    # Calculate MFU against peak hardware FLOPS
                    mfu = calculate_mfu(
                        step_flops=step_flops,
                        elapsed_time=elapsed,
                        grad_accum_steps=args.grad_accum_steps,
                        world_size=ddp_world_size
                    )
                    
                    logger.info(
                        f"Epoch {epoch+1} | Step {step_counter}/{total_steps//args.grad_accum_steps} | "
                        f"Loss: {loss.item()*args.grad_accum_steps:.4f} | LR: {lr:.2e} | "
                        f"Speed: {throughput:.0f} tok/s | MFU: {mfu:.1f}% | Time: {elapsed:.2f}s"
                    )
                    start_time = time.time()
                    
                if is_master and step_counter % args.save_steps == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_sft_step_{step_counter}.pt")
                    logger.info(f"Saving step checkpoint to {checkpoint_path}...")
                    
                    raw_model = model.module if ddp else model
                    torch.save({
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": model_config,
                        "step": step_counter
                    }, checkpoint_path)

        avg_epoch_loss = epoch_loss / len(dataloader)
        if is_master:
            logger.info(f"--- Epoch {epoch+1} completed! Average Loss: {avg_epoch_loss:.4f} ---")
            
    # Save final model state
    if is_master:
        final_path = os.path.join(args.output_dir, "checkpoint_sft.pt")
        logger.info(f"Saving final model checkpoint to {final_path}...")
        raw_model = model.module if ddp else model
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "config": model_config,
            "step": step_counter
        }, final_path)
        
    if ddp:
        dist.destroy_process_group()
    logger.info("nano-llm: SFT Succeeded!")

if __name__ == "__main__":
    train()
