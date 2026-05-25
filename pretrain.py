import os
import time
import math
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import ModelConfig, Transformer, convert_to_fp8
from utils.profiler import EstimateStepFlops

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Causal Language Pre-training Engine (Karpathy Style, Zero-Dependency DDP)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="nano-llm: High-Performance Multi-GPU FP8 Pre-training Engine")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--block_size", type=int, default=1024, help="Sequence context block length")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum number of pre-training optimization steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Number of linear learning rate warmup steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate under cosine decay")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="L2 weight decay penalty")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Global gradient norm clipping value")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--use_fp8", type=str, default="True", help="Enable Native float8 matrix multiplication mapping")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing packed train.bin and val.bin")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory to save pre-trained checkpoints")
    args = parser.parse_args()

    # 0. Initialize hardware telemetry monitor
    from utils.system_monitor import SystemMonitor
    monitor = SystemMonitor()

    # 1. Initialize Distributed Data Parallel (DDP) environment
    ddp = "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        # Fallback to single GPU or CPU
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed_offset = 0

    if master_process:
        os.makedirs(args.out_dir, exist_ok=True)
        logger.info("Initializing nano-llm pre-training environment...")
        logger.info(f"DDP Status: {ddp} (World Size: {ddp_world_size})")

    # Set random seeds for deterministic reproducibility
    torch.manual_seed(1337 + seed_offset)
    np.random.seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 2. Memory-map train.bin and val.bin packed token streams
    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")

    if not os.path.exists(train_bin):
        # Local mock corpus for dry-runs if no bin dataset is packed yet
        if master_process:
            logger.warning(f"Packed binary array not found at '{train_bin}'. Generating high-quality dummy data...")
        os.makedirs(args.data_dir, exist_ok=True)
        # Create a mock 100,000 token array of mock vocab IDs
        mock_tokens = np.random.randint(0, 1000, size=(100000,), dtype=np.uint16)
        mock_tokens.tofile(train_bin)
        mock_tokens[:10000].tofile(val_bin)

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        # Select random starting token indexes
        ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + args.block_size]).astype(np.int64)) for i in ix])
        
        # Pin memory for high-throughput GPU transfer
        if "cuda" in device:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # 3. Build model Config & Model
    config = ModelConfig(
        block_size=args.block_size,
        vocab_size=1200, # vocab size of custom trained tokenizer
        n_layer=4,       # Light config for fast startup & pre-training stability
        n_head=8,
        n_embd=512,
        vision_dim=None  # Pure Causal Text Pre-training
    )
    
    model = Transformer(config)
    
    # 4. Native FP8 Conversion Optimization
    use_fp8_bool = args.use_fp8.lower() == "true"
    if use_fp8_bool:
        if master_process:
            logger.info("⚡ Native FP8 Mixed-Precision scaling active! Converting linear layers...")
        convert_to_fp8(model)

    model.to(device)

    # Log trainable parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if master_process:
        logger.info(f"Model Architecture: LLaMA Causal Transformer | Parameters: {total_params:,}")

    # Set up optimizer ( nanoGPT-style weight decay grouping )
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # 5. Shard model through DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # 6. Cosine learning rate scheduling calculator
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / (args.warmup_steps + 1)
        if step > args.max_steps:
            return args.min_lr
        # Cosine decay factor calculation
        decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.lr - args.min_lr)

    # 7. Pre-training Optimization Loop
    model.train()
    step = 0
    t0 = time.time()

    # Pre-calculate steps flops for real-time MFU reporting
    raw_model = model.module if ddp else model
    step_flops = EstimateStepFlops(
        batch_size=args.batch_size * ddp_world_size * args.grad_accum_steps,
        seq_len=args.block_size,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        vocab_size=config.vocab_size
    )

    if master_process:
        logger.info("🚀 Launching optimization steps. Streaming real-time telemetry metrics...")

    while step < args.max_steps:
        # Determine learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Batch accumulation loop
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(args.grad_accum_steps):
            x, y = get_batch("train")
            
            # Disable DDP gradient sync on intermediate micro-steps
            if ddp:
                model.require_backward_grad_sync = (micro_step == args.grad_accum_steps - 1)
                
            # Forward pass under native bfloat16 AMP
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, loss = model(x, targets=y)
                loss = loss / args.grad_accum_steps
                
            loss_accum += loss.detach().item()
            
            # Backward pass
            loss.backward()

        # Clip global gradient norm
        if args.grad_clip > 0.0:
            if ddp:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimization step
        optimizer.step()
        
        # Calculate time & hardware saturation metrics (MFU)
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        # Convert DT step time to MFU using H800 peak theoretical FLOPs (312 TFLOPS bfloat16)
        flops_per_sec = step_flops / (dt + 1e-8)
        mfu_percentage = (flops_per_sec / (312e12)) * 100.0
        mfu_percentage = min(100.0, max(0.0, mfu_percentage))

        if master_process:
            telemetry_str = monitor.get_formatted_telemetry()
            logger.info(
                f"Step {step+1}/{args.max_steps} | "
                f"Loss: {loss_accum:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {dt*1000:.1f}ms | "
                f"MFU: {mfu_percentage:.1f}% | "
                f"{telemetry_str}"
            )
            
            # Print detailed ASCII telemetry cockpit every 50 steps
            if (step + 1) % 50 == 0:
                monitor.print_dashboard()
            
            # Write structured JSON to stdout so FastAPI server captures step metrics
            print(f"METRICS_JSON: {{\"step\": {step+1}, \"loss\": {loss_accum:.4f}, \"lr\": {lr:.2e}, \"mfu\": {mfu_percentage:.1f}}}", flush=True)
            
            # Save system telemetry to outputs directory
            import json
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/system_telemetry.json", "w") as f:
                json.dump(monitor.get_telemetry_report(), f, indent=2)

        step += 1

    # 8. Save final pre-trained state dictionary
    if master_process:
        checkpoint_path = os.path.join(args.out_dir, "checkpoint_pretrain.pt")
        logger.info(f"💾 Saving final pre-trained checkpoint to: {checkpoint_path}")
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "config": config,
                "step": step
            },
            checkpoint_path
        )
        logger.info("✅ Pre-training stage successfully completed!")

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
