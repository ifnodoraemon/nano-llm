import os
# Inject high-performance environment variables before PyTorch initializes
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["TORCHINDUCTOR_AUTOTUNE_MAX"] = "1"

import time
import math
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import ModelConfig, Transformer, convert_to_fp8
from utils.profiler import estimate_step_flops, calculate_mfu
from config import load_config, config_to_model_config
from utils.training_utils import get_cosine_lr, configure_optimizers

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
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--ep_size", type=int, default=1, help="Expert Parallel size")
    parser.add_argument("--use_triton_mla", type=str, default="False", help="Use Triton MLA kernel")
    parser.add_argument("--use_triton", type=str, default="False", help="Use Triton RMSNorm and SwiGLU kernels")
    parser.add_argument("--use_checkpoint", type=str, default="False", help="Enable activation checkpointing to save VRAM")
    parser.add_argument("--use_compile", type=str, default="False", help="Use torch.compile to optimize runtime performance")
    parser.add_argument("--data_mix", type=str, default=None,
                        choices=["balanced", "code_heavy", "zh_focused", "english_only"],
                        help="Data mixing preset for multi-domain pretraining")
    parser.add_argument("--model_size", type=str, default=None,
                        choices=["tiny", "1.5B-dense", "2B-dense", "2.7B-dense"],
                        help="Model preset size configuration")
    args = parser.parse_args()

    # 0. Initialize hardware telemetry monitor
    from utils.system_monitor import SystemMonitor
    monitor = SystemMonitor()

    # Dist environment auto-tuning
    from utils.dist_helper import autotune_nccl
    autotune_nccl()

    from utils.checkpoint_saver import BackgroundCheckpointSaver, ElasticRestoreManager
    saver = BackgroundCheckpointSaver()
    restore_mgr = ElasticRestoreManager(args.out_dir)

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
        
        # Initialize 3D Parallel groups
        from utils.tensor_parallel import init_tp_process_group
        from utils.pipeline_parallel import init_pp_process_group
        from utils.expert_parallel import init_ep_process_group
        
        init_tp_process_group(args.tp_size)
        init_pp_process_group(args.pp_size)
        init_ep_process_group(args.ep_size)
        
        # Initialize DP Process Groups
        dp_size = ddp_world_size // (args.tp_size * args.pp_size)
        dp_group = None
        if dp_size > 1:
            for tp in range(args.tp_size):
                for pp in range(args.pp_size):
                    dp_ranks = [dp * (args.pp_size * args.tp_size) + pp * args.tp_size + tp for dp in range(dp_size)]
                    group = dist.new_group(dp_ranks)
                    if ddp_rank in dp_ranks:
                        dp_group = group
    else:
        # Fallback to single GPU or CPU
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed_offset = 0

    from utils.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(
        project="nano-llm",
        config=vars(args),
        mode="offline" if master_process else "disabled",
        log_dir="./logs",
    )

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

    # Guard assert: ensure dataset has enough tokens for gradient accumulation steps
    tokens_needed = args.batch_size * args.block_size * args.grad_accum_steps * ddp_world_size
    if len(train_data) < tokens_needed:
        raise AssertionError(
            f"Pretraining dataset {train_bin} has {len(train_data):,} tokens, which is smaller "
            f"than the {tokens_needed:,} tokens required for a single gradient accumulation step."
        )

    def get_batch(split, batch_size=None):
        data = train_data if split == "train" else val_data
        bs = batch_size if batch_size is not None else args.batch_size
        
        # Vectorized generation of starting offsets in NumPy for speed
        ix = np.random.randint(0, len(data) - args.block_size - 1, size=(bs,))
        
        # Batch slice fast numpy array stack and convert to int32 (saving 50% PCIe host bandwidth vs int64)
        x_np = np.stack([data[i : i + args.block_size] for i in ix]).astype(np.int32)
        y_np = np.stack([data[i + 1 : i + 1 + args.block_size] for i in ix]).astype(np.int32)
        
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        # Pin memory for low latency GPU transfers, casting to long on GPU to satisfy cross_entropy requirements
        if "cuda" in device:
            x, y = x.pin_memory().to(device, non_blocking=True).long(), y.pin_memory().to(device, non_blocking=True).long()
        else:
            x, y = x.to(device).long(), y.to(device).long()
        return x, y

    # 2b. Optional multi-domain data mixing via DynamicDataMixer
    data_mixer_iter = None
    if args.data_mix is not None:
        manifest_path = os.path.join(args.data_dir, "data_manifest.json")
        if not os.path.exists(manifest_path):
            if master_process:
                logger.warning(f"Data manifest not found at {manifest_path} — falling back to single-domain data")
        else:
            import json as _json
            with open(manifest_path, "r") as f:
                manifest = _json.load(f)
            preset = manifest.get("presets", {}).get(args.data_mix)
            if preset is None:
                if master_process:
                    logger.warning(f"Data mix preset '{args.data_mix}' not found in manifest — falling back to default")
            else:
                if master_process:
                    logger.info(f"Data mix preset '{args.data_mix}': {preset['description']}")
                    logger.info(f"Sources: {preset['sources']}")

                from utils.tokenizer_loader import load_tokenizer
                tokenizer = load_tokenizer(fallback_model_name="gpt2")

                from utils.data_mixer import DynamicDataMixer
                data_mixer = DynamicDataMixer(
                    sources=preset["sources"],
                    tokenizer=tokenizer,
                    block_size=args.block_size,
                    buffer_size=5000,
                    seed=1337 + seed_offset,
                )
                data_mixer_iter = iter(data_mixer)

                def get_batch(split, batch_size=None):
                    """Get a batch from the multi-domain data mixer (overrides memmap version)."""
                    bs = batch_size if batch_size is not None else args.batch_size
                    xs, ys = [], []
                    for _ in range(bs):
                        sample = next(data_mixer_iter)
                        xs.append(sample["input_ids"])
                        ys.append(sample["labels"])
                    x = torch.stack(xs)
                    y = torch.stack(ys)
                    if "cuda" in device:
                        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
                    else:
                        x, y = x.to(device), y.to(device)
                    return x, y

                if master_process:
                    logger.info(f"Multi-domain data mixer active — preset: {args.data_mix}")

    cfg = load_config(mode="prod" if args.max_steps > 500 else "dev", model_size=args.model_size, num_gpus=ddp_world_size)
    config = config_to_model_config(
        cfg,
        block_size=args.block_size,
        vision_dim=None,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        ep_size=args.ep_size,
        use_triton_mla=args.use_triton_mla.lower() == "true",
        use_triton=args.use_triton.lower() == "true",
        use_checkpoint=args.use_checkpoint.lower() == "true",
    )
    model = Transformer(config)
    
    # 4. Native FP8 Conversion Optimization
    use_fp8_bool = args.use_fp8.lower() == "true"
    if use_fp8_bool:
        if master_process:
            logger.info("⚡ Native FP8 Mixed-Precision scaling active! Converting linear layers...")
        convert_to_fp8(model)

    model.to(device)

    if args.use_compile.lower() == "true":
        if master_process:
            logger.info("🔥 Enabling PyTorch Inductor compilation (torch.compile) with default mode...")
        model = torch.compile(model)

    # Log trainable parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if master_process:
        logger.info(f"Model Architecture: LLaMA Causal Transformer | Parameters: {total_params:,}")

    # Set up optimizer ( nanoGPT-style weight decay grouping )
    optimizer = configure_optimizers(
        model=model,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        device_type="cuda" if "cuda" in device else "cpu"
    )

    # 5. Shard model through DDP or Pipeline Parallel
    from utils.pipeline_parallel import PipelineStage, OneFOneBScheduler, get_pp_size, get_pp_rank
    pp_size = get_pp_size()
    pp_rank = get_pp_rank()
    
    if pp_size > 1:
        # Slice layers for current PP stage
        num_layers = len(model.layers)
        layers_per_stage = num_layers // pp_size
        start_idx = pp_rank * layers_per_stage
        end_idx = start_idx + layers_per_stage if pp_rank < pp_size - 1 else num_layers
        
        local_layers = nn.ModuleList([model.layers[i] for i in range(start_idx, end_idx)])
        tok_embeddings = model.tok_embeddings if pp_rank == 0 else None
        head_wrapper = nn.Sequential(model.norm, model.output) if pp_rank == pp_size - 1 else None
        
        stage = PipelineStage(local_layers, embedding=tok_embeddings, head=head_wrapper, freqs_cis=model.freqs_cis)
        scheduler = OneFOneBScheduler(stage, num_microbatches=args.grad_accum_steps, d_model=config.n_embd)
        
        if ddp and dp_size > 1:
            stage = DDP(stage, device_ids=[ddp_local_rank], process_group=dp_group, gradient_as_bucket_view=True, static_graph=False)
        model_to_opt = stage
    else:
        stage = None
        scheduler = None
        if ddp and dp_size > 1:
            model = DDP(model, device_ids=[ddp_local_rank], process_group=dp_group, gradient_as_bucket_view=True, static_graph=False)
        model_to_opt = model

    # Auto-detect checkpoint to self-heal and hot-restore on failure
    has_ckpt, restored_step, restored_epoch = restore_mgr.auto_detect_checkpoint()
    if has_ckpt:
        restored_step, _, _ = restore_mgr.restore_training_state(model, optimizer)

    @torch.no_grad()
    def evaluate_val_loss():
        model.eval()
        losses = []
        for _ in range(30):
            x_val, y_val = get_batch("val")
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x_val, targets=y_val)
            losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)
        if ddp:
            loss_tensor = torch.tensor(mean_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            mean_loss = (loss_tensor / ddp_world_size).item()
        model.train()
        return mean_loss

    # 7. Pre-training Optimization Loop
    model.train()
    step = 0
    if has_ckpt:
        step = restored_step
    t0 = time.time()

    # Pre-calculate steps flops for real-time MFU reporting
    raw_model = model_to_opt.module if hasattr(model_to_opt, "module") else model_to_opt
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    step_flops = estimate_step_flops(
        n_parameters=n_params,
        batch_size=args.batch_size,
        seq_len=args.block_size,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        n_head=config.n_head,
        use_activation_checkpointing=(args.use_checkpoint.lower() == "true")
    )

    if master_process:
        logger.info("🚀 Launching optimization steps. Streaming real-time telemetry metrics...")
        tracker.log_config({
            "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd,
            "total_params": n_params, "batch_size": args.batch_size,
            "block_size": args.block_size, "max_steps": args.max_steps,
        })

    class DataPrefetcher:
        def __init__(self, get_batch_fn, split, grad_accum_steps):
            self.get_batch_fn = get_batch_fn
            self.split = split
            self.grad_accum_steps = grad_accum_steps
            self.stream = torch.cuda.Stream()
            self.next_batches = []
            self.preload()

        def preload(self):
            with torch.cuda.stream(self.stream):
                self.next_batches = [self.get_batch_fn(self.split) for _ in range(self.grad_accum_steps)]

        def next(self):
            torch.cuda.current_stream().wait_stream(self.stream)
            batches = self.next_batches
            self.preload()
            return batches

    prefetcher = DataPrefetcher(get_batch, "train", args.grad_accum_steps)

    try:
        while step < args.max_steps:
            # Determine learning rate
            lr = get_cosine_lr(step=step, max_lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, decay_steps=args.max_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Batch accumulation loop
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0

            if pp_size > 1:
                # Slice input batch into micro-batches for 1F1B
                x, y = get_batch("train", batch_size=args.batch_size * args.grad_accum_steps)
                micro_batches_x = list(x.chunk(args.grad_accum_steps, dim=0))
                micro_batches_y = list(y.chunk(args.grad_accum_steps, dim=0))

                def loss_fn(pred_logits, targets_mb):
                    return F.cross_entropy(pred_logits.view(-1, pred_logits.size(-1)), targets_mb.view(-1), ignore_index=-100)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses = scheduler.run_1f1b(
                        micro_batches=micro_batches_x,
                        targets=micro_batches_y,
                        loss_fn=loss_fn,
                        device=torch.device(device)
                    )

                loss_accum = sum(l.detach().item() for l in losses) / args.grad_accum_steps if losses else 0.0
            else:
                batches = prefetcher.next()
                if ddp and args.grad_accum_steps > 1:
                    # Pre-N-1 steps: use model.no_sync() to strictly avoid any DDP communications hooks
                    for micro_step in range(args.grad_accum_steps - 1):
                        x, y = batches[micro_step]
                        with model.no_sync():
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                logits, loss, _ = model(x, targets=y)
                                loss = loss / args.grad_accum_steps
                            loss_accum += loss.detach().item()
                            loss.backward()
                    
                    # Final step: run outside no_sync to trigger All-Reduce gradient synchronization
                    x, y = batches[-1]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, loss, _ = model(x, targets=y)
                        loss = loss / args.grad_accum_steps
                    loss_accum += loss.detach().item()
                    loss.backward()
                else:
                    for micro_step in range(args.grad_accum_steps):
                        x, y = batches[micro_step]
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits, loss, _ = model(x, targets=y)
                            loss = loss / args.grad_accum_steps
                        loss_accum += loss.detach().item()
                        loss.backward()

            # Clip global gradient norm
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model_to_opt.parameters(), args.grad_clip)

            # Optimization step
            optimizer.step()

            # Calculate time & hardware saturation metrics (MFU)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Convert DT step time to MFU using H800 peak theoretical FLOPs (312 TFLOPS bfloat16)
            mfu_percentage = calculate_mfu(
                step_flops=step_flops,
                elapsed_time=dt,
                grad_accum_steps=args.grad_accum_steps,
                world_size=ddp_world_size,
                peak_flops=312e12
            )
            mfu_percentage = min(100.0, max(0.0, mfu_percentage))

            # Run validation loss evaluation on all ranks if we hit the interval
            val_loss = None
            if (step + 1) % 500 == 0:
                val_loss = evaluate_val_loss()

            if master_process:
                log_data = {
                    "pretrain/loss": loss_accum,
                    "pretrain/lr": lr,
                    "pretrain/mfu": mfu_percentage,
                    "pretrain/step_time_ms": dt * 1000,
                }
                if val_loss is not None:
                    log_data["pretrain/val_loss"] = val_loss
                    logger.info(f"📊 Validation Step {step+1} | Val Loss: {val_loss:.4f}")
                tracker.log(log_data, step=step)
                telemetry_str = monitor.get_formatted_telemetry()
                logger.info(
                    f"Step {step+1}/{args.max_steps} | "
                    f"Loss: {loss_accum:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {dt*1000:.1f}ms | "
                    f"MFU: {mfu_percentage:.1f}% | "
                    f"{telemetry_str}"
                )

                # Print detailed ASCII telemetry cockpit every 500 steps
                if (step + 1) % 500 == 0:
                    monitor.print_dashboard()

                # Write structured JSON to stdout so FastAPI server captures step metrics
                print(f"METRICS_JSON: {{\"step\": {step+1}, \"loss\": {loss_accum:.4f}, \"lr\": {lr:.2e}, \"mfu\": {mfu_percentage:.1f}}}", flush=True)

                # Save system telemetry to outputs directory
                import json
                os.makedirs("outputs", exist_ok=True)
                with open("outputs/system_telemetry.json", "w") as f:
                    json.dump(monitor.get_telemetry_report(), f, indent=2)

                # Non-blocking asynchronous checkpoint and manifest update every 500 steps
                if (step + 1) % 500 == 0:
                    saver.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=None,
                        config=config,
                        step=step + 1,
                        epoch=0,
                        loss=loss_accum,
                        out_dir=args.out_dir
                    )

            step += 1

        # 8. Save final pre-trained state dictionary
        if master_process:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(args.out_dir, "checkpoint_pretrain.pt")
            timestamped_path = os.path.join(args.out_dir, f"checkpoint_pretrain_{timestamp}.pt")
            
            logger.info(f"💾 Saving final pre-trained checkpoint to: {checkpoint_path} and {timestamped_path}")
            checkpoint_payload = {
                "model_state_dict": raw_model.state_dict(),
                "config": config,
                "step": step
            }
            torch.save(checkpoint_payload, checkpoint_path)
            torch.save(checkpoint_payload, timestamped_path)
            logger.info("✅ Pre-training stage successfully completed!")

            # Automated stage evaluation benchmark hook
            import subprocess
            import sys
            try:
                logger.info("🔥 Starting automated evaluation benchmark hook...")
                subprocess.run([sys.executable, "eval_benchmarks.py", "--checkpoint_path", checkpoint_path], check=True)
                logger.info("✅ Automated evaluation benchmark hook finished.")
            except Exception as e:
                logger.error(f"Failed to run automated benchmark hook: {e}")


    finally:
        if master_process:
            tracker.finish()
        if ddp:
            try:
                dist.destroy_process_group()
            except AssertionError:
                pass
if __name__ == "__main__":
    main()
