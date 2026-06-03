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

from model import ModelConfig, Transformer, get_deepseek_config
from data import SFTDataset, SequencePackingCollator
from utils.model_utils import configure_logging, count_parameters
from utils.profiler import estimate_step_flops, calculate_mfu
from utils.ddp_helper import init_ddp
from utils.tokenizer_loader import load_tokenizer
from utils.checkpoint_utils import load_checkpoint_with_fp8_translation
from utils.training_utils import validate_dataset, assert_grad_accum_safe, get_cosine_lr, configure_optimizers

logger = logging.getLogger(__name__)


def construct_block_diagonal_mask(seqlens_list, max_length, device):
    """
    Constructs a 2D block-diagonal causal attention mask of shape [B, 1, max_length, max_length].
    seqlens_list is a list of lists, where each inner list contains segment lengths for that batch item.
    """
    batch_size = len(seqlens_list)
    mask = torch.full((batch_size, 1, max_length, max_length), float("-inf"), device=device)
    
    for b, seqlens in enumerate(seqlens_list):
        start_idx = 0
        for seq_len in seqlens:
            end_idx = start_idx + seq_len
            if seq_len > 0:
                block_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
                mask[b, 0, start_idx:end_idx, start_idx:end_idx] = block_mask
            start_idx = end_idx
            
    return mask


def construct_multimodal_mask(attention_mask, num_patches, device):
    """
    Constructs a 2D causal + padding mask of shape [B, 1, total_len, total_len] for VLM training.
    attention_mask: [B, max_seq_len] (1 for valid text token, 0 for padding)
    """
    batch_size, max_seq_len = attention_mask.shape
    total_len = num_patches + max_seq_len
    
    mask = torch.full((batch_size, 1, total_len, total_len), float("-inf"), device=device)
    causal_grid = torch.triu(torch.full((total_len, total_len), True, device=device), diagonal=1)
    
    for b in range(batch_size):
        valid_text_len = int(attention_mask[b].sum().item())
        valid_total_len = num_patches + valid_text_len
        allowed = ~causal_grid[:valid_total_len, :valid_total_len]
        mask[b, 0, :valid_total_len, :valid_total_len] = torch.where(allowed, 0.0, float("-inf"))
        
    return mask


@torch.no_grad()
def evaluate_val_loss(model, dataloader, device, ddp, max_length, use_multimodal=False):
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        mask = None
        pixel_values = batch.get("pixel_values", None)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device, non_blocking=True)
            
        if use_multimodal:
            num_patches = pixel_values.size(1) if pixel_values is not None else 0
            mask = construct_multimodal_mask(
                batch["attention_mask"].to(device),
                num_patches=num_patches,
                device=device
            )
        elif "seqlens" in batch:
            mask = construct_block_diagonal_mask(batch["seqlens"], max_length, device)
            
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            _, loss, _ = model(input_ids, pixel_values=pixel_values, targets=labels, mask=mask)
        total_loss += loss.item()
        count += 1
    
    if count == 0:
        return 0.0
    
    # Average across GPUs if DDP is active
    if ddp:
        loss_tensor = torch.tensor(total_loss / count, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()
    else:
        avg_loss = total_loss / count
    
    model.train()
    return avg_loss


# ==============================================================================
# Main Orchestrated SFT DDP Training Script
# ==============================================================================

def train():
    parser = argparse.ArgumentParser(description="nano-llm: High-Performance Distributed SFT Engine")
    parser.add_argument("--model_name_or_path", type=str, default="qwen/Qwen2.5-7B", help="ModelScope base model identifier")
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
    parser.add_argument("--use_lora", action="store_true", help="Train only custom LoRA adapters from scratch")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank dimension size")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha scaling multiplier")
    parser.add_argument("--use_multimodal", action="store_true", help="Enable VLM multimodal SFT data loader")
    args = parser.parse_args()

    # 1. Initialize PyTorch DDP
    dist_info = init_ddp()
    ddp = dist_info["ddp"]
    ddp_rank = dist_info["rank"]
    ddp_local_rank = dist_info["local_rank"]
    ddp_world_size = dist_info["world_size"]
    device = dist_info["device"]
    is_master = dist_info["is_master"]

    # Setup master logging
    log_level = logging.INFO if is_master else logging.WARNING
    configure_logging(level=log_level)

    from utils.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(
        project="nano-llm",
        config=vars(args),
        mode="offline" if is_master else "disabled",
        log_dir="./logs",
    )

    logger.info(f"--- DDP Node initialized | Rank: {ddp_rank} | Local Rank: {ddp_local_rank} | World Size: {ddp_world_size} ---")
    
    torch.manual_seed(args.seed + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + ddp_rank)
        
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Tokenizer & Model Configuration Loader
    logger.info("Initializing tokenizer...")
    tokenizer = load_tokenizer(fallback_model_name=args.model_name_or_path)

    logger.info("Assembling custom LLaMA Transformer architecture config...")
    base_checkpoint_path = args.model_name_or_path
    if not os.path.exists(base_checkpoint_path) and os.path.exists("./outputs/checkpoint_pretrain.pt"):
        base_checkpoint_path = "./outputs/checkpoint_pretrain.pt"
    
    model_config = None
    state_dict = None
    has_ckpt = False
    if os.path.exists(base_checkpoint_path):
        logger.info(f"Found base pre-trained checkpoint at {base_checkpoint_path}. Loading configuration and weights...")
        model_config, state_dict = load_checkpoint_with_fp8_translation(base_checkpoint_path, map_location=device)
        has_ckpt = True
    else:
        logger.warning("Base pre-trained checkpoint not found. Instantiating configuration from scratch...")
        model_config = get_deepseek_config(
            "tiny",
            block_size=args.max_length,
            vocab_size=len(tokenizer),
            lora_r=args.lora_r if args.use_lora else 0,
            lora_alpha=args.lora_alpha
        )

    # Initialize Dataset & Collator
    if args.use_multimodal:
        from data import MultimodalSFTDataset, MultimodalSequenceCollator
        vision_dim = getattr(model_config, "vision_dim", 1152) or 1152
        dataset = MultimodalSFTDataset(
            data_path=args.data_path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            vision_dim=vision_dim
        )
        collator = MultimodalSequenceCollator(
            pad_token_id=tokenizer.pad_token_id,
            vision_dim=vision_dim
        )
    else:
        dataset = SFTDataset(data_path=args.data_path, tokenizer=tokenizer, max_length=args.max_length)
        validate_dataset(dataset, tokenizer=tokenizer, name="SFT training dataset")
        collator = SequencePackingCollator(
            pad_token_id=tokenizer.pad_token_id,
            max_length=args.max_length,
            batch_size=args.batch_size
        )

    # Split dataset into train and validation sets (90% train, 10% val)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False) if ddp else None

    # Enlarge DataLoader micro-batch size by 6x to ensure collator gets enough samples to pack fully if not multimodal
    dataloader_batch_size = args.batch_size * 6 if not args.use_multimodal else args.batch_size
    val_dataloader_batch_size = args.batch_size * 6 if not args.use_multimodal else args.batch_size

    dataloader = DataLoader(
        train_dataset, 
        batch_size=dataloader_batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        collate_fn=collator,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=val_dataloader_batch_size, 
        sampler=val_sampler, 
        shuffle=False,
        collate_fn=collator,
        pin_memory=True
    )
    assert_grad_accum_safe(dataloader, args.grad_accum_steps, dataloader_batch_size)

    # 3. Load Modern LLaMA Model with custom LoRA configurations
    logger.info("Assembling custom LLaMA Transformer architecture...")
    restored_step = 0
    model_config.use_checkpoint = True  # Enable memory-saving activation checkpointing!
    model = Transformer(model_config).to(device)
    if has_ckpt and state_dict is not None:
        model.load_state_dict(state_dict)
    
    # If use_lora is active, freeze base layer parameters before creating optimizer
    if args.use_lora:
        logger.info(f"Toggling parameter freezing: FREEZING base layers, unfreezing custom LoRA (r={args.lora_r}) adapters...")
        trainable_params = model.configure_lora_trainable()
        logger.info(f"LoRA config unfreezing succeeded! Trainable parameters: {trainable_params:,}")
    
    param_report = count_parameters(model)
    logger.info(f"Model Parameters: {param_report['total_params']:,} total | {param_report['trainable_params']:,} trainable ({param_report['percentage_trainable']:.2f}% active)")

    # Wrap model in standard DDP with high-performance parameters
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank, gradient_as_bucket_view=True, static_graph=False)

    # Enable compile on policies to maximize forward/backward arithmetic intensity
    # Since model is now trace-safe and graph-break-free, this provides a massive speedup!
    logger.info("🔥 Enabling PyTorch Inductor compilation (torch.compile) for SFT...")
    model = torch.compile(model)

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
    step_counter = restored_step if has_ckpt else 0
    
    if is_master:
        tracker.log_config({
            "n_layer": model_config.n_layer, "n_head": model_config.n_head, "n_embd": model_config.n_embd,
            "total_params": param_report["total_params"], "trainable_params": param_report["trainable_params"],
            "batch_size": args.batch_size, "max_length": args.max_length,
            "grad_accum_steps": args.grad_accum_steps, "epochs": args.epochs,
            "max_lr": args.max_lr,
        })
    logger.info(f"🚀 Launching SFT DDP loop: total steps={total_steps} | micro-batch={args.batch_size} | accum steps={args.grad_accum_steps}")
    
    try:
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            epoch_loss = 0.0

            optimizer.zero_grad(set_to_none=True)
            start_time = time.time()

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                
                pixel_values = batch.get("pixel_values", None)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device, non_blocking=True)

                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                mask = None
                if args.use_multimodal:
                    num_patches = pixel_values.size(1) if pixel_values is not None else 0
                    mask = construct_multimodal_mask(
                        batch["attention_mask"].to(device),
                        num_patches=num_patches,
                        device=device
                    )
                elif "seqlens" in batch:
                    mask = construct_block_diagonal_mask(batch["seqlens"], args.max_length, device)

                with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                    logits, loss, _ = model(input_ids, pixel_values=pixel_values, targets=labels, mask=mask)
                    loss = loss / args.grad_accum_steps

                epoch_loss += loss.item() * args.grad_accum_steps

                if ddp:
                    model.require_backward_grad_sync = (batch_idx + 1) % args.grad_accum_steps == 0

                if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % args.grad_accum_steps == 0:
                    lr = get_cosine_lr(
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

                    # Periodic Validation Loop
                    if step_counter % 100 == 0:
                        val_loss = evaluate_val_loss(model, val_dataloader, device, ddp, args.max_length, use_multimodal=args.use_multimodal)
                        if is_master:
                            tracker.log({"sft/val_loss": val_loss}, step=step_counter)
                            logger.info(f"🏆 Step {step_counter} | Validation Loss: {val_loss:.4f}")

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

                        tracker.log({
                            "sft/loss": loss.item() * args.grad_accum_steps,
                            "sft/lr": lr,
                            "sft/throughput": throughput,
                            "sft/mfu": mfu,
                        }, step=step_counter)

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
            val_loss = evaluate_val_loss(model, val_dataloader, device, ddp, args.max_length, use_multimodal=args.use_multimodal)
            if is_master:
                logger.info(f"--- Epoch {epoch+1} completed! Average Loss: {avg_epoch_loss:.4f} | Validation Loss: {val_loss:.4f} ---")
                tracker.log({"sft/epoch_val_loss": val_loss}, step=step_counter)

        # Save final model state
        if is_master:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(args.output_dir, "checkpoint_sft.pt")
            timestamped_path = os.path.join(args.output_dir, f"checkpoint_sft_{timestamp}.pt")
            
            logger.info(f"Saving final model checkpoint to {final_path} and {timestamped_path}...")
            raw_model = model.module if ddp else model
            checkpoint_payload = {
                "model_state_dict": raw_model.state_dict(),
                "config": model_config,
                "step": step_counter
            }
            torch.save(checkpoint_payload, final_path)
            torch.save(checkpoint_payload, timestamped_path)

            # Automated stage evaluation benchmark hook
            import subprocess
            import sys
            try:
                logger.info("🔥 Starting automated evaluation benchmark hook...")
                subprocess.run([sys.executable, "eval_benchmarks.py", "--checkpoint_path", final_path], check=True)
                logger.info("✅ Automated evaluation benchmark hook finished.")
            except Exception as e:
                logger.error(f"Failed to run automated benchmark hook: {e}")

        if ddp:
            dist.destroy_process_group()
        logger.info("nano-llm: SFT Succeeded!")


    finally:
        if is_master:
            tracker.finish()
if __name__ == "__main__":
    train()
