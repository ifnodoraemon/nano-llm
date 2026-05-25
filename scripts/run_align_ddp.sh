#!/bin/bash

# ==============================================================================
# nano-llm: Distributed DPO Alignment Launcher (8xH800 DDP)
# ==============================================================================

# Highly optimized NCCL variables to fully utilize H800 400 GB/s NVLink
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Output directory and logs
OUTPUT_DIR="./outputs_dpo"
LOG_FILE="./outputs_dpo/dpo_training.log"
mkdir -p "$OUTPUT_DIR"

# SFT starting checkpoint (fined-tuned in Stage 1)
SFT_CHECKPOINT="./outputs/checkpoint_sft.pt"

# DPO Preference training data path (JSON Lines pairwise chosen/rejected format)
DATA_PATH="./data/train_dpo.jsonl"

echo "======================================================================="
echo "🚀 Starting Karpathy-Style Distributed DPO Alignment (Native DDP)"
echo "-----------------------------------------------------------------------"
echo "🖥️  GPUs Sharded: 8 (H800 80GB via torchrun)"
echo "📦 SFT Checkpoint: $SFT_CHECKPOINT"
echo "🗂️  Dataset: $DATA_PATH"
echo "📝 Logging to: $LOG_FILE"
echo "======================================================================="

# Launch DPO training using torchrun across all 8 GPUs sharded on local node
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29602 \
    align.py \
    --sft_checkpoint_path "$SFT_CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --max_length 4096 \
    --max_prompt_length 2048 \
    --epochs 1 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --beta 0.1 \
    --max_lr 5e-6 \
    --weight_decay 0.01 \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo "======================================================================="
echo "✅ DPO Complete! Checkpoint saved as: $OUTPUT_DIR/checkpoint_dpo.pt"
echo "======================================================================="
