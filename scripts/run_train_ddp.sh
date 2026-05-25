#!/bin/bash

# ==============================================================================
# nano-llm: Distributed SFT Training Launcher (8xH800 DDP)
# ==============================================================================

# Highly optimized NCCL variables to fully utilize H800 400 GB/s NVLink
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Output directory and logs
OUTPUT_DIR="./outputs"
LOG_FILE="./outputs/sft_training.log"
mkdir -p "$OUTPUT_DIR"

# Base model path (adjust to your local model path or HuggingFace ID)
# Used to instantiate the correct token vocabulary size and tokenizer configs
MODEL_PATH="Qwen/Qwen2.5-7B" 

# Training data path (JSON Lines conversational format)
DATA_PATH="./data/train_sft.jsonl"

echo "======================================================================="
echo "🚀 Starting Karpathy-Style Distributed SFT Training (Native DDP)"
echo "-----------------------------------------------------------------------"
echo "🖥️  GPUs Sharded: 8 (H800 80GB via torchrun)"
echo "📦 Config reference: $MODEL_PATH"
echo "🗂️  Dataset: $DATA_PATH"
echo "📝 Logging to: $LOG_FILE"
echo "======================================================================="

# Launch training using torchrun across all 8 GPUs sharded on local node
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29601 \
    train.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --max_length 4096 \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_lr 2e-5 \
    --min_lr 2e-6 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --clip_grad 1.0 \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 200 \
    --seed 42 \
    2>&1 | tee "$LOG_FILE"

echo "======================================================================="
echo "✅ Training Complete! Checkpoint saved as: $OUTPUT_DIR/checkpoint_sft.pt"
echo "======================================================================="
