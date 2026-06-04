#!/bin/bash

# ==============================================================================
# nano-llm: Distributed Base Pre-training Launcher (8xH800 DDP)
# ==============================================================================

# Highly optimized NCCL variables to fully utilize H800 400 GB/s NVLink
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Output directory and logs
OUTPUT_DIR="./outputs_pretrain"
LOG_FILE="./outputs_pretrain/pretrain_training.log"
mkdir -p "$OUTPUT_DIR"

# Pre-training dataset directory (containing our merged 77.8B tokens train.bin/val.bin)
DATA_DIR="/data/nano-llm-data/binaries_1t"

echo "======================================================================="
echo "🚀 Starting Karpathy-Style Distributed Base Pre-training from Scratch"
echo "-----------------------------------------------------------------------"
echo "🖥️  GPUs Sharded: 8 (H800 80GB via torchrun)"
echo "📦 Model preset: 2B-dense (36 Layers, 32 Heads, 2048 Dim, 4096 Context)"
echo "🗂️  Dataset: $DATA_DIR"
echo "📝 Logging to: $LOG_FILE"
echo "======================================================================="

# Launch pre-training using torchrun across all 8 GPUs sharded on local node
torchrun --nnodes=1 --nproc_per_node=8 --master_port=29602 \
    pretrain.py \
    --model_size 2B-dense \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUTPUT_DIR" \
    --batch_size 8 \
    --block_size 4096 \
    --max_steps 50000 \
    --lr 3e-4 \
    --min_lr 3e-5 \
    --warmup_steps 1000 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --grad_accum_steps 4 \
    --use_fp8 True \
    --use_checkpoint True \
    --use_compile True \
    2>&1 | tee "$LOG_FILE"

echo "======================================================================="
echo "✅ Pre-training Phase Complete!"
echo "======================================================================="
