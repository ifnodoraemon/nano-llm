#!/usr/bin/env bash
# ==============================================================================
# nano-llm: Full End-to-End Pipeline Script
# ==============================================================================
# This script demonstrates and executes the COMPLETE training-to-serving pipeline
# for nano-llm on a multi-GPU cluster (tested on 8x H800).
#
# Pipeline Stages:
#   1. Data Crawling       → Raw corpus collection
#   2. Tokenizer Training  → Custom BPE tokenizer (10k vocab)
#   3. Binary Packing      → Memory-mapped training datasets
#   4. Pre-training        → From-scratch LLaMA transformer training
#   5. SFT                 → Supervised fine-tuning on instruction data
#   6. DPO                 → Direct Preference Optimization alignment
#   7. GRPO                → Group Relative Policy Optimization (reasoning RL)
#   8. Evaluation          → MMLU + GSM8K + Elo Arena benchmarks
#   9. Quantization        → INT8/INT4 weight-only RTN compression
#  10. Serving             → KV-cached autoregressive inference with RAG
#
# Usage:
#   bash scripts/run_full_pipeline.sh [--stage N] [--gpus 8] [--skip-training]
# ==============================================================================

set -euo pipefail

# Parse arguments
STAGE_START=${1:-1}
NUM_GPUS=${2:-8}
MASTER_PORT=${3:-29509}

# Environment setup for H800 cluster
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=0
export NANO_HUB_PROVIDER=ms
export OMP_NUM_THREADS=4

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              nano-llm Full Pipeline Runner                   ║"
echo "║              GPUs: ${NUM_GPUS} | Start Stage: ${STAGE_START}                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ==============================================================================
# Stage 1: Data Crawling
# ==============================================================================
if [ "$STAGE_START" -le 1 ]; then
    echo ""
    echo "━━━ Stage 1/10: Data Crawling ━━━"
    python3 crawl_data.py --sources wikitext zhihu_subset arxiv_subset \
        --output_dir ./data/raw_crawled \
        --max_pages 500
    echo "✅ Stage 1 complete: Raw corpus saved to ./data/raw_crawled/"
fi

# ==============================================================================
# Stage 2: Tokenizer Training
# ==============================================================================
if [ "$STAGE_START" -le 2 ]; then
    echo ""
    echo "━━━ Stage 2/10: Custom BPE Tokenizer Training ━━━"
    python3 train_tokenizer.py \
        --corpus_dir ./data/raw_crawled \
        --output_path ./data/custom_tokenizer.json \
        --vocab_size 10000 \
        --special_tokens '<|im_start|>,<|im_end|>,<|pad|>,<|unk|>,<|endoftext|>'
    echo "✅ Stage 2 complete: Tokenizer saved to ./data/custom_tokenizer.json"
fi

# ==============================================================================
# Stage 3: Binary Dataset Packing
# ==============================================================================
if [ "$STAGE_START" -le 3 ]; then
    echo ""
    echo "━━━ Stage 3/10: Binary Dataset Packing ━━━"
    python3 pack_binaries.py \
        --corpus_dir ./data/raw_crawled \
        --tokenizer_path ./data/custom_tokenizer.json \
        --output_dir ./data \
        --val_ratio 0.1
    echo "✅ Stage 3 complete: train.bin and val.bin saved to ./data/"
fi

# ==============================================================================
# Stage 4: Pre-training from Scratch
# ==============================================================================
if [ "$STAGE_START" -le 4 ]; then
    echo ""
    echo "━━━ Stage 4/10: Pre-training (8x GPU DDP) ━━━"
    mkdir -p ./outputs
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
        pretrain.py \
        --tp_size 1 --pp_size 1 --ep_size 1 \
        --max_steps 200 \
        --batch_size 4 \
        --block_size 512 \
        --learning_rate 3e-4 \
        --warmup_steps 20
    echo "✅ Stage 4 complete: Pre-trained checkpoint saved to ./outputs/checkpoint_pretrain.pt"
fi

# ==============================================================================
# Stage 5: Supervised Fine-Tuning (SFT)
# ==============================================================================
if [ "$STAGE_START" -le 5 ]; then
    echo ""
    echo "━━━ Stage 5/10: SFT ━━━"
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
        train.py \
        --train_data ./data/train_sft.jsonl \
        --output_dir ./outputs \
        --num_epochs 3 \
        --batch_size 4 \
        --max_length 512 \
        --learning_rate 2e-5 \
        --use_lora false
    echo "✅ Stage 5 complete: SFT checkpoint saved to ./outputs/checkpoint_sft.pt"
fi

# ==============================================================================
# Stage 6: DPO Preference Alignment
# ==============================================================================
if [ "$STAGE_START" -le 6 ]; then
    echo ""
    echo "━━━ Stage 6/10: DPO Alignment ━━━"
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
        align.py \
        --train_data ./data/train_dpo.jsonl \
        --checkpoint_path ./outputs/checkpoint_sft.pt \
        --output_dir ./outputs_dpo \
        --num_epochs 1 \
        --batch_size 2 \
        --learning_rate 5e-6 \
        --beta 0.1
    echo "✅ Stage 6 complete: DPO checkpoint saved to ./outputs_dpo/checkpoint_dpo.pt"
fi

# ==============================================================================
# Stage 7: GRPO Reasoning Reinforcement Learning
# ==============================================================================
if [ "$STAGE_START" -le 7 ]; then
    echo ""
    echo "━━━ Stage 7/10: GRPO (Reasoning RL) ━━━"
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \
        grpo.py \
        --train_data ./data/train_grpo.jsonl \
        --checkpoint_path ./outputs_dpo/checkpoint_dpo.pt \
        --output_dir ./outputs_grpo \
        --num_epochs 1 \
        --batch_size 2 \
        --group_size 4 \
        --kl_coeff 0.04
    echo "✅ Stage 7 complete: GRPO checkpoint saved to ./outputs_grpo/checkpoint_grpo.pt"
fi

# ==============================================================================
# Stage 8: Benchmark Evaluation
# ==============================================================================
if [ "$STAGE_START" -le 8 ]; then
    echo ""
    echo "━━━ Stage 8/10: Benchmark Evaluation ━━━"
    # Use the latest checkpoint available (GRPO > DPO > SFT > Pretrain)
    EVAL_CKPT="./outputs_grpo/checkpoint_grpo.pt"
    if [ ! -f "$EVAL_CKPT" ]; then
        EVAL_CKPT="./outputs_dpo/checkpoint_dpo.pt"
    fi
    if [ ! -f "$EVAL_CKPT" ]; then
        EVAL_CKPT="./outputs/checkpoint_sft.pt"
    fi
    if [ ! -f "$EVAL_CKPT" ]; then
        EVAL_CKPT="./outputs/checkpoint_pretrain.pt"
    fi

    python3 eval_benchmarks.py \
        --checkpoint_path "$EVAL_CKPT" \
        --baseline_checkpoint_path ./outputs/checkpoint_pretrain.pt
    echo "✅ Stage 8 complete: Evaluation report saved to ./outputs/eval_report.json"
fi

# ==============================================================================
# Stage 9: Quantization (INT8 + INT4)
# ==============================================================================
if [ "$STAGE_START" -le 9 ]; then
    echo ""
    echo "━━━ Stage 9/10: Quantization ━━━"
    QUANT_SRC="$EVAL_CKPT"

    # 8-bit quantization
    python3 quantize.py \
        --src "$QUANT_SRC" \
        --dest ./outputs/checkpoint_quantized_int8.pt \
        --bits 8
    echo "  → INT8 quantized model saved."

    # 4-bit quantization
    python3 quantize.py \
        --src "$QUANT_SRC" \
        --dest ./outputs/checkpoint_quantized_int4.pt \
        --bits 4
    echo "  → INT4 quantized model saved."
    echo "✅ Stage 9 complete: Quantized checkpoints saved."
fi

# ==============================================================================
# Stage 10: Inference Serving Demo
# ==============================================================================
if [ "$STAGE_START" -le 10 ]; then
    echo ""
    echo "━━━ Stage 10/10: Inference Serving Demo ━━━"
    SERVE_CKPT="$EVAL_CKPT"

    # Test with RAG
    python3 serve.py \
        --checkpoint_path "$SERVE_CKPT" \
        --rag_sources README.md \
        --prompt "What is nano-llm and what are its key features?" \
        --max_new_tokens 256 \
        --temperature 0.7

    echo ""
    echo "✅ Stage 10 complete: Inference serving verified."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        🎉 Full Pipeline Complete! 🎉                         ║"
echo "║                                                              ║"
echo "║  Checkpoints:                                                ║"
echo "║    Pre-train : ./outputs/checkpoint_pretrain.pt              ║"
echo "║    SFT       : ./outputs/checkpoint_sft.pt                   ║"
echo "║    DPO       : ./outputs_dpo/checkpoint_dpo.pt               ║"
echo "║    GRPO      : ./outputs_grpo/checkpoint_grpo.pt             ║"
echo "║    INT8      : ./outputs/checkpoint_quantized_int8.pt        ║"
echo "║    INT4      : ./outputs/checkpoint_quantized_int4.pt        ║"
echo "║    Eval      : ./outputs/eval_report.json                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
