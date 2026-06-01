#!/usr/bin/env bash
set -euo pipefail

# nano-llm: One-click environment setup and verification
# Usage: bash scripts/setup.sh [--dev|--prod]

MODE="${1:---dev}"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo " nano-llm Environment Setup ($MODE)"
echo "=============================================="

# --- Python ---
echo -e "\n${YELLOW}[1/6]${NC} Checking Python..."
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null || echo "")
if [ -z "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python not found. Install Python 3.10+${NC}"
    exit 1
fi
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
echo -e "  ${GREEN}OK${NC} Python $PY_VER ($PYTHON)"

# --- CUDA ---
echo -e "\n${YELLOW}[2/6]${NC} Checking CUDA..."
if $PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_COUNT=$($PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    GPU_MEM=$($PYTHON -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}')" 2>/dev/null)
    echo -e "  ${GREEN}OK${NC} $GPU_COUNT x $GPU_NAME (${GPU_MEM} GB each)"

    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        echo "  CUDA Driver: $CUDA_VER"
    fi
else
    echo -e "  ${YELLOW}WARN${NC} CUDA not available. Running CPU-only."
fi

# --- Dependencies ---
echo -e "\n${YELLOW}[3/6]${NC} Installing dependencies..."
$PYTHON -m pip install --upgrade pip -q
$PYTHON -m pip install -e ".[dev]" -q 2>/dev/null || $PYTHON -m pip install -r requirements.txt -q
echo -e "  ${GREEN}OK${NC} Dependencies installed"

# --- Tokenizer ---
echo -e "\n${YELLOW}[4/6]${NC} Checking tokenizer..."
if [ -f "./data/fast_tokenizer.json" ] || [ -f "./data/custom_tokenizer.json" ]; then
    echo -e "  ${GREEN}OK${NC} Tokenizer found"
else
    echo -e "  ${YELLOW}WARN${NC} No tokenizer found. Run: python data/train_fast_tokenizer.py"
fi

# --- Data ---
echo -e "\n${YELLOW}[5/6]${NC} Checking datasets..."
for ds in "train_sft_premium.jsonl" "train_dpo_premium.jsonl" "train_grpo_premium.jsonl"; do
    if [ -f "./data/$ds" ]; then
        lines=$(wc -l < "./data/$ds" 2>/dev/null || echo "0")
        echo -e "  ${GREEN}OK${NC} $ds ($lines samples)"
    else
        echo -e "  ${YELLOW}MISSING${NC} $ds - run: python data/pipeline.py --premium-only"
    fi
done

# --- Quick Smoketest ---
echo -e "\n${YELLOW}[6/6]${NC} Running smoketest..."
if $PYTHON -c "
import torch
from model import ModelConfig, Transformer
c = ModelConfig(block_size=64, vocab_size=1000, n_layer=2, n_head=4, n_embd=64, use_mla=False, use_moe=False)
m = Transformer(c)
x = torch.randint(0, 1000, (1, 32))
l, _, _ = m(x, targets=x)
print('Smoketest passed, loss:', l.item())
" 2>/dev/null; then
    echo -e "  ${GREEN}OK${NC} Model forward pass works"
else
    echo -e "  ${RED}FAIL${NC} Model smoketest failed - check PyTorch installation"
fi

# --- Summary ---
echo ""
echo "=============================================="
echo -e " ${GREEN}Setup complete!${NC}"
echo ""
echo " Quick start:"
echo "   make pretrain   # or:  python pretrain.py --model_size tiny"
echo "   make sft        # or:  torchrun --nproc_per_node=1 train.py --data_path ./data/train_sft_premium.jsonl"
echo "   make serve      # or:  python serve.py --checkpoint_path ./outputs/checkpoint_sft.pt"
echo "=============================================="
