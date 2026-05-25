# nano-llm: Clean, Minimalist PyTorch LLM Training & Alignment Node

Welcome to **nano-llm**! This repository is a clean, minimal, and highly educational PyTorch implementation for pre-training, Supervised Fine-Tuning (SFT), and Preference Alignment (DPO) of modern Large Language Models (LLMs).

Directly inspired by Andrej Karpathy's famous repositories (`nanoGPT`, `llm.c`, `llama2.c`), **nano-llm** strips away the heavy layers of third-party libraries (Hugging Face Trainer, DeepSpeed configs, TRL wrappers) to expose the raw, fundamental PyTorch code beneath. 

This repository is optimized for high-performance scale-up on **8x80GB H800 GPU** nodes using **PyTorch DDP (Distributed Data Parallel)**.

---

## 🎨 Design Philosophy

* 🧠 **Zero Abstraction**: No giant Trainer configurations. SFT and DPO training loops are written explicitly in pure PyTorch (`train.py` and `align.py`), showing exactly how optimizer updates, gradients, and log-probabilities are calculated.
* 🛠️ **From-Scratch Architecture**: Modern LLaMA-style architecture implemented in `model.py` containing custom layers for RMSNorm, SwiGLU, and Rotary Position Embeddings (RoPE).
* 🏎️ **Native Multi-GPU Scaling**: Distributed training uses PyTorch's native `DDP` and `torchrun` to demonstrate GPU scaling and gradient synchronization.
* 📦 **Educational & Production Ready**: Compact, readable, well-commented modules that run at maximum efficiency on H800 high-bandwidth NVLink hardware.

---

## 📂 Project Structure

```text
nano-llm/
├── README.md               # This Guide
├── requirements.txt        # Minimal PyTorch, Transformers, WandB, and TikToken dependencies
├── model.py                # LLaMA-style Transformer (RMSNorm, RoPE, SwiGLU, Attention)
├── data.py                 # Custom packed dataset and memory-mapped bin loader
├── train.py                # Raw PyTorch SFT/Pre-training loop with DDP (DDP setup, manual grad accumulation)
├── align.py                # Raw PyTorch DPO preference alignment training loop from scratch
├── serve.py                # Minimal text generator with KV-cache optimization
├── scripts/
│   ├── run_train_ddp.sh    # SFT/Pre-train DDP multi-GPU launch script
│   └── run_align_ddp.sh    # DPO DDP multi-GPU launch script
└── tests/
    └── test_model.py       # Unit tests for RMSNorm and RoPE computations
```

---

## 📐 Mathematical Underpinnings

### 1. Rotary Position Embeddings (RoPE)
Instead of adding absolute position embeddings, we rotate the Query ($Q$) and Key ($K$) vectors in 2D sub-spaces by angle $\theta$:
$$R_{\Theta, m}^d = \text{diag}\left( R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m} \right)$$
where $R_{\theta_i, m} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}$. This preserves relative distance relationship between tokens naturally.

### 2. RMSNorm
RMSNorm is a highly efficient alternative to LayerNorm that drops the mean-centering step, saving GPU computation:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

### 3. SwiGLU MLP
Modern LLMs replace traditional ReLU/GELU feed-forward networks with gated linear units:
$$\text{SwiGLU}(x) = \left( \text{Swish}(x W_g) \odot x W_u \right) W_d$$

### 4. DPO Loss
DPO optimizes conversational preference alignment directly over policy $\pi_\theta$ and reference $\pi_{\text{ref}}$ log-probabilities:
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

---

## 🚀 Quick Start Guide

### 1. Multi-GPU Distributed SFT Training
Ensure your dataset is ready at `data/train_sft.jsonl` and run `torchrun` via:
```bash
bash scripts/run_train_ddp.sh
```

### 2. Multi-GPU DPO Alignment
Once SFT finishes, align the policy with preference datasets at `data/train_dpo.jsonl`:
```bash
bash scripts/run_align_ddp.sh
```

### 3. KV-Cached Fast Generation Server
Expose your trained checkpoint for streaming inference:
```bash
python3 serve.py --model_path ./outputs/checkpoint_sft.pt --prompt "Explain quantum physics in a simple way"
```
