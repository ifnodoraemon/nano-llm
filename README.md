<p align="center">
  <img src="assets/logo.png" width="350" alt="nano-llm logo"/>
</p>

# nano-llm: Clean, Minimalist PyTorch LLM Training & Alignment Node

Welcome to **nano-llm** (now upgraded with **nano-deepseek** capabilities)! This repository is a clean, minimal, and highly educational PyTorch implementation for pre-training, Supervised Fine-Tuning (SFT), Preference Alignment (DPO), and long-context/FP8 optimizations of modern LLMs, designed for **8x80GB H800 GPU** clusters.

Read this in other languages: [English](README.md) | [简体中文](README_ZH.md)

---

## 🎨 Master Pipeline & Core Architecture

```mermaid
graph TD
    %% Dataset Prep Phase
    subgraph Data Prep tab [Data Preparation & Processing]
        A["TinyStories / WikiText / URLs"] -->|download_dataset.py / crawl_data.py| B["Sanitized corpus (.txt)"]
        B -->|deduplicate.py MinHash LSH| C["Unique Cleansed Corpus"]
        C -->|train_tokenizer.py BPE| D["custom_tokenizer.json"]
        C & D -->|pack_binaries.py| E["Memory-Mapped train.bin & val.bin"]
    end

    %% Training Pipeline Phase
    subgraph Training cockpit [Distributed Multi-GPU Engine]
        E -->|pretrain.py + FP8 + NTK-RoPE| F["checkpoint_pretrain.pt (Pre-trained Base)"]
        F -->|train.py + SFT Packing| G["checkpoint_sft.pt (SFT Policy)"]
        G -->|align.py + DPO Preference| H["checkpoint_dpo.pt (Aligned Policy)"]
    end

    %% Optimization & Deployment Phase
    subgraph Compression & serving [Autopilot serving Room]
        H -->|quantize.py Asymmetric RTN| I["checkpoint_dpo_q4.pt (4-bit Compression)"]
        I -->|serve.py Static KV-Cache + VLM| J["High-Throughput Streaming API"]
        I -->|upload_hub.py Publish| K["Hugging Face Hub / ModelScope Hub"]
    end
    
    style E fill:#00f2fe,stroke:#000,stroke-width:2px,color:#000
    style H fill:#b927fc,stroke:#000,stroke-width:2px,color:#fff
    style J fill:#42e695,stroke:#000,stroke-width:2px,color:#000
```

---

## 📘 Visual blue prints & Architectures

We have organized all documentation into **100% Visual blueprints**, replacing large text blocks entirely with design diagrams:

| Documentation Blueprint | Visual Content Included |
| :--- | :--- |
| 👁️ **[DeepSeek MLA & DeepSeekMoE Blueprint](file:///home/ifnodoraemon/myagent/nano-llm/docs/deepseek_upgrade.md)** | • GQA vs. DeepSeek MLA Memory Flow<br>• Shared + Gated MoE Routing Architecture<br>• UML Class mapping of `model.py` components |
| ⚡ **[Attention & Position Coordinates Blueprint](file:///home/ifnodoraemon/myagent/nano-llm/docs/attention_and_position.md)** | • Softmax Attention Sharpening peaks comparison<br>• NTK-Aware Positional Coordinate stretching<br>• High-Resolution Double-Grid Visual projector flow<br>• Hard Negative SFT Data generation sequence |
| 📈 **[1M Long-Context Blueprint](file:///home/ifnodoraemon/myagent/nano-llm/docs/context_engineering.md)** | • Dynamic NTK phase precompute sequence<br>• KV-Cache memory footprint metrics (GQA vs. MLA)<br>• Ring-Attention Context Parallel circular network<br>• Ring Attention computational loop states |
| 🏎️ **[Performance & Self-Iteration Blueprint](file:///home/ifnodoraemon/myagent/nano-llm/docs/performance_benchmarks.md)** | • Activation Checkpointing forward/backward VRAM graph<br>• FSDP Parameter/Optimizer sharding comparison<br>• Fused AdamW hardware CUDA kernel sweeps<br>• Autonomous Model Self-Play DPO iteration loop<br>• Evaluation Benchmark logics (MMLU, GSM8K, ARC, etc.) |
| 🛡️ **[LLM Risk Mitigation Blueprint](file:///home/ifnodoraemon/myagent/nano-llm/docs/risk_mitigation.md)** | • Loss Spikes & Training Collapse triggers & safeguards<br>• MoE Routing Collapse & Shared Expert Gated router balance<br>• DPO Reward Hacking & KL-regularization sequence loops<br>• FP8 dynamic range mapping underflow/overflow mitigations |

---

## 📂 Project Architecture

```mermaid
graph TD
    classDef file fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;
    classDef dir fill:#e2e2e2,stroke:#000,stroke-width:2px,color:#000;
    classDef root fill:#00f2fe,stroke:#005c66,stroke-width:2px,color:#000;

    Root["nano-llm /"]:::root
    
    %% Config and environment
    Root --> DOCKER["Dockerfile / docker-compose.yml <br> (Multi-GPU CUDA host setup)"]:::file
    
    %% Core Model
    Root --> MODEL["model.py <br> (MLA + MoE + FP8 + Checkpointing)"]:::file
    
    %% Main pipelines
    Root --> PRETRAIN["pretrain.py <br> (Causal Pre-training loop)"]:::file
    Root --> TRAIN["train.py <br> (DDP SFT Packing)"]:::file
    Root --> ALIGN["align.py <br> (DPO Alignment)"]:::file
    Root --> SERVE["serve.py <br> (VLM static-cached streaming serving)"]:::file
    
    %% Subfolders
    Root --> UTILS["utils/"]:::dir
    Root --> WEB["web/"]:::dir
    Root --> DOCS["docs/"]:::dir
    
    UTILS --> U1["download_dataset.py"]:::file
    UTILS --> U2["vision_helper.py"]:::file
    UTILS --> U3["upload_hub.py"]:::file
    
    WEB --> W1["Premium glassmorphic web dashboard HUD"]:::file
    
    DOCS --> DOC1["deepseek_upgrade.md"]:::file
    DOCS --> DOC2["attention_and_position.md"]:::file
    DOCS --> DOC3["context_engineering.md"]:::file
    DOCS --> DOC4["performance_benchmarks.md"]:::file
    DOCS --> DOC5["risk_mitigation.md"]:::file
```

---

## 📐 Mathematical Frameworks in Code

```mermaid
classDiagram
    class TransformerArchitecture {
        +RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
        +SwiGLU(x) = (Swish(x * W_g) * (x * W_u)) * W_d
        +MLA_KV_Compression(x) = W_DKV * x
        +Attention_Softmax_Sharpening(Q, K) = Softmax( (Q * K^T) / sqrt(d) * attn_scale_multiplier )
        +DPO_Loss(pi_theta, pi_ref) = -E[ log_sigmoid( beta * log(ratio_chosen) - beta * log(ratio_rejected) ) ]
        +NTK_RoPE_Base_Scale(theta) = theta * (scaling_factor ^ (d / (d-2)))
    }
```

---

## 🚀 Quick Start

### A. Environment Spin-Up

```mermaid
sequenceDiagram
    autonumber
    actor Operator as operator
    participant Compose as docker-compose.yml
    participant Docker as CUDA Docker Container
    
    Operator->>Compose: docker compose up --build -d
    Compose->>Docker: Launch GPU pass-through container (16GB shared memory)
    Docker-->>Operator: Web HUD Dashboard listening on http://localhost:8000
```

### B. Autopilot Web Console Operations Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Autopilot Operator
    participant BE as FastAPI Control Panel (Container)
    participant HW as 8xH800 GPU Cluster (Host)
    
    User->>BE: Trigger Data Preparation
    BE->>BE: Run download_dataset.py / pack_binaries.py
    BE-->>User: Binary packing finished! train.bin mapped.
    
    User->>BE: Trigger FP8 Pre-training / SFT
    BE->>HW: Spawn torchrun pretrain.py --use_fp8 True
    HW-->>BE: Stream stdout logs & MFU metrics via WebSockets
    BE-->>User: Active loss curve animations
    
    User->>BE: Ask chat terminal (Text/Image prompt)
    BE->>HW: Prefill static KV-Cache with image patches + prompt tokens
    HW-->>User: SSE streamed autoregressive tokens (TTFT ms logs)
```
