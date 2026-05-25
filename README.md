<p align="center">
  <img src="assets/logo.png" width="350" alt="nano-llm logo"/>
</p>

# nano-llm: Clean, Minimalist PyTorch LLM Training & Alignment Node

<p align="center">
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://github.com/ifnodoraemon/nano-llm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"/></a>
  <a href="https://nvidia.com"><img src="https://img.shields.io/badge/Hardware-NVIDIA%20H800%20%7C%20A100-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="NVIDIA GPU"/></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="Black"/></a>
</p>

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
        G -->|grpo.py + RLHF Reasoning| I["checkpoint_grpo.pt (Reasoning Policy)"]
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
| 🐳 **[Docker Swarm Multi-Node Deployment Blueprint](file:///home/ifnodoraemon/myagent/nano-llm/docs/swarm_multi_node_deployment.md)** | • Swarm Manager vs. Workers topology map<br>• Node labeling & overlay-bypassing host-network sharding<br>• PyTorch `torchrun` multi-node environment variables & launch scripts |
| 📡 **[Stage 3 Telemetry Walkthrough Report](file:///home/ifnodoraemon/.gemini/antigravity-cli/brain/b9731614-1ca7-4068-b098-e69f35aea81a/walkthrough_telemetry.md)** | • Direct `/proc` telemetry reading & `nvidia-smi` CSV queries<br>• Zero-dependency CPU/RAM/Net/GPU monitor cockpit<br>• Dynamic fallbacks and simulator designs |
| 🚀 **[Stage 4 Multi-Dimension Upgrades Walkthrough](file:///home/ifnodoraemon/.gemini/antigravity-cli/brain/b9731614-1ca7-4068-b098-e69f35aea81a/walkthrough_stage4.md)** | • NCCL autotuning & RoCE network parameters<br>• Asynchronous non-blocking checkpointing thread queues<br>• Elastic self-healing states & auto fault-tolerance<br>• High-throughput linear-time MinHash-LSH deduplication ($O(N)$)<br>• PagedAttention & Continuous Batching concurrent scheduling |

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
    Root --> SWARM["docker-compose-swarm.yml <br> (Docker Swarm Multi-Node)"]:::file
    
    %% Core Model
    Root --> MODEL["model.py <br> (MLA + MoE + FP8 + Checkpointing)"]:::file
    
    %% Main pipelines
    Root --> PRETRAIN["pretrain.py <br> (Causal Pre-training loop)"]:::file
    Root --> TRAIN["train.py <br> (DDP SFT Packing)"]:::file
    Root --> ALIGN["align.py <br> (DPO Alignment)"]:::file
    Root --> GRPO["grpo.py <br> (DeepSeek-R1 GRPO RLHF loop)"]:::file
    Root --> SERVE["serve.py <br> (VLM static-cached streaming serving)"]:::file
    
    %% Subfolders
    Root --> UTILS["utils/"]:::dir
    Root --> WEB["web/"]:::dir
    Root --> DOCS["docs/"]:::dir
    Root --> TESTS["tests/"]:::dir
    
    UTILS --> U1["download_dataset.py"]:::file
    UTILS --> U2["vision_helper.py"]:::file
    UTILS --> U3["upload_hub.py"]:::file
    UTILS --> U4["hub_adapter.py"]:::file
    UTILS --> U5["parallel_3d.py"]:::file
    UTILS --> U6["paged_attention.py"]:::file
    UTILS --> U7["audio_projection.py"]:::file
    UTILS --> U8["grpo_critic.py"]:::file
    UTILS --> U9["triton_fa3.py"]:::file
    UTILS --> U10["kv_eviction.py"]:::file
    UTILS --> U11["search_tree.py"]:::file
    
    WEB --> W1["Premium glassmorphic web dashboard HUD"]:::file
    
    DOCS --> DOC1["deepseek_upgrade.md"]:::file
    DOCS --> DOC2["attention_and_position.md"]:::file
    DOCS --> DOC3["context_engineering.md"]:::file
    DOCS --> DOC4["performance_benchmarks.md"]:::file
    DOCS --> DOC5["risk_mitigation.md"]:::file
    DOCS --> DOC6["swarm_multi_node_deployment.md"]:::file

    TESTS --> T1["test_model.py"]:::file
    TESTS --> T2["test_data.py"]:::file
    TESTS --> T3["test_grpo.py"]:::file
    TESTS --> T4["test_upgrades.py"]:::file
    TESTS --> T5["test_search_eviction.py"]:::file
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

---

## 🗺️ Visual Project Roadmap & Milestones

```mermaid
graph TD
    classDef comp fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    classDef active fill:#00f2fe,stroke:#005c66,stroke-width:2px,color:#000;
    classDef planned fill:#e2e2e2,stroke:#555,stroke-width:1px,color:#777;

    C1["1. Custom BPE & flat binary packers<br>(data.py / deduplicate.py)"]:::comp
    C2["2. DeepSeek MLA + DeepSeekMoE architectures<br>(model.py)"]:::comp
    C3["3. Dynamic NTK-RoPE 1M context & FP8 calculations<br>(pretrain.py / serve.py)"]:::comp
    C4["4. Interactive glassmorphic autopilot HUD<br>(FastAPI Control Panel)"]:::comp
    C5["5. DeepSeek-R1 style RLHF & GRPO Alignment<br>(grpo.py)"]:::comp
    
    %% Planned milestones
    P1["6. 3D Model Parallelism <br>(TP + PP + DP sharding setups)"]:::active
    P2["7. Quantized static serving kernels <br>(TensorRT-LLM / vLLM hooks)"]:::planned
    
    C1 --> C2 --> C3 --> C4 --> C5
    C5 -->|Current Focus| P1
    P1 --> P2
```

---

## 💖 Show Your Support & Star History

We are committed to building the cleanest, most performant education-first PyTorch LLM implementation in the world. If you find this project valuable, **please give it a star!** It helps other researchers discover this repository and fuels our open-source developments. ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=ifnodoraemon/nano-llm&type=Date)](https://github.com/ifnodoraemon/nano-llm)
