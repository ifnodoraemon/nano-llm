# Contributing to nano-llm

First off, thank you for considering contributing to **nano-llm** (upgraded with **nano-deepseek**)! It is people like you who make this codebase a spectacular resource for the deep learning community.

---

## 🗺️ Visual Project Roadmap & Future Milestones

We welcome contributions across all areas of the project. Here is our current development roadmap:

```mermaid
graph TD
    classDef comp fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    classDef active fill:#00f2fe,stroke:#005c66,stroke-width:2px,color:#000;
    classDef planned fill:#e2e2e2,stroke:#555,stroke-width:1px,color:#777;

    %% Completed milestones
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

## 🛠️ Contribution Guidelines

To keep the repository highly clean, academic, and high-performance, we maintain four core guidelines:

1. 🧠 **Zero Heavy Abstractions**
   * Keep SFT, DPO, and Pre-training loops written in **explicit, pure PyTorch**.
   * Avoid wrapping code in thick, black-box libraries (like Hugging Face Trainer, DeepSpeed, or Megatron). All operations should be traceable.

2. 🏎️ **Hopper & NVIDIA Optimization**
   * Prioritize dynamic FP8 calculations, Fused AdamW kernels, and PyTorch Activation Checkpointing hooks.
   * Make sure memory-mapped datasets are loaded using fast NumPy pointers to bypass CPU overhead.

3. 📐 **Theoretical Rigor**
   * Ensure that mathematical algorithms (such as low-rank Multi-Head Latent Attention compression, SwiGLU gated FFNs, and NTK positional stretching) are represented accurately.

4. 🎨 **Visual-First Documentation**
   * If you are modifying architectures, please update the corresponding visual markdown blueprints in `docs/` using **Mermaid diagrams**. 
   * Avoid adding large blocks of text; represent processes visually!

---

## 🚀 Getting Started with Pull Requests

1. **Fork the repository** on GitHub.
2. **Create a feature branch** from `main` (`git checkout -b feature/your-awesome-upgrade`).
3. **Write and verify your changes**:
   * Run the test suite to ensure compile and logic success:
     ```bash
     python -m unittest tests/test_model.py
     python -m unittest tests/test_data.py
     ```
4. **Commit with descriptive messages** following Conventional Commits (e.g. `feat: add GRPO loss calculator`, `docs: update MLA blueprint with KV projections`).
5. **Push to your fork** and submit a **Pull Request** to the `main` branch.

Thank you for building the future of minimal, powerful, open-source deep learning with us! 🚀
