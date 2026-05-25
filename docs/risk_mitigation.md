# LLM Training Risks & Mitigation Blueprint

```mermaid
graph TB
    subgraph Legend
        FailureStyle[Risk / Failure Mode]:::failNode
        SafeguardStyle[Proactive Prevention / Solution]:::safeNode
        classDef failNode fill:#ff4d4d,stroke:#990000,stroke-width:2px,color:#fff;
        classDef safeNode fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    end
```

---

## 💥 Risk 1: Loss Spikes & Training Collapse (训练崩盘 / 损失骤增)

```mermaid
graph TD
    classDef risk fill:#ff4d4d,stroke:#990000,stroke-width:1.5px,color:#fff;
    classDef safe fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    classDef nodeStyle fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;

    %% Triggers
    NoisyData["Noisy / Corrupted Batches"]:::nodeStyle --> Explode["Gradient Explodes (NaN/Inf)"]:::risk
    DeepLayers["Parameter Norm Outliers in Deep Layers"]:::nodeStyle --> Explode
    LRSpike["Unwarmed Learning Rate Updates"]:::nodeStyle --> Explode
    
    Explode --> TrainingCrash["Catastrophic Divergence (Loss Spikes)"]:::risk

    %% Safeguards in nano-llm
    MinHash["1. MinHash LSH Deduplication <br> (Cleanses noisy and repetitive text)"]:::safe --> Explode
    RMSNorm["2. Pre-Layer RMSNorm <br> (Caps activation scale before every attention/MLP block)"]:::safe --> Explode
    LinearWarmup["3. Linear Cosine Warmup <br> (Gently settles randomized weights in first 500 steps)"]:::safe --> Explode
    GradClip["4. Dynamic Gradient Clipping <br> (torch.nn.utils.clip_grad_norm_ at max_norm=1.0)"]:::safe --> Explode
```

---

## 🔀 Risk 2: MoE Expert Collapse & Load Balancing (路由坍坍 / 专家闲置)

```mermaid
graph TD
    classDef risk fill:#ff4d4d,stroke:#990000,stroke-width:1.5px,color:#fff;
    classDef safe fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    classDef nodeStyle fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;

    Token["Token Embeddings"] --> Router["Gated Softmax Router"]:::nodeStyle
    
    %% Standard MoE Risk
    Router -->|Imbalanced Gating Scores| Collapse["Routing Collapse: Router favors only 1-2 experts"]:::risk
    Collapse --> HeavyWeights["Untrained experts idle, active experts overloaded"]:::risk
    HeavyWeights --> Suboptimal["Model behaves like a standard Dense model (No specialization)"]:::risk

    %% DeepSeekMoE Safeguards
    Router -->|Auxiliary Gating Penalty| BalanceLoss["1. Auxiliary Load-Balancing Loss <br> (Forces uniform token distribution)"]:::safe
    
    Token -->|Always Active Pathway| SharedExperts["2. Shared Experts Module <br> (Captures generic knowledge, acting as a fallback baseline)"]:::safe
    
    BalanceLoss & SharedExperts --> Specialization["Stable Routing & Highly Optimized Specialized Experts"]:::safe
```

---

## 🤖 Risk 3: Reward Hacking & Alignment Degeneracy in DPO (奖励黑客)

```mermaid
sequenceDiagram
    autonumber
    actor Engine as DPO Loss Calculator (align.py)
    participant Policy as Policy Model (pi_theta)
    participant Ref as Reference Model (pi_ref)
    participant Loss as DPO Loss & KL Penalty
    
    Engine->>Policy: Generate conversational responses on preferences
    
    alt Standard Optimization without anchors
        Policy->>Policy: Exploit low-perplexity tokens / repeat punctuation <br> (Cheats reward model score)
        Policy-->>Engine: Reward Hacked Responses (low quality, high score)
    else DPO Anchor Regularization (nano-llm)
        Engine->>Ref: Forward pass chosen & rejected completions
        Ref-->>Loss: Reference Log-Probabilities (pi_ref)
        Engine->>Policy: Forward pass chosen & rejected completions
        Policy-->>Loss: Policy Log-Probabilities (pi_theta)
        Loss->>Loss: Apply rigid KL Regularization (beta = 0.1) <br> log(pi_theta / pi_ref)
        Loss-->>Policy: Dynamic weight updates keeping policy close to base checkpoint
    end
```

---

## 💾 Risk 4: FP8 Dynamic Range Underflow/Overflow (FP8 截断与溢出)

```mermaid
graph LR
    classDef risk fill:#ff4d4d,stroke:#990000,stroke-width:1.5px,color:#fff;
    classDef safe fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    classDef scale fill:#ffaa00,stroke:#8c5d00,stroke-width:1.5px,color:#000;

    FP8_Range["FP8 Dynamic Range (float8_e4m3fn)"]
    
    %% Issues
    FP8_Range -->|Gradients too small| Underflow["Quantization Underflow <br> (Values collapse to absolute zero)"]:::risk
    FP8_Range -->|Activations too large| Overflow["Quantization Overflow <br> (Values clip to Inf, causing NaNs)"]:::risk
    
    %% Dynamic Scaling solutions
    Underflow & Overflow --> DynamicScaling["1. Dynamic Tensor-Wise Scaling <br> (Calculates max scale factors per matrix block)"]:::scale
    DynamicScaling --> FitRange["Map values safely into [1e-4, 240] FP8 limits"]:::safe
    
    %% Mixed precision solutions
    Underflow & Overflow --> MixedPrecision["2. High-Precision Hybrid Backing <br> (Keep SwiGLU gates & RMSNorm layers in BF16/FP32)"]:::safe
    MixedPrecision --> SafeNorms["Zero-divergence linear matrix multiplication acceleration"]:::safe
```
