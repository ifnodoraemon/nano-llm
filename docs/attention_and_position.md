# Attention Detail & Position Coordinates Optimization Blueprint

```mermaid
graph TB
    subgraph Legend
        DataNode[Data / Token]:::data
        OpNode[Mathematical Operator]:::op
        ModelNode[Neural Block / Layer]:::model
        classDef data fill:#00f2fe,stroke:#005c66,stroke-width:1.5px,color:#000;
        classDef op fill:#ffaa00,stroke:#8c5d00,stroke-width:1.5px,color:#000;
        classDef model fill:#e2e2e2,stroke:#555,stroke-width:1px,color:#000;
    end
```

---

## ⚡ 1. Softmax Attention Sharpening (熵约束注意力)

### Standard Attention Logits Scaling vs. Sharpened Attention

```mermaid
graph TD
    classDef logits fill:#00f2fe,stroke:#000,stroke-width:1px,color:#000;
    classDef opt fill:#ffaa00,stroke:#8c5d00,stroke-width:2px,color:#000;
    classDef distribution fill:#ff4d4d,stroke:#990000,stroke-width:1.5px,color:#fff;
    classDef sharpened fill:#42e695,stroke:#005c1e,stroke-width:2.5px,color:#000;

    Q["Query Vector (Q)"] & K["Key Vector (K)"] --> MatMul["QK^T (Matrix Product)"]:::opt
    
    %% Standard Path
    MatMul -->|Divide by sqrt(d)| RawLogits["Standard Attention Logits"]:::logits
    RawLogits --> Softmax1["Standard Softmax(x)"]:::opt
    Softmax1 --> FlatDist["Entropy Dilution: Flat, Wide Probability Curve <br> (Attention scattered across irrelevant tokens)"]:::distribution

    %% Sharpened Path
    MatMul -->|Divide by sqrt(d) * gamma (gamma = 1.3)| SharpLogits["Sharpened Attention Logits"]:::logits
    SharpLogits --> Softmax2["Standard Softmax(x)"]:::opt
    Softmax2 --> SharpDist["Entropy Compression: High, Narrow Peak <br> (Attention mathematically focused on precision tokens)"]:::sharpened
```

---

## 📐 2. Positional Coordinate Stretching (Decoupled RoPE Base Scaling)

```mermaid
graph TD
    classDef state fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;
    classDef stretch fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    classDef collapse fill:#ff4d4d,stroke:#990000,stroke-width:2px,color:#fff;

    Base["Base Frequency Base theta = 10,000"]:::state
    Context["Context Length Extended: 4K --> 1M (Scale Factor alpha = 250)"]:::state
    
    Base & Context --> StandardInterpolation["Linear Positional Interpolation"]:::state
    StandardInterpolation --> InterpCollapse["Orthogonal Polar Phase Collapses <br> (Nearby tokens share identical coordinate values)"]:::collapse
    
    Base & Context --> NTK_Formula["NTK-Aware Frequency Scaling <br> theta_scaled = theta * (alpha * S / block_size - alpha + 1) ^ (d / (d-2))"]:::state
    NTK_Formula --> StretchedCoordinates["Positional Coordinates Dynamically Stretched <br> (Stretches adjacent token angles, resolving micro-distances)"]:::stretch
```

---

## 👁️ 3. Multimodal Detail Ingestion: Double-Grid Visual Projector

```mermaid
graph TD
    classDef img fill:#ffaa00,stroke:#000,stroke-width:1.5px,color:#000;
    classDef proc fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;
    classDef out fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;

    Image["High-Resolution Source Image <br> (e.g. 448 x 448 pixels)"]:::img
    
    %% Split pathway
    Image -->|Global Resize 224x224| ViewGlobal["Global View <br> (1x downsampled grid)"]:::img
    Image -->|Crop Grid Top-Left| ViewTL["Grid Patch 1 (224x224)"]:::img
    Image -->|Crop Grid Top-Right| ViewTR["Grid Patch 2 (224x224)"]:::img
    Image -->|Crop Grid Bottom-Left| ViewBL["Grid Patch 3 (224x224)"]:::img
    Image -->|Crop Grid Bottom-Right| ViewBR["Grid Patch 4 (224x224)"]:::img
    
    %% Vision Transformer Projection
    ViewGlobal --> ViT["Vision Transformer (ViT-H)"]:::proc
    ViewTL --> ViT
    ViewTR --> ViT
    ViewBL --> ViT
    ViewBR --> ViT
    
    ViT --> FeatGlobal["Global Features <br> [16 tokens, 1024-dim]"]
    ViT --> FeatTL["Patch 1 Features <br> [16 tokens, 1024-dim]"]
    ViT --> FeatTR["Patch 2 Features <br> [16 tokens, 1024-dim]"]
    ViT --> FeatBL["Patch 3 Features <br> [16 tokens, 1024-dim]"]
    ViT --> FeatBR["Patch 4 Features <br> [16 tokens, 1024-dim]"]
    
    %% Multimodal Connector Projection
    FeatGlobal & FeatTL & FeatTR & FeatBL & FeatBR --> MLP["Dynamic Vision-to-Language Projection Layer <br> (Two-layer SwiGLU MLP)"]:::proc
    
    MLP --> Concat["Concatenation Block"]:::proc
    Concat --> LanguageInput["Visual Sequence Input to Transformer <br> [80 Patches, Embedding Dim]"]:::out
```

---

## 🤖 4. Data-Engine Detail Injection: Hard Negative Prompting

```mermaid
sequenceDiagram
    autonumber
    actor Engine as Self-Instruction Data Engine (self_instruct.py)
    participant Seed as Seed Instruction Bank
    participant Critic as Large Critic LLM (API / Judge)
    participant File as train_sft.jsonl & train_dpo.jsonl
    
    Engine->>Seed: Pull base conversational query
    Engine->>Critic: Submit query with strict structured prompt modifiers <br> (Force explicit math steps, character count constraints, strict code markers)
    Critic-->>Engine: Generate high-resolution Detail-Intensive SFT Pairs
    Engine->>Engine: Run Regular Expression validator on output markers
    
    alt Verification Successful
        Engine->>File: Write validated Detail-SFT pair to train_sft.jsonl
    else Verification Fails
        Engine->>Critic: Re-generate using hard-negative corrective prompt
    end
```
