# DeepSeek MLA & DeepSeekMoE Architecture Blueprint

```mermaid
graph TB
    subgraph Legend
        InputStyle[Input/Output]:::inputNode
        ProcessStyle[Linear Projection / Operation]:::processNode
        CacheStyle[(Memory Cache)]:::cacheNode
        classDef inputNode fill:#00f2fe,stroke:#005c66,stroke-width:2px,color:#000;
        classDef processNode fill:#e2e2e2,stroke:#555,stroke-width:1px,color:#000;
        classDef cacheNode fill:#b927fc,stroke:#4a0072,stroke-width:2px,color:#fff;
    end
```

---

## 👁️ 1. GQA (Grouped Query Attention) vs. DeepSeek MLA (Multi-Head Latent Attention)

### A. Grouped Query Attention (Standard LLaMA)

```mermaid
graph TD
    classDef input fill:#00f2fe,stroke:#000,stroke-width:1.5px,color:#000;
    classDef process fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;
    classDef cache fill:#ff4d4d,stroke:#990000,stroke-width:2px,color:#fff;
    
    H["Hidden States (h_t) [B, S, D]"]:::input
    
    %% Key projection path
    H --> Wk["W_k [D, G*D_h]"]:::process
    Wk --> K["Keys [B, S, G, D_h]"]
    K --> RoPE["Apply RoPE (Rotary Position)"]:::process
    RoPE --> K_cache[("KV-Cache (VRAM) <br> [B, S, G, D_h]")]:::cache
    
    %% Value projection path
    H --> Wv["W_v [D, G*D_h]"]:::process
    Wv --> V["Values [B, S, G, D_h]"]
    V --> V_cache[("KV-Cache (VRAM) <br> [B, S, G, D_h]")]:::cache
    
    %% Query projection path
    H --> Wq["W_q [D, H*D_h]"]:::process
    Wq --> Q["Queries [B, S, H, D_h]"]:::input
    Q --> RoPE_Q["Apply RoPE"]:::process
    
    %% Attention
    RoPE_Q & K_cache --> SDPA["FlashAttention (QK^T)"]:::process
    V_cache --> SDPA
    SDPA --> Out["Output [B, S, D]"]:::input
```

### B. DeepSeek MLA (Compressed KV-Cache & Decoupled RoPE)

```mermaid
graph TD
    classDef input fill:#00f2fe,stroke:#000,stroke-width:1.5px,color:#000;
    classDef process fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;
    classDef cache fill:#b927fc,stroke:#4a0072,stroke-width:2px,color:#fff;
    classDef extra fill:#ffcc00,stroke:#996600,stroke-width:1px,color:#000;

    H["Hidden States (h_t) [B, S, D]"]:::input
    
    %% Latent Compression
    H --> W_DKV["Down-Projection (W_DKV) <br> [D, D_c]"]:::process
    W_DKV --> C_KV["Latent KV (c_t^KV) <br> [B, S, D_c]"]:::cache
    
    %% Decompression Pathway (Forward pass only, no storage!)
    C_KV -->|In-Flight Up-Projection| W_UK["Up-Projection Keys (W_UK) <br> [D_c, H*D_h]"]:::process
    C_KV -->|In-Flight Up-Projection| W_UV["Up-Projection Values (W_UV) <br> [D_c, H*D_h]"]:::process
    
    W_UK --> K_c["Decompressed Keys (k_t^C) <br> [B, S, H, D_h]"]
    W_UV --> V_c["Decompressed Values (v_t^C) <br> [B, S, H, D_h]"]
    
    %% Decoupled RoPE Key pathway
    H --> W_KR["RoPE Key Proj (W_KR) <br> [D, D_r]"]:::process
    W_KR --> K_pe["Positional Keys (k_t^R) <br> [B, S, D_r]"]
    K_pe --> RoPE["Apply RoPE (Positional)"]:::process
    RoPE --> K_pe_cached[("RoPE Key-Cache (VRAM) <br> [B, S, D_r]")]:::cache
    
    %% Decoupled RoPE Query pathway
    H --> W_QR["RoPE Query Proj (W_QR) <br> [D, D_r]"]:::process
    W_QR --> Q_pe["Positional Queries (q_t^R) <br> [B, S, D_r]"]
    Q_pe --> RoPE_Q["Apply RoPE (Positional)"]:::process
    
    %% Query latent compression/decompression
    H --> W_DQ["Q Down-Proj (W_DQ) <br> [D, D_c_q]"]:::process
    W_DQ --> C_Q["Latent Q (c_t^Q) <br> [B, S, D_c_q]"]
    C_Q --> W_UQ["Q Up-Proj (W_UQ) <br> [D_c_q, H*D_h]"]:::process
    W_UQ --> Q_c["Decompressed Queries (q_t^C) <br> [B, S, H, D_h]"]
    
    %% Fused Concatenated Attention Calculation
    Q_c & K_c --> Dot1["MatMul (q_t^C * k_t^C)"]:::process
    Q_pe & K_pe_cached --> Dot2["MatMul (q_t^R * k_t^R)"]:::process
    
    Dot1 & Dot2 --> Add["Add Attn Logits"]:::process
    Add --> Scale["Scale (1 / sqrt(d) * gamma)"]:::process
    Scale --> Softmax["Softmax (Entropy Constraint)"]:::process
    Softmax --> Attn_W["Attention Weights"]
    
    Attn_W & V_c --> FinalMatMul["Weighted Value Sum"]:::process
    FinalMatMul --> W_o["Out Projection (W_o)"]:::process
    W_o --> Out["Output Hidden State"]:::input
```

---

## 🔀 2. DeepSeekMoE (Mixture of Experts) Architecture

```mermaid
graph TD
    classDef input fill:#00f2fe,stroke:#000,stroke-width:1.5px,color:#000;
    classDef process fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;
    classDef expert fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    classDef router fill:#ffaa00,stroke:#8c5d00,stroke-width:2px,color:#000;

    X["Token Representation (x)"]:::input
    
    %% Split to Shared and Routed pathways
    X --> Shared1["Shared Expert 1 <br> (Always Active SwiGLU)"]:::expert
    X --> Shared2["Shared Expert 2 <br> (Always Active SwiGLU)"]:::expert
    
    X --> Router["Gated Router Module <br> (Softmax + Top-K Filtering)"]:::router
    
    %% Router selection
    Router -->|Gate Score g_1| E1["Routed Expert 1"]:::expert
    Router -->|Gate Score g_2| E2["Routed Expert 2"]:::expert
    Router -->|Gate Score g_3| E3["Routed Expert 3"]:::expert
    Router -->|Gate Score g_N| En["Routed Expert N"]:::expert
    
    %% Dynamic Masking based on Top-2
    Router -.->|Dynamic Routing Mask| Switch1["Expert Gate Switch"]:::router
    Switch1 -->|y_1 = g_1 * E_1(x)| SumRouted["Sum of Top-K Routed Outputs"]:::process
    Switch1 -->|y_2 = g_2 * E_2(x)| SumRouted
    
    Shared1 & Shared2 --> SumShared["Sum of Shared Outputs"]:::process
    
    %% Blend everything
    SumShared & SumRouted --> Blend["Add Outputs"]:::process
    Blend --> Output["Output Token y"]:::input
```

---

## ⚙️ 3. Structural Mapping in Code

```mermaid
classDiagram
    class ModelConfig {
        +int n_layer
        +int n_head
        +int dim
        +int block_size
        +float attn_scale_multiplier
        +bool use_mla
        +int q_lora_rank
        +int kv_lora_rank
        +int qe_rope_dim
        +bool use_moe
        +int n_shared_experts
        +int n_routed_experts
        +int num_active_experts
    }

    class Transformer {
        +RMSNorm norm
        +Embedding wte
        +ModuleList layers
        +Linear lm_head
        +forward(tokens, targets)
    }

    class Block {
        +RMSNorm input_layernorm
        +CausalSelfAttention attn
        +RMSNorm post_attention_layernorm
        +SwiGLU_MLP / DeepSeekMoE mlp
        +forward(x)
    }

    class CausalSelfAttention {
        +MLA_Projections
        +DecoupledRoPE
        +forward(x)
    }

    class DeepSeekMoE {
        +ModuleList shared_experts
        +ModuleList routed_experts
        +GateRouter gate
        +forward(x)
    }

    Transformer *-- Block
    Block *-- CausalSelfAttention
    Block *-- DeepSeekMoE
    ModelConfig ..> Transformer
```
