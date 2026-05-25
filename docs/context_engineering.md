# 1M Long-Context Engineering Blueprint

```mermaid
graph TB
    subgraph Legend
        DataNode[Data Buffer / Tensor]:::data
        CommNode[NCCL Communication]:::comm
        CalcNode[Compute Kernel / Operation]:::calc
        classDef data fill:#00f2fe,stroke:#005c66,stroke-width:1.5px,color:#000;
        classDef comm fill:#b927fc,stroke:#4a0072,stroke-width:2px,color:#fff;
        classDef calc fill:#42e695,stroke:#005c1e,stroke-width:1.5px,color:#000;
    end
```

---

## 📈 1. Dynamic NTK-Aware RoPE Scaling Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Input as Sequence Tokens [Batch, SeqLen]
    participant Pos as Position Indices
    participant Phase as Phase precompute_freqs_cis
    participant RoPE as Rotary Embedding Application
    
    Input->>Pos: Generate indices [0, 1, ..., S]
    alt SeqLen <= base_block_size (4K)
        Pos->>Phase: Compute standard phase angles using theta = 10,000
    else SeqLen > base_block_size (up to 1M)
        Pos->>Phase: Compute dynamically scaled base theta_scaled = theta * (scale_factor) ^ (d/(d-2))
    end
    Phase-->>RoPE: Precomputed complex rotary vectors (freqs_cis)
    RoPE->>RoPE: Broadcast and rotate Query/Key dimensions element-wise
```

---

## 💾 2. KV-Cache Memory Footprint: GQA vs. DeepSeek MLA

```mermaid
graph TD
    classDef baseline fill:#ff4d4d,stroke:#990000,stroke-width:2px,color:#fff;
    classDef optimized fill:#42e695,stroke:#005c1e,stroke-width:2.5px,color:#000;
    classDef nodeStyle fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;

    1M_Context["1,000,000 Token Context Memory Profile"]:::nodeStyle
    
    %% Baseline GQA
    1M_Context --> GQA["Grouped Query Attention (Standard GQA)"]:::baseline
    GQA --> GQA_Formula["VRAM = 2 * Layers (32) * KV_Heads (8) * Head_Dim (128) * 1,000,000"]:::nodeStyle
    GQA_Formula --> GQA_Total["32.7 GB VRAM per sequence <br> (Immediate Out-Of-Memory!)"]:::baseline
    
    %% Optimized MLA
    1M_Context --> MLA["Multi-Head Latent Attention (DeepSeek MLA)"]:::optimized
    MLA --> MLA_Formula["VRAM = Layers (32) * Latent_Dim (128) * 1,000,000"]:::nodeStyle
    MLA_Formula --> MLA_Total["2.29 GB VRAM per sequence <br> (93% KV-Cache compression)"]:::optimized
```

---

## 🔀 3. Context Parallelism (Ring-Attention Virtual GPU Ring)

During the multi-GPU attention forward pass, the 1M token sequence is distributed across all 8 GPUs ($125,000$ tokens per GPU). The GPUs continuously stream Keys and Values in a circular ring topology using async NCCL P2P operations:

```mermaid
graph LR
    classDef gpu fill:#e2e2e2,stroke:#000,stroke-width:2px,color:#000;
    classDef p2p fill:#b927fc,stroke:#4a0072,stroke-width:2.5px,color:#fff;

    subgraph 8-GPU Host Node Communication Rings
        GPU1["GPU 1 <br> (Tokens 0 - 125K) <br> Q_1 / K_1 / V_1"]:::gpu
        GPU2["GPU 2 <br> (Tokens 125K - 250K) <br> Q_2 / K_2 / V_2"]:::gpu
        GPU3["GPU 3 <br> (Tokens 250K - 375K) <br> Q_3 / K_3 / V_3"]:::gpu
        GPU4["GPU 4 <br> (Tokens 375K - 500K) <br> Q_4 / K_4 / V_4"]:::gpu
        GPU5["GPU 5 <br> (Tokens 500K - 625K) <br> Q_5 / K_5 / V_5"]:::gpu
        GPU6["GPU 6 <br> (Tokens 625K - 750K) <br> Q_6 / K_6 / V_6"]:::gpu
        GPU7["GPU 7 <br> (Tokens 750K - 875K) <br> Q_7 / K_7 / V_7"]:::gpu
        GPU8["GPU 8 <br> (Tokens 875K - 1M) <br> Q_8 / K_8 / V_8"]:::gpu
    end

    %% P2P Streams
    GPU1 -->|NCCL send K_1/V_1| GPU2:::p2p
    GPU2 -->|NCCL send K_2/V_2| GPU3:::p2p
    GPU3 -->|NCCL send K_3/V_3| GPU4:::p2p
    GPU4 -->|NCCL send K_4/V_4| GPU5:::p2p
    GPU5 -->|NCCL send K_5/V_5| GPU6:::p2p
    GPU6 -->|NCCL send K_6/V_6| GPU7:::p2p
    GPU7 -->|NCCL send K_7/V_7| GPU8:::p2p
    GPU8 -->|NCCL send K_8/V_8| GPU1:::p2p
```

### The Ring Attention Computational Steps (Asynchronous Loop)

```mermaid
stateDiagram-v2
    [*] --> Stage0: Local Chunk MatMul Q_i * K_i^T
    
    state RingLoop {
        Stage0 --> SendKV: Ring Send KV tensor slice (i) to Ring Next (i+1)
        SendKV --> RecvKV: Async Receive KV tensor slice (i-1) from Ring Prev (i-1)
        RecvKV --> ComputAttn: Compute attention on current Q_i and received KV slice
        ComputAttn --> Accumulate: Accumulate local attention weights & logits
        Accumulate --> SendKV: Loop for all 8 virtual steps
    }
    
    RingLoop --> Finish: Local output tensors compiled & softmax normalized
    Finish --> [*]
```
