# Performance Benchmarks & Self-Iteration Optimization Blueprint

```mermaid
graph TB
    subgraph Legend
        VramNode[VRAM / Memory State]:::vram
        CpuNode[CPU / Host Action]:::cpu
        GpuNode[GPU Core Execution]:::gpu
        classDef vram fill:#ff4d4d,stroke:#990000,stroke-width:1.5px,color:#fff;
        classDef cpu fill:#ffaa00,stroke:#8c5d00,stroke-width:1.5px,color:#000;
        classDef gpu fill:#42e695,stroke:#005c1e,stroke-width:1.5px,color:#000;
    end
```

---

## ⚡ 1. Activation Checkpointing: VRAM Memory Profile

```mermaid
graph TD
    classDef baseline fill:#ff4d4d,stroke:#990000,stroke-width:1.5px,color:#fff;
    classDef optimized fill:#42e695,stroke:#005c1e,stroke-width:1.5px,color:#000;
    classDef checkpoint fill:#00f2fe,stroke:#005c66,stroke-width:2px,color:#000;

    subgraph Standard Forward / Backward Propagation
        H0["Input Tokens"] --> L1["Layer 1 Forward"]
        L1 --> L1_Act["Layer 1 Activations stored in VRAM"]:::baseline
        L1_Act --> L2["Layer 2 Forward"]
        L2 --> L2_Act["Layer 2 Activations stored in VRAM"]:::baseline
        L2_Act --> L3["Layer 3 Forward"]
        L3 --> L3_Act["Layer 3 Activations stored in VRAM"]:::baseline
        L3_Act --> Loss1["Loss Computation"]
        Loss1 --> L3_Grad["Layer 3 Backprop"]:::baseline
        L3_Grad --> L2_Grad["Layer 2 Backprop"]:::baseline
        L2_Grad --> L1_Grad["Layer 1 Backprop"]:::baseline
    end

    subgraph Optimized Activation Checkpointing
        H0_cp["Input Tokens"] --> L1_cp["Layer 1 Forward"]
        L1_cp --> C1[("Checkpoint 1: Layer 1 Stored")]:::checkpoint
        C1 -->|Skip intermediate storing| L2_cp["Layer 2 Forward"]
        L2_cp -->|Skip intermediate storing| L3_cp["Layer 3 Forward"]
        L3_cp --> C2[("Checkpoint 2: Layer 3 Stored")]:::checkpoint
        C2 --> Loss2["Loss Computation"]
        
        Loss2 --> L3_Grad_cp["Layer 3 Backprop <br> (Directly reads Checkpoint 2)"]:::optimized
        L3_Grad_cp --> Recompute["Dynamically Recompute Layer 2 Forward"]:::optimized
        Recompute --> L2_Grad_cp["Layer 2 Backprop"]:::optimized
        L2_Grad_cp --> L1_Grad_cp["Layer 1 Backprop <br> (Directly reads Checkpoint 1)"]:::optimized
    end
```

---

## 🔀 2. FSDP (Fully Sharded Data Parallel) Memory Allocation

```mermaid
graph LR
    classDef replica fill:#ff4d4d,stroke:#990000,stroke-width:1.5px,color:#fff;
    classDef shard fill:#42e695,stroke:#005c1e,stroke-width:1.5px,color:#000;

    subgraph Standard DDP (Model Replication on 8 GPUs)
        GPU1_ddp["GPU 1: Full Model [14GB] + Full Optimizer States [56GB]"]:::replica
        GPU2_ddp["GPU 2: Full Model [14GB] + Full Optimizer States [56GB]"]:::replica
        GPU8_ddp["GPU 8: Full Model [14GB] + Full Optimizer States [56GB]"]:::replica
    end

    subgraph PyTorch FSDP (Sharded Parameters & Optimizer)
        GPU1_fsdp["GPU 1: 1/8 Model Shard [1.7GB] + 1/8 Optimizer Shard [7GB]"]:::shard
        GPU2_fsdp["GPU 2: 1/8 Model Shard [1.7GB] + 1/8 Optimizer Shard [7GB]"]:::shard
        GPU8_fsdp["GPU 8: 1/8 Model Shard [1.7GB] + 1/8 Optimizer Shard [7GB]"]:::shard
    end
```

---

## 🏎️ 3. Hardware Optimization: Fused AdamW CUDA Sweeping

```mermaid
graph TD
    classDef memory fill:#ffaa00,stroke:#8c5d00,stroke-width:1.5px,color:#000;
    classDef kernel fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    classDef bad fill:#ff4d4d,stroke:#990000,stroke-width:1.5px,color:#fff;

    subgraph Standard AdamW (High Memory Traffic)
        Step1["Read weights, gradients, momentums from HBM to SRAM"]:::memory
        Step1 --> CalcMomentum["Compute First Moment (m_t)"]:::bad
        CalcMomentum --> WriteHBM1["Write m_t back to HBM"]:::memory
        WriteHBM1 --> Step2["Read m_t and weights from HBM to SRAM"]:::memory
        Step2 --> CalcVariance["Compute Second Moment (v_t)"]:::bad
        CalcVariance --> WriteHBM2["Write v_t back to HBM"]:::memory
        WriteHBM2 --> Step3["Read v_t, m_t, and weights from HBM to SRAM"]:::memory
        Step3 --> ApplyUpdate["Apply weight update step"]:::bad
        ApplyUpdate --> WriteHBM3["Write updated weights to HBM"]:::memory
    end

    subgraph Fused AdamW (Single-Sweep CUDA Kernel)
        Step1_f["Read weights, gradients, momentums, variances from HBM once"]:::memory
        Step1_f --> FusedKernel["Fused CUDA Kernel Sweep <br> (Compute m_t + v_t + Apply Update in GPU Registers)"]:::kernel
        FusedKernel --> WriteHBM_f["Write updated weights to HBM once"]:::memory
    end
```

---

## 🔄 4. Autonomous Model Self-Iteration Protocol

```mermaid
graph TD
    classDef active fill:#00f2fe,stroke:#005c66,stroke-width:1.5px,color:#000;
    classDef loopNode fill:#b927fc,stroke:#4a0072,stroke-width:2px,color:#fff;
    classDef codeNode fill:#42e695,stroke:#005c1e,stroke-width:1.5px,color:#000;

    Checkpoint0["Base DeepSeek MLA Checkpoint"]:::active
    
    %% Bootstrapping Data
    Checkpoint0 -->|1. Generate Prompt Seeds| PromptGen["Self-Instruction Prompt Engine"]:::active
    PromptGen -->|Prompt Seeds| RejectionSampling["Rejection Sampling (Generate y_A and y_B)"]:::active
    
    %% Critique
    RejectionSampling -->|Candidate Response Pairs| Judge["LLM-as-a-Judge / External Sandbox execution"]:::active
    Judge -->|Autofeedback Scores| Filter["Consensus Filter & Cleanse"]:::active
    
    %% Alignment
    Filter -->|Build train_dpo.jsonl| DpoDataset[("DPO Aligned Dataset (Chosen vs Rejected)")]:::loopNode
    DpoDataset -->|2. Execute align.py| DpoTrain["DPO Preference Optimization Loop"]:::codeNode
    DpoTrain -->|Update model weights| CheckpointUpgraded["Upgraded Checkpoint"]:::active
    
    CheckpointUpgraded -->|Self-Iteration Loop| Checkpoint0
```

---

## 📈 5. Multi-Benchmark Evaluation Logic

```mermaid
graph TD
    classDef benchmark fill:#e2e2e2,stroke:#000,stroke-width:1.5px,color:#000;
    classDef evaluator fill:#ffaa00,stroke:#8c5d00,stroke-width:1.5px,color:#000;
    classDef metrics fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;

    Model["Trained Checkpoint (model.py)"]
    
    %% Benchmarks
    Model --> MMLU["MMLU Benchmark"]:::benchmark
    Model --> GSM8K["GSM8K Benchmark"]:::benchmark
    Model --> ARC["ARC Benchmark"]:::benchmark
    Model --> HellaSwag["HellaSwag Benchmark"]:::benchmark
    Model --> HumanEval["HumanEval Benchmark"]:::benchmark
    Model --> MMMU["MMMU VLM Benchmark"]:::benchmark

    %% Internal logics
    MMLU --> MC_Logits["Extract MCQ Logits for keys (A, B, C, D) <br> P(opt) = Softmax(logits)"]:::evaluator
    GSM8K --> Greedy_Dec["Greedy Autoregressive Decoding"]:::evaluator
    ARC --> Conditional_Perp["Compute Sentence-level Likelihood <br> Log P(Completion | Prompt)"]:::evaluator
    HellaSwag --> Last_Token_Perp["Calculate average log-probability of last completion tokens"]:::evaluator
    HumanEval --> Sandbox_Run["Autoregressive Python Generation <br> + Sandbox execution validation"]:::evaluator
    MMMU --> Double_Grid["Pass Global + 4 patches through <br> Vision MLP Connector"]:::evaluator

    %% Results
    MC_Logits & Greedy_Dec & Conditional_Perp & Last_Token_Perp & Sandbox_Run & Double_Grid --> Leaderboard["Unified Leaderboard metrics compiled"]:::metrics
```
