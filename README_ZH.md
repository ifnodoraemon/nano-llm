<p align="center">
  <img src="assets/logo.png" width="350" alt="nano-llm logo"/>
</p>

# nano-llm: 极简 PyTorch 大语言模型预训练与对齐计算节点

<p align="center">
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://github.com/ifnodoraemon/nano-llm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"/></a>
  <a href="https://nvidia.com"><img src="https://img.shields.io/badge/Hardware-NVIDIA%20H800%20%7C%20A100-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="NVIDIA GPU"/></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="Black"/></a>
</p>

欢迎来到 **nano-llm**（现已升级 **nano-deepseek** 架构支持）！本仓库是一个纯净、极简且高度模块化的 PyTorch 全流程实现，涵盖现代大语言模型的预训练（Pre-training）、监督微调（SFT）、偏好对齐（DPO）以及百万级长上下文与 FP8 混合精度优化。专为 **8x80GB H800 GPU** 高性能集群设计。

阅读其他语言版本: [English](README.md) | [简体中文](README_ZH.md)

---

## 🎨 核心流水线与分布式架构

```mermaid
graph TD
    %% 数据准备阶段
    subgraph 数据准备 [数据清洗与预处理]
        A["原始语料 (TinyStories / WikiText / URLs)"] -->|download_dataset.py / crawl_data.py| B["净化文本语料 (.txt)"]
        B -->|deduplicate.py MinHash LSH| C["去重清洗数据集"]
        C -->|train_tokenizer.py BPE| D["定制 BPE 分词器 (custom_tokenizer.json)"]
        C & D -->|pack_binaries.py| E["内存映射二进制序列 (train.bin & val.bin)"]
    end

    %% 分布式训练阶段
    subgraph 训练中心 [分布式多卡 GPU 计算引擎]
        E -->|pretrain.py + FP8 + NTK-RoPE| F["预训练基座权重 (checkpoint_pretrain.pt)"]
        F -->|train.py + SFT 样本打包| G["微调策略权重 (checkpoint_sft.pt)"]
        G -->|align.py + DPO 偏好学习| H["偏好对齐策略 (checkpoint_dpo.pt)"]
        G -->|grpo.py + RLHF 推理对齐| I["推理策略权重 (checkpoint_grpo.pt)"]
    end

    %% 压缩与部署阶段
    subgraph 部署服务 [自动驾驶Serving服务]
        H -->|quantize.py 4-bit RTN量化| I["压缩版权重 (checkpoint_dpo_q4.pt)"]
        I -->|serve.py 静态KV缓存 + VLM| J["高吞吐流式 API 服务"]
        I -->|upload_hub.py 推送| K["Hugging Face Hub / ModelScope 社区"]
    end
    
    style E fill:#00f2fe,stroke:#000,stroke-width:2px,color:#000
    style H fill:#b927fc,stroke:#000,stroke-width:2px,color:#fff
    style J fill:#42e695,stroke:#000,stroke-width:2px,color:#000
```

---

## 📘 100% 视觉化架构设计蓝图

为了实现最直观的工程掌控力，我们将所有底层技术文档重构为**全视觉图表蓝图**，彻底移除冗长文本：

| 架构技术蓝图 | 图解展示的核心技术设计 |
| :--- | :--- |
| 👁️ **[DeepSeek MLA & DeepSeekMoE 架构蓝图](docs/deepseek_upgrade.md)** | • 标准 GQA 与 DeepSeek MLA 显存读写拓扑对比<br>• 共享专家与动态门控路由专家的细粒度融合流程<br>• `model.py` 中神经网络组件的类依赖结构图 |
| ⚡ **[注意力细节与旋转位置坐标优化蓝图](docs/attention_and_position.md)** | • Softmax 熵约束注意力锐化前后概率分布曲线对比<br>• NTK-Aware Positional 旋转相角微距拉伸原理<br>• Double-Grid 多网格超分辨率视觉投影器数据流<br>• Hard Negative 自我迭代时序校验与指令生成闭环 |
| 📈 **[1M 超长上下文工程蓝图](docs/context_engineering.md)** | • 动态 NTK 频率基底预计算流程时序<br>• 百万 Token 上下文下 GQA 与 MLA 显存开销对比<br>• 基于 Ring-Attention 的 8 卡 GPU 虚拟环通信拓扑<br>• 环形注意力异步计算与缓存传递状态机 |
| 🏎️ **[性能指标与自我迭代蓝图](docs/performance_benchmarks.md)** | • 激活检查点（Activation Checkpointing）显存均摊流<br>• PyTorch FSDP 权重与优化器参数分布式切分状态<br>• Fused AdamW 硬件 CUDA 单扫（Single-Sweep）提速机制<br>• 自动化 Self-Play DPO 自我进化与对齐回路流程<br>• 多榜单评测（MMLU, GSM8K, ARC 等）提取与评分流程 |
| 🛡️ **[LLM 训练风险与对齐规避蓝图](docs/risk_mitigation.md)** | • Loss 骤增与训练收敛崩盘的诱因与四重防御流图<br>• MoE 路由崩溃与专家闲置的动态门控均衡与共享专家保障<br>• DPO 偏好对齐奖励黑客与基于 Reference Anchor 的 KL 正则惩罚<br>• FP8 动态范围截断与溢出噪声的 Tensor-wise 动态比例映射 |
| 🐳 **[Docker Swarm 多机部署与调度蓝图](docs/swarm_multi_node_deployment.md)** | • Swarm 管理节点与工作节点分布式拓扑图<br>• 节点标签设置与绕过容器虚拟化的 Host 网络映射<br>• PyTorch `torchrun` 跨多机环境变量配置与容器启动时序 |

---

## 📂 项目代码结构

```mermaid
graph TD
    classDef file fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;
    classDef dir fill:#e2e2e2,stroke:#000,stroke-width:2px,color:#000;
    classDef root fill:#00f2fe,stroke:#005c66,stroke-width:2px,color:#000;

    Root["nano-llm /"]:::root
    
    %% 环境与配置
    Root --> DOCKER["Dockerfile / docker-compose.yml <br> (多卡 NVIDIA 容器直通环境)"]:::file
    Root --> SWARM["docker-compose-swarm.yml <br> (Docker Swarm 跨多机部署)"]:::file
    
    %% 模型内核
    Root --> MODEL["model.py <br> (MLA + MoE + FP8 + Checkpointing)"]:::file
    
    %% 主训练管线
    Root --> PRETRAIN["pretrain.py <br> (因果语言模型预训练循环)"]:::file
    Root --> TRAIN["train.py <br> (SFT 分布式多卡打包训练)"]:::file
    Root --> ALIGN["align.py <br> (DPO 分布式偏好对齐)"]:::file
    Root --> GRPO["grpo.py <br> (DeepSeek-R1 GRPO 强化学习对齐)"]:::file
    Root --> SERVE["serve.py <br> (流式高吞吐 VLM 静态缓存推理服务)"]:::file
    
    %% 辅助模块
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
    
    WEB --> W1["高端拟物玻璃化控制台前端 (FastAPI HUD)"]:::file
    
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

## 📐 数学算子在代码中的硬核实现

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

## 🚀 容器快速启动

### A. GPU 直通环境拉起

```mermaid
sequenceDiagram
    autonumber
    actor Operator as 算力管理员
    participant Compose as docker-compose.yml
    participant Docker as CUDA Docker 容器
    
    Operator->>Compose: docker compose up --build -d
    Compose->>Docker: 映射 host 卡直通及 16GB 共享显存 (shm_size)
    Docker-->>Operator: 自动激活 FastAPI HUD 控制台监听 http://localhost:8000
```

### B. 控制台自动驾驶流水线

```mermaid
sequenceDiagram
    autonumber
    actor User as 系统操作员
    participant BE as FastAPI 控制中心 (Container)
    participant HW as 8xH800 GPU 集群 (Host)
    
    User->>BE: 点击 “开始数据准备”
    BE->>BE: 自动执行 download_dataset.py / pack_binaries.py
    BE-->>User: 二进制映射文件打包完成 (train.bin)
    
    User->>BE: 点击 “启动预训练” / “启动微调”
    BE->>HW: 分配启动 torchrun pretrain.py --use_fp8 True
    HW-->>BE: 通过 WebSockets 实时推流标准输出与 MFU 计算指标
    BE-->>User: 动态 Loss 损失收敛动画呈现在 HUD 控制面板上
    
    User->>BE: 向终端发送推理请求 (文本 / 图像 URL)
    BE->>HW: 执行静态 KV-Cache 图像投影合并与 Prefill 首词计算
    HW-->>User: 通过 SSE 流式返回回复 Token (打印首词生成毫秒 TTFT)
```

---

## 🗺️ 项目技术路线图与重要里程碑 (Visual Roadmap)

```mermaid
graph TD
    classDef comp fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;
    classDef active fill:#00f2fe,stroke:#005c66,stroke-width:2px,color:#000;
    classDef planned fill:#e2e2e2,stroke:#555,stroke-width:1px,color:#777;

    C1["1. 定制 BPE 与二进制数据打包<br>(data.py / deduplicate.py)"]:::comp
    C2["2. DeepSeek MLA + DeepSeekMoE 架构<br>(model.py)"]:::comp
    C3["3. 动态 NTK-RoPE 1M 上下文 & FP8 计算<br>(pretrain.py / serve.py)"]:::comp
    C4["4. 拟物化反重力大模型控制台 HUD<br>(FastAPI 仪表盘)"]:::comp
    C5["5. DeepSeek-R1 风格的 RLHF 与 GRPO 强化学习对齐<br>(grpo.py)"]:::comp
    
    %% Planned milestones
    P1["6. 3D 并行分布式切分训练 <br>(TP + PP + DP 混合并行)"]:::active
    P2["7. 高吞吐静态量化推理内核 <br>(TensorRT-LLM / vLLM hooks)"]:::planned
    
    C1 --> C2 --> C3 --> C4 --> C5
    C5 -->|当前研发重心| P1
    P1 --> P2
```

---

## 💖 您的支持是我们开源的动力 (Show Your Support)

我们致力于打造全世界最纯净、最高性能、最易于学术与工程理解的纯 PyTorch 大模型预训练与对齐实现。如果您觉得这个项目对您有帮助，**请为我们点亮一颗 Star！** 您的支持是鼓励我们不断迭代和开源更强大功能的最大动力。⭐

[![Star History Chart](https://api.star-history.com/svg?repos=ifnodoraemon/nano-llm&type=Date)](https://github.com/ifnodoraemon/nano-llm)
