# nano-llm 项目全流程待办事项 (ToDo & Gaps Roadmap)

目标：打通预训练、微调(SFT)、对齐(DPO)以及强化学习(GRPO)全链路，打磨极速、高性能、多模态的顶级 2B 稠密基座模型。

---

## 🎯 1. 预训练阶段 (Pre-training Gaps & Innovations)

### 1.1 监控 1.5B 预训练收敛 (P0 - 正在运行)
*   **任务 ID**：`task-2212` (目标 5000 步，当前进度 755+)。
*   **状态**：已解决 NVML 锁问题，单步耗时稳定在 **4.2 秒**，Loss 在 5.5 左右，收敛正常。
*   **待办**：持续跟进 Loss收敛趋势与梯度稳定性，预计在 2026-06-01 23:50 完成。

### 1.2 修复 MFU (Model FLOPs Utilization) 计算错误 (P1)
*   **问题**：[pretrain.py](file:///home/ifnodoraemon/myagent/nano-llm/pretrain.py#L374) 在计算 `mfu_percentage` 时，直接将集群总 FLOPs/sec 除以了单卡 H800 的峰值性能 (`312e12`)，导致算出的 MFU 偏大 `ddp_world_size` (8)倍。
*   **待办**：修改 MFU 计算公式，将分母更新为 `312e12 * ddp_world_size`（或直接使用 `calculate_mfu` 助手函数），保证在 DDP 多卡模式下精确报告单卡算力利用率（预计真实 MFU 约为 ~52.7%）。

### 1.3 引入预训练周期性验证机制 (P1)
*   **问题**：目前代码虽加载了验证集数据 `val_data`，但训练过程中完全没有运行验证循环（Validation Loop），无法定量追踪验证 Loss (val_loss)。
*   **待办**：在 [pretrain.py](file:///home/ifnodoraemon/myagent/nano-llm/pretrain.py) 中实现 `evaluate_val_loss`：
    *   每 500 步在验证集上采样 50 个 batches 计算平均 loss。
    *   在 DDP 模式下，通过 `dist.all_reduce` 跨卡聚合 val_loss，由 master 节点打印并上报至 `ExperimentTracker`。

### 1.4 💡 预训练算法与数据级硬核技术创新 (Pre-training Deep Innovations)
*   **1) GNS-Adaptive Batch Size Scheduling (基于梯度噪声强度的自适应 Batch Size 调度)**：
    *   *设计*：在线实时估算当前梯度的 Gradient Noise Scale (GNS)：$\text{GNS} = \text{tr}(\Sigma) / \|g\|^2$。在训练初期，GNS 较小，使用较小的 Batch Size（如 `--grad_accum_steps 1`）节省算力并快速发散；随着训练步数深入和梯度收敛，GNS 增大，自动成倍调大梯度累积步数以扩大全局 Batch Size。
    *   *目的*：从理论上将预训练收敛效率提升 1.5 ~ 2.0 倍，并在中后期有效抑止梯度发散。
*   **2) Benchmark Leakage Prevention & Semantic Blocking (语义指纹阻断防泄露引擎)**：
    *   *设计*：在 Dataloader 阶段集成一个语义指纹过滤器。将 MMLU、GSM8K、ARC 等评测基准的问题库在内存中构建为轻量级的局部敏感哈希（LSH）或 5-gram 索引。在载入预训练数据块时进行实时对比，一旦重合度触发阈值，直接对该块执行 semantic masking（将对应 labels 设为 -100），彻底阻断 benchmark 泄露。
    *   *目的*：确保基座模型在下游 benchmark 上的评估成绩为纯粹的 Zero-Shot 泛化，而非数据记忆。
*   **3) LG-Opt: Loss-Gradient Decoupled Rescaling (基于 Loss 偏导的梯度自适应重缩放)**：
    *   *设计*：在反向传播前，根据当前 batch 的 loss 偏离 EMA (指数移动平均) 历史 loss 的绝对偏差 $\delta = |\text{Loss} - \text{EMA\_Loss}|$，动态计算动态重缩放因子：
        *   若 $\delta > 3\sigma$（偏离过高，判定为二进制噪声或系统日志等脏数据），对该 Batch 的梯度乘 0.1 衰减权重，防止参数学坏；
        *   若 Loss 接近于 0 且 $\delta \approx 0$（判定为版权免责声明等高度冗余数据），进行轻微惩罚。
    *   *目的*：从算法底层提供预训练稳定性护栏，彻底摆脱对繁琐 Adam 超参数的经验调参依赖。
*   **4) Spectral Decoupled RoPE Base Scaling (谱分离位置编码基频预外推)**：
    *   *设计*：不同于 YaRN 等对称插值，在预训练阶段应用谱分离位置编码：
        *   对高频维度部分（代表短程局部注意力）保持基频 $\theta=10000$ 绝对不缩放，确保 1K 范围内的近距离高精度逻辑和结构感知；
        *   对中低频维度部分（代表长程语境）使用指数衰减因子进行插值外推。
    *   *目的*：实现在 4K 的常规预训练中，让模型直接具备外推 16K+ 长上下文的能力，且短文本性能零退化。
*   **5) Annealing CoT Injection (退火阶段逻辑夹带)**：
    *   *设计*：在余弦学习率衰减至 Min_LR 的退火（Annealing）阶段，混入 5% 的高质量多步推理合成数据，从而在预训练中就激活强大的结构化推理表达。

### 1.5 💡 对标 Qwen3-VL：预训练 Interleaved-MRoPE 架构集成 (P2)
*   **问题（老技术淘汰）**：传统多模态位置编码MRoPE虽然有时间（T）、高度（H）和宽度（W）的分立，但是分块进行的，容易造成视频长文本中的时空分辨率退化。
*   **待办**：在预训练底层实现 **Interleaved-MRoPE (交错式多模态旋转位置编码)**：
    *   将时空维度的旋转角度（$\theta_t, \theta_h, \theta_w$）在通道维度上进行交错式分布，确保每个注意力头的高低频频段同时感知 3D 关联，为后期直接具备原生 256K 超长视频理解打下物理位置表征底座。

### 1.6 💡 引入 2026 前沿：Muon 优化器与 Manifold-Constrained 信号流 (P2)
*   **Muon 正交梯度优化器集成 (Muon Optimizer)**：
    *   *设计*：用全新的 Muon 一阶优化器替代传统的 AdamW 来更新所有的 Transformer 权重矩阵。在每一步对梯度执行正交化（Orthogonalization），使更新几何分布正交。
    *   *目的*：**在相同的训练 Token 数量下，提升预训练收敛速度 2.5 ~ 3.0 倍**，大幅压缩算力开销。
*   **Manifold-Constrained Hyper-Connections (mHC, 流形约束超连接)**：
    *   *设计*：废弃传统的简单残差直连（Residual Connection），对多模态前向激活值引入流形约束映射。
    *   *目的*：在处理极长上下文（8K/16K）和大规模跨模态注入时，彻底稳定神经信号传播，防范深层表征退化。

---

## 📊 2. 评估系统升级 (Comprehensive Evaluation System Gaps)

### 2.1 引入真实评测集代替 Mock 数据 (P1)
*   **问题**：[eval_benchmarks.py](file:///home/ifnodoraemon/myagent/nano-llm/eval_benchmarks.py#L18) 中的 MMLU (3 题)、GSM8K (3 题) 和 Arena (6 题) 全是硬编码 Mock数据，属于“伪评估”，无法客观反映模型的真实智能水平。
*   **待办**：
    *   从本地加载或自动下载一个更具代表性的小型真实评估数据集（如 100 题 bilingual MMLU, 50 题 GSM8K）。
    *   支持 `eval_benchmarks.py` 动态加载本地 JSON/JSONL 评估文件，避免硬编码样本。

### 2.2 丰富多尺度大海捞针 (NIAH) 与长文本 Perplexity 测试 (P1)
*   **问题**：目前的 `NeedleInAHaystackEvaluator` 长度限制为 `[1024, 2048, 4096]`，无法直接用于评估 8K/16K 长上下文模型。
*   **待办**：
    *   修改脚本支持参数化长度测试（可配置为 `8192` 和 `16384`）。
    *   增加长文本 Perplexity (PPL) 递进测试，追踪模型在上下文拉长时，Attention 分布与交叉熵损失的演变情况。

### 2.3 💡 工具调用与长任务轨迹级评估升级 (Agent & Tool-Use Evaluation) (P2)
*   **问题**：当前评估系统仅评估最终答案（Outcome-based），缺乏对工具调用（Tool Use）和多步长任务规划能力的诊断，无法评估模型在遇到 API 报错等真实噪声时的容错自愈能力。
*   **待办**：
    *   **Trace-Level Action Match (轨迹级动作匹配评测)**：实现工具调用轨迹对比引擎，不仅评估最终输出，还要比对中间生成的 Tool Name、JSON 参数的准确度（F1-Score），精确评估中间规划是否有偏离或幻觉参数。
    *   **Interactive Sandbox Environment (交互式沙盒与报错注入测试)**：构建沙盒交互测试集（模拟 API 查询），在评估时随机注入 API 报错噪声（如 `Rate Limit` 或 `Invalid Date`），检测模型在长程任务中是否能够感知错误并执行 Self-Correction 自我修复，最终达成全局目标。

---

## 🔧 3. SFT / DPO / GRPO 流程诊断与优化 (Fine-tuning & Alignment Gaps)

### 3.1 消除 SFT 路径硬编码并建立 Validation Loop (P1)
*   **问题**：[train.py](file:///home/ifnodoraemon/myagent/nano-llm/train.py#L103) 中加载 pretrain pt 文件时，路径 `./outputs/checkpoint_pretrain.pt` 被写死，忽略了 `--model_name_or_path` 参数；且没有划分验证集来计算 val loss。
*   **待办**：
    *   重构 `train.py` 使用 `--model_name_or_path` 作为基座模型加载路径。
    *   在 `train.py` 载入数据时划分 10% 作为验证集，在 Epoch 结束或每 100 步评估一次 Validation Loss，监控过拟合。

### 3.2 建立 DPO 验证机制与 Tokenizer 动态适配 (P1)
*   **问题**：DPO 对齐阶段极易产生过拟合或“模式崩溃”，目前 [align.py](file:///home/ifnodoraemon/myagent/nano-llm/align.py) 没有任何验证逻辑，且 Tokenizer 标识硬编码为远程 `Qwen/Qwen2.5-7B`。
*   **待办**：
    *   划分独立的 Chosen-Rejected 验证对，每 50 步评估一次 Validation DPO Loss 和 Chosen vs Rejected Accuracy。
    *   使 Tokenizer 加载路径与 `--sft_checkpoint_path` / `--model_name_or_path` 动态绑定。

### 3.3 重构 GRPO 相对优势计算与在线 Evaluation (P1)
*   **问题**：
    *   [grpo.py](file:///home/ifnodoraemon/myagent/nano-llm/grpo.py#L226) 中 `advantages` 计算在局部做了一次 EMA 归一化后，又立刻算了一遍组内均值方差进行了二次归一化。在数学上，这直接抵消并消除了 `GRPORewardScaler` 的 EMA 均值/方差作用，使其退化为局部归一化。
    *   缺少对强化学习的实时监控（如防止 Reward Hacking，防止模型为刷分堆砌无意义的符号）。
*   **待办**：
    *   重构优势度归一化代码，去除不必要的双重归一化；在 DDP 模式下，支持跨卡搜集 rewards (`dist.all_gather`) 计算全局 Group Advantage。
    *   增加在线 Evaluation 阶段：每 50 步使用 20 个独立验证 prompt，评估平均生成长度、正确率、及 Sandbox 编译率，提供安全防崩塌护栏。

### 3.4 💡 代理型对齐：步进奖励与约束规划 GRPO (Agentic GRPO & Constrained Alignment) (P2)
*   **问题**：在工具调用和长程任务中，单纯依靠最终结果打分存在严重的“奖励稀疏（Sparse Reward）”问题；同时模型缺乏“预算/限制”意识，容易在多步调用中产生死循环。
*   **待办**：
    *   **Process-Supervised Step Reward (PRM 步进奖励)**：对 GRPO 训练引入多步过程监督。对于生成中途的每一个 `tool_call` 执行正则与结构化参数检验，对格式正确且逻辑连续的动作实时给予小额正向步进奖励（Step Reward $\gamma^t R_{step}$），加速复杂长任务的收敛。
    *   **Budget-Constrained Penalty (预算约束惩罚机制)**：在 Prompt 中混入显式的全局约束条件（如：“限定 5 步交互” 或 “费用预算限制”），一旦模型在 GRPO 的采样轨迹中出现调用超频、死循环或超预算，给予惩罚性负奖励（如 $-3.0$），强迫模型学会“剪枝规划”与“最小代价求解”。

### 3.5 💥 消除数据打包注意力交叉污染 (Solve Sequence Packing Mask Leakage) (P1)
*   **问题**：[data.py](file:///home/ifnodoraemon/myagent/nano-llm/data.py#L93) 的 `SequencePackingCollator` 中将多个独立会话打包进同一个长度为 `max_length` 的 sequence。但在 [model/__init__.py](file:///home/ifnodoraemon/myagent/nano-llm/model/__init__.py#L393) 的 forward 循环中，调用 `layer` 时将 `mask` 强制写死为了 `None`，导致注意力退化为全局的下三角 Causal Mask。这导致 packed sequence 中的后一会话在计算自注意力时，能完全看到前一会话的上下文，产生严重的跨样本注意力交叉泄露（Attention Cross-talk）与逻辑污染。
*   **待办**：
    *   重构 `SequencePackingCollator` 输出会话的 `cumulative_seqlens` 边界信息。
    *   在 `train.py` 中构造 **Block-Diagonal Causal Attention Mask (块对角线因果掩码)**，对于不同会话之间的 token 交叉将注意力置为 $-\infty$。
    *   修改 [model/__init__.py](file:///home/ifnodoraemon/myagent/nano-llm/model/__init__.py)，使 `Transformer.forward` 接受外接 `mask` 参数并透传至各层注意力机制。

### 3.6 补齐多模态 (VLM) 视觉对齐全链路训练入口 (Unlock Multimodal Alignment Entry) (P2)
*   **问题**：虽然 `data.py` 实现了 `MultimodalSFTDataset`，且模型底座包含了 `vision_projection`。但在 SFT、DPO 和 GRPO 的训练脚本中完全缺失多模态数据的调用开关，导致模型无法实际参与视觉对齐训练。
*   **待办**：在 `train.py` / `align.py` 等脚本中补齐 `--use_multimodal` 开关，动态替换为多模态 Dataset 与 Collator，并将提取的 `pixel_values` 传入模型 forward 接口。

---

## 🚀 4. 长上下文扩展微调计划 (P1 - 预训练后)

### 4.1 批大小与分布式累加步数调整 (防范 OOM)
*   **8K 上下文 (8192)**：配置 `--block_size 8192 --batch_size 8 --grad_accum_steps 2`（全局 Batch Size = 128）。
*   **16K 上下文 (16384)**：配置 `--block_size 16384 --batch_size 4 --grad_accum_steps 4`（全局 Batch Size = 128）。

### 4.2 位置编码外推与注意力分布锐化
*   **NTK-Aware RoPE 缩放**：在 [precompute_freqs_cis](file:///home/ifnodoraemon/myagent/nano-llm/model/__init__.py#L90) 中，根据 args.block_size 对基频进行动态外推缩放：`base_theta = base * (scaling_factor ** (dim / (dim - 2)))`。
*   **注意力分布锐化**：开启自适应 logits 锐化 [attn_scale_multiplier=1.2](file:///home/ifnodoraemon/myagent/nano-llm/model/config.py#L28)，对抗 Softmax 的熵增，使小模型能紧锁远端信息。

### 4.3 数据混合配比设计
*   在 `data_mixer` 中配入 **30% 长文档问答 + 20% 随机合成“大海捞针”检索样本**，打破小模型的近因偏差，使其学会全局检索。

---

## 💡 5. 业界领先模型对标与后训练技术创新计划 (P2 - 后续迭代)

为了在 2B 稠密模型级别上超越 Qwen3-VL、MiniCPM 和 DeepSeek-R1-Distill 等顶尖模型，我们规划了以下后训练技术创新改造路线：

### 5.1 创新方向一：Debated-GRPO (辩论与对抗纠错强化学习)
*   **设计**：
    *   **Peer-Review GRPO (对等评审机制)**：将采样得到的 $G$ 个 completions 划分为“生成器”与“评审员”。评审员接受生成器输出，在其内部的 `<think>` 中纠正逻辑漏洞，以纠错后的最终表现作为 Reward。
    *   **Mutual Information Reward (互信息约束)**：引入多样性与熵奖励，惩罚在 `<think>` block 内大量堆砌符号、空格和行循环的行为。
    *   **Cold-Start Reasoner CoT SFT (冷启动前置引导)**：在 GRPO 前混入约 1k 条带推理轨迹的高质量数学与代码 SFT 样本，引导小模型快速收敛。

### 5.2 创新方向二：Hybrid Sparse-MLA (混合稀疏-低秩注意力)
*   **设计**：
    *   **Sliding Window MLA (滑动窗口低秩注意力)**：在 Transformer 的前半部分（低层与中层）使用局部滑动窗口注意力（如 window_size=2048），使计算和 KV 缓存复杂度降为 $O(S)$。
    *   **Global MLA**：仅在 Transformer 的后半部分（高层）保留全局 MLA，用以捕获跨序列的长程检索特征，在保持“大海捞针”精确度的同时减少 60% 算力开销。

### 5.3 创新方向三：Spatial-RoPE & Pixel-Shuffle (视觉多模态压缩与定位)
*   **设计**：
    *   **Pixel-Shuffle Downsampler (像素洗牌下采样器)**：用逆向 Pixel-Shuffle 算子对视觉 tokens 进行 $2 \times 2$ 空间拼入通道，将 token 数量从 1024 降为 256，节省 75% 序列长度。
    *   **2D Decoupled RoPE (2D 分解 RoPE)**：在视觉 Transformer 和 LLM 的多模态前几层应用 2D 旋转位置编码，将维度对半应用 $X$ 和 $Y$ 轴旋转：
        $$\mathbf{R}_{2D}(x, y) = \mathbf{R}_{1D}(x) \oplus \mathbf{R}_{1D}(y)$$
        保留图像的高清空间位置敏感度。

### 5.4 创新方向四：Logits Softcapping 与 FP8 精度护栏 (FP8 Safeguard)
*   **设计**：
    *   **Attention Logits Softcapping (软截断)**：在 Softmax 之前引入 tanh 运算，将 Scores 限制在 `[-cap, +cap]` 区间：
        $$\text{Scores} = \text{cap} \cdot \tanh\left(\frac{Q K^T}{\sqrt{d} \cdot \text{cap}}\right)$$
        稳定注意力权重，防止梯度爆炸。
    *   **FP8 Safeguard Auto-Scaler (动态校准器)**：在训练期间每 100 步对权重和激活分布进行动态统计，更新 FP8 的 Scale Factor，保障低比特运算精度。

### 5.5 💡 对标 Qwen3-VL：DeepStack 多层级特征级联融合与非因果图像掩码 (P2)
*   **设计**：
    *   **DeepStack Integration (深度层叠级联融合)**：废弃单纯的 Input 拼接投影。在前段多层（如第 2, 4, 6 层）设置级联层，将 Vision Transformer (ViT) 不同层级的隐层特征跨层直接融合注入，加深低阶特征与文字的强绑定。
    *   **Non-Causal Image Masking (非因果图像自注意力掩码)**：在 LLM 的 Attention 矩阵中，允许图像 tokens 之间进行无方向的双向注意力交互（Non-Causal），仅对文本部分应用 Causal Mask，极大释放图像空间物理定位理解精度。

### 5.6 💡 对标 DeepSeek-VL2：Dynamic Tiling 动态图块切片与 Global-Local 混合注意力 (P2)
*   **设计**：
    *   **Dynamic Tiling Strategy (动态图块切片)**：输入图像首先进行等比缩放。对于高分辨率大图，自适应切割为多个 $384 \times 384$ 的局部 Tiles，同时生成一张全局 Thumbnail 缩略图 Tile，共同输入同一 Vision Encoder，以保留全局上下文和局部的极细微细节。
    *   **Global-Local Hybrid Attention (全局-局部混合注意力)**：将局部的 tiles 语义向量和全局缩略图向量以并联形式映射为 visual tokens，在 LLM 侧建立针对不同 tiles 区域的空间路由机制，实现对长 PDF 和复杂图表的无损精确 OCR 与语义问答。

### 5.7 💡 对标 DeepSeek 2026《Thinking with Visual Primitives》：基于空间标记思考 (Thinking with Points & Bounding Boxes) (P2)
*   **背景 (指代鸿沟)**：DeepSeek 最新多模态研究提出，模型常面临“指代鸿沟（Reference Gap）”。语言无法精准描述连续空间，导致密集计数（Pixmo-Count）、迷宫导航（DS_Maze_Navigation）和路径追踪（DS_Path_Tracing）在纯语言 CoT 时完全崩溃。
*   **设计**：
    *   **Thinking with Visual Primitives (基于空间标记思考)**：将空间标记 —— 代表位置的 **Point (点)** 与代表范围的 **Bounding Box (边界框)** 提升为“思考的最小单元”。训练模型在 `<think>` 思考链中交替输出点 `&lt;｜point｜&gt;[[y, x]]&lt;｜/point｜&gt;` 和框 `&lt;｜box｜&gt;[[y1, x1, y2, x2]]&lt;｜/box｜&gt;`，把语言逻辑死死锚定在物理像素坐标上。
    *   **训练流水线与 RL 引导**：先在 Web 抓取的 bbox 框数据集上进行专家 SFT（因为框的标注是确定性的，信息丰满，更易学习），随后在后训练阶段通过强化学习（RL）引导模型学习 point。对迷宫导航等设计细粒度的 RL 奖励（路径覆盖度、回溯探索完整度、避障墙壁判定率），以形成自愈寻路闭环。
    *   **7056倍像素超强压缩 (极致 token 效率)**：在 Vision Encoder 出口执行 3×3 空间降采样（每 9 个 patch 合成 1 个），送入 LLM 前再次利用 CSA (压缩稀疏注意力) 将 KV cache 压缩 4 倍，将 756×756 图像最终压缩至 **81 个 KV 条目**，以 1/8 的 token 占用跑赢 GPT-5.4 与 Claude 4.6 级别的长上下文视觉推理。

---

## 🚀 6. 其它系统级与速度优化 (P2 - 推理加速)

*   **生产级 REST API 流式输出**：实现 `serve/` 中 `/api/chat` 的正式流式输出。
*   **双模型投机采样 (Speculative Decoding)**：设计 100M 级的 draft model，通过共享 KV Cache 和投机采样，将 1.5B 模型的推理吞吐提升 **2~3 倍**。
*   **自定义 Triton Flash-MLA 算子**：开发专用的 Triton MLA 算子，融合低秩投影和 Attention 操作，最大化减少 VRAM 带宽瓶颈。
