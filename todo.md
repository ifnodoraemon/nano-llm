# nano-llm 项目全流程待办事项 (ToDo & Gaps Roadmap)

目标：打通预训练、微调(SFT)、对齐(DPO)以及强化学习(GRPO)全链路，打磨极速、高性能、多模态的顶级 1.5B 稠密基座模型。

---

## 🎯 1. 预训练阶段 (Pre-training Gaps & Innovations)

### 1.1 监控 1.5B 预训练收敛 (P0 - 正在运行)
*   **任务 ID**：`task-2212` (目标 5000 步，当前进度 755+)。
*   **状态**：已解决 NVML 锁问题，单步耗时稳定在 **4.2 秒**，Loss 在 5.5 左右，收敛正常。
*   **待办**：持续跟进 Loss 收敛趋势与梯度稳定性，预计在 2026-06-01 23:50 完成。

### 1.2 修复 MFU (Model FLOPs Utilization) 计算错误 (P1)
*   **问题**：[pretrain.py](file:///home/ifnodoraemon/myagent/nano-llm/pretrain.py#L374) 在计算 `mfu_percentage` 时，直接将集群总 FLOPs/sec 除以了单卡 H800 的峰值性能 (`312e12`)，导致算出的 MFU 偏大 `ddp_world_size` (8)倍。
*   **待办**：修改 MFU 计算公式，将分母更新为 `312e12 * ddp_world_size`（或直接使用 `calculate_mfu` 助手函数），保证在 DDP 多卡模式下精确报告单卡算力利用率（预计真实 MFU 约为 ~52.7%）。

### 1.3 引入预训练周期性验证机制 (P1)
*   **问题**：目前代码虽加载了验证集数据 `val_data`，但训练过程中完全没有运行验证循环（Validation Loop），无法定量追踪验证 Loss (val_loss)。
*   **待办**：在 [pretrain.py](file:///home/ifnodoraemon/myagent/nano-llm/pretrain.py) 中实现 `evaluate_val_loss`：
    *   每 500 步在验证集上采样 50 个 batches 计算平均 loss。
    *   在 DDP 模式下，通过 `dist.all_reduce` 跨卡聚合 val_loss，由 master 节点打印并上报至 `ExperimentTracker`。

### 1.4 💡 预训练算法与数据级技术创新 (Pre-training Innovations) (P2)
为了全面提升 1.5B 稠密基座模型的无监督压缩上限、抗遗忘能力和推理先验，我们规划了以下预训练创新：
*   **1) 预训练退火阶段逻辑夹带 (Annealing CoT Injection)**：
    *   *设计*：在预训练的 Cosine LR 余弦衰减（Decay）阶段，模型的参数可塑性极高。我们在最后 10% 的退火阶段混入 5% 的高质量合成长推理（CoT）轨迹数据与代码执行树数据。
    *   *目的*：无需等待后训练，直接在预训练基座中前置注入逻辑思维链结构表征，提升 zero-shot 推理倾向。
*   **2) 动态课程数据混合 (Curriculum Data Mixing)**：
    *   *设计*：摒弃静态数据比例。前期以高比例的百科、代码语法等低熵高事实数据建立基础语法网络；中期逐步提升复杂代码、数理逻辑和长文档占比，强迫模型提取结构化关联；后期执行热力退火。
*   **3) Token-Level Loss 差异化在线噪声过滤 (Loss-Based Filtering)**：
    *   *设计*：在 DataLoader 批次分发中，对每一个 micro-batch 进行 loss 初探。如果当前 loss 极低（说明为极度重复的无效广告/协议模板）或 loss 异常高（说明是噪声/乱码），则动态收缩该 batch 的梯度更新权重，节省 15% 的无效算力。
*   **4) 随机深度预训练 (Pre-training Stochastic Depth)**：
    *   *设计*：在 1.5B 小模型的 32 层网络中引入随机层级丢弃（Stochastic Layer Dropout）。在训练过程中以特定概率 $p(l)$ 随机跳过部分 Transformer Block（通过残差直连），随着训练步数逐步降低丢弃率。
    *   *目的*：强迫模型各层学习到更独立、解耦的特征，表现为网络集成（Ensemble）效应，大幅提升模型在后训练微调时的抗过拟合与抗灾难性遗忘能力。

---

## 📊 2. 评估系统升级 (Comprehensive Evaluation System Gaps)

### 2.1 引入真实评测集代替 Mock 数据 (P1)
*   **问题**：[eval_benchmarks.py](file:///home/ifnodoraemon/myagent/nano-llm/eval_benchmarks.py#L18) 中的 MMLU (3 题)、GSM8K (3 题) 和 Arena (6 题) 全是硬编码 Mock 数据，属于“伪评估”，无法客观反映模型的真实智能水平。
*   **待办**：
    *   从本地加载或自动下载一个更具代表性的小型真实评估数据集（如 100 题 bilingual MMLU, 50 题 GSM8K）。
    *   支持 `eval_benchmarks.py` 动态加载本地 JSON/JSONL 评估文件，避免硬编码样本。

### 2.2 丰富多尺度大海捞针 (NIAH) 与长文本 Perplexity 测试 (P1)
*   **问题**：目前的 `NeedleInAHaystackEvaluator` 长度限制为 `[1024, 2048, 4096]`，无法直接用于评估 8K/16K 长上下文模型。
*   **待办**：
    *   修改脚本支持参数化长度测试（可配置为 `8192` 和 `16384`）。
    *   增加长文本 Perplexity (PPL) 递进测试，追踪模型在上下文拉长时，Attention 分布与交叉熵损失的演变情况。

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

为了在 1.5B 稠密模型级别上超越 Qwen2.5、MiniCPM 和 DeepSeek-R1-Distill 等顶尖模型，我们规划了以下四大后训练技术创新改造路线：

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

---

## 🚀 6. 其它系统级与速度优化 (P2 - 推理加速)

*   **生产级 REST API 流式输出**：实现 `serve/` 中 `/api/chat` 的正式流式输出。
*   **双模型投机采样 (Speculative Decoding)**：设计 100M 级的 draft model，通过共享 KV Cache 和投机采样，将 1.5B 模型的推理吞吐提升 **2~3 倍**。
*   **自定义 Triton Flash-MLA 算子**：开发专用的 Triton MLA 算子，融合低秩投影和 Attention 操作，最大化减少 VRAM 带宽瓶颈。
