# nano-llm 项目全流程待办事项 (ToDo & Gaps Roadmap)

目标：打通预训练、微调(SFT)、对齐(DPO)以及强化学习(GRPO)全链路，打磨极速、高性能、多模态的顶级 1.5B 稠密基座模型。

---

## 🎯 1. 预训练阶段 (Pre-training Gaps & Monitoring)

### 1.1 监控 1.5B 预训练收敛 (P0 - 正在运行)
*   **任务 ID**：`task-2212` (目标 5000 步，当前进度 755+)。
*   **状态**：已解决 NVML 锁问题，单步耗时稳定在 **4.2 秒**，Loss 在 5.5 左右，收敛正常。
*   **待办**：持续跟进 Loss 收敛趋势与梯度稳定性，预计在 2026-06-01 23:50 完成。

### 1.2 修复 MFU (Model FLOPs Utilization) 计算错误 (P1)
*   **问题**：[pretrain.py](file:///home/ifnodoraemon/myagent/nano-llm/pretrain.py#L374) 在计算 `mfu_percentage` 时，直接将集群总 FLOPs/sec 除以了单卡 H800 的峰值性能 (`312e12`)，导致算出的 MFU 偏大 `ddp_world_size` (8) 倍。
*   **待办**：修改 MFU 计算公式，将分母更新为 `312e12 * ddp_world_size`（或直接使用 `calculate_mfu` 助手函数），保证在 DDP 多卡模式下精确报告单卡算力利用率（预计真实 MFU 约为 ~52.7%）。

### 1.3 引入预训练周期性验证机制 (P1)
*   **问题**：目前代码虽加载了验证集数据 `val_data`，但训练过程中完全没有运行验证循环（Validation Loop），无法定量追踪验证 Loss (val_loss)。
*   **待办**：在 [pretrain.py](file:///home/ifnodoraemon/myagent/nano-llm/pretrain.py) 中实现 `evaluate_val_loss`：
    *   每 500 步在验证集上采样 50 个 batches 计算平均 loss。
    *   在 DDP 模式下，通过 `dist.all_reduce` 跨卡聚合 val_loss，由 master 节点打印并上报至 `ExperimentTracker`。

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

## 💡 5. 进阶特性研发计划 (P2 - 后续迭代)

### 5.1 更好 (Better) —— 智能与推理极限
*   **GRPO 推理强化学习 (Reasoning RL)**：利用 [grpo/](file:///home/ifnodoraemon/myagent/nano-llm/grpo) 模块，通过数学/代码任务设计高保真规则编译器和格式奖励（Accuracy & Format Rewards），训练 1.5B 模型自主涌现类似 DeepSeek-R1 的 `<think>` 思考链（CoT）与自我纠错能力。
*   **高质量蒸馏数据混合 (Distillation)**：从顶级推理模型中提取推理轨迹，混合到微调数据中，提升小模型的逻辑泛化力。
*   **在线 DPO (On-policy DPO)**：在对齐阶段，将离线对比样本升级为在线即时采样，克服离线偏置。

### 5.2 更快 (Faster) —— 算力与速度极限
*   **生产级 REST API 流式输出**：实现 `serve/` 中 `/api/chat` 的正式流式输出（目前为 mock 逻辑）。
*   **自定义 Triton 算子优化**：编写定制的 Triton Flash-MLA 算子，在注意力层面融合低秩压缩和 FlashAttention，削减长上下文的访存带宽瓶颈。
*   **双模型投机采样 (Speculative Decoding)**：使用一个更小的 draft model（如 100M-200M），利用其轻量级 KV-cache 和 MLA 特性，加速 1.5B 目标模型的自回归生成，在无损精度下将推理吞吐提升 **2~3 倍**。
*   **FP8 混合精度推理量化**：支持在线 FP8 的权重与激活值（W8A8）量化部署，最大化压缩显存并饱和释放 H800 的 FP8 Tensor Core 吞吐性能。

### 5.3 更强 (Stronger) —— 视觉多模态与系统鲁棒性
*   **多模态视觉能力解锁 (VLM)**：将配置中的 `vision_dim`（目前预留了 `1152`）与 SigLIP 等预训练视觉编码器对齐。通过双阶段对齐（Projector 对齐 -> 图像-文本联合微调），将基座升级为具备高保真图文理解力的多模态模型，对标 DeepSeek-VL。
*   **DoRA 无损微调支持**：支持权重分解低秩适应（DoRA），为下游垂直行业提供更强、更稳定的参数高效微调。
