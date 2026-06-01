# nano-llm 项目待办事项

目标：模型训练全链路打通，极速、智能、多模态，打造专属的顶级 1.5B 稠密基座模型。

---

## 🎯 当前预训练监控与评估

### 1. 监控 1.5B 预训练收敛 (P0 - 正在运行)
* **任务 ID**：`task-2212` (从第 700 步弹性断点恢复，目标 5000 步)。
* **状态**：集成了 NVML 系统遥测缓存（5秒缓存 + 仅查询 GPU 0），单步耗时稳定在 **4.2 秒**，无任何卡顿。
* **待办**：持续跟进 Loss 收敛趋势（当前在 5.7 左右），监控是否出现梯度异常。预计在 2026-06-01 23:50 左右完成。

### 2. 开展下游基准测试评估 (P1 - 预训练后)
* **待办**：预训练完成后，在最终的 `checkpoint_pretrain.pt` 上执行 downstream evaluation 脚本（例如 `eval_benchmarks.py`），收集 MMLU、GSM8K、PPL 和 NIAH (大海捞针) 基础表现，输出评估报告。

---

## 🚀 长上下文扩展微调计划 (P1 - 预训练后)

### 1. 批大小与分布式累加步数调整 (防范 OOM)
* **8K 上下文 (8192)**：配置 `--block_size 8192 --batch_size 8 --grad_accum_steps 2`（全局 Batch Size = 128）。
* **16K 上下文 (16384)**：配置 `--block_size 16384 --batch_size 4 --grad_accum_steps 4`（全局 Batch Size = 128）。

### 2. 位置编码外推与注意力分布锐化
* **NTK-Aware RoPE 缩放**：在 [pretrain.py](file:///home/ifnodoraemon/myagent/nano-llm/pretrain.py) 中，动态根据序列长度计算 RoPE 扩展乘子：`rope_scaling = max(1.0, args.block_size / 4096.0)`，对 [precompute_freqs_cis](file:///home/ifnodoraemon/myagent/nano-llm/model/__init__.py#L90) 进行基频缩放。
* **注意力分布锐化**：开启自适应 logits 锐化 [attn_scale_multiplier=1.2](file:///home/ifnodoraemon/myagent/nano-llm/model/config.py#L28)，对抗 Softmax 的熵增，使小模型能紧锁远端信息。

### 3. 数据混合配比设计
* 在 `data_mixer` 中配入 **30% 长文档问答 + 20% 随机合成“大海捞针”检索样本**，打破小模型的近因偏差，使其学会全局检索。

---

## 💡 进阶特性研发计划 (P2 - 后续迭代)

### 1. 更好 (Better) —— 智能与推理极限
* [ ] **GRPO 推理强化学习 (Reasoning RL)**：利用 [grpo/](file:///home/ifnodoraemon/myagent/nano-llm/grpo) 模块，通过数学/代码任务设计高保真规则编译器和格式奖励（Accuracy & Format Rewards），训练 1.5B 模型自主涌现类似 DeepSeek-R1 的 `<think>` 思考链（CoT）与自我纠错能力。
* [ ] **高质量蒸馏数据混合 (Distillation)**：从顶级推理模型中提取推理轨迹，混合到微调数据中，提升小模型的逻辑泛化力。
* [ ] **在线 DPO (On-policy DPO)**：在对齐阶段，将离线对比样本升级为在线即时采样，克服离线偏置。

### 2. 更快 (Faster) —— 算力与速度极限
* [ ] **生产级 REST API 流式输出**：实现 `serve/` 中 `/api/chat` 的正式流式输出（目前为 mock 逻辑）。
* [ ] **自定义 Triton 算子优化**：编写定制的 Triton Flash-MLA 算子，在注意力层面融合低秩压缩和 FlashAttention，削减长上下文的访存带宽瓶颈。
* [ ] **双模型投机采样 (Speculative Decoding)**：使用一个更小的 draft model（如 100M-200M），利用其轻量级 KV-cache 和 MLA 特性，加速 1.5B 目标模型的自回归生成，在无损精度下将推理吞吐提升 **2~3 倍**。
* [ ] **FP8 混合精度推理量化**：支持在线 FP8 的权重与激活值（W8A8）量化部署，最大化压缩显存并饱和释放 H800 的 FP8 Tensor Core 吞吐性能。

### 3. 更强 (Stronger) —— 视觉多模态与系统鲁棒性
* [ ] **多模态视觉能力解锁 (VLM)**：将配置中的 `vision_dim`（目前预留了 `1152`）与 SigLIP 等预训练视觉编码器对齐。通过双阶段对齐（Projector 对齐 -> 图像-文本联合微调），将基座升级为具备高保真图文理解力的多模态模型，对标 DeepSeek-VL。
* [ ] **DoRA 无损微调支持**：支持权重分解低秩适应（DoRA），为下游垂直行业提供更强、更稳定的参数高效微调。
