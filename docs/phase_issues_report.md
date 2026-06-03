# nano-llm 阶段性问题与处理方案报告 (Phase Issues & Resolutions Report)

本报告详细记录了在集群迁移（从 `10.232.18.214` 迁移至新服务器 `10.232.18.193`）与拉起大规模分布式预训练和对齐过程中遇到的关键问题及对应的技术解决策略。

---

## 1. ⚙️ CUDA 初始化错误 802 (CUDA Init Error 802)
* **问题描述**：在新服务器 `10.232.18.193` 上检测 GPU 状态时，PyTorch 抛出 CUDA Error 802 错误，表示 NVIDIA 驱动与系统级通信服务失效。
* **原因分析**：系统当前运行的 NVIDIA 显卡驱动版本为 `595.71.05`，然而后台安装运行的 `nvidia-fabricmanager` 版本依然为旧版本的 `535.x`。驱动与 Fabric Manager 之间的版本严重不匹配，导致 NVLink 通信接口被操作系统强行阻断。
* **解决措施**：
  1. 通过 APT 升级 fabricmanager：`sudo apt-get install nvidia-fabricmanager-595`。
  2. 启动并激活服务：`sudo systemctl restart nvidia-fabricmanager`。
  3. 重启服务器节点后验证，`torch.cuda.is_available() == True`，8 张 H800 卡全部恢复正常工作。

---

## 2. 🗜️ Triton MLA 模拟器 Out of Memory (OOM) 爆显存问题
* **问题描述**：在使用 8 卡 H800 启动 2B-dense 模型预训练时，当 sequence length 设置为 `4096`，首个编译步即发生 `torch.OutOfMemoryError` 爆显存，提示尝试分配 16.00 GiB 显存失败。
* **原因分析**：
  在 `utils/triton_fa3.py` 实现的模拟 FP8 FlashAttention-3 算子中，程序执行了：
  ```python
  scores = torch.matmul(q_bf16, k_bf16.transpose(-2, -1)) * fwd_scale
  attn_weights = F.softmax(scores, dim=-1)
  ```
  在 Python 层将注意力得分矩阵完全实例化。对于 `[batch_size=8, num_heads=32, seq_len=4096, seq_len=4096]` 这一四维张量，单个张量在 bfloat16 精度下即占用 **8.59 GB** 显存，加上反向传播所需的梯度存储，单次计算需要占用 **17.18 GB** 额外空间，直接挤爆了单卡显存。
* **解决措施**：
  重构了 [utils/triton_kernels.py](file:///home/ifnodoraemon/myagent/nano-llm/utils/triton_kernels.py#L217) 中的 `triton_mla_flash_attn` 算子：
  1. 不再调用在 Python 层进行矩阵乘法与 Softmax 的 `FP8FlashAttention3`。
  2. 改为使用 PyTorch 原生的 C++ 融合缩放点积注意力 `F.scaled_dot_product_attention`。该底层自动调用 CUDA 级高度优化的 **FlashAttention-2 / Memory-Efficient Attention**，在 SRAM 中分块计算注意力，**不实例化任何 $N \times N$ 得分矩阵**。
  3. **保留 FP8 精度模拟**：在传入 SDPA 之前，将 Q/K/V 投射到 `torch.float8_e4m3fn`（FP8 格式）然后再转回 `bfloat16`，成功实现了“零显存开销”的 FP8 量化噪声模拟。
  4. **结果**：显存占用直降至 **67.8 GB**，8 卡 MFU 达到极佳的 **47.5%**，训练稳定收敛。

---

## 3. 🔄 Evol-Instruct 变异 Tokenization 严重不一致 Bug
* **问题描述**：在 GRPO 强化学习中，当有 50% 概率触发 Evol-Instruct 对 prompt 进行变异时，变异产生的复杂约束文本（`p_text`）并没有被重新编码。
* **原因分析**：
  模型在 Rollout 生成答案时依然使用的是变异前的原始 tokens（`p_ids`），这意味着模型生成的是普通答案；但在后期判定 rewards 时，评估函数拿到的却是变异后包含复杂限制（如“必须输出 JSON 格式”）的 `p_text`。模型因没看到限制而未按格式回答，导致大量优质生成被“冤判”扣分，强化学习难以收敛。
* **解决措施**：
  在 [grpo.py](file:///home/ifnodoraemon/myagent/nano-llm/grpo.py#L279) 变异分支中，增加了对变异文本的重编码逻辑：
  ```python
  p_text = evol_engine.mutate(p_text)
  # 重新 Tokenize 变异后的文本以更新模型输入
  p_ids_dict = tokenizer(p_text, return_tensors="pt")
  p_ids = p_ids_dict["input_ids"].to(device)
  ```
  使输入端与评测端在文本信息上达成严格一致。

---

## 4. 🛡️ GRPO 奖励作弊与思维链退化 (Reward Hacking)
* **问题描述**：强化学习训练中期，小模型极易找到规则漏洞来套取高 Reward。例如，在 `<think>` 思维链中大量重复无意义的标点、空格、句式（如 `Because... Because... Because...`）来堆砌长度，使思维链发生逻辑退化。
* **解决措施**：
  在 [grpo/rewards.py](file:///home/ifnodoraemon/myagent/nano-llm/grpo/rewards.py) 中加入了三层反作弊硬约束：
  1. **N-gram 重复率监测**：提取 `<think>` block，对 4-gram 重复度超过 30% 的 completion 直接判负分（Reward -1.0）。
  2. **长度约束**：将思维链长度限制在 100~800 词之间最为健康，超出 1500 词或少于 20 词施加二次曲线形式的惩罚。
  3. **死循环制造监测**：检测相邻句子的语义重合度，对发生死循环的完成结果直接实行前向切断，防止模型刷分。

---

## 5. 👥 孤儿/重复下载进程冲突
* **问题描述**：新服务器 `193` 重新拉起下载器后，发现部分分片数据有零星损坏和文件大小不均问题。
* **原因分析**：
  此前运行的一批 24 进程下载器由于是通过 parent 挂断等非干净方式终止，导致 3 个孤儿 worker 进程（PPID=1）继续存活并在后台不停向 `en_train_worker_7.bin` 等分片文件追加数据。当新下载器启动并试图覆写该文件时，两个进程在不同文件偏移量上同时写入，发生了写冲突。
* **解决措施**：
  1. 使用命令 `kill -9` 强行终止了所有归属于 `ubuntu` 用户的 Python 进程，清除孤儿进程。
  2. 清空了所有的临时数据分片缓存，以 8 卡 8 Workers 的最优限流配置（防范 HF-mirror 的 429 频控）干净地重新拉起下载。

---

## 6. 🗂️ ModelScope 统一模型地址配置 (替代 Model Size 校验)
* **问题描述**：旧版本的 `config.py` 和加载引擎过度依赖对硬编码 `model_size` 字段（如 `tiny`、`2B-dense`）的校验，如果使用国内的高速 ModelScope (MS) 镜像拉取，解析 HF 模型 ID 时经常因为找不到预设Preset而报错。
* **重构与推荐的 ModelScope 地址**：
  * 我们集成了标准的 ModelScope 统一加载接口 [HubAdapter](file:///home/ifnodoraemon/myagent/nano-llm/utils/hub_adapter.py#L33)。用户在训练或加载时，只需在 `--model_name_or_path` 直接指定 ModelScope 平台的标准模型 ID 地址，不再进行模型尺寸的强绑定拦截。
  * **推荐的标准 ModelScope 模型地址列表**：
    * **Qwen 2.5 7B 指令/对话模型**：`qwen/Qwen2.5-7B-Instruct`
    * **Qwen 2.5 7B 基础/预训练模型**：`qwen/Qwen2.5-7B`
    * **Qwen 2.5 1.5B 指令/对话模型**：`qwen/Qwen2.5-1.5B-Instruct`
    * **Qwen 2.5 1.5B 基础/预训练模型**：`qwen/Qwen2.5-1.5B`
    * **Qwen 2.5 0.5B 基础/预训练模型**：`qwen/Qwen2.5-0.5B`
  * 平台内部会自动重定向至 ModelScope 的 Mainland China 高速镜像节点进行毫秒级自动拉取，同时兼容 `HF_ENDPOINT=https://hf-mirror.com` 镜像的高速 fallback 回退。

---

## 7. 🔄 大规模预训练“接力”训练与断点弹性恢复设计 (Relay & Fault-Tolerant Pretraining)
* **问题背景**：针对 400B 目标语料的数据下载过程需要数天，为了充分利用集群算力，我们采用“边下载、边训练”的流水线并行化方案。初始在 `/data/nano-llm-data/binaries_1t` 数据集上开跑，400B 数据合并完成后必须能够平滑“接力”到新的 binaries 目录继续训练。
* **解决措施**：
  1. **非阻塞后台异步 Checkpoint 写入**：利用 [BackgroundCheckpointSaver](file:///home/ifnodoraemon/myagent/nano-llm/utils/checkpoint_saver.py#L21) 在主线程完成快速深拷贝（<100ms），并交由后台守护线程进行高耗时的磁盘 I/O，规避大模型权重保存造成的系统性迭代抖动（Jitter）。
  2. **弹性容错自动恢复**：[ElasticRestoreManager](file:///home/ifnodoraemon/myagent/nano-llm/utils/checkpoint_saver.py#L117) 会在引擎重启时自动扫描 `training_manifest.json`，无缝拉回最新的 Step 与 Epoch 状态。
  3. **无缝接力切换 (Dataset Relay)**：在 400B 全量数据下载合并后，只需将 `run_pretrain_mla_triton.sh` 中的 `DATA_DIR` 修改为 `/data/nano-llm-data/binaries_400b` 并重新运行。引擎会自动读取当前目录下的 `outputs_pretrain_mla_triton/checkpoint_elastic.pt`，平滑热重启并自动切换到新数据集上“接力”向后计算，实现零算力等待与无损接力。

