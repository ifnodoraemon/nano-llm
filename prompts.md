# 修复提示词 — 复制给其他 AI 使用

---

## 提示词 1：修复 train.py（优先级 P0+P1）

```
项目目录: /home/ifnodoraemon/myagent/nano-llm

任务：修复 train.py 中回退的 9 处代码，让它使用项目中已有的工具模块。

需要改的地方和对应的正确写法：

1. 第 85 行 `--use_lora` 参数：`type=bool, default=False` → `action="store_true"`

2. DDP 初始化（第 91-105 行）：删除旧的 if/else 块，改为：
   from utils.ddp_helper import init_ddp
   ctx = init_ddp()
   ddp, ddp_rank, ddp_local_rank, ddp_world_size = ctx["ddp"], ctx["rank"], ctx["local_rank"], ctx["world_size"]
   device = ctx["device"]
   is_master = ctx["is_master"]

3. Tokenizer 加载（第 128-141 行）：删除旧的 CustomTokenizerAdapter / CustomBPETokenizer / AutoTokenizer 逻辑，改为：
   from utils.tokenizer_loader import load_tokenizer
   tokenizer = load_tokenizer(fallback_model_name=args.model_name_or_path)

4. Checkpoint 加载（第 158-164 行）：删除 raw torch.load，改为：
   from utils.checkpoint_utils import load_checkpoint_with_fp8_translation
   model_config, state_dict = load_checkpoint_with_fp8_translation(
       base_checkpoint_path, map_location=device, state_keys=("model_state_dict",)
   )
   model = Transformer(model_config).to(device)
   model.load_state_dict(state_dict)

5. 从零创建模型（第 166-173 行）：删除硬编码的 n_layer/n_head/n_embd，改为：
   from model import get_deepseek_config
   model_config = get_deepseek_config("7B-equivalent",
       block_size=args.max_length, vocab_size=len(tokenizer),
       lora_r=args.lora_r if args.use_lora else 0,
       lora_alpha=args.lora_alpha,
   )
   model = Transformer(model_config).to(device)

6. Optimizer（第 37-60 行）：删除整个 configure_optimizers 函数，改为：
   from utils.training_utils import configure_optimizers, get_cosine_lr as get_learning_rate
   然后在 train() 中调用 configure_optimizers(model, args.weight_decay, args.max_lr, betas=(0.9, 0.95), device_type="cuda" if "cuda" in device else "cpu")

7. 在 DataLoader 创建后（第 154 行后）添加：
   from utils.training_utils import validate_dataset, assert_grad_accum_safe
   validate_dataset(dataset, tokenizer=tokenizer, name="SFT")
   assert_grad_accum_safe(dataloader, args.grad_accum_steps, args.batch_size)

8. step_counter（第 212 行附近，循环开始前）：改为：
   step_counter = restored_step if has_ckpt else 0
   （需要先有 ElasticRestoreManager，见下一条）

9. ElasticRestoreManager（在 optimizer 创建之后，训练循环之前添加）：
   from utils.checkpoint_saver import ElasticRestoreManager
   elastic = ElasticRestoreManager(args.output_dir)
   has_ckpt, restored_step, restored_epoch = elastic.auto_detect_checkpoint()
   start_epoch = 0
   if has_ckpt:
       raw_model_for_restore = model.module if ddp else model
       restored_step, restored_epoch, _ = elastic.restore_training_state(raw_model_for_restore, optimizer, None)
       start_epoch = restored_epoch
   # 然后用 range(start_epoch, args.epochs) 替代 range(args.epochs)

要求：
- 不要删除 ExperimentTracker 相关代码（第 111-117 行和已有的 tracker.log/tracker.finish 调用）
- 不要改动文件的其他部分
- 修改后文件能 import 通过（python -c "import train" 不报错即可，训练逻辑不需要在 CPU 上跑通）
```

---

## 提示词 2：修复 align.py（优先级 P0 — 有运行时崩溃）

```
项目目录: /home/ifnodoraemon/myagent/nano-llm

任务：修复 align.py 中回退的代码，特别是会导致运行时崩溃的 3-tuple 解包问题。

关键背景：model.Transformer.forward() 现在返回 3 元组 (logits, loss, aux_loss)，所有调用 model() 的地方必须用 3 元组解包。

需要改的地方：

1. **P0-崩溃** policy 前向传播（约第 248-255 行）：
   policy_chosen_logits, _ = policy_model(c_input_ids)  →  policy_chosen_logits, _, _ = policy_model(...)
   policy_rejected_logits, _ = policy_model(r_input_ids)  →  policy_rejected_logits, _, _ = policy_model(...)

2. **P0-崩溃** reference 前向传播（约第 258-264 行）：
   ref_chosen_logits, _ = reference_model(c_input_ids)  →  ref_chosen_logits, _, _ = reference_model(...)
   ref_rejected_logits, _ = reference_model(r_input_ids)  →  ref_rejected_logits, _, _ = reference_model(...)

3. DDP 初始化：删除旧的 if/else 块，改为：
   from utils.ddp_helper import init_ddp
   ctx = init_ddp()
   ddp, ddp_rank, ddp_local_rank, ddp_world_size = ctx["ddp"], ctx["rank"], ctx["local_rank"], ctx["world_size"]
   device = ctx["device"]
   is_master = ctx["is_master"]

4. Tokenizer 加载（约第 136-149 行）：删除旧的 CustomTokenizerAdapter / CustomBPETokenizer / HubAdapter 逻辑，改为：
   from utils.tokenizer_loader import load_tokenizer
   tokenizer = load_tokenizer()

5. Checkpoint 加载（约第 186-191 行）：删除 raw torch.load，改为：
   from utils.checkpoint_utils import load_checkpoint_with_fp8_translation
   model_config, state_dict = load_checkpoint_with_fp8_translation(
       args.sft_checkpoint_path, map_location="cpu", state_keys=("model_state_dict",)
   )
   # 然后用 state_dict 加载两个模型

6. Optimizer（约第 207-215 行）：删除内联的 param_dict/decay_params/nodecay_params 逻辑，改为：
   from utils.training_utils import configure_optimizers
   optimizer = configure_optimizers(policy_model, args.weight_decay, args.max_lr, betas=(0.9, 0.95))

7. ElasticRestoreManager（在 optimizer 创建后添加）：
   from utils.checkpoint_saver import ElasticRestoreManager
   elastic = ElasticRestoreManager(args.output_dir)
   has_ckpt, restored_step, restored_epoch = elastic.auto_detect_checkpoint()
   start_epoch = 0
   if has_ckpt:
       restored_step, restored_epoch, _ = elastic.restore_training_state(policy_model, optimizer, None)
       start_epoch = restored_epoch

8. step_counter（约第 219 行）：改为：
   step_counter = restored_step if has_ckpt else 0

9. epoch 循环：`for epoch in range(args.epochs):` 改为 `for epoch in range(start_epoch, args.epochs):`

10. 最终 checkpoint 保存处：写入 training_manifest.json 以便质量门控读取：
    import json
    with open(os.path.join(args.output_dir, "training_manifest.json"), "w") as f:
        json.dump({"latest_step": step_counter, "latest_epoch": epoch, "final_accuracy": accuracy.item() * 100}, f)
    （accuracy 变量在训练循环中已有，取最后一个 step 的值即可）

要求：
- 不要删除 ExperimentTracker 相关代码
- 不要改动 dpo_collator、compute_logprobs、compute_dpo_loss 函数
- 不要改动文件的 import 部分（已有 utils.model_utils 等）
```

---

## 提示词 3：修复 pretrain.py（优先级 P1）

```
项目目录: /home/ifnodoraemon/myagent/nano-llm

任务：修复 pretrain.py，接入项目已有的 config 系统和 training_utils。

需要改的地方：

1. 模型配置（约第 150-177 行区域，当前是硬编码的 config = ModelConfig(...)）：
   改为使用 config 系统：
   from config import load_config, config_to_model_config, print_config
   cfg = load_config(
       mode="prod",
       model_size=args.model_size,
       num_gpus=ddp_world_size,
       batch_size=args.batch_size,
       max_steps=args.max_steps,
       max_lr=args.lr,
       min_lr=args.min_lr,
       warmup_steps=args.warmup_steps,
       weight_decay=args.weight_decay,
       clip_grad=args.grad_clip,
       grad_accum_steps=args.grad_accum_steps,
       use_fp8=args.use_fp8.lower() == "true",
       tp_size=args.tp_size, pp_size=args.pp_size, ep_size=args.ep_size,
       output_dir=args.out_dir, data_dir=args.data_dir, block_size=args.block_size,
   )
   config = config_to_model_config(cfg, vision_dim=None)
   print_config(cfg)

2. Optimizer（约第 184-192 行的内联 optim_groups）：改为：
   from utils.training_utils import configure_optimizers
   optimizer = configure_optimizers(model, args.weight_decay, args.lr, betas=(0.9, 0.95))

3. LR scheduler（约第 229-237 行的内联 get_lr 函数）：改为：
   from utils.training_utils import get_cosine_lr
   def get_lr(step):
       return get_cosine_lr(step, args.lr, args.min_lr, args.warmup_steps, args.max_steps)

4. 在 get_batch 函数定义之后、模型构建之前，添加数据大小检查：
   assert len(train_data) > args.block_size * 2, f"Training data ({len(train_data)} tokens) insufficient for block_size {args.block_size}"

要求：
- pretrain.py 有特殊的 3D 并行初始化（TP/PP/EP），保留不动
- ExperimentTracker 和 data_mixer 代码保留不动
- 所有 torch.amp.autocast("cuda", ...) 保留不动（已经是正确的）
- 所有 3-tuple 解包 logits, loss, _ = model(...) 保留不动（已是正确的）
```

---

## 提示词 4：修复 grpo.py（优先级 P1+P2）

```
项目目录: /home/ifnodoraemon/myagent/nano-llm

任务：简化 grpo.py，让它导入项目中已有的 grpo/ 子模块，并接入工具模块。

当前 grpo.py 有 766 行，把 GRPODataset、evaluate_completion_rewards、generate_completions 等全部内联了。但项目里已经有这些模块：
- grpo/dataset.py — GRPODataset 类 + grpo_collate_fn
- grpo/rewards.py — evaluate_completion_rewards, GRPORewardScaler, AdaptiveKLTuner, extract_answer
- grpo/engine.py — generate_completions, compute_action_logprobs

需要改的地方：

1. 删除文件中的 GRPODataset 类定义、grpo_collate_fn 函数、evaluate_completion_rewards 函数、GRPORewardScaler 类、AdaptiveKLTuner 类、extract_answer 函数、generate_completions 函数、compute_action_logprobs 函数。

2. 改为从子模块导入（在文件顶部 import 区域添加）：
   from grpo.dataset import GRPODataset, grpo_collate_fn
   from grpo.rewards import evaluate_completion_rewards, GRPORewardScaler, AdaptiveKLTuner
   from grpo.engine import generate_completions, compute_action_logprobs

3. DDP 初始化：删除旧的 if/else 块，改为：
   from utils.ddp_helper import init_ddp
   ctx = init_ddp()
   ddp, ddp_rank, ddp_local_rank, ddp_world_size = ctx["ddp"], ctx["rank"], ctx["local_rank"], ctx["world_size"]
   device = ctx["device"]
   is_master = ctx["is_master"]
   seed_offset = ddp_rank

4. Tokenizer 加载：删除旧的 HubAdapter/CustomBPETokenizer 逻辑，改为：
   from utils.tokenizer_loader import load_tokenizer
   tokenizer = load_tokenizer(fallback_model_name="gpt2")

5. Checkpoint 加载：删除 raw torch.load，改为：
   from utils.checkpoint_utils import load_checkpoint_with_fp8_translation
   config, model_state = load_checkpoint_with_fp8_translation(
       args.sft_checkpoint_path, map_location=device, state_keys=("model_state_dict", "model")
   )

6. 在 generate_completions 调用之后、policy forward pass 之前，添加训练模式恢复：
   raw_model.train()
   （generate_completions 内部会调用 model.eval()）

7. 确保所有 model() 调用使用 3 元组解包：`logits, _, _`

要求：
- ExperimentTracker 代码保留（约第 481 行）
- SandboxExecutor 导入保留
- EvolInstructEngine 导入保留（约第 564 行）
- 删除内联代码后，确保 train() 函数的逻辑流不变
```

---

## 使用说明

1. 按优先级顺序执行：**提示词 2 → 提示词 1 → 提示词 3 → 提示词 4**
   - 提示词 2 最紧急（align.py 有运行时崩溃）
2. 每个提示词是独立的，可以分别给不同的 AI
3. 每个 AI 只需要修改 1 个文件
4. 改完后运行 `python -c "import train"` / `python -c "import align"` 等验证 import 不报错即可
