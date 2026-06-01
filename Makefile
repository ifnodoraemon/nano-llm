.PHONY: help setup pretrain sft dpo grpo serve eval export pipeline publish clean

help:
	@echo "nano-llm Makefile"
	@echo "  make setup      - Install dependencies and verify environment"
	@echo "  make pretrain   - Run pretraining"
	@echo "  make sft        - Run SFT fine-tuning"
	@echo "  make dpo        - Run DPO alignment"
	@echo "  make grpo       - Run GRPO RL training"
	@echo "  make serve      - Start inference server"
	@echo "  make eval       - Run benchmarks"
	@echo "  make export     - Export to Safetensors"
	@echo "  make pipeline   - Run full pipeline (data through export)"
	@echo "  make publish    - Export model and upload to Hugging Face Hub"
	@echo "  make clean      - Remove outputs"

setup:
	bash scripts/setup.sh

pretrain:
	python pretrain.py --model_size tiny --max_steps 200

sft:
	torchrun --nproc_per_node=1 train.py \
		--data_path ./data/train_sft_premium.jsonl \
		--output_dir ./outputs \
		--epochs 3 \
		--batch_size 4 \
		--max_length 512 \
		--max_lr 2e-5

dpo:
	torchrun --nproc_per_node=1 align.py \
		--data_path ./data/train_dpo_premium.jsonl \
		--sft_checkpoint_path ./outputs/checkpoint_sft.pt \
		--output_dir ./outputs_dpo \
		--epochs 1 \
		--batch_size 2 \
		--max_lr 5e-6

grpo:
	torchrun --nproc_per_node=1 grpo.py \
		--data_path ./data/train_grpo_premium.jsonl \
		--sft_checkpoint_path ./outputs/checkpoint_sft.pt \
		--output_dir ./outputs_grpo \
		--epochs 1 \
		--batch_size 2

serve:
	python serve.py --checkpoint_path ./outputs/checkpoint_sft.pt

eval:
	python eval_benchmarks.py --checkpoint_path ./outputs/checkpoint_sft.pt

export:
	python export_dora.py

pipeline:
	python run_unified_pipeline.py --mode dev

smoke:
	python -m pytest tests/ -x -v || python -m unittest discover -s tests -p "test_*.py" -v

publish:
	python export_dora.py --quantize --push_to_hub

clean:
	rm -rf ./outputs ./outputs_dev ./outputs_dpo ./outputs_grpo ./logs/*.log
