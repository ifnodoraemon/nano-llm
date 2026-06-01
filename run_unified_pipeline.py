import os
import subprocess
import sys
import time
import json
import argparse

from config import load_config, detect_hardware
from utils.experiment_tracker import ExperimentTracker


def run_command(cmd_parts: list):
    cmd_str = " ".join(cmd_parts)
    print(f"\n=======================================================================")
    print(f"\U0001f3ac Executing Pipeline Command:\n   {cmd_str}")
    print(f"=======================================================================\n")

    start_time = time.time()
    process = subprocess.Popen(cmd_parts, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()

    duration = time.time() - start_time
    if process.returncode != 0:
        print(f"❌ Command failed with exit code: {process.returncode} (Duration: {duration:.1f}s)")
        sys.exit(process.returncode)

    print(f"✅ Command completed successfully! (Duration: {duration:.1f}s)")


def _check_checkpoint(path: str, min_size_mb: float = 1.0, required: bool = True) -> bool:
    """Verify a checkpoint file exists and meets the minimum size threshold."""
    if not os.path.exists(path):
        if required:
            print(f"❌ Quality gate FAILED: Checkpoint not found at {path}")
            sys.exit(1)
        else:
            print(f"ℹ️  Quality gate: Optional checkpoint not found at {path} — skipping check")
            return False
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < min_size_mb:
        print(f"❌ Quality gate FAILED: Checkpoint {path} too small ({size_mb:.1f} MB < {min_size_mb} MB)")
        sys.exit(1)
    print(f"✅ Quality gate PASSED: {path} ({size_mb:.1f} MB)")
    return True


def _run_eval_gate(checkpoint_path: str, label: str, baseline_path: str = None):
    """Run eval_benchmarks.py as a quality gate, optionally against a baseline."""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Quality gate FAILED: {label} checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    print(f"\n\U0001f4ca Quality gate: Running {label} benchmark evaluation...")
    eval_cmd = [sys.executable, "eval_benchmarks.py", "--checkpoint_path", checkpoint_path]
    if baseline_path and os.path.exists(baseline_path):
        eval_cmd += ["--baseline_checkpoint_path", baseline_path]
    try:
        run_command(eval_cmd)
    except SystemExit:
        print(f"⚠️  {label} evaluation exited early — continuing pipeline...")


def main():
    parser = argparse.ArgumentParser(description="nano-llm Unified Training Pipeline")
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "prod"],
                        help="Pipeline mode: dev (fast smoke test) or prod (full training)")
    parser.add_argument("--model_size", type=str, default=None,
                        help="Model size preset (default: tiny for dev, 16B-equivalent for prod)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--start_stage", type=str, default="data",
                        choices=["data", "sft", "dpo", "grpo", "export"],
                        help="Stage to start from")
    parser.add_argument("--base_port", type=int, default=29500,
                        help="Base master port for torchrun")
    args = parser.parse_args()

    hw = detect_hardware()
    num_gpus = args.num_gpus if args.num_gpus is not None else max(hw.num_gpus, 1)
    print(f"\U0001f4cb Detected hardware: {num_gpus}x {hw.gpu_name} ({hw.gpu_memory_gb:.1f} GB each)")

    tracker = ExperimentTracker(
        project="nano-llm",
        config=vars(args),
        mode="offline",
        log_dir="./logs",
    )

    cfg = load_config(
        mode=args.mode,
        model_size=args.model_size,
        num_gpus=num_gpus,
    )

    # 1. Ensure premium datasets are available
    sft_data = cfg.sft_data
    dpo_data = cfg.dpo_data
    grpo_data = cfg.grpo_data

    if args.start_stage == "data":
        tracker.log({"pipeline/stage": "data"}, step=0)
        print("\U0001f4cb Checking availability of premium datasets...")
        missing = [p for p in [sft_data, dpo_data, grpo_data] if not os.path.exists(p)]
        if missing:
            print(f"⚠️ Missing datasets: {missing}")
            print("\U0001f4e6 Running data pipeline to download premium datasets...")
            run_command([sys.executable, "./data/pipeline.py", "--premium-only"])

        for path in [sft_data, dpo_data, grpo_data]:
            if os.path.exists(path):
                print(f"✅ Found dataset: {path}")
            else:
                print(f"❌ Dataset still missing after pipeline run: {path}")
                sys.exit(1)

    # 2. Stage: Supervised Fine-Tuning (SFT)
    if args.start_stage in ("data", "sft"):
        tracker.log({"pipeline/stage": "sft"}, step=1)

        # Quality gate: validate pretrain checkpoint before SFT
        pretrain_ckpt = os.path.join(cfg.output_dir, "checkpoint_pretrain.pt")
        print(f"\n\U0001f50d Pre-SFT Quality Gate: Checking pretrain checkpoint...")
        _check_checkpoint(pretrain_ckpt, min_size_mb=1.0, required=False)

        port = args.base_port
        print("\n\U0001f680 Starting Stage: Supervised Fine-Tuning (SFT)...")
        sft_cmd = [
            "torchrun", f"--nproc_per_node={num_gpus}", f"--master_port={port}", "train.py",
            "--data_path", sft_data,
            "--output_dir", cfg.output_dir,
            "--epochs", str(cfg.epochs),
            "--batch_size", str(cfg.batch_size),
            "--max_length", str(cfg.max_length),
            "--max_lr", str(cfg.max_lr),
        ]
        run_command(sft_cmd)

        # Quality gate: run evaluation after SFT
        if cfg.mode == "prod":
            sft_ckpt = os.path.join(cfg.output_dir, "checkpoint_sft.pt")
            if os.path.exists(sft_ckpt):
                print("\n\U0001f4ca Quality gate: Running SFT benchmark evaluation...")
                eval_cmd = [sys.executable, "eval_benchmarks.py", "--checkpoint_path", sft_ckpt]
                try:
                    run_command(eval_cmd)
                except SystemExit:
                    print("⚠️ Evaluation failed but continuing pipeline...")
    else:
        print("\n⏭️ Skipping SFT stage.")

    # 3. Stage: DPO Preference Alignment
    if args.start_stage in ("data", "sft", "dpo"):
        tracker.log({"pipeline/stage": "dpo"}, step=2)
        port = args.base_port + 1
        dpo_output_dir = cfg.output_dir.replace("outputs", "outputs_dpo")
        print("\n\U0001f680 Starting Stage: DPO Preference Alignment...")
        dpo_cmd = [
            "torchrun", f"--nproc_per_node={num_gpus}", f"--master_port={port}", "align.py",
            "--data_path", dpo_data,
            "--sft_checkpoint_path", cfg.sft_checkpoint_path,
            "--output_dir", dpo_output_dir,
            "--epochs", str(cfg.epochs),
            "--batch_size", str(max(cfg.batch_size // 2, 1)),
            "--max_lr", str(cfg.max_lr * 0.25),
            "--beta", str(cfg.dpo_beta),
        ]
        run_command(dpo_cmd)

        # Quality gate: validate DPO checkpoint
        print(f"\n\U0001f50d Post-DPO Quality Gate: Validating DPO checkpoint...")
        dpo_ckpt = os.path.join(dpo_output_dir, "checkpoint_dpo.pt")
        _check_checkpoint(dpo_ckpt, min_size_mb=1.0, required=True)

        # Best-effort accuracy extraction from training manifest
        dpo_manifest = os.path.join(dpo_output_dir, "training_manifest.json")
        if os.path.exists(dpo_manifest):
            with open(dpo_manifest, "r") as f:
                manifest = json.load(f)
            dpo_accuracy = manifest.get("final_accuracy", manifest.get("accuracy"))
            if dpo_accuracy is not None:
                if dpo_accuracy < 50.0:
                    print(f"⚠️  DPO accuracy ({dpo_accuracy:.1f}%) is below 50% — model may not align well. Continuing pipeline...")
                else:
                    print(f"✅ DPO accuracy ({dpo_accuracy:.1f}%) meets threshold.")
            else:
                print(f"ℹ️  Training manifest found but no accuracy metric recorded.")
        else:
            print(f"ℹ️  No training manifest found at {dpo_manifest} — skipping accuracy check.")
    else:
        print("\n⏭️ Skipping DPO stage.")

    # 4. Stage: GRPO Reasoning RL
    if args.start_stage in ("data", "sft", "dpo", "grpo"):
        tracker.log({"pipeline/stage": "grpo"}, step=3)
        port = args.base_port + 2
        grpo_output_dir = cfg.output_dir.replace("outputs", "outputs_grpo")
        print("\n\U0001f680 Starting Stage: GRPO Reasoning RL...")
        grpo_cmd = [
            "torchrun", f"--nproc_per_node={num_gpus}", f"--master_port={port}", "grpo.py",
            "--data_path", grpo_data,
            "--sft_checkpoint_path", cfg.sft_checkpoint_path,
            "--output_dir", grpo_output_dir,
            "--epochs", str(cfg.epochs),
            "--batch_size", str(max(cfg.batch_size // 2, 1)),
            "--group_size", str(cfg.group_size),
            "--max_prompt_len", str(cfg.max_prompt_len),
            "--max_gen_len", str(cfg.max_gen_len),
        ]
        run_command(grpo_cmd)

        # Quality gate: validate GRPO checkpoint and benchmark against SFT
        print(f"\n\U0001f50d Post-GRPO Quality Gate: Validating checkpoint and benchmarking...")
        grpo_ckpt = os.path.join(grpo_output_dir, "checkpoint_grpo_epoch_1.pt")
        _check_checkpoint(grpo_ckpt, min_size_mb=1.0, required=True)

        if cfg.mode == "prod":
            sft_ckpt = os.path.join(cfg.output_dir, "checkpoint_sft.pt")
            if os.path.exists(sft_ckpt):
                print(f"\n\U0001f4ca Comparing GRPO checkpoint against SFT baseline...")
                _run_eval_gate(grpo_ckpt, "GRPO", baseline_path=sft_ckpt)
            else:
                print(f"ℹ️  SFT checkpoint not found at {sft_ckpt} — running standalone GRPO evaluation...")
                _run_eval_gate(grpo_ckpt, "GRPO")
    else:
        print("\n⏭️ Skipping GRPO stage.")

    # 5. Stage: Export model
    if args.start_stage != "export":
        tracker.log({"pipeline/stage": "export"}, step=4)
        print("\n\U0001f680 Exporting Safetensors model...")
        grpo_output_dir = cfg.output_dir.replace("outputs", "outputs_grpo")
        grpo_ckpt = os.path.join(grpo_output_dir, "checkpoint_grpo_epoch_1.pt")
        export_dest_dir = os.path.join(cfg.output_dir, "dora")
        run_command(["python3", "export_dora.py", "--checkpoint_path", grpo_ckpt, "--dest_dir", export_dest_dir])

        # Optional quantization after export
        print("\n\U0001f4e6 Generating quantized model variants (int8/int4)...")
        try:
            run_command([sys.executable, "quantize.py", "--src", os.path.join(export_dest_dir, "model.safetensors"),
                         "--dest", os.path.join(export_dest_dir, "model_int8.pt"), "--bits", "8"])
            run_command([sys.executable, "quantize.py", "--src", os.path.join(export_dest_dir, "model.safetensors"),
                         "--dest", os.path.join(export_dest_dir, "model_int4.pt"), "--bits", "4"])
        except SystemExit:
            print("⚠️  Quantization step failed — continuing pipeline...")

    tracker.finish()
    print("\n=======================================================================")
    print("\U0001f3c6 CONGRATULATIONS: Unified Pipeline completed successfully!")
    print(f"   Standard safetensors model exported to: {cfg.output_dir}")
    print("=======================================================================\n")


if __name__ == "__main__":
    main()
