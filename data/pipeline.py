"""
Unified data processing pipeline: crawl -> clean -> deduplicate -> tokenize -> pack.
Also handles premium dataset download for SFT/DPO/GRPO.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import logging
import subprocess
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_step(cmd_parts: list, step_name: str) -> None:
    logger.info(f"=== Starting: {step_name} ===")
    process = subprocess.Popen(cmd_parts, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    if process.returncode != 0:
        logger.error(f"Step '{step_name}' failed with exit code {process.returncode}")
        sys.exit(process.returncode)
    logger.info(f"=== Completed: {step_name} ===")


def download_premium_datasets(
    sft_size: int = 5000,
    code_size: int = 2000,
    dpo_size: int = 2000,
    grpo_size: int = 2000,
) -> None:
    logger.info("Downloading premium SFT/DPO/GRPO datasets...")
    run_step(
        [
            sys.executable, "./data/download_premium_data.py",
            "--sft_size", str(sft_size),
            "--code_size", str(code_size),
            "--dpo_size", str(dpo_size),
            "--grpo_size", str(grpo_size),
        ],
        "premium dataset download",
    )


def crawl_data(output_dir: str = "./data/cleaned_corpus") -> None:
    logger.info("Crawling raw text corpus...")
    run_step(
        [sys.executable, "./crawl_data.py", "--output_dir", output_dir],
        "crawl raw corpus",
    )


def clean_data(src_dir: str = "./data/cleaned_corpus") -> None:
    logger.info("Cleaning crawled corpus (<unk> removal, formatting)...")
    run_step(
        [sys.executable, "./data/clean_unk.py"],
        "clean corpus",
    )


def deduplicate_data(
    src_dir: str = "./data/super_cleaned_corpus",
    dest_dir: str = "./data/deduplicated_corpus",
) -> None:
    logger.info("Deduplicating cleaned corpus...")
    run_step(
        [sys.executable, "./deduplicate.py", "--src_dir", src_dir, "--dest_dir", dest_dir],
        "deduplicate corpus",
    )


def train_tokenizer(
    corpus_dir: str = "./data/deduplicated_corpus",
    vocab_size: int = 32000,
    output_path: str = "./data/fast_tokenizer.json",
) -> None:
    logger.info("Training fast BPE tokenizer...")
    run_step(
        [
            sys.executable, "./data/train_fast_tokenizer.py",
            "--corpus_dir", corpus_dir,
            "--vocab_size", str(vocab_size),
            "--output", output_path,
        ],
        "train tokenizer",
    )


def pack_binaries(
    src_dir: str = "./data/deduplicated_corpus",
    tokenizer_path: str = "./data/fast_tokenizer.json",
    dest_dir: str = "./data/binaries",
) -> None:
    logger.info("Packing tokenized binaries for pretraining...")
    run_step(
        [
            sys.executable, "./data/pack_fast_binaries.py",
            "--src_dir", src_dir,
            "--tokenizer", tokenizer_path,
            "--dest_dir", dest_dir,
        ],
        "pack binaries",
    )


def download_chinese_corpus(output_dir: str = "./data/cleaned_corpus") -> None:
    """Download Chinese text corpus with graceful fallback, writing to individual article files."""
    logger.info("Downloading/preparing Chinese text corpus...")
    os.makedirs(output_dir, exist_ok=True)

    # Check if we already have sufficient individual files in output_dir
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("zh_art_") and f.endswith(".txt")]
    if len(existing_files) >= 10000:
        logger.info(f"Chinese corpus already has {len(existing_files)} files in {output_dir} — skipping download")
        return

    # Path to local snapshot json
    snapshot_path = "/root/.cache/huggingface/hub/datasets--pleisto--wikipedia-cn-20230720-filtered/snapshots/4cef256a3f426ae1d3f6930c8cd59a32d785d99d/wikipedia-cn-20230720-filtered.json"
    if os.path.exists(snapshot_path):
        logger.info(f"Loading Chinese corpus from local snapshot: {snapshot_path}")
        try:
            import json as _json
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            count = 0
            for sample in data:
                text = sample.get("completion") or sample.get("text") or sample.get("content") or ""
                text = text.strip()
                if len(text) > 50:
                    art_file = os.path.join(output_dir, f"zh_art_{count}.txt")
                    with open(art_file, "w", encoding="utf-8") as out:
                        out.write(text + "\n")
                    count += 1
                    if count >= 50000:
                        break
            logger.info(f"Loaded {count} Chinese text samples from local snapshot as individual files in {output_dir}")
            return
        except Exception as e:
            logger.warning(f"Failed to load from local snapshot: {e} — trying online datasets...")

    # Try downloading online using HubAdapter (ModelScope first, then HF fallback)
    success = False
    try:
        from utils.hub_adapter import HubAdapter
        adapter = HubAdapter()
        logger.info("Trying to download Chinese corpus via HubAdapter...")
        dataset = adapter.load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train")
        count = 0
        for sample in dataset:
            text = sample.get("completion") or sample.get("text") or sample.get("content") or ""
            text = text.strip()
            if len(text) > 50:
                art_file = os.path.join(output_dir, f"zh_art_{count}.txt")
                with open(art_file, "w", encoding="utf-8") as out:
                    out.write(text + "\n")
                count += 1
                if count >= 50000:
                    break
        logger.info(f"Downloaded and split {count} Chinese text samples via HubAdapter.")
        success = True
    except Exception as e:
        logger.warning(f"HubAdapter download failed: {e}. Trying streaming fallback...")

    if not success:
        try:
            from datasets import load_dataset

            datasets_to_try = [
                ("pleisto/wikipedia-cn-20230720-filtered", "train"),
            ]
            for ds_name, ds_split in datasets_to_try:
                try:
                    logger.info(f"Trying dataset: {ds_name} (streaming)...")
                    dataset = load_dataset(ds_name, split=ds_split, streaming=True)
                    count = 0
                    for sample in dataset:
                        text = sample.get("completion") or sample.get("text") or sample.get("content") or ""
                        text = text.strip()
                        if len(text) > 50:
                            art_file = os.path.join(output_dir, f"zh_art_{count}.txt")
                            with open(art_file, "w", encoding="utf-8") as f:
                                f.write(text + "\n")
                            count += 1
                            if count >= 50000:
                                break
                    logger.info(f"Downloaded and split {count} Chinese text samples to {output_dir}")
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Dataset {ds_name} failed: {e}")
                    continue

            if not success:
                _print_chinese_corpus_instructions(output_dir)
        except ImportError:
            logger.warning("`datasets` library not installed — cannot download Chinese corpus")
            _print_chinese_corpus_instructions(output_dir)
        except Exception as e:
            logger.warning(f"Chinese corpus download failed: {e}")
            _print_chinese_corpus_instructions(output_dir)


def _print_chinese_corpus_instructions(output_dir: str) -> None:
    """Print manual download instructions for Chinese text corpus."""
    logger.info("=" * 60)
    logger.info("Manual Chinese Corpus Setup Instructions:")
    logger.info(f"  1. Create directory: mkdir -p {output_dir}")
    logger.info(f"  2. Install datasets: pip install datasets")
    logger.info(f"  3. Run: python -c \"")
    logger.info(f"       from datasets import load_dataset")
    logger.info(f"       ds = load_dataset('pleisto/wikipedia-cn-20230720-filtered', split='train', streaming=True)")
    logger.info(f"       with open('{output_dir}/zh_corpus.txt','w') as f:")
    logger.info(f"           for i,s in enumerate(ds):")
    logger.info(f"               if i>=50000: break")
    logger.info(f"               f.write(s['text']+'\\n')\"")
    logger.info(f"  Or place any .txt files in {output_dir}/")
    logger.info("=" * 60)


def validate_datasets() -> dict:
    """Validate generated datasets and return a quality report."""
    report = {}
    checks = {
        "train_sft_premium.jsonl": "./data/train_sft_premium.jsonl",
        "train_dpo_premium.jsonl": "./data/train_dpo_premium.jsonl",
        "train_grpo_premium.jsonl": "./data/train_grpo_premium.jsonl",
        "fast_tokenizer.json": "./data/fast_tokenizer.json",
    }

    for name, path in checks.items():
        if os.path.exists(path):
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                report[name] = {
                    "exists": True,
                    "samples": len(lines),
                    "size_bytes": os.path.getsize(path),
                }
                # Check format
                invalid = 0
                for line in lines[:100]:
                    try:
                        sample = json.loads(line)
                        if "messages" not in sample and "prompt" not in sample:
                            invalid += 1
                    except json.JSONDecodeError:
                        invalid += 1
                report[name]["format_errors_first_100"] = invalid
            else:
                report[name] = {
                    "exists": True,
                    "size_bytes": os.path.getsize(path),
                }
        else:
            report[name] = {"exists": False}

    return report


def full_pipeline(
    skip_crawl: bool = False,
    skip_dedup: bool = False,
    premium_only: bool = False,
) -> None:
    """Run the full data pipeline end-to-end."""
    logger.info("=== Starting full data pipeline ===")

    # 1. Download premium datasets (always needed)
    download_premium_datasets()

    # 1b. Download Chinese corpus
    download_chinese_corpus()

    if premium_only:
        logger.info("Premium-only mode: skipping crawl->pack pipeline.")
        report = validate_datasets()
        logger.info(f"Dataset validation report:\n{json.dumps(report, indent=2)}")
        logger.info("=== Data pipeline complete (premium only) ===")
        return

    # 2. Crawl raw corpus
    if not skip_crawl:
        crawl_data()

    # 3. Clean corpus
    clean_data()

    # 4. Deduplicate
    if not skip_dedup:
        deduplicate_data()

    # 5. Train tokenizer
    train_tokenizer()

    # 6. Pack binaries for pretraining
    pack_binaries()

    # Validate
    report = validate_datasets()
    logger.info(f"Dataset validation report:\n{json.dumps(report, indent=2)}")
    logger.info("=== Data pipeline complete ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified data processing pipeline")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip raw corpus crawling")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication step")
    parser.add_argument("--premium-only", action="store_true", help="Only download premium SFT/DPO/GRPO datasets")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing datasets")
    args = parser.parse_args()

    if args.validate_only:
        report = validate_datasets()
        print(json.dumps(report, indent=2))
    else:
        full_pipeline(
            skip_crawl=args.skip_crawl,
            skip_dedup=args.skip_dedup,
            premium_only=args.premium_only,
        )
