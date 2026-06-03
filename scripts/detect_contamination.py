#!/usr/bin/env python3
"""
Data Contamination Detector:
Performs 3-gram overlap checks between training datasets and evaluation benchmarks.
Identifies and automatically filters out contaminated training samples.
"""

import os
import re
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_ngrams(text: str, n: int = 3):
    """Normalized word-level n-gram generator."""
    # Normalize punctuation and convert to lowercase
    text = text.lower()
    words = re.findall(r'\w+', text)
    if len(words) < n:
        return set()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

def calculate_overlap(train_ngrams: set, eval_ngrams: set) -> float:
    """Calculates overlap ratio of train n-grams found in eval n-grams."""
    if not train_ngrams:
        return 0.0
    intersection = train_ngrams.intersection(eval_ngrams)
    return len(intersection) / len(train_ngrams)

def load_eval_texts(eval_dir: str) -> set:
    """Aggregates all 3-grams from benchmark evaluation files."""
    eval_ngrams = set()
    if not os.path.exists(eval_dir):
        logger.warning(f"Evaluation directory {eval_dir} not found. Using pre-baked fallback texts.")
        # Fallback benchmark strings
        fallbacks = [
            "Root Mean Square Normalization (RMSNorm) over LayerNorm",
            "Rotary Position Embeddings (RoPE) to incorporate token positions",
            "Direct Preference Optimization (DPO) beta parameter",
            "Weng earns $12 an hour babysitting.",
            "A box contains 5 red balls, 3 blue balls, and 2 green balls.",
            "farmer has 15 apple trees. Each tree yields 20 apples.",
            "physical change? Burning wood, Melting ice, Rusting iron, Baking a cake",
            "powerhouse of the cell? Nucleus, Mitochondria, Ribosome",
            "sawing a wooden plank. holds the plank with one hand"
        ]
        for text in fallbacks:
            eval_ngrams.update(get_ngrams(text))
        return eval_ngrams

    # Iterate through all JSON/JSONL evaluation benchmarks
    for filename in os.listdir(eval_dir):
        if not filename.endswith((".json", ".jsonl")):
            continue
        filepath = os.path.join(eval_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                if filename.endswith(".jsonl"):
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            for k, v in data.items():
                                if isinstance(v, str):
                                    eval_ngrams.update(get_ngrams(v))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    if isinstance(v, str):
                                        eval_ngrams.update(get_ngrams(v))
            logger.info(f"Loaded evaluation benchmark: {filename}")
        except Exception as e:
            logger.error(f"Failed to read evaluation benchmark {filename}: {e}")

    return eval_ngrams

def scan_and_clean_dataset(train_path: str, eval_ngrams: set, threshold: float) -> list:
    """Scans train dataset, filters out samples exceeding overlap threshold, and returns clean samples."""
    clean_samples = []
    contaminated_samples = []
    
    if not os.path.exists(train_path):
        logger.error(f"Training dataset path {train_path} does not exist.")
        return []

    try:
        with open(train_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                sample = json.loads(line)
                
                # Combine all text fields in the sample for total representation
                sample_text = ""
                if "instruction" in sample:
                    sample_text += " " + sample["instruction"]
                if "input" in sample:
                    sample_text += " " + sample["input"]
                if "output" in sample:
                    sample_text += " " + sample["output"]
                if "prompt" in sample:
                    sample_text += " " + sample["prompt"]
                if "chosen" in sample:
                    sample_text += " " + sample["chosen"]
                if "rejected" in sample:
                    sample_text += " " + sample["rejected"]
                if "conversations" in sample:
                    for turn in sample["conversations"]:
                        if "value" in turn:
                            sample_text += " " + turn["value"]

                train_ngrams = get_ngrams(sample_text)
                overlap = calculate_overlap(train_ngrams, eval_ngrams)
                
                if overlap > threshold:
                    contaminated_samples.append({
                        "index": idx,
                        "overlap": overlap,
                        "sample": sample
                    })
                else:
                    clean_samples.append(sample)
    except Exception as e:
        logger.error(f"Error reading training dataset: {e}")
        return []

    # Write clean data back
    if contaminated_samples:
        logger.warning(f"⚠️ Detected {len(contaminated_samples)} contaminated samples! Rewriting clean dataset...")
        try:
            with open(train_path, "w", encoding="utf-8") as f:
                for sample in clean_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            logger.info(f"Successfully rewrote {train_path} without contaminated samples.")
        except Exception as e:
            logger.error(f"Failed to write clean dataset: {e}")
    else:
        logger.info("✅ No contaminated samples detected in training dataset.")

    # Save report
    os.makedirs("./outputs", exist_ok=True)
    report_path = "./outputs/contamination_report.json"
    report = {
        "dataset": train_path,
        "total_scanned": len(clean_samples) + len(contaminated_samples),
        "total_contaminated": len(contaminated_samples),
        "contamination_ratio": len(contaminated_samples) / max(1, len(clean_samples) + len(contaminated_samples)),
        "contaminated_samples_info": [
            {"index": item["index"], "overlap": item["overlap"]} for item in contaminated_samples
        ]
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Contamination report successfully saved to {report_path}")
    
    return clean_samples

def main():
    parser = argparse.ArgumentParser(description="nano-llm: Data Contamination Detector")
    parser.add_argument("--train_file", type=str, required=True, help="Path to SFT or DPO training JSONL file")
    parser.add_argument("--eval_dir", type=str, default="./data/eval/", help="Directory containing eval benchmark files")
    parser.add_argument("--threshold", type=float, default=0.7, help="3-gram overlap threshold (0.0 to 1.0)")
    args = parser.parse_args()

    logger.info("--- Initiating Data Contamination Scan ---")
    eval_ngrams = load_eval_texts(args.eval_dir)
    logger.info(f"Compiled {len(eval_ngrams)} evaluation benchmark 3-grams.")
    
    scan_and_clean_dataset(args.train_file, eval_ngrams, args.threshold)

if __name__ == "__main__":
    main()
