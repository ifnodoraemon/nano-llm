import os
import json
import logging
import random
import argparse
from utils.hub_adapter import HubAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_mmlu(adapter: HubAdapter, output_dir: str, limit: int = 100):
    logger.info(f"Downloading MMLU subset (elementary_mathematics)...")
    try:
        # Load elementary_mathematics subset for high-quality mathematical reasoning
        mmlu_ds = adapter.load_dataset("cais/mmlu", split="test", name="elementary_mathematics")
        samples = []
        for idx, sample in enumerate(mmlu_ds):
            if idx >= limit:
                break
            
            question = sample.get("question", "")
            choices = sample.get("choices", [])
            answer_idx = sample.get("answer", 0) # 0 to 3
            
            # Map choice index to ABCD
            answer_letter = ["A", "B", "C", "D"][answer_idx]
            
            formatted_choices = []
            for letter, choice in zip(["A", "B", "C", "D"], choices):
                formatted_choices.append(f"{letter}) {choice}")
                
            samples.append({
                "question": question,
                "choices": formatted_choices,
                "answer": answer_letter
            })
            
        out_path = os.path.join(output_dir, "mmlu.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"✅ Saved {len(samples)} MMLU samples to {out_path}")
    except Exception as e:
        logger.error(f"Failed to download MMLU: {e}")

def download_gsm8k(adapter: HubAdapter, output_dir: str, limit: int = 100):
    logger.info(f"Downloading GSM8K test set...")
    try:
        gsm_ds = adapter.load_dataset("gsm8k", split="test", name="main")
        samples = []
        for idx, sample in enumerate(gsm_ds):
            if idx >= limit:
                break
            
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            
            # Parse final numeric answer
            if "####" in answer:
                ground_truth = answer.split("####")[-1].strip()
            else:
                ground_truth = answer.strip()
                
            samples.append({
                "question": question,
                "answer": ground_truth,
                "reasoning": answer
            })
            
        out_path = os.path.join(output_dir, "gsm8k.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"✅ Saved {len(samples)} GSM8K samples to {out_path}")
    except Exception as e:
        logger.error(f"Failed to download GSM8K: {e}")

def download_arc(adapter: HubAdapter, output_dir: str, limit: int = 100):
    logger.info(f"Downloading ARC-Challenge test set...")
    try:
        arc_ds = adapter.load_dataset("ai2_arc", split="test", name="ARC-Challenge")
        samples = []
        for idx, sample in enumerate(arc_ds):
            if idx >= limit:
                break
            
            question = sample.get("question", "")
            choices_dict = sample.get("choices", {})
            labels = choices_dict.get("label", [])
            texts = choices_dict.get("text", [])
            answer_key = sample.get("answerKey", "")
            
            # Map labels to A, B, C, D if they are numbers or different letters
            # Normal label keys: A, B, C, D or 1, 2, 3, 4
            mapped_choices = []
            standard_letters = ["A", "B", "C", "D", "E", "F"]
            
            ans_letter = answer_key
            label_to_letter = {}
            
            for i, (lbl, text) in enumerate(zip(labels, texts)):
                standard_lbl = standard_letters[i] if i < len(standard_letters) else lbl
                label_to_letter[lbl] = standard_lbl
                mapped_choices.append(f"{standard_lbl}) {text}")
                
            ans_letter = label_to_letter.get(answer_key, answer_key)
            
            samples.append({
                "question": question,
                "choices": mapped_choices,
                "answer": ans_letter
            })
            
        out_path = os.path.join(output_dir, "arc.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"✅ Saved {len(samples)} ARC samples to {out_path}")
    except Exception as e:
        logger.error(f"Failed to download ARC: {e}")

def download_hellaswag(adapter: HubAdapter, output_dir: str, limit: int = 100):
    logger.info(f"Downloading HellaSwag validation set...")
    try:
        hs_ds = adapter.load_dataset("hellaswag", split="validation")
        samples = []
        for idx, sample in enumerate(hs_ds):
            if idx >= limit:
                break
            
            ctx = sample.get("ctx", "")
            endings = sample.get("endings", [])
            label_raw = sample.get("label", 0)
            
            try:
                label_idx = int(label_raw)
            except ValueError:
                label_idx = 0
                
            answer_letter = ["A", "B", "C", "D"][label_idx] if label_idx < 4 else "A"
            
            formatted_choices = []
            for i, ending in enumerate(endings[:4]):
                letter = ["A", "B", "C", "D"][i]
                formatted_choices.append(f"{letter}) {ending}")
                
            samples.append({
                "context": ctx,
                "choices": formatted_choices,
                "answer": answer_letter
            })
            
        out_path = os.path.join(output_dir, "hellaswag.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"✅ Saved {len(samples)} HellaSwag samples to {out_path}")
    except Exception as e:
        logger.error(f"Failed to download HellaSwag: {e}")

def download_humaneval(adapter: HubAdapter, output_dir: str, limit: int = 50):
    logger.info(f"Downloading HumanEval test set...")
    try:
        he_ds = adapter.load_dataset("openai_humaneval", split="test")
        samples = []
        for idx, sample in enumerate(he_ds):
            if idx >= limit:
                break
                
            samples.append({
                "task_id": sample.get("task_id", ""),
                "prompt": sample.get("prompt", ""),
                "test": sample.get("test", ""),
                "entry_point": sample.get("entry_point", ""),
                "canonical_solution": sample.get("canonical_solution", "")
            })
            
        out_path = os.path.join(output_dir, "humaneval.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"✅ Saved {len(samples)} HumanEval samples to {out_path}")
    except Exception as e:
        logger.error(f"Failed to download HumanEval: {e}")

def main():
    parser = argparse.ArgumentParser(description="nano-llm: Industry Benchmark Dataset Downloader")
    parser.add_argument("--provider", type=str, default=None, choices=["hf", "ms"], help="Hub provider override")
    parser.add_argument("--output_dir", type=str, default="./data/eval", help="Target output folder")
    parser.add_argument("--limit", type=int, default=100, help="Limit of samples per dataset")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = HubAdapter(provider=args.provider)
    
    download_mmlu(adapter, args.output_dir, limit=args.limit)
    download_gsm8k(adapter, args.output_dir, limit=args.limit)
    download_arc(adapter, args.output_dir, limit=args.limit)
    download_hellaswag(adapter, args.output_dir, limit=args.limit)
    download_humaneval(adapter, args.output_dir, limit=min(args.limit, 50))
    
    logger.info("=======================================================================")
    logger.info(f"✅ Evaluation benchmark download finished!")
    logger.info(f"📂 Saved files to folder: {args.output_dir}")
    logger.info("=======================================================================")

if __name__ == "__main__":
    main()
