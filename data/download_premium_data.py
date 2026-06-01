import os
import json
import logging
import argparse
from utils.hub_adapter import HubAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_all(sft_size=5000, code_size=2000, dpo_size=2000, grpo_size=2000):
    adapter = HubAdapter()
    os.makedirs("./data", exist_ok=True)
    
    # 1. Download Premium SFT Dataset (tatsu-lab/alpaca subset + CodeAlpaca subset)
    sft_samples = []
    
    # 1a. General Alpaca SFT
    logger.info(f"📦 Fetching premium SFT data (tatsu-lab/alpaca subset: {sft_size} samples)...")
    try:
        sft_ds = adapter.load_dataset("tatsu-lab/alpaca", split="train")
        count = 0
        for sample in sft_ds:
            if count >= sft_size:
                break
            instruction = sample.get("instruction", "").strip()
            inp = sample.get("input", "").strip()
            output = sample.get("output", "").strip()
            
            if instruction and output:
                user_content = f"{instruction}\n{inp}" if inp else instruction
                chatml_messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]
                sft_samples.append({"messages": chatml_messages})
                count += 1
        logger.info(f"✅ Loaded {count} samples from tatsu-lab/alpaca.")
    except Exception as e:
        logger.error(f"Failed SFT general fetch: {e}")

    # 1b. Programming CodeAlpaca SFT
    logger.info(f"📦 Fetching premium SFT coding data (sahil2801/CodeAlpaca-20k subset: {code_size} samples)...")
    try:
        code_ds = adapter.load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        count = 0
        for sample in code_ds:
            if count >= code_size:
                break
            instruction = sample.get("instruction", "").strip()
            inp = sample.get("input", "").strip()
            output = sample.get("output", "").strip()
            
            if instruction and output:
                user_content = f"{instruction}\n{inp}" if inp else instruction
                chatml_messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]
                sft_samples.append({"messages": chatml_messages})
                count += 1
        logger.info(f"✅ Loaded {count} samples from lucasmccabe-lmi/CodeAlpaca-20k.")
    except Exception as e:
        logger.error(f"Failed SFT coding fetch: {e}")

    # Shuffle the combined dataset for optimal training mixture
    import random
    random.shuffle(sft_samples)
    
    try:
        with open("./data/train_sft_premium.jsonl", "w", encoding="utf-8") as f:
            for item in sft_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"✅ Combined premium SFT data saved: {len(sft_samples)} samples to ./data/train_sft_premium.jsonl")
    except Exception as e:
        logger.error(f"Failed SFT save: {e}")
        
    # 2. Download Premium DPO Dataset (orca_dpo_pairs subset)
    logger.info(f"📦 Fetching premium DPO data (orca_dpo_pairs subset: {dpo_size} samples)...")
    try:
        dpo_ds = adapter.load_dataset("Intel/orca_dpo_pairs", split="train")
        dpo_samples = []
        count = 0
        for sample in dpo_ds:
            if count >= dpo_size:
                break
            system = sample.get("system", "")
            question = sample.get("question", "")
            chosen = sample.get("chosen", "")
            rejected = sample.get("rejected", "")
            
            if question and chosen and rejected:
                prompt_messages = []
                if system:
                    prompt_messages.append({"role": "system", "content": system})
                prompt_messages.append({"role": "user", "content": question})
                
                dpo_samples.append({
                    "prompt": prompt_messages,
                    "chosen": {"role": "assistant", "content": chosen},
                    "rejected": {"role": "assistant", "content": rejected}
                })
                count += 1
                
        with open("./data/train_dpo_premium.jsonl", "w", encoding="utf-8") as f:
            for item in dpo_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"✅ Premium DPO data saved: {count} samples to ./data/train_dpo_premium.jsonl")
    except Exception as e:
        logger.error(f"Failed DPO fetch: {e}")
        
    # 3. Download Premium GRPO Dataset (GSM8K math reasoning)
    logger.info(f"📦 Fetching premium GRPO reasoning data (GSM8K subset: {grpo_size} samples)...")
    try:
        grpo_ds = adapter.load_dataset("gsm8k", split="train", name="main")
        grpo_samples = []
        count = 0
        for sample in grpo_ds:
            if count >= grpo_size:
                break
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            # Extract final number from GSM8K answer (format: "#### 42")
            if "####" in answer:
                ground_truth = answer.split("####")[-1].strip()
                prompt_seed = (
                    f"Question: {question} Think step-by-step. "
                    f"Wrap your step-by-step reasoning process inside <think>...</think> tags, "
                    f"and wrap your final correct number output inside <answer>...</answer> tags."
                )
                grpo_samples.append({
                    "prompt": prompt_seed,
                    "ground_truth": ground_truth
                })
                count += 1
                
        with open("./data/train_grpo_premium.jsonl", "w", encoding="utf-8") as f:
            for item in grpo_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"✅ Premium GRPO data saved: {count} samples to ./data/train_grpo_premium.jsonl")
    except Exception as e:
        logger.error(f"Failed GRPO fetch: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-llm: Premium Dataset Downloader")
    parser.add_argument("--sft_size", type=int, default=5000, help="SFT sample count")
    parser.add_argument("--code_size", type=int, default=2000, help="SFT coding sample count")
    parser.add_argument("--dpo_size", type=int, default=2000, help="DPO sample count")
    parser.add_argument("--grpo_size", type=int, default=2000, help="GRPO reasoning count")
    args = parser.parse_args()
    
    download_all(
        sft_size=args.sft_size,
        code_size=args.code_size,
        dpo_size=args.dpo_size,
        grpo_size=args.grpo_size
    )
