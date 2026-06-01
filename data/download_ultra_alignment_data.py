import os
import re
import json
import random
import argparse
import logging
from utils.hub_adapter import HubAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_boxed_answer(solution: str) -> str:
    """Extracts answer inside \boxed{...} for MATH dataset."""
    match = re.search(r"\\boxed\{(.*?)\}", solution)
    if match:
        return match.group(1).strip()
    return ""

def download_sft_data(adapter: HubAdapter, output_dir: str, sft_size: int, code_size: int):
    logger.info("📦 Fetching premium SFT data (OpenHermes-2.5 + CodeAlpaca)...")
    sft_samples = []
    
    # 1. OpenHermes-2.5 SFT
    try:
        logger.info(f"Loading OpenHermes-2.5 subset ({sft_size} samples)...")
        # OpenHermes-2.5 can be large, we stream or load and slice
        hermes_ds = adapter.load_dataset("teknium/OpenHermes-2.5", split="train")
        count = 0
        for sample in hermes_ds:
            if count >= sft_size:
                break
                
            conversations = sample.get("conversations", [])
            # Convert conversations format (from/value) to ChatML (role/content)
            chatml_messages = []
            valid = True
            for msg in conversations:
                from_role = msg.get("from", "")
                val = msg.get("value", "")
                if from_role == "human":
                    chatml_messages.append({"role": "user", "content": val})
                elif from_role == "gpt":
                    chatml_messages.append({"role": "assistant", "content": val})
                elif from_role == "system":
                    chatml_messages.append({"role": "system", "content": val})
                else:
                    valid = False
                    break
                    
            if valid and len(chatml_messages) >= 2:
                sft_samples.append({"messages": chatml_messages})
                count += 1
        logger.info(f"✅ Loaded {count} samples from OpenHermes-2.5.")
    except Exception as e:
        logger.error(f"Failed to fetch OpenHermes-2.5 SFT: {e}. Falling back to default Alpaca.")
        # Fallback to standard alpaca SFT if OpenHermes is inaccessible
        try:
            alpaca_ds = adapter.load_dataset("tatsu-lab/alpaca", split="train")
            count = 0
            for sample in alpaca_ds:
                if count >= sft_size:
                    break
                instruction = sample.get("instruction", "").strip()
                inp = sample.get("input", "").strip()
                output = sample.get("output", "").strip()
                if instruction and output:
                    user_content = f"{instruction}\n{inp}" if inp else instruction
                    sft_samples.append({
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": output}
                        ]
                    })
                    count += 1
        except Exception as ex:
            logger.error(f"SFT fallback download also failed: {ex}")

    # 2. Programming CodeAlpaca
    try:
        logger.info(f"Loading CodeAlpaca-20k subset ({code_size} samples)...")
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
                sft_samples.append({
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output}
                    ]
                })
                count += 1
        logger.info(f"✅ Loaded {count} samples from CodeAlpaca.")
    except Exception as e:
        logger.error(f"Failed to fetch CodeAlpaca: {e}")

    # Mix and Shuffle
    random.shuffle(sft_samples)
    
    out_path = os.path.join(output_dir, "train_sft_premium.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for item in sft_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"🚀 Premium SFT data saved: {len(sft_samples)} samples to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save premium SFT file: {e}")

def download_dpo_data(adapter: HubAdapter, output_dir: str, dpo_size: int):
    logger.info("📦 Fetching premium DPO data (UltraFeedback Binarized)...")
    dpo_samples = []
    
    try:
        logger.info(f"Loading ultrafeedback_binarized subset ({dpo_size} samples)...")
        uf_ds = adapter.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        count = 0
        for sample in uf_ds:
            if count >= dpo_size:
                break
                
            prompt = sample.get("prompt", "")
            chosen = sample.get("chosen", [])
            rejected = sample.get("rejected", [])
            
            # Extract content from final message
            chosen_content = chosen[-1].get("content", "") if chosen else ""
            rejected_content = rejected[-1].get("content", "") if rejected else ""
            
            if prompt and chosen_content and rejected_content:
                dpo_samples.append({
                    "prompt": [{"role": "user", "content": prompt}],
                    "chosen": {"role": "assistant", "content": chosen_content},
                    "rejected": {"role": "assistant", "content": rejected_content}
                })
                count += 1
        logger.info(f"✅ Loaded {count} samples from UltraFeedback.")
    except Exception as e:
        logger.error(f"Failed to fetch UltraFeedback: {e}. Falling back to Intel/orca_dpo_pairs.")
        try:
            orca_ds = adapter.load_dataset("Intel/orca_dpo_pairs", split="train")
            count = 0
            for sample in orca_ds:
                if count >= dpo_size:
                    break
                question = sample.get("question", "")
                chosen_ans = sample.get("chosen", "")
                rejected_ans = sample.get("rejected", "")
                if question and chosen_ans and rejected_ans:
                    dpo_samples.append({
                        "prompt": [{"role": "user", "content": question}],
                        "chosen": {"role": "assistant", "content": chosen_ans},
                        "rejected": {"role": "assistant", "content": rejected_ans}
                    })
                    count += 1
        except Exception as ex:
            logger.error(f"DPO fallback download also failed: {ex}")
            
    out_path = os.path.join(output_dir, "train_dpo_premium.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for item in dpo_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"🚀 Premium DPO data saved: {len(dpo_samples)} samples to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save premium DPO file: {e}")

def download_grpo_data(adapter: HubAdapter, output_dir: str, math_size: int, code_size: int):
    logger.info("📦 Fetching premium GRPO reasoning data (GSM8K + MATH + MBPP)...")
    grpo_samples = []
    
    # 1. GSM8K Reasoning
    try:
        logger.info(f"Loading GSM8K training set ({math_size} samples)...")
        gsm_ds = adapter.load_dataset("gsm8k", split="train", name="main")
        count = 0
        for sample in gsm_ds:
            if count >= math_size:
                break
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            if "####" in answer:
                ground_truth = answer.split("####")[-1].strip()
                prompt_seed = (
                    f"Question: {question} Think step-by-step. "
                    f"Wrap your step-by-step reasoning process inside <think>...</think> tags, "
                    f"and wrap your final correct number output inside <answer>...</answer> tags."
                )
                grpo_samples.append({
                    "prompt": prompt_seed,
                    "ground_truth": ground_truth,
                    "task_type": "math"
                })
                count += 1
        logger.info(f"✅ Loaded {count} samples from GSM8K.")
    except Exception as e:
        logger.error(f"Failed to fetch GSM8K for GRPO: {e}")

    # 2. MATH dataset
    try:
        logger.info(f"Loading Competition MATH training set ({math_size} samples)...")
        math_ds = adapter.load_dataset("competition_math", split="train")
        count = 0
        for sample in math_ds:
            if count >= math_size:
                break
            problem = sample.get("problem", "")
            solution = sample.get("solution", "")
            ans = extract_boxed_answer(solution)
            if problem and ans:
                prompt_seed = (
                    f"Problem: {problem}\nSolve the problem step-by-step. "
                    f"Wrap your reasoning process inside <think>...</think> tags, "
                    f"and wrap your final boxed or numerical answer inside <answer>...</answer> tags."
                )
                grpo_samples.append({
                    "prompt": prompt_seed,
                    "ground_truth": ans,
                    "task_type": "math"
                })
                count += 1
        logger.info(f"✅ Loaded {count} samples from MATH.")
    except Exception as e:
        logger.error(f"Failed to fetch Competition MATH: {e}")

    # 3. MBPP Coding dataset
    try:
        logger.info(f"Loading MBPP coding training set ({code_size} samples)...")
        mbpp_ds = adapter.load_dataset("mbpp", split="train")
        count = 0
        for sample in mbpp_ds:
            if count >= code_size:
                break
            text = sample.get("text", "")
            test_list = sample.get("test_list", [])
            code = sample.get("code", "")
            if text and test_list:
                prompt_seed = (
                    f"Write a Python function to solve the following problem:\n{text}\n"
                    f"Make sure to wrap your reasoning inside <think>...</think> tags, "
                    f"and wrap your final Python code block inside <answer>...</answer> tags."
                )
                grpo_samples.append({
                    "prompt": prompt_seed,
                    "ground_truth": json.dumps(test_list), # Store assertions as JSON string
                    "task_type": "code"
                })
                count += 1
        logger.info(f"✅ Loaded {count} samples from MBPP.")
    except Exception as e:
        logger.error(f"Failed to fetch MBPP: {e}")

    # Mix and Shuffle
    random.shuffle(grpo_samples)
    
    out_path = os.path.join(output_dir, "train_grpo_premium.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for item in grpo_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"🚀 Premium GRPO data saved: {len(grpo_samples)} samples to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save premium GRPO file: {e}")

def main():
    parser = argparse.ArgumentParser(description="nano-llm: SOTA Alignment Dataset Downloader")
    parser.add_argument("--provider", type=str, default=None, choices=["hf", "ms"], help="Hub provider override")
    parser.add_argument("--output_dir", type=str, default="./data", help="Target output folder")
    parser.add_argument("--sft_size", type=int, default=5000, help="SFT OpenHermes sample count")
    parser.add_argument("--code_size", type=int, default=2000, help="SFT CodeAlpaca sample count")
    parser.add_argument("--dpo_size", type=int, default=3000, help="DPO UltraFeedback sample count")
    parser.add_argument("--math_size", type=int, default=2000, help="GRPO math sample count")
    parser.add_argument("--grpo_code_size", type=int, default=1000, help="GRPO coding sample count")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = HubAdapter(provider=args.provider)
    
    download_sft_data(adapter, args.output_dir, args.sft_size, args.code_size)
    download_dpo_data(adapter, args.output_dir, args.dpo_size)
    download_grpo_data(adapter, args.output_dir, args.math_size, args.grpo_code_size)
    
    logger.info("=======================================================================")
    logger.info("✅ All SOTA alignment datasets successfully downloaded and compiled!")
    logger.info("=======================================================================")

if __name__ == "__main__":
    main()
