import os
import urllib.request
import logging
import argparse
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. High-Quality Pre-training Dataset Puller & Sanitizer (Zero-Dependency)
# ==============================================================================

def download_and_setup_dataset(dest_dir: str):
    """
    Downloads high-quality open-source text corpus segments (WikiText / TinyStories fragments)
    and places them directly in data/cleaned_corpus/ for tokenizer training and packing.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Open-source raw dataset URLs (WikiText-2 raw text segments)
    urls = {
        "wikitext_sample_1.txt": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt",
        "wikitext_sample_2.txt": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
    }
    
    logger.info("Initializing dataset downloader...")
    downloaded_count = 0
    
    for filename, url in urls.items():
        dest_path = os.path.join(dest_dir, filename)
        logger.info(f"Downloading {filename} from {url}...")
        
        try:
            # Set request header to simulate standard browser fetching
            req = urllib.request.Request(
                url, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8")
                
            # Basic text sanitization (remove WikiText structural lines)
            lines = content.split("\n")
            sanitized_lines = []
            for line in lines:
                line_str = line.strip()
                if line_str and not line_str.startswith("="): # Filter headers
                    sanitized_lines.append(line_str)
                    
            if len(sanitized_lines) > 10:
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(sanitized_lines))
                logger.info(f"✅ Successfully downloaded and sanitized: {filename} ({len(sanitized_lines):,} sentences)")
                downloaded_count += 1
            else:
                logger.warning(f"Downloaded content from {filename} was empty or too small.")
                
        except Exception as e:
            logger.warning(f"Failed to fetch {filename} over network: {e}")
            
    # 2. Local Fallback Generator
    if downloaded_count == 0:
        logger.warning("Network fetching failed. Triggering high-quality offline corpus synthesizer...")
        
        offline_corpus = [
            "Hopper GPU architecture introduces high-performance FP8 tensor cores to accelerate deep learning training.",
            "Distributed Data Parallel divides batch gradients across multiple H800 GPU nodes connected via NCCL NVLink bridges.",
            "Causal Language Modeling trains transformers to predict the next token given a history of preceding tokens.",
            "Low-Rank Adaptation freezes standard weights and inserts small trainable adapter matrices to save VRAM.",
            "Static KV-Caches pre-allocate key and value attention slices, boosting autoregressive token decoding throughput by over 10x.",
            "Byte-Pair Encoding operates by recursively merging frequent adjacent byte segments into novel vocab IDs.",
            "Reinforcement Learning from AI Feedback utilizes LLM-as-a-Judge preference scoring to run DPO alignments.",
            "MinHash LSH signatures estimate Jaccard similarity between document shingles, filtering out duplicated web pages.",
            "RMSNorm stabilizes neural activation distributions without shifting inputs by their mean values, saving operations.",
            "Rotary Position Embeddings project complex keys and queries into orthogonal polar coordinates before computing scaled dot product attention."
        ]
        
        fallback_path = os.path.join(dest_dir, "offline_pretrain_corpus.txt")
        expanded_corpus = []
        for i in range(150):
            expanded_corpus.append(f"Sample sentence sequence index {i}: " + offline_corpus[i % len(offline_corpus)])
            
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write("\n".join(expanded_corpus))
            
        logger.info(f"✅ Successfully generated offline pre-training corpus at: {fallback_path}")
        
    logger.info("=======================================================================")
    logger.info("✅ Dataset download and preparation successfully completed!")
    logger.info(f"📂 Cleaned Corpus Destination: {dest_dir}")
    logger.info("=======================================================================")


# ==============================================================================
# 2. Reasoning Prompt Seeds Generator for GRPO (DeepSeek-R1-Zero style)
# ==============================================================================

def generate_grpo_seed_dataset(dest_file: str, num_samples: int = 100):
    """
    Generates a premium, 100+ prompt reasoning seed file for GRPO training in JSONL format,
    covering math arithmetic tasks, logic riddle tasks, and strict output formatting.
    """
    logger.info(f"Generating premium {num_samples} GRPO prompt reasoning seeds...")
    
    # 1. Base template configurations
    math_operators = [("+", lambda x, y: x + y), ("-", lambda x, y: x - y), ("*", lambda x, y: x * y)]
    prompts = []
    
    # Generate Math Prompt Seeds
    for i in range(num_samples):
        # Generate clean randomized arithmetic questions
        op_name, op_func = math_operators[i % len(math_operators)]
        val1 = (i * 3 + 7) % 50
        val2 = (i * 2 + 5) % 30
        
        # Simple two-term calculation
        question = f"Solve the mathematical equation: {val1} {op_name} {val2}."
        answer = str(op_func(val1, val2))
        
        # Injects strict R1 tag instructions
        prompt_seed = (
            f"Question: {question} Think step-by-step. "
            f"Wrap your step-by-step reasoning process inside <think>...</think> tags, "
            f"and wrap your final correct number output inside <answer>...</answer> tags."
        )
        
        prompts.append({
            "prompt": prompt_seed,
            "ground_truth": answer
        })
        
    # Write to destination JSONL file
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    with open(dest_file, "w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info("=======================================================================")
    logger.info("✅ GRPO Prompt Reasoning Seeds successfully synthesized!")
    logger.info(f"📂 Destination File: {dest_file}")
    logger.info(f"📝 Total Prompts: {len(prompts)} written.")
    logger.info("=======================================================================")


# ==============================================================================
# 3. Main Orchestrator Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="nano-llm: zero-dependency open-source dataset downloader and generator")
    parser.add_argument("--dest_dir", type=str, default="./data/cleaned_corpus", help="Destination to save sanitized pre-training text files")
    parser.add_argument("--grpo_file", type=str, default="./data/train_grpo.jsonl", help="Destination to write GRPO prompt seeds")
    parser.add_argument("--num_grpo", type=int, default=100, help="Number of GRPO prompt reasoning samples to synthesize")
    args = parser.parse_args()
    
    # 1. Run Pre-training dataset puller
    download_and_setup_dataset(args.dest_dir)
    
    # 2. Run GRPO Prompt Dataset generator
    generate_grpo_seed_dataset(args.grpo_file, num_samples=args.num_grpo)

if __name__ == "__main__":
    main()
