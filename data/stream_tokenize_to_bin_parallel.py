import os
import random
import argparse
import logging
import multiprocessing as mp
import numpy as np
import requests
from urllib.parse import urlparse

# ==============================================================================
# Transparent Hugging Face Mirror Rewriter (Monkeypatch)
# ==============================================================================
# In regions where huggingface.co is blocked, huggingface_hub uses HF_ENDPOINT (e.g. hf-mirror.com).
# However, Hugging Face API pagination uses the HTTP Link header, which returns absolute URLs
# pointing directly to huggingface.co. To prevent pagination from hanging, we rewrite all
# requests targeting huggingface.co to use the configured mirror.
original_request = requests.Session.request

def patched_request(self, method, url, *args, **kwargs):
    if isinstance(url, str) and "huggingface.co" in url:
        endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").strip()
        parsed = urlparse(endpoint)
        domain = parsed.netloc or parsed.path
        if domain:
            url = url.replace("huggingface.co", domain)
    return original_request(self, method, url, *args, **kwargs)

requests.Session.request = patched_request

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def worker_tokenize_task(
    worker_idx: int,
    num_workers: int,
    dataset_id: str,
    config_name: str,
    split: str,
    target_tokens: int,
    output_dir: str,
    prefix: str,
    val_ratio: float,
    skip_docs: int = 0,
):
    # Set mirror in environment for sub-processes
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    # Configure logger for this process
    proc_logger = logging.getLogger(f"Worker-{worker_idx}")
    proc_logger.setLevel(logging.INFO)
    
    proc_logger.info(f"🚀 Worker {worker_idx}/{num_workers} starting for {dataset_id} ({prefix}). Target: {target_tokens:,} tokens.")
    
    try:
        from utils.tokenizer_loader import load_tokenizer
        from utils.hub_adapter import HubAdapter
        
        # Load fast tokenizer
        tokenizer = load_tokenizer()
        
        import time
        import random
        
        # Add random startup jitter to prevent concurrent api rate-limiting spikes
        startup_delay = random.uniform(1.0, 15.0)
        proc_logger.info(f"Waiting {startup_delay:.1f}s jitter delay to prevent API rate limiting...")
        time.sleep(startup_delay)
        
        # Load dataset in streaming mode with retry loop
        adapter = HubAdapter(provider="hf")
        dataset = None
        max_retries = 8
        for retry in range(max_retries):
            try:
                dataset = adapter.load_dataset(dataset_id, name=config_name, split=split, streaming=True)
                break
            except Exception as e:
                if retry < max_retries - 1:
                    sleep_time = random.uniform(10.0, 30.0) * (retry + 1)
                    proc_logger.warning(f"Failed to load dataset (retry {retry+1}/{max_retries}) due to: {e}. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    raise e
        
        # Shard the dataset among workers
        sharded_dataset = dataset.shard(num_shards=num_workers, index=worker_idx)
        if skip_docs > 0:
            proc_logger.info(f"Skipping first {skip_docs:,} documents for worker {worker_idx}...")
            sharded_dataset = sharded_dataset.skip(skip_docs)
        
        # Output binary paths
        train_path = os.path.join(output_dir, f"{prefix}_train_worker_{worker_idx}.bin")
        val_path = os.path.join(output_dir, f"{prefix}_val_worker_{worker_idx}.bin")
        
        # Open write streams
        f_train = open(train_path, "wb")
        f_val = open(val_path, "wb")
        
        total_tokens_written = 0
        train_buffer = []
        val_buffer = []
        batch_write_size = 20000
        
        doc_count = 0
        for sample in sharded_dataset:
            text = sample.get("text", "").strip()
            if not text:
                continue
                
            # Tokenize text
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
            except Exception as e:
                proc_logger.warning(f"Error encoding sample {doc_count}: {e}")
                continue
                
            if not tokens:
                continue
                
            doc_count += 1
            num_tokens = len(tokens)
            
            # Determine split (probabilistic SFT/val split)
            if random.random() < val_ratio:
                val_buffer.extend(tokens)
            else:
                train_buffer.extend(tokens)
                
            total_tokens_written += num_tokens
            
            # Flush buffers to file if they exceed batch_write_size
            if len(train_buffer) >= batch_write_size:
                arr = np.array(train_buffer, dtype=np.uint16)
                f_train.write(arr.tobytes())
                train_buffer = []
                
            if len(val_buffer) >= batch_write_size:
                arr = np.array(val_buffer, dtype=np.uint16)
                f_val.write(arr.tobytes())
                val_buffer = []
                
            if doc_count % 10000 == 0:
                proc_logger.info(f"Processed {doc_count:,} docs | Total tokens written: {total_tokens_written:,}")
                
            if total_tokens_written >= target_tokens:
                proc_logger.info(f"🎯 Target token count reached ({total_tokens_written:,} tokens). Stopping.")
                break
                
        # Flush remaining buffers
        if train_buffer:
            arr = np.array(train_buffer, dtype=np.uint16)
            f_train.write(arr.tobytes())
        if val_buffer:
            arr = np.array(val_buffer, dtype=np.uint16)
            f_val.write(arr.tobytes())
            
        f_train.close()
        f_val.close()
        
        proc_logger.info(f"✅ Worker {worker_idx} completed. Final tokens: {total_tokens_written:,} | Docs: {doc_count:,}")
        return total_tokens_written
        
    except Exception as e:
        proc_logger.error(f"❌ Worker {worker_idx} crashed: {e}")
        import traceback
        proc_logger.error(traceback.format_exc())
        return 0

def merge_binary_files(output_dir: str, prefix: str, num_workers: int, dest_dir: str):
    logger.info(f"🔗 Merging worker binary shards for prefix '{prefix}'...")
    
    train_dest = os.path.join(dest_dir, f"{prefix}_train_merged.bin")
    val_dest = os.path.join(dest_dir, f"{prefix}_val_merged.bin")
    
    # 1. Merge train shards
    logger.info(f"Combining train shards into {train_dest}...")
    with open(train_dest, "wb") as out_f:
        for i in range(num_workers):
            shard_path = os.path.join(output_dir, f"{prefix}_train_worker_{i}.bin")
            if os.path.exists(shard_path) and os.path.getsize(shard_path) > 0:
                with open(shard_path, "rb") as in_f:
                    while True:
                        chunk = in_f.read(64 * 1024 * 1024) # 64MB chunks
                        if not chunk:
                            break
                        out_f.write(chunk)
                # Cleanup shard
                os.remove(shard_path)
                
    # 2. Merge val shards
    logger.info(f"Combining val shards into {val_dest}...")
    with open(val_dest, "wb") as out_f:
        for i in range(num_workers):
            shard_path = os.path.join(output_dir, f"{prefix}_val_worker_{i}.bin")
            if os.path.exists(shard_path) and os.path.getsize(shard_path) > 0:
                with open(shard_path, "rb") as in_f:
                    while True:
                        chunk = in_f.read(64 * 1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
                # Cleanup shard
                os.remove(shard_path)
                
    logger.info(f"✅ Finished merging {prefix} shards. Train: {train_dest} | Val: {val_dest}")
    return train_dest, val_dest

def final_concatenate_and_rename(dest_dir: str, final_dir: str):
    logger.info("📦 Step 3: Performing final merge of English and Chinese corpora...")
    
    os.makedirs(final_dir, exist_ok=True)
    final_train_path = os.path.join(final_dir, "train.bin")
    final_val_path = os.path.join(final_dir, "val.bin")
    
    # Combine en and zh train
    logger.info(f"Writing final concatenated pretraining data to {final_train_path}...")
    with open(final_train_path, "wb") as out_f:
        for prefix in ["en", "zh"]:
            path = os.path.join(dest_dir, f"{prefix}_train_merged.bin")
            if os.path.exists(path):
                logger.info(f"Appending {path}...")
                with open(path, "rb") as in_f:
                    while True:
                        chunk = in_f.read(64 * 1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
                os.remove(path)
                
    # Combine en and zh val
    logger.info(f"Writing final concatenated validation data to {final_val_path}...")
    with open(final_val_path, "wb") as out_f:
        for prefix in ["en", "zh"]:
            path = os.path.join(dest_dir, f"{prefix}_val_merged.bin")
            if os.path.exists(path):
                logger.info(f"Appending {path}...")
                with open(path, "rb") as in_f:
                    while True:
                        chunk = in_f.read(64 * 1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
                os.remove(path)
                
    logger.info("=======================================================================")
    logger.info("🎉 All operations completed! Trillion-scale pretraining binaries compiled.")
    logger.info(f"📂 Train Binary: {final_train_path} | Size: {os.path.getsize(final_train_path)/(1024**3):.2f} GB")
    logger.info(f"📂 Val Binary: {final_val_path} | Size: {os.path.getsize(final_val_path)/(1024**3):.2f} GB")
    logger.info("=======================================================================")

def main():
    parser = argparse.ArgumentParser(description="nano-llm: Parallel Trillion-Scale Dataset Downloader and Tokenizer")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel downloading worker processes")
    parser.add_argument("--max_tokens", type=float, default=1e12, help="Target total tokens (e.g. 1e12 for 1 Trillion)")
    parser.add_argument("--en_ratio", type=float, default=0.8, help="English token ratio (0.0 to 1.0)")
    parser.add_argument("--val_ratio", type=float, default=0.005, help="Validation data split ratio")
    parser.add_argument("--temp_dir", type=str, default="./data/pretrain_temp", help="Temporary directory for shards")
    parser.add_argument("--dest_dir", type=str, default="./data/pretrain_merged", help="Merged files output directory")
    parser.add_argument("--final_dir", type=str, default="./data/binaries_1t", help="Final output folder for train.bin / val.bin")
    parser.add_argument("--skip_docs", type=int, default=0, help="Number of documents to skip per worker to resume/deduplicate stream")
    args = parser.parse_args()
    
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.dest_dir, exist_ok=True)
    os.makedirs(args.final_dir, exist_ok=True)
    
    max_tokens = int(args.max_tokens)
    target_en = int(max_tokens * args.en_ratio)
    target_zh = int(max_tokens * (1.0 - args.en_ratio))
    
    logger.info("=======================================================================")
    logger.info("🔥 Starting Parallel Trillion-Scale Streaming Pipeline")
    logger.info(f"   Target Total Tokens: {max_tokens:,}")
    logger.info(f"   English Tokens (FineWeb-Edu): {target_en:,} ({args.en_ratio*100}%)")
    logger.info(f"   Chinese Tokens (SkyPile-150B): {target_zh:,} ({(1.0-args.en_ratio)*100}%)")
    logger.info(f"   Workers: {args.num_workers} | Val Ratio: {args.val_ratio}")
    logger.info("=======================================================================")
    
    mp.set_start_method("spawn", force=True)
    
    # --------------------------------------------------------------------------
    # PHASE 1: English Corpus (FineWeb-Edu)
    # --------------------------------------------------------------------------
    if target_en > 0:
        logger.info("📦 Phase 1/2: Streaming English educational corpus (HuggingFaceFW/fineweb-edu)...")
        target_per_worker = target_en // args.num_workers
        
        processes = []
        for i in range(args.num_workers):
            p = mp.Process(
                target=worker_tokenize_task,
                args=(
                    i,
                    args.num_workers,
                    "HuggingFaceFW/fineweb-edu",
                    "default", # Default is the full 1.3T token dataset
                    "train",
                    target_per_worker,
                    args.temp_dir,
                    "en",
                    args.val_ratio,
                    args.skip_docs,
                )
            )
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            
        # Merge English shards
        merge_binary_files(args.temp_dir, "en", args.num_workers, args.dest_dir)
        
    # --------------------------------------------------------------------------
    # PHASE 2: Chinese Corpus (SkyPile-150B)
    # --------------------------------------------------------------------------
    if target_zh > 0:
        logger.info("📦 Phase 2/2: Streaming Chinese corpus (Skywork/SkyPile-150B)...")
        target_per_worker = target_zh // args.num_workers
        
        processes = []
        for i in range(args.num_workers):
            p = mp.Process(
                target=worker_tokenize_task,
                args=(
                    i,
                    args.num_workers,
                    "Skywork/SkyPile-150B",
                    None,
                    "train",
                    target_per_worker,
                    args.temp_dir,
                    "zh",
                    args.val_ratio,
                    args.skip_docs,
                )
            )
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            
        # Merge Chinese shards
        merge_binary_files(args.temp_dir, "zh", args.num_workers, args.dest_dir)
        
    # --------------------------------------------------------------------------
    # PHASE 3: Concatenate En & Zh
    # --------------------------------------------------------------------------
    final_concatenate_and_rename(args.dest_dir, args.final_dir)

if __name__ == "__main__":
    main()
