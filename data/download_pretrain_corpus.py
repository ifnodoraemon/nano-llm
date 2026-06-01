import os
import argparse
import logging
from utils.hub_adapter import HubAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_fineweb_edu(adapter: HubAdapter, output_dir: str, num_docs: int):
    logger.info(f"Streaming FineWeb-Edu (sample-10BT) to download {num_docs} documents...")
    try:
        # Load dataset in streaming mode to save local disk/memory
        dataset = adapter.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        count = 0
        os.makedirs(output_dir, exist_ok=True)
        
        # Batch write documents to text files
        batch_size = 5000
        current_text = []
        file_idx = 1
        
        for sample in dataset:
            text = sample.get("text", "").strip()
            if text:
                current_text.append(text)
                count += 1
                
            if len(current_text) >= batch_size:
                out_path = os.path.join(output_dir, f"fineweb_edu_{file_idx}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n\n=== DOCUMENT SPLIT ===\n\n".join(current_text))
                logger.info(f"✅ Saved batch {file_idx} ({len(current_text)} documents) to {out_path}")
                current_text = []
                file_idx += 1
                
            if count >= num_docs:
                break
                
        # Write remaining
        if current_text:
            out_path = os.path.join(output_dir, f"fineweb_edu_{file_idx}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n=== DOCUMENT SPLIT ===\n\n".join(current_text))
            logger.info(f"✅ Saved final batch {file_idx} ({len(current_text)} documents) to {out_path}")
            
        logger.info(f"🎉 Successfully downloaded {count} FineWeb-Edu documents.")
    except Exception as e:
        logger.error(f"Failed to stream FineWeb-Edu: {e}")

def download_skypile(adapter: HubAdapter, output_dir: str, num_docs: int):
    logger.info(f"Streaming SkyPile-150B to download {num_docs} Chinese documents...")
    try:
        # Load Skywork/SkyPile-150B in streaming mode
        dataset = adapter.load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)
        
        count = 0
        os.makedirs(output_dir, exist_ok=True)
        
        batch_size = 5000
        current_text = []
        file_idx = 1
        
        for sample in dataset:
            text = sample.get("text", "").strip()
            if text:
                current_text.append(text)
                count += 1
                
            if len(current_text) >= batch_size:
                out_path = os.path.join(output_dir, f"skypile_{file_idx}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n\n=== DOCUMENT SPLIT ===\n\n".join(current_text))
                logger.info(f"✅ Saved Chinese batch {file_idx} ({len(current_text)} documents) to {out_path}")
                current_text = []
                file_idx += 1
                
            if count >= num_docs:
                break
                
        # Write remaining
        if current_text:
            out_path = os.path.join(output_dir, f"skypile_{file_idx}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n=== DOCUMENT SPLIT ===\n\n".join(current_text))
            logger.info(f"✅ Saved final Chinese batch {file_idx} ({len(current_text)} documents) to {out_path}")
            
        logger.info(f"🎉 Successfully downloaded {count} SkyPile documents.")
    except Exception as e:
        logger.error(f"Failed to stream SkyPile-150B: {e}")

def main():
    parser = argparse.ArgumentParser(description="nano-llm: SOTA Pre-training Corpus Downloader")
    parser.add_argument("--provider", type=str, default=None, choices=["hf", "ms"], help="Hub provider override")
    parser.add_argument("--output_dir", type=str, default="./data/pretrain_corpus", help="Target output folder")
    parser.add_argument("--num_en_docs", type=int, default=100000, help="Number of English documents (FineWeb-Edu)")
    parser.add_argument("--num_zh_docs", type=int, default=50000, help="Number of Chinese documents (SkyPile-150B)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = HubAdapter(provider=args.provider)
    
    # 1. Download English educational documents
    if args.num_en_docs > 0:
        download_fineweb_edu(adapter, args.output_dir, args.num_en_docs)
        
    # 2. Download Chinese general documents
    if args.num_zh_docs > 0:
        download_skypile(adapter, args.output_dir, args.num_zh_docs)
        
    logger.info("=======================================================================")
    logger.info(f"✅ Pre-training corpus streaming downloads successfully completed!")
    logger.info(f"📂 Corpus saved under folder: {args.output_dir}")
    logger.info("=======================================================================")

if __name__ == "__main__":
    main()
