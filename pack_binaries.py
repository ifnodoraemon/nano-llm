import os
import array
import logging
import argparse
from typing import List

from train_tokenizer import CustomBPETokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Pure Python Binary Token Packing (Zero-Dependency)
# ==============================================================================

def pack_corpus_to_binaries(
    src_dir: str, 
    tokenizer_file: str, 
    dest_dir: str, 
    val_ratio: float = 0.1
):
    """
    Loads custom BPE rules, tokenizes all cleaned text files, 
    and packs them into flat uint16 binary files train.bin and val.bin.
    """
    logger.info(f"Loading custom trained BPE Tokenizer from: {tokenizer_file}")
    tokenizer = CustomBPETokenizer()
    tokenizer.load(tokenizer_file)
    
    # 1. Read all unique clean text files
    doc_paths = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".txt")]
    logger.info(f"Tokenizing {len(doc_paths)} unique corpus documents...")
    
    all_tokens = []
    
    for doc_path in sorted(doc_paths):
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            
        if not text:
            continue
            
        # Encode string to list of BPE token IDs
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        
        logger.debug(f"Encoded '{os.path.basename(doc_path)}': {len(tokens)} tokens.")
        
    total_tokens = len(all_tokens)
    logger.info(f"Total tokens in compiled corpus: {total_tokens:,} tokens.")
    
    if total_tokens == 0:
        logger.error("No tokens found. Ensure you crawled and deduplicated text first.")
        return
        
    # 2. Split into Training (90%) and Validation (10%)
    split_idx = int(total_tokens * (1 - val_ratio))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    logger.info(f"Split: train={len(train_tokens):,} tokens | val={len(val_tokens):,} tokens")
    
    # 3. Serialize to binary files using standard python array (unsigned 16-bit 'H' arrays)
    # This matches numpy.uint16 format perfectly and works out-of-the-box with zero-dependencies!
    train_bin_path = os.path.join(dest_dir, "train.bin")
    val_bin_path = os.path.join(dest_dir, "val.bin")
    
    logger.info(f"Writing packed binary array to: {train_bin_path}...")
    with open(train_bin_path, "wb") as f:
        # 'H' represents unsigned short (16-bit integer, values 0 to 65535)
        train_arr = array.array("H", train_tokens)
        train_arr.tofile(f)
        
    logger.info(f"Writing packed binary array to: {val_bin_path}...")
    with open(val_bin_path, "wb") as f:
        val_arr = array.array("H", val_tokens)
        val_arr.tofile(f)
        
    logger.info("=======================================================================")
    logger.info("✅ Binary Token Packing Complete!")
    logger.info(f"💾 Train Binary: {train_bin_path} | Size: {os.path.getsize(train_bin_path)/(1024):.1f} KB")
    logger.info(f"💾 Val Binary: {val_bin_path} | Size: {os.path.getsize(val_bin_path)/(1024):.1f} KB")
    logger.info("=======================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-llm: zero-dependency binary pre-training tokenizer packer")
    parser.add_argument("--src_dir", type=str, default="./data/cleaned_corpus", help="Directory containing deduplicated .txt files")
    parser.add_argument("--tokenizer", type=str, default="./data/custom_tokenizer.json", help="Path to trained custom_tokenizer.json file")
    parser.add_argument("--dest_dir", type=str, default="./data", help="Output directory to save train.bin and val.bin")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation dataset split ratio")
    args = parser.parse_args()
    
    pack_corpus_to_binaries(
        src_dir=args.src_dir,
        tokenizer_file=args.tokenizer,
        dest_dir=args.dest_dir,
        val_ratio=args.val_ratio
    )
