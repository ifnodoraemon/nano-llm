import os
import array
import argparse
from tokenizers import Tokenizer

def pack_fast_binaries(src_dir, tokenizer_path, dest_dir, val_ratio=0.1):
    print(f"⚡ Loading fast BPE Tokenizer from: {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Read unique cleaned files
    doc_paths = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".txt")]
    print(f"🧩 Tokenizing {len(doc_paths)} super cleaned corpus documents...")
    
    all_tokens = []
    for doc_path in sorted(doc_paths):
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
            
        # Encode text to standard IDs
        encoded = tokenizer.encode(text)
        all_tokens.extend(encoded.ids)
        
    total_tokens = len(all_tokens)
    print(f"📊 Total tokens compiled in noise-free corpus: {total_tokens:,} tokens.")
    
    if total_tokens == 0:
        print("❌ Error: No tokens found. Verify clean text files exist.")
        return
        
    # Split training and validation subsets
    split_idx = int(total_tokens * (1 - val_ratio))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    print(f"📊 Split: train={len(train_tokens):,} tokens | val={len(val_tokens):,} tokens")
    
    train_bin_path = os.path.join(dest_dir, "train.bin")
    val_bin_path = os.path.join(dest_dir, "val.bin")
    
    os.makedirs(dest_dir, exist_ok=True)
    print(f"💾 Saving packed binary array to: {train_bin_path}...")
    with open(train_bin_path, "wb") as f:
        train_arr = array.array("H", train_tokens)
        train_arr.tofile(f)
        
    print(f"💾 Saving packed binary array to: {val_bin_path}...")
    with open(val_bin_path, "wb") as f:
        val_arr = array.array("H", val_tokens)
        val_arr.tofile(f)
        
    print("=======================================================================")
    print("✅ Noise-free binary token packing successfully completed!")
    print(f"   Train Binary: {train_bin_path} | Size: {os.path.getsize(train_bin_path)/(1024):.1f} KB")
    print(f"   Val Binary: {val_bin_path} | Size: {os.path.getsize(val_bin_path)/(1024):.1f} KB")
    print("=======================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-llm: fast binary pre-training packer")
    parser.add_argument("--src_dir", type=str, default="./data/super_cleaned_corpus", help="Directory containing super cleaned .txt files")
    parser.add_argument("--tokenizer", type=str, default="./data/fast_tokenizer.json", help="Path to fast_tokenizer.json")
    parser.add_argument("--dest_dir", type=str, default="./data", help="Output directory to save train.bin and val.bin")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    args = parser.parse_args()
    
    pack_fast_binaries(
        src_dir=args.src_dir,
        tokenizer_path=args.tokenizer,
        dest_dir=args.dest_dir,
        val_ratio=args.val_ratio
    )
