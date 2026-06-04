import os
import glob
import re

def merge_split(temp_dir, prefix, split, dest_path):
    print(f"Merging {prefix} {split} shards...")
    # Find all shard files matching prefix_split_worker_*.bin
    pattern = os.path.join(temp_dir, f"{prefix}_{split}_worker_*.bin")
    files = glob.glob(pattern)
    
    # Sort files by worker index
    def get_index(path):
        match = re.search(r'worker_(\d+)\.bin$', path)
        return int(match.group(1)) if match else 0
    files = sorted(files, key=get_index)
    
    print(f"Found {len(files)} shards to merge.")
    if not files:
        print("No shards found. Skipping.")
        return
        
    with open(dest_path, "wb") as out_f:
        for fpath in files:
            size_gb = os.path.getsize(fpath) / (1024**3)
            print(f"Appending {os.path.basename(fpath)} ({size_gb:.2f} GB)...")
            with open(fpath, "rb") as in_f:
                while True:
                    chunk = in_f.read(64 * 1024 * 1024) # 64MB chunk size
                    if not chunk:
                        break
                    out_f.write(chunk)
    print(f"Successfully merged shards into {dest_path}")

def main():
    temp_dir = "/data/nano-llm-data/pretrain_temp"
    dest_dir = "/data/nano-llm-data/binaries_1t"
    os.makedirs(dest_dir, exist_ok=True)
    
    train_dest = os.path.join(dest_dir, "train.bin")
    val_dest = os.path.join(dest_dir, "val.bin")
    
    # Merge train shards
    merge_split(temp_dir, "en", "train", train_dest)
    
    # Merge val shards
    merge_split(temp_dir, "en", "val", val_dest)
    
    print("=======================================================================")
    print("🎉 Merging completed successfully!")
    print(f"📂 Final Train Binary: {train_dest} | Size: {os.path.getsize(train_dest)/(1024**3):.2f} GB")
    print(f"📂 Final Val Binary: {val_dest} | Size: {os.path.getsize(val_dest)/(1024**3):.2f} GB")
    print("=======================================================================")

if __name__ == "__main__":
    main()
