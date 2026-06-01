import os
import re

def clean_corpus():
    src_dir = "./data/cleaned_corpus"
    dest_dir = "./data/super_cleaned_corpus"
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"🧹 Cleaning literal '<unk>' strings and formatting noise from {src_dir}...")
    
    files = [f for f in os.listdir(src_dir) if f.endswith(".txt")]
    for filename in sorted(files):
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # 1. Replace `<unk>` strings (and their surrounding spaces) with a single space
        cleaned_text = re.sub(r'\s*<unk>\s*', ' ', text)
        
        # 2. Standardize duplicate spaces
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text.strip() + "\n")
            
        print(f"   Processed: {filename} ➡️ {os.path.basename(dest_path)}")
        
    print("✅ Noise-free corpus pre-processing complete! Super cleaned corpus saved to: ./data/super_cleaned_corpus")

if __name__ == "__main__":
    clean_corpus()
