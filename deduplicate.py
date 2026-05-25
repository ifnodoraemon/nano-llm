import os
import re
import random
import logging
import argparse
from typing import List, Set, Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Large prime number used in our custom LCG hash generator
LARGE_PRIME = 4294967291  # 2^32 - 5

# ==============================================================================
# MinHash near-deduplication from scratch in pure Python
# ==============================================================================

class MinHashDeduplicator:
    """
    Computes MinHash signatures for documents to estimate Jaccard similarity
    and deduplicate near-duplicate web materials at O(N) scale.
    """
    def __init__(self, num_hashes: int = 64, shingle_size: int = 3):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        
        # Generate random coefficients 'a' and 'b' for our 64 hash functions from scratch
        # Hash formula: h(x) = (a * x + b) % LARGE_PRIME
        random.seed(42)
        self.hash_a = [random.randint(1, LARGE_PRIME - 1) for _ in range(num_hashes)]
        self.hash_b = [random.randint(0, LARGE_PRIME - 1) for _ in range(num_hashes)]

    def get_shingles(self, text: str) -> Set[str]:
        """
        Converts text into a unique set of word n-grams (shingles).
        e.g., "the quick brown fox" -> {"the quick brown", "quick brown fox"} (shingle_size=3)
        """
        # Clean text: remove non-alphanumeric chars and lowercase
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i : i + self.shingle_size])
            shingles.add(shingle)
            
        return shingles

    def compute_signature(self, shingles: Set[str]) -> List[int]:
        """
        Computes the MinHash signature of length `num_hashes` for a set of shingles.
        """
        signature = [float("inf")] * self.num_hashes
        
        for shingle in shingles:
            # We use Python's built-in hash() as the base seed value
            shingle_hash = hash(shingle) & 0xFFFFFFFF
            
            # Apply our 64 custom hash functions
            for i in range(self.num_hashes):
                # Linear Congruential Generator (LCG) mapping
                h_val = (self.hash_a[i] * shingle_hash + self.hash_b[i]) % LARGE_PRIME
                
                # Keep the minimum hash value
                if h_val < signature[i]:
                    signature[i] = h_val
                    
        return signature

    def estimate_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Estimates Jaccard similarity by measuring the overlap fraction between signatures.
        """
        assert len(sig1) == len(sig2)
        match_count = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return match_count / len(sig1)


# ==============================================================================
# Pipeline Deduplication Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="nano-llm: MinHash document deduplicator")
    parser.add_argument("--src_dir", type=str, default="./data/raw_crawled", help="Source directory containing raw crawled .txt files")
    parser.add_argument("--dest_dir", type=str, default="./data/cleaned_corpus", help="Destination directory to save unique text files")
    parser.add_argument("--threshold", type=float, default=0.75, help="Jaccard similarity threshold above which docs are duplicate")
    args = parser.parse_args()
    
    os.makedirs(args.dest_dir, exist_ok=True)
    
    # 1. Read all crawled documents
    doc_paths = [os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir) if f.endswith(".txt")]
    logger.info(f"Loaded {len(doc_paths)} documents for near-deduplication checks...")
    
    dedup = MinHashDeduplicator(num_hashes=64, shingle_size=3)
    
    unique_docs = []
    seen_signatures = []
    
    duplicate_count = 0
    
    for doc_path in sorted(doc_paths):
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            
        if not text:
            continue
            
        shingles = dedup.get_shingles(text)
        
        # If document is extremely short (fewer shingles than size), keep it
        if len(shingles) < 3:
            shingles = {text}
            
        # Compute signature
        sig = dedup.compute_signature(shingles)
        
        # 2. Check Jaccard similarity against all unique documents seen so far
        is_duplicate = False
        for idx, ref_sig in enumerate(seen_signatures):
            similarity = dedup.estimate_jaccard(sig, ref_sig)
            if similarity > args.threshold:
                logger.info(
                    f"⚠️  Deduplication Triggered: '{os.path.basename(doc_path)}' matches "
                    f"'{unique_docs[idx]['name']}' with Jaccard Similarity of {similarity*100:.1f}%"
                )
                is_duplicate = True
                duplicate_count += 1
                break
                
        if not is_duplicate:
            # Document is unique: save to dest_dir and append to signatures
            unique_docs.append({
                "name": os.path.basename(doc_path),
                "text": text
            })
            seen_signatures.append(sig)
            
            # Export unique document
            dest_path = os.path.join(args.dest_dir, os.path.basename(doc_path))
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(text)
                
    logger.info("=======================================================================")
    logger.info("✅ Near-Deduplication Complete!")
    logger.info(f"💾 Unique clean documents saved: {len(unique_docs)} files")
    logger.info(f"🚫 Duplicate documents discarded: {duplicate_count} files")
    logger.info("=======================================================================")

if __name__ == "__main__":
    main()
