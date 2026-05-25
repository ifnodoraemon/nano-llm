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

class MinHashDeduplicator:
    """
    Computes MinHash signatures for documents to estimate Jaccard similarity
    and deduplicate near-duplicate web materials at O(N) scale.
    """
    def __init__(self, num_hashes: int = 64, shingle_size: int = 3, num_bands: int = 8, rows_per_band: int = 8):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        assert num_bands * rows_per_band == num_hashes, "Bands * Rows must equal total number of hashes!"
        
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
            shingle_hash = hash(shingle) & 0xFFFFFFFF
            
            # Apply our 64 custom hash functions
            for i in range(self.num_hashes):
                h_val = (self.hash_a[i] * shingle_hash + self.hash_b[i]) % LARGE_PRIME
                if h_val < signature[i]:
                    signature[i] = h_val
                    
        return [int(x) if x != float("inf") else 0 for x in signature]

    def estimate_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Estimates Jaccard similarity by measuring the overlap fraction between signatures.
        """
        assert len(sig1) == len(sig2)
        match_count = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return match_count / len(sig1)

    def get_lsh_buckets(self, signature: List[int]) -> List[Tuple[int, int]]:
        """
        Splits a signature into bands and returns the band index along with the hashed bucket of that band.
        Returns a list of tuples: [(band_idx, bucket_hash_id), ...]
        """
        buckets = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(signature[start:end])
            
            # Hash the band slice values using built-in hash function
            bucket_hash = hash(band_slice) & 0xFFFFFFFF
            buckets.append((band_idx, bucket_hash))
        return buckets


class HighFidelityQualityFilter:
    """
    Industry-grade heuristic filter to discard low-quality web text before deduplication.
    """
    def __init__(self, min_words: int = 15, max_words: int = 80000, max_symbol_ratio: float = 0.15):
        self.min_words = min_words
        self.max_words = max_words
        self.max_symbol_ratio = max_symbol_ratio
        
        # Blacklist of standard garbage text patterns
        self.blacklist_patterns = [
            r"lorem ipsum", r"click here to", r"cookie policy",
            r"javascript is disabled", r"all rights reserved"
        ]

    def is_high_quality(self, text: str) -> Tuple[bool, str]:
        words = text.split()
        word_count = len(words)
        
        # 1. Length constraint
        if word_count < self.min_words:
            return False, f"Too short ({word_count} words)"
        if word_count > self.max_words:
            return False, f"Too long ({word_count} words)"
            
        # 2. Symbol ratio check (too many symbols like #, $, %, @, * usually indicates code or garbage)
        symbols = re.findall(r"[#\$%\^\&\*_\+=\{\}\[\]\|<>~`]", text)
        symbol_ratio = len(symbols) / max(1, len(text))
        if symbol_ratio > self.max_symbol_ratio:
            return False, f"Too many symbols (ratio: {symbol_ratio:.3f})"
            
        # 3. Punctuation sanity check (too few periods/commas in long text is typical of garbage crawled text)
        if word_count > 100:
            punctuation = re.findall(r"[\.,\?!\d]", text)
            if len(punctuation) < 2:
                return False, "Lack of basic sentence structures"
                
        # 4. Blacklisted content patterns check
        lowercase_text = text.lower()
        for pattern in self.blacklist_patterns:
            if re.search(pattern, lowercase_text):
                return False, f"Matches blacklisted pattern '{pattern}'"
                
        return True, "Passed quality gate"


def main():
    parser = argparse.ArgumentParser(description="nano-llm: MinHash-LSH Deduplicator and Quality Pipeline")
    parser.add_argument("--src_dir", type=str, default="./data/raw_crawled", help="Source directory containing raw crawled .txt files")
    parser.add_argument("--dest_dir", type=str, default="./data/cleaned_corpus", help="Destination directory to save unique text files")
    parser.add_argument("--threshold", type=float, default=0.75, help="Jaccard similarity threshold above which docs are duplicate")
    args = parser.parse_args()
    
    os.makedirs(args.dest_dir, exist_ok=True)
    
    # Generate some dummy data if directory does not exist to avoid crashing
    if not os.path.exists(args.src_dir) or len(os.listdir(args.src_dir)) == 0:
        os.makedirs(args.src_dir, exist_ok=True)
        dummy_docs = {
            "doc1.txt": "The quick brown fox jumps over the lazy dog. A beautiful sunny day in San Francisco.",
            "doc2.txt": "The quick brown fox jumped over that lazy dog! Beautiful sunny day in San Francisco.", # near-duplicate of doc1
            "doc3.txt": "lorem ipsum dolor sit amet. javascript is disabled on this web browser.", # garbage document
            "doc4.txt": "nano-llm pre-training framework supports multiple GPUs and dynamic data mixing strategies.",
            "doc5.txt": "Short." # too short
        }
        for name, content in dummy_docs.items():
            with open(os.path.join(args.src_dir, name), "w", encoding="utf-8") as f:
                f.write(content)
                
    doc_paths = [os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir) if f.endswith(".txt")]
    logger.info(f"Loaded {len(doc_paths)} documents for LSH-de-duplication & Quality checks...")
    
    dedup = MinHashDeduplicator(num_hashes=64, shingle_size=3, num_bands=8, rows_per_band=8)
    q_filter = HighFidelityQualityFilter()
    
    # LSH index mapping: (band_idx, bucket_hash) -> list of unique document IDs
    lsh_index: Dict[Tuple[int, int], List[int]] = {}
    
    unique_docs: List[Dict[str, Any]] = []
    seen_signatures: List[List[int]] = []
    
    filtered_count = 0
    duplicate_count = 0
    
    for doc_path in sorted(doc_paths):
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            
        if not text:
            continue
            
        # 1. Run High-Fidelity Quality Filter
        is_ok, reason = q_filter.is_high_quality(text)
        if not is_ok:
            logger.info(f"🚫 Quality Filter Discarded '{os.path.basename(doc_path)}': {reason}")
            filtered_count += 1
            continue
            
        shingles = dedup.get_shingles(text)
        if len(shingles) < 3:
            shingles = {text}
            
        # Compute MinHash signature
        sig = dedup.compute_signature(shingles)
        
        # 2. Query LSH to find candidate duplicates (O(1) lookups)
        buckets = dedup.get_lsh_buckets(sig)
        candidates = set()
        
        for band_idx, b_hash in buckets:
            key = (band_idx, b_hash)
            if key in lsh_index:
                candidates.update(lsh_index[key])
                
        # 3. Check Jaccard similarity ONLY against candidate matches!
        is_duplicate = False
        for cand_idx in candidates:
            ref_sig = seen_signatures[cand_idx]
            similarity = dedup.estimate_jaccard(sig, ref_sig)
            if similarity > args.threshold:
                logger.info(
                    f"⚠️ LSH Near-Duplicate Detected: '{os.path.basename(doc_path)}' is identical to "
                    f"'{unique_docs[cand_idx]['name']}' with Jaccard Similarity of {similarity*100:.1f}%"
                )
                is_duplicate = True
                duplicate_count += 1
                break
                
        if not is_duplicate:
            # Document is verified unique: append to seen list
            doc_id = len(unique_docs)
            unique_docs.append({
                "name": os.path.basename(doc_path),
                "text": text
            })
            seen_signatures.append(sig)
            
            # Register this unique document's buckets in our LSH index
            for band_idx, b_hash in buckets:
                key = (band_idx, b_hash)
                if key not in lsh_index:
                    lsh_index[key] = []
                lsh_index[key].append(doc_id)
                
            # Export to clean destination path
            dest_path = os.path.join(args.dest_dir, os.path.basename(doc_path))
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(text)
                
    logger.info("=" * 60)
    logger.info("✅ High-Fidelity MinHash-LSH Deduplication Complete!")
    logger.info(f"  🔹 Unique clean documents saved : {len(unique_docs)} files")
    logger.info(f"  🔹 Discarded low-quality docs   : {filtered_count} files")
    logger.info(f"  🔹 Near-duplicate docs removed  : {duplicate_count} files")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
