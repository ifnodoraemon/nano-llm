import os
import json
import logging
import argparse
from typing import Dict, List, Tuple, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Pure Python BPE (Byte Pair Encoding) Tokenizer (minbpe-style)
# ==============================================================================

def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Counts the frequencies of all adjacent pairs of integers."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Replaces all occurrences of `pair` in `ids` with the new integer `idx` using fast C-level index searching."""
    p0, p1 = pair
    newids = []
    i = 0
    n = len(ids)
    
    try:
        while i < n:
            # Fast C-level search for p0
            next_p0 = ids.index(p0, i)
            # Copy everything from current position i to next_p0
            newids.extend(ids[i:next_p0])
            
            # Check if it forms the pair
            if next_p0 < n - 1 and ids[next_p0 + 1] == p1:
                newids.append(idx)
                i = next_p0 + 2
            else:
                newids.append(p0)
                i = next_p0 + 1
    except ValueError:
        # p0 is not found in the rest of the list
        newids.extend(ids[i:])
        
    return newids


class CustomBPETokenizer:
    """
    Byte Pair Encoding Tokenizer trained and executed from scratch.
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[int, int], int] = {}
        # Vocab maps token ID to byte sequences
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        # Standard special tokens
        self.special_tokens = {
            "<|im_start|>": 10000,
            "<|im_end|>": 10001,
            "<|pad|>": 10002
        }

    def train(self, text: str, vocab_size: int):
        """
        Trains BPE tokenizer on raw input text by iteratively finding and merging
        the most frequent adjacent byte pairs.
        """
        self.vocab_size = vocab_size
        # 1. Convert raw text to list of raw bytes (integers 0..255)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        num_merges = vocab_size - 256
        logger.info(f"Starting BPE training on corpus size {len(ids)} bytes...")
        logger.info(f"Target Vocabulary size: {vocab_size} ({num_merges} merges to learn)")
        
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
                
            # Find the most frequent adjacent pair
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            logger.debug(f"Learning merge rule {i+1}/{num_merges}: {best_pair} -> {new_id} (count: {stats[best_pair]})")
            
            # Record merge rule
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Merge in-place in our token list
            ids = merge(ids, best_pair, new_id)
            
        logger.info(f"BPE training complete! Final vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into BPE token IDs by iteratively applying merges in learn-order.
        """
        # Convert string to raw bytes
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        # Maintain a set of unique token IDs currently present in the sequence
        ids_set = set(ids)
        
        # Apply merges in the exact order they were learned during BPE training
        for pair, new_id in self.merges.items():
            if pair[0] not in ids_set or pair[1] not in ids_set:
                continue
                
            old_len = len(ids)
            ids = merge(ids, pair, new_id)
            if len(ids) < old_len:
                # Merge succeeded, add the new token ID to our active set
                ids_set.add(new_id)
            
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into standard UTF-8 string by expanding bytes.
        """
        byte_list = []
        for idx in ids:
            if idx in self.vocab:
                byte_list.append(self.vocab[idx])
            elif idx in self.special_tokens.values():
                # For special tokens, decode to their string representation
                inv_special = {v: k for k, v in self.special_tokens.items()}
                byte_list.append(inv_special[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID: {idx}")
                
        # Combine bytes and decode to UTF-8, replacing errors
        return b"".join(byte_list).decode("utf-8", errors="replace")

    def save(self, file_path: str):
        """Saves the tokenizer rules and vocab to a local JSON file."""
        # Convert tuple keys in merges to string representations for JSON compatibility
        json_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        # Convert bytes in vocab to latin-1 strings
        json_vocab = {k: v.decode("latin-1") for k, v in self.vocab.items()}
        
        data = {
            "vocab_size": self.vocab_size,
            "merges": json_merges,
            "vocab": json_vocab,
            "special_tokens": self.special_tokens
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Custom BPE Tokenizer saved successfully to {file_path}")

    def load(self, file_path: str):
        """Loads BPE tokenizer merges and vocab from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.vocab_size = data["vocab_size"]
        self.special_tokens = data["special_tokens"]
        
        # Load merges
        self.merges = {}
        for k, v in data["merges"].items():
            pair_ints = tuple(map(int, k.split(",")))
            self.merges[pair_ints] = v
            
        # Load vocab
        self.vocab = {}
        for k, v in data["vocab"].items():
            self.vocab[int(k)] = v.encode("latin-1")
            
        logger.info(f"Custom BPE Tokenizer loaded successfully from {file_path}")


# ==============================================================================
# Training Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="nano-llm: custom SentencePiece/minbpe Tokenizer Trainer")
    parser.add_argument("--src_dir", type=str, default="./data/cleaned_corpus", help="Directory containing deduplicated .txt files")
    parser.add_argument("--output_file", type=str, default="./data/custom_tokenizer.json", help="Path to save trained tokenizer rules")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Target vocabulary size (e.g. 32000 or 65536)")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 1. Accumulate all text in our cleaned corpus
    doc_paths = [os.path.join(args.src_dir, f) for f in os.listdir(args.src_dir) if f.endswith(".txt")]
    
    corpus_text = ""
    for path in sorted(doc_paths):
        with open(path, "r", encoding="utf-8") as f:
            corpus_text += f.read() + "\n"
            
    if not corpus_text.strip():
        logger.error("No training text found in corpus directory. Run crawl_data.py first.")
        return
        
    # 2. Instantiate and train BPE
    tokenizer = CustomBPETokenizer()
    tokenizer.train(corpus_text, vocab_size=args.vocab_size)
    
    # 3. Export Rules
    tokenizer.save(args.output_file)
    
    # 4. Verify round-trip encoding/decoding accuracy
    test_str = "Deep learning breakthrough: Albert Einstein invented BPE tokenization in 1957!"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    
    logger.info("=======================================================================")
    logger.info("📊 Tokenizer Verification Test:")
    logger.info(f"📝 Test Input: '{test_str}'")
    logger.info(f"🪙 Token IDs: {encoded}")
    logger.info(f"🤖 Decoded Output: '{decoded}'")
    logger.info(f"✅ Round-Trip Match: {test_str == decoded}")
    logger.info("=======================================================================")

if __name__ == "__main__":
    main()
