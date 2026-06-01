import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import AddedToken

logger = logging.getLogger(__name__)

class CustomBPETokenizer(PreTrainedTokenizer):
    """
    HuggingFace compliant Custom BPE Tokenizer for nano-llm.
    Supports auto_map loading natively.
    """
    vocab_files_names = {"vocab_file": "custom_tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(self, vocab_file=None, **kwargs):
        self.special_tokens = {
            "<|im_start|>": 10000,
            "<|im_end|>": 10001,
            "<|pad|>": 10002
        }
        
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab_size_val = 10005
        
        # Load from file if available
        if vocab_file is None:
            # Look in the same directory as this file
            dir_path = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(dir_path, "custom_tokenizer.json")
            if os.path.exists(default_path):
                vocab_file = default_path

        if vocab_file and os.path.exists(vocab_file):
            try:
                with open(vocab_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.vocab_size_val = data.get("vocab_size", 10005)
                self.special_tokens = data.get("special_tokens", self.special_tokens)
                
                # Load merges
                self.merges = {}
                for k, v in data.get("merges", {}).items():
                    pair_ints = tuple(map(int, k.split(",")))
                    self.merges[pair_ints] = v
                    
                # Load vocab
                self.vocab = {}
                for k, v in data.get("vocab", {}).items():
                    self.vocab[int(k)] = v.encode("latin-1")
            except Exception as e:
                logger.error(f"Error loading custom BPE vocabulary file {vocab_file}: {e}")

        # Define special token definitions
        bos_token = AddedToken("<|im_start|>", lstrip=False, rstrip=False, normalized=False, special=True)
        eos_token = AddedToken("<|im_end|>", lstrip=False, rstrip=False, normalized=False, special=True)
        pad_token = AddedToken("<|pad|>", lstrip=False, rstrip=False, normalized=False, special=True)
        
        # Inverted lookups
        self.ids_to_tokens = {v: k for k, v in self.special_tokens.items()}
        for k, v in self.vocab.items():
            self.ids_to_tokens[k] = v.decode("utf-8", errors="replace")
            
        self.tokens_to_ids = {v: k for k, v in self.ids_to_tokens.items()}
        
        # Pop special tokens from kwargs if they exist to prevent multiple-values conflict
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("pad_token", None)
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        return self.vocab_size_val

    def get_vocab(self) -> Dict[str, int]:
        return self.tokens_to_ids

    def _tokenize(self, text: str) -> List[str]:
        ids = self._encode_text(text)
        return [self.ids_to_tokens.get(idx, "<|pad|>") for idx in ids]

    def _encode_text(self, text: str) -> List[int]:
        # Raw string UTF-8 bytes to list of ints
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        ids_set = set(ids)
        for pair, new_id in self.merges.items():
            if pair[0] not in ids_set or pair[1] not in ids_set:
                continue
                
            newids = []
            i = 0
            n = len(ids)
            p0, p1 = pair
            try:
                while i < n:
                    next_p0 = ids.index(p0, i)
                    newids.extend(ids[i:next_p0])
                    if next_p0 < n - 1 and ids[next_p0 + 1] == p1:
                        newids.append(new_id)
                        i = next_p0 + 2
                    else:
                        newids.append(p0)
                        i = next_p0 + 1
            except ValueError:
                newids.extend(ids[i:])
            ids = newids
            ids_set = set(ids)
        return ids

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokens_to_ids.get(token, self.special_tokens.get("<|pad|>"))

    def _convert_id_to_token(self, index: int) -> str:
        try:
            index = int(index)
        except (ValueError, TypeError):
            pass
        return self.ids_to_tokens.get(index, "<|pad|>")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        byte_list = []
        for token in tokens:
            if token in self.special_tokens:
                # Do not write special tokens directly as raw BPE bytes
                continue
            token_id = self.tokens_to_ids.get(token)
            if token_id is not None and token_id in self.vocab:
                byte_list.append(self.vocab[token_id])
            else:
                byte_list.append(token.encode("utf-8"))
        return b"".join(byte_list).decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        vocab_file = os.path.join(save_directory, "custom_tokenizer.json")
        json_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        json_vocab = {k: v.decode("latin-1") for k, v in self.vocab.items()}
        data = {
            "vocab_size": self.vocab_size_val,
            "merges": json_merges,
            "vocab": json_vocab,
            "special_tokens": self.special_tokens
        }
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return (vocab_file,)
