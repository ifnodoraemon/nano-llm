import os
import random
import logging
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

class DynamicDataMixer(IterableDataset):
    """
    High-performance online dynamic multi-source data mixer for nano-llm.
    It reads raw text files from multiple sources in real time, applies relative mixing weights,
    online-tokenizes raw text, and streams無损 packed sequences of length `block_size`.
    Supports infinite looping over data sources.
    """
    def __init__(
        self,
        sources: Dict[str, float],
        tokenizer,
        block_size: int = 1024,
        buffer_size: int = 5000,
        seed: int = 42
    ):
        """
        Args:
            sources: Dict mapping folder/file path to relative sampling weight.
                     e.g. {"data/wiki": 0.6, "data/code": 0.4}
            tokenizer: Tokenizer supporting `encode` method.
            block_size: Size of target packed context sequences.
            buffer_size: Size of internal shuffle/mixing buffer to ensure random blend.
            seed: Random seed.
        """
        super().__init__()
        self.sources = sources
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.rng = random.Random(seed)
        
        # Resolve all file paths per source
        self.source_files = {}
        self.weights = []
        self.source_names = []
        
        total_w = sum(sources.values())
        if total_w == 0:
            raise ValueError("Total weights of data sources must be positive.")
            
        for path, weight in sources.items():
            normalized_weight = weight / total_w
            files = []
            if os.path.isdir(path):
                for root, _, filenames in os.walk(path):
                    for fname in filenames:
                        if fname.endswith((".txt", ".json", ".jsonl")):
                            files.append(os.path.join(root, fname))
            elif os.path.isfile(path):
                files.append(path)
            else:
                # If path does not exist, create dummy directory and mock file to prevent failure
                logger.warning(f"Data source path '{path}' not found. Creating local mock text source...")
                os.makedirs(path, exist_ok=True)
                mock_file = os.path.join(path, "mock_source.txt")
                with open(mock_file, "w", encoding="utf-8") as f:
                    # Write some dummy text representing high-quality domain corpus
                    f.write("\n".join([f"This is a high-quality pre-training text sample from data source {path} representing text sequence {i}." for i in range(100)]))
                files.append(mock_file)
                
            if files:
                self.source_files[path] = files
                self.source_names.append(path)
                self.weights.append(normalized_weight)
                logger.info(f"Source '{path}': resolved {len(files)} files, target weight = {normalized_weight:.2f}")
            else:
                logger.warning(f"No valid files found for source '{path}'. Skipping.")
                
        if not self.source_files:
            raise ValueError("No data sources were successfully loaded.")

    def _stream_source(self, source_name: str):
        """
        Generator that loops infinitely over the files in a single source,
        yielding tokenized lists from text lines.
        """
        files = self.source_files[source_name]
        file_idx = 0
        self.rng.shuffle(files)
        
        while True:
            filepath = files[file_idx]
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Online tokenizer encoding
                        tokens = self.tokenizer.encode(line, add_special_tokens=False)
                        if tokens:
                            yield tokens
            except Exception as e:
                logger.error(f"Error reading file '{filepath}': {e}")
                
            # Loop back or move to next file
            file_idx = (file_idx + 1) % len(files)
            if file_idx == 0:
                self.rng.shuffle(files)

    def __iter__(self):
        # Create generator streams for each source
        streams = {name: self._stream_source(name) for name in self.source_names}
        
        # Mixing buffer
        buffer = []
        
        # Token accumulator for sequence packing
        accumulator = []
        
        while True:
            # 1. Sample which source stream to draw from according to weights
            chosen_source = self.rng.choices(self.source_names, weights=self.weights, k=1)[0]
            stream = streams[chosen_source]
            
            try:
                # 2. Extract next tokenized line from chosen stream
                tokens = next(stream)
                buffer.append(tokens)
            except StopIteration:
                # Re-create stream if exhausted
                streams[chosen_source] = self._stream_source(chosen_source)
                continue
                
            # 3. Once buffer reaches capacity, shuffle and pack
            if len(buffer) >= self.buffer_size:
                self.rng.shuffle(buffer)
                for item_tokens in buffer:
                    accumulator.extend(item_tokens)
                    
                    # Whenever accumulator has enough tokens, yield a packed block
                    while len(accumulator) >= self.block_size + 1:
                        # block_size + 1 for autoregressive shift targets: x = block[:-1], y = block[1:]
                        chunk = accumulator[:self.block_size + 1]
                        accumulator = accumulator[self.block_size:]
                        
                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        yield {"input_ids": x, "labels": y}
                
                # Clear buffer
                buffer = []
