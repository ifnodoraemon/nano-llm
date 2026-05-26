import re
import math
import torch
from typing import List, Dict, Any, Tuple

# ==============================================================================
# 1. Document Chunk Processor
# ==============================================================================

class ChunkProcessor:
    """
    Splits documents into overlapping chunks based on sentence/word boundaries.
    """
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Keep overlap sentences
                overlap_words = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    s_words = s.split()
                    if overlap_len + len(s_words) <= self.chunk_overlap:
                        overlap_words.insert(0, s)
                        overlap_len += len(s_words)
                    else:
                        break
                current_chunk = overlap_words
                current_length = overlap_len
                
            current_chunk.append(sentence)
            current_length += sentence_length
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks


# ==============================================================================
# 2. Pure PyTorch Dense Retriever (Zero extra dependencies)
# ==============================================================================

class DenseRetriever:
    """
    Dense Semantic Retriever. Uses a simple bag-of-words / TF-IDF or random projection embeddings
    in PyTorch to calculate semantic cosine similarities, avoiding external FAISS dependencies.
    """
    def __init__(self, embed_dim: int = 384):
        self.embed_dim = embed_dim
        self.chunks: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None

    def _deterministic_hash(self, s: str) -> int:
        h = 0
        for c in s:
            h = (h * 31 + ord(c)) & 0xFFFFFFFF
        return h

    def fit(self, chunks: List[str]):
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return
            
        # Simulating dense embedding generation via random hashing projections
        # In a real environment, sentence-transformers would be invoked.
        # This keeps the environment 100% lightweight and fast.
        num_chunks = len(chunks)
        self.embeddings = torch.zeros(num_chunks, self.embed_dim)
        
        for idx, chunk in enumerate(chunks):
            # Compute a hash vector representing semantic frequencies
            words = chunk.lower().split()
            vector = torch.zeros(self.embed_dim)
            for word in words:
                word = word.strip(".,!?()[]{}\"';:")
                if not word:
                    continue
                word_hash = self._deterministic_hash(word) % self.embed_dim
                vector[word_hash] += 1.0
            # Normalize to unit length
            norm = vector.norm(p=2)
            if norm > 0:
                vector = vector / norm
            self.embeddings[idx] = vector

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Returns top K indices and scores of matched chunks."""
        if self.embeddings is None or not self.chunks:
            return []
            
        # Generate query vector
        query_vector = torch.zeros(self.embed_dim)
        for word in query.lower().split():
            word = word.strip(".,!?()[]{}\"';:")
            if not word:
                continue
            word_hash = self._deterministic_hash(word) % self.embed_dim
            query_vector[word_hash] += 1.0
        norm = query_vector.norm(p=2)
        if norm > 0:
            query_vector = query_vector / norm
            
        # Calculate cosine similarity using PyTorch matrix multiplication
        # shape: (num_chunks,)
        similarities = torch.matmul(self.embeddings, query_vector)
        
        # Sort and select Top-K
        scores, indices = torch.topk(similarities, k=min(top_k, len(self.chunks)))
        
        return list(zip(indices.tolist(), scores.tolist()))


# ==============================================================================
# 3. Pure Python BM25 Sparse Retriever
# ==============================================================================

class SparseRetriever:
    """
    Okapi BM25 Keyword-based Sparse Retriever.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[str] = []
        self.doc_len: List[int] = []
        self.avg_doc_len = 0.0
        self.idf: Dict[str, float] = {}
        self.tf: List[Dict[str, int]] = []

    def fit(self, chunks: List[str]):
        self.chunks = chunks
        if not chunks:
            return
            
        self.doc_len = [len(chunk.lower().split()) for chunk in chunks]
        self.avg_doc_len = sum(self.doc_len) / len(chunks)
        self.tf = []
        
        # Document frequencies for IDF calculation
        df: Dict[str, int] = {}
        for chunk in chunks:
            words = chunk.lower().split()
            chunk_tf = {}
            seen = set()
            for word in words:
                chunk_tf[word] = chunk_tf.get(word, 0) + 1
                if word not in seen:
                    df[word] = df.get(word, 0) + 1
                    seen.add(word)
            self.tf.append(chunk_tf)
            
        # Calculate IDF
        num_docs = len(chunks)
        for word, freq in df.items():
            # BM25 IDF formula
            self.idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        if not self.chunks:
            return []
            
        query_words = query.lower().split()
        scores = []
        
        for idx, chunk_tf in enumerate(self.tf):
            score = 0.0
            d_len = self.doc_len[idx]
            for word in query_words:
                if word in chunk_tf:
                    tf_val = chunk_tf[word]
                    idf_val = self.idf.get(word, 0.0)
                    # BM25 scoring function
                    numerator = idf_val * tf_val * (self.k1 + 1.0)
                    denominator = tf_val + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                    score += numerator / denominator
            scores.append((idx, score))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ==============================================================================
# 4. Reciprocal Rank Fusion (RRF) Hybrid Retriever
# ==============================================================================

class HybridRetriever:
    """
    Hybrid Retriever combining Dense (vector) and Sparse (BM25) search results
    via Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.chunks: List[str] = []

    def fit(self, chunks: List[str]):
        self.chunks = chunks
        self.dense.fit(chunks)
        self.sparse.fit(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self.chunks:
            return []
            
        # Retrieve candidate slices from both retrievers (requesting twice top_k to fuse effectively)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k=top_k * 2)
        
        # Calculate Reciprocal Rank Fusion (RRF) scores
        # RRF(d) = sum_{retriever} 1 / (k + rank)
        rrf_scores: Dict[int, float] = {}
        
        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            
        # Sort fused scores
        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
        
        # Return top-k chunk text values
        return [self.chunks[idx] for idx in sorted_indices[:top_k]]
