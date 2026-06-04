"""
Semantic Fingerprint Blocking (语义指纹阻断防泄露)

Prevents benchmark data leakage by building a character-level n-gram fingerprint
index from evaluation benchmarks (MMLU, GSM8K, ARC etc.) and checking training
data against it. Contaminated token positions have their labels set to -100
(ignored by cross-entropy loss).

核心原理:
    1. 初始化时从 data/eval/ 目录加载评测基准题目
    2. 构建字符级 13-gram 指纹集合
    3. 训练时解码 token → 文本, 计算 13-gram, 与指纹集对比
    4. 将污染位置的 labels 设为 -100, 从 loss 计算中剔除

Implementation uses simple deterministic n-gram overlap (NOT LSH).
"""

import os
import json
import logging
from typing import Set, Optional, List, Tuple

import torch

logger = logging.getLogger(__name__)

# Default n-gram length for fingerprinting (指纹 n-gram 长度)
DEFAULT_NGRAM_SIZE = 13
# If this fraction of a sequence's n-grams overlap, mark the whole region
# 如果序列中超过此比例的 n-gram 重叠, 则标记整个区域
DEFAULT_CONTAMINATION_THRESHOLD = 0.1


class BenchmarkLeakageBlocker:
    """Blocks benchmark data leakage by n-gram fingerprint matching.

    Builds a fingerprint set from evaluation benchmark files and provides
    a `check_and_mask()` method that detects contaminated training samples
    and sets their labels to -100 for loss masking.

    Args:
        eval_data_dir: Path to directory containing benchmark JSONL files.
        tokenizer: A tokenizer with `decode()` method for token→text conversion.
        ngram_size: Character-level n-gram size for fingerprinting.
        contamination_threshold: Fraction of n-grams that must match to
            consider a sample contaminated.
    """

    def __init__(
        self,
        eval_data_dir: str = "./data/eval",
        tokenizer=None,
        ngram_size: int = DEFAULT_NGRAM_SIZE,
        contamination_threshold: float = DEFAULT_CONTAMINATION_THRESHOLD,
    ):
        self.eval_data_dir = eval_data_dir
        self.tokenizer = tokenizer
        self.ngram_size = ngram_size
        self.contamination_threshold = contamination_threshold

        # Build the fingerprint index (构建指纹索引)
        self.fingerprint_set: Set[str] = set()
        self._build_fingerprint_index()

        self.total_checked = 0
        self.total_blocked = 0

    def _build_fingerprint_index(self) -> None:
        """Load evaluation benchmark data and build n-gram fingerprint set.

        从评测基准数据加载文本并构建 n-gram 指纹集合.
        Supports JSONL files with 'question', 'text', 'prompt', or 'input' fields.
        """
        if not os.path.isdir(self.eval_data_dir):
            logger.warning(
                f"Eval data directory '{self.eval_data_dir}' not found — "
                f"benchmark leakage blocking will be inactive"
            )
            return

        file_count = 0
        question_count = 0

        for filename in sorted(os.listdir(self.eval_data_dir)):
            if not filename.endswith(".jsonl"):
                continue

            filepath = os.path.join(self.eval_data_dir, filename)
            file_count += 1

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Extract text from common benchmark fields
                        # 从常见基准字段中提取文本
                        text = (
                            record.get("question")
                            or record.get("text")
                            or record.get("prompt")
                            or record.get("input")
                            or ""
                        )
                        if not text or len(text) < self.ngram_size:
                            continue

                        # Add n-grams to fingerprint set (添加 n-gram 到指纹集)
                        ngrams = self._extract_ngrams(text)
                        self.fingerprint_set.update(ngrams)
                        question_count += 1

            except Exception as e:
                logger.warning(f"Error reading benchmark file {filepath}: {e}")
                continue

        if file_count > 0:
            logger.info(
                f"🔒 Benchmark leakage blocker initialized: "
                f"{file_count} files, {question_count} questions, "
                f"{len(self.fingerprint_set):,} unique {self.ngram_size}-gram fingerprints"
            )
        else:
            logger.warning(
                f"No JSONL benchmark files found in '{self.eval_data_dir}' — "
                f"leakage blocking inactive"
            )

    def _extract_ngrams(self, text: str) -> Set[str]:
        """Extract character-level n-grams from text.

        对文本提取字符级 n-gram. 预先做归一化: 转小写, 去除多余空白.

        Args:
            text: Input text string.

        Returns:
            Set of n-gram strings.
        """
        # Normalize: lowercase and collapse whitespace (归一化处理)
        normalized = " ".join(text.lower().split())
        if len(normalized) < self.ngram_size:
            return set()
        return {
            normalized[i : i + self.ngram_size]
            for i in range(len(normalized) - self.ngram_size + 1)
        }

    def check_and_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check training batch for benchmark contamination and mask labels.

        Decodes input tokens back to text, computes n-grams, checks overlap with
        the fingerprint set, and sets labels to -100 for contaminated positions.

        检查训练 batch 是否包含基准测试数据泄露, 将污染位置 labels 设为 -100.

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len).
            labels: Target labels, shape (batch_size, seq_len).

        Returns:
            Tuple of (input_ids, labels) with contaminated labels masked.
        """
        # Skip if no fingerprints loaded or no tokenizer (无指纹或无分词器则跳过)
        if not self.fingerprint_set or self.tokenizer is None:
            return input_ids, labels

        batch_size = input_ids.size(0)
        labels = labels.clone()  # Avoid in-place modification (避免原地修改)

        for i in range(batch_size):
            self.total_checked += 1

            # Decode token IDs to text (解码 token 为文本)
            try:
                text = self.tokenizer.decode(input_ids[i].tolist())
            except Exception:
                continue

            # Extract n-grams and check overlap (提取 n-gram 并检查重叠)
            sample_ngrams = self._extract_ngrams(text)
            if not sample_ngrams:
                continue

            overlap = sample_ngrams & self.fingerprint_set
            overlap_ratio = len(overlap) / len(sample_ngrams)

            if overlap_ratio >= self.contamination_threshold:
                # Mask all labels in contaminated sample (标记污染样本所有 labels)
                labels[i, :] = -100
                self.total_blocked += 1

                if self.total_blocked <= 10:  # Log first 10 blocked samples
                    logger.info(
                        f"🚫 Blocked contaminated sample {self.total_blocked}: "
                        f"overlap={overlap_ratio:.2%} ({len(overlap)}/{len(sample_ngrams)} n-grams)"
                    )

        return input_ids, labels

    def get_stats(self) -> dict:
        """Return blocking statistics for logging.

        返回阻断统计信息用于日志记录.
        """
        return {
            "pretrain/leakage_checked": self.total_checked,
            "pretrain/leakage_blocked": self.total_blocked,
            "pretrain/leakage_block_rate": (
                self.total_blocked / max(1, self.total_checked)
            ),
            "pretrain/fingerprint_count": len(self.fingerprint_set),
        }
