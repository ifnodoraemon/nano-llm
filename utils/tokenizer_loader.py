"""
Unified tokenizer loader shared across all training/inference scripts.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_tokenizer(
    fallback_model_name: Optional[str] = None,
    use_fast: bool = True,
) -> "PreTrainedTokenizerBase":
    """
    Load tokenizer with priority: fast_tokenizer.json > custom_tokenizer.json > fallback.

    Args:
        fallback_model_name: HF model name for AutoTokenizer fallback (e.g. "Qwen/Qwen2.5-7B").
                             If None, uses HubAdapter with Qwen default.
        use_fast: Whether to use fast tokenizer implementations.

    Returns:
        A HuggingFace-compatible tokenizer instance.
    """
    from transformers import PreTrainedTokenizerFast

    if os.path.exists("./data/fast_tokenizer.json"):
        logger.info("Found fast standard 32k BPE tokenizer. Loading natively...")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="./data/fast_tokenizer.json",
            bos_token="<|im_start|>",
            eos_token="<|im_end|>",
            pad_token="<|pad|>",
        )
    elif os.path.exists("./data/custom_tokenizer.json"):
        logger.info("Found custom trained BPE tokenizer. Loading from ./data/custom_tokenizer.json...")
        from serve import CustomTokenizerAdapter
        from train_tokenizer import CustomBPETokenizer
        raw_tok = CustomBPETokenizer()
        raw_tok.load("./data/custom_tokenizer.json")
        tokenizer = CustomTokenizerAdapter(raw_tok)
    else:
        logger.warning("Custom tokenizer not found. Falling back to AutoTokenizer...")
        if fallback_model_name:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=use_fast)
        else:
            from utils.hub_adapter import HubAdapter
            hub = HubAdapter()
            tokenizer = hub.load_tokenizer_or_model(
                "Qwen/Qwen2.5-7B" if hub.provider == "hf" else "qwen/Qwen2.5-7B",
                load_type="tokenizer",
                use_fast=use_fast,
            )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    return tokenizer
