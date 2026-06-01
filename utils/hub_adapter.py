import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# Unified Hub Adapter: Dynamic Hugging Face & ModelScope Switcher
# ==============================================================================

class HubAdapter:
    """
    Unified Hub Adapter that dynamically switches between Hugging Face (HF) and
    ModelScope (MS) based on command-line flags, environment variables, or
    network geographical connectivity.
    
    Provides unified interfaces for downloading datasets, tokenizers, and model weights.
    """
    def __init__(self, provider: str = None):
        # Determine provider: command-line > environment variable > default ('ms')
        env_provider = os.environ.get("NANO_HUB_PROVIDER", "ms").lower()
        self.provider = (provider or env_provider).lower()
        
        if self.provider not in ["hf", "ms"]:
            logger.warning(f"Unsupported provider '{self.provider}'. Falling back to 'ms'.")
            self.provider = "ms"
            
        # Proactively configure mainland China high-speed mirrors for Hugging Face
        # Always set HF_ENDPOINT if it is not present, to ensure high speed fallback
        if "HF_ENDPOINT" not in os.environ:
            logger.info("Setting HF_ENDPOINT mirror to https://hf-mirror.com for high-speed downloads...")
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                
        logger.info(f"Initialized HubAdapter targeting provider: {self.provider.upper()}")

    def load_dataset(self, dataset_id: str, split: str = "train", **kwargs) -> Any:
        """
        Loads an open-source dataset natively from either ModelScope MsDataset
        or Hugging Face datasets dynamically.
        """
        logger.info(f"Loading dataset '{dataset_id}' (split: {split}) from {self.provider.upper()}...")
        
        if self.provider == "ms":
            try:
                from modelscope.msdatasets import MsDataset
                # MsDataset loads mirrored datasets natively from Alibaba CDNs
                dataset = MsDataset.load(dataset_id, split=split, **kwargs)
                logger.info(f"✅ Successfully loaded dataset '{dataset_id}' from ModelScope.")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load dataset '{dataset_id}' from ModelScope: {e}. Attempting Hugging Face fallback...")
                
        # Hugging Face default path (with mirror enabled)
        try:
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from datasets import load_dataset
            dataset = load_dataset(dataset_id, split=split, **kwargs)
            logger.info(f"✅ Successfully loaded dataset '{dataset_id}' from Hugging Face.")
            return dataset
        except ImportError:
            logger.error("The 'datasets' library is required to load datasets from Hugging Face.")
            logger.info("Please install via: pip install datasets")
            raise

    def load_tokenizer_or_model(self, repo_id: str, load_type: str = "tokenizer", **kwargs) -> Any:
        """
        Downloads and loads a tokenizer or pre-trained base model from either
        ModelScope Model Hub or Hugging Face Model Hub seamlessly.
        """
        logger.info(f"Loading {load_type} for '{repo_id}' from {self.provider.upper()}...")
        
        if self.provider == "ms":
            try:
                from modelscope import AutoTokenizer as MSAutoTokenizer
                from modelscope import AutoModelForCausalLM as MSAutoModel
                from modelscope import snapshot_download
                
                # Fetch model folder from ModelScope domestic SNAPSHOT mirror
                local_dir = snapshot_download(repo_id)
                logger.info(f"✅ Model snapshot downloaded to local path: {local_dir}")
                
                if load_type == "tokenizer":
                    return MSAutoTokenizer.from_pretrained(local_dir, **kwargs)
                else:
                    return MSAutoModel.from_pretrained(local_dir, **kwargs)
            except Exception as e:
                logger.warning(f"ModelScope {load_type} snapshot load failed for '{repo_id}': {e}. Falling back to Hugging Face...")

        # Hugging Face default path (utilizes HF_ENDPOINT mirror seamlessly)
        try:
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from transformers import AutoTokenizer, AutoModelForCausalLM
            if load_type == "tokenizer":
                tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
                logger.info(f"✅ Tokenizer '{repo_id}' loaded from Hugging Face.")
                return tokenizer
            else:
                model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
                logger.info(f"✅ Model '{repo_id}' loaded from Hugging Face.")
                return model
        except ImportError:
            logger.error(f"The 'transformers' library is required for Hugging Face {load_type} loading.")
            logger.info("Please install via: pip install transformers")
            raise

# ==============================================================================
# Helper verification runner
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="nano-llm: Hub Adapter Verification Utility")
    parser.add_argument("--provider", type=str, default=None, choices=["hf", "ms"], help="Hub provider override")
    parser.add_argument("--repo_id", type=str, default="gpt2", help="Repository identifier")
    args = parser.parse_args()
    
    adapter = HubAdapter(provider=args.provider)
    
    try:
        # Dry run loading tokenizer
        tokenizer = adapter.load_tokenizer_or_model(args.repo_id, load_type="tokenizer")
        print(f"\n=======================================================================")
        print(f"✅ HubAdapter initialized successfully!")
        print(f"🎯 Targeted Provider: {adapter.provider.upper()}")
        print(f"📝 Loaded Tokenizer: {tokenizer}")
        print(f"=======================================================================\n")
    except Exception as e:
        print(f"Dry-run failed: {e}")
