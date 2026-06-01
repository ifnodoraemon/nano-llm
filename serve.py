"""nano-llm: Autoregressive KV-Cached Serving Server with MLA/MoE support."""

import sys
import argparse
import logging
import torch
from model import Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Re-export from split modules for backward compatibility
from serve.generation import CustomTokenizerAdapter, generate_with_static_cache
from serve.speculative import generate_with_speculative_decoding
from serve.paged_attention import generate_with_paged_attention_continuous_batching


def main():
    parser = argparse.ArgumentParser(description="nano-llm: Autoregressive KV-Cached Serving Server")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved .pt model checkpoint file")
    parser.add_argument("--prompt", type=str, default="Tell me a joke about computer programming.", help="Prompt content")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation diversity temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K tail filter limits")
    parser.add_argument("--image", type=str, default=None, help="Optional image path or URL for multimodal VLM serving")
    parser.add_argument("--speculative", action="store_true", help="Enable Speculative Decoding acceleration")
    parser.add_argument("--paged_continuous", action="store_true", help="Enable PagedAttention & Continuous Batching")
    parser.add_argument("--reasoning_effort", type=str, default="low", choices=["low", "medium", "high"], help="Reasoning effort level")
    parser.add_argument("--rag_sources", type=str, default=None, help="Comma-separated file paths for RAG knowledge base")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading custom checkpoint state from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]

    model = Transformer(model_config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Initializing tokenizer...")
    from utils.tokenizer_loader import load_tokenizer
    tokenizer = load_tokenizer(fallback_model_name="gpt2")

    if args.image is not None:
        from utils.vision_helper import extract_image_features
        pixel_values = extract_image_features(
            image_path_or_url=args.image,
            vision_dim=model_config.vision_dim or 1152,
            num_patches=16,
            device=device,
        )
    else:
        pixel_values = None

    prompt = args.prompt

    if args.rag_sources:
        import os
        from utils.rag_retriever import ChunkProcessor, HybridRetriever
        processor = ChunkProcessor()
        retriever = HybridRetriever()
        chunks = []
        for p in [p.strip() for p in args.rag_sources.split(",")]:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        text = f.read()
                        chunks.extend(processor.split_text(text))
                except Exception as e:
                    logger.error(f"Failed to read RAG source {p}: {e}")
            else:
                logger.warning(f"RAG source path {p} does not exist.")
        if chunks:
            retriever.fit(chunks)
            logger.info(f"Loaded {len(chunks)} chunks into RAG retriever.")
            retrieved_chunks = retriever.retrieve(args.prompt, top_k=2)
            if retrieved_chunks:
                context = "\n---\n".join(retrieved_chunks)
                prompt = f"Use the following reference documents to answer the question:\n{context}\n\nQuestion: {args.prompt}"
                logger.info("RAG Context injected successfully.")

    if args.reasoning_effort == "high":
        from utils.mcts_engine import MCTSEngine
        engine = MCTSEngine(model, tokenizer, num_simulations=16)
        logger.info("Running Monte Carlo Tree Search reasoning...")
        response = engine.search(prompt)
        print(f"\nAssistant (MCTS Search Result): {response}")
    elif args.paged_continuous:
        logger.info(f"Running PagedAttention & Continuous Batching on {device}...")
        test_prompts = [
            prompt,
            "Explain DeepSeek Multi-Head Latent Attention in 2 sentences.",
            "Write a simple Python function to calculate Fibonacci series.",
        ]
        generate_with_paged_attention_continuous_batching(
            model=model, tokenizer=tokenizer, prompts=test_prompts,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device,
        )
    elif args.speculative:
        logger.info(f"Running Speculative Decoding on {device}...")
        generate_with_speculative_decoding(
            model=model, tokenizer=tokenizer, prompt=prompt, pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, device=device,
        )
    else:
        logger.info(f"Running static KV-cached autoregressive serving on {device}...")
        generate_with_static_cache(
            model=model, tokenizer=tokenizer, prompt=prompt, pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, device=device,
        )


if __name__ == "__main__":
    main()
