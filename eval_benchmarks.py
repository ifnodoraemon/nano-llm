import re
import argparse
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Pre-Baked Benchmark Question Corpora
# ==============================================================================

MMLU_SAMPLES = [
    {
        "question": "What is the primary optimization benefit of Root Mean Square Normalization (RMSNorm) over LayerNorm?",
        "choices": [
            "A) It completely eliminates activation scale drift",
            "B) It drops the mean-centering step, saving computational operations",
            "C) It replaces all linear layers with low-rank convolutions",
            "D) It naturally supports dynamic quantization layers"
        ],
        "answer": "B"
    },
    {
        "question": "Which mathematical operation is used in Rotary Position Embeddings (RoPE) to incorporate token positions?",
        "choices": [
            "A) Adding absolute position values to the input embeddings",
            "B) Element-wise multiplying keys and queries by a sinusoidal scale matrix",
            "C) Rotating the Query and Key vector pairs in 2D sub-spaces by positional angles",
            "D) Convolving token embeddings along the causal sequence dimension"
        ],
        "answer": "C"
    },
    {
        "question": "In Direct Preference Optimization (DPO), what does the beta parameter regulate?",
        "choices": [
            "A) The gradient accumulation scaling constant",
            "B) The learning rate Cosine warmup duration",
            "C) The KL divergence regularization weight relative to the reference model",
            "D) The bits size for asymmetric RTN weight compression"
        ],
        "answer": "C"
    }
]

GSM8K_SAMPLES = [
    {
        "question": "Weng earns $12 an hour babysitting. Yesterday, she babysat for 5 hours. How much money did she earn?",
        "answer": "60"
    },
    {
        "question": "A box contains 5 red balls, 3 blue balls, and 2 green balls. If John takes out all of them except 4 balls, how many balls does he take out?",
        "answer": "6"
    },
    {
        "question": "A farmer has 15 apple trees. Each tree yields 20 apples. If the farmer sells half of all apples, how many apples does he keep?",
        "answer": "150"
    }
]

# ==============================================================================
# 2. Logits-Based MMLU Evaluator
# ==============================================================================

@torch.no_grad()
def evaluate_mmlu(model: Transformer, tokenizer, device: str = "cuda") -> float:
    """
    Evaluates MMLU multiple choice questions.
    Extracts the logits for tokens 'A', 'B', 'C', 'D' at the single next step.
    The letter with the highest logit score is the prediction.
    """
    logger.info(f"--- Running MMLU Leaderboard Benchmarking ({len(MMLU_SAMPLES)} questions) ---")
    correct_count = 0
    
    # Pre-encode option tokens
    option_tokens = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ["A", "B", "C", "D"]]
    
    for idx, sample in enumerate(MMLU_SAMPLES):
        # Format the question and options
        prompt = f"Question: {sample['question']}\n"
        for choice in sample["choices"]:
            prompt += f"{choice}\n"
        prompt += "Answer: "
        
        # Format matching ChatML standard
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Forward pass: get logits of the last token
        logits, _ = model(x)
        last_token_logits = logits[0, -1, :] # Shape: (vocab_size)
        
        # Extract logits corresponding only to tokens A, B, C, D
        option_logits = last_token_logits[option_tokens]
        prediction_idx = torch.argmax(option_logits).item()
        predicted_letter = ["A", "B", "C", "D"][prediction_idx]
        
        is_correct = predicted_letter == sample["answer"]
        if is_correct:
            correct_count += 1
            
        logger.info(
            f"Q{idx+1}: Predicted={predicted_letter} | Correct={sample['answer']} | "
            f"Result={'✅ CORRECT' if is_correct else '❌ WRONG'}"
        )
        
    accuracy = correct_count / len(MMLU_SAMPLES)
    return accuracy

# ==============================================================================
# 3. Chain-of-Thought GSM8K Evaluator (Exact Match)
# ==============================================================================

@torch.no_grad()
def evaluate_gsm8k(model: Transformer, tokenizer, device: str = "cuda") -> float:
    """
    Evaluates Grade School Math (GSM8K) word problems.
    Triggers Chain-of-Thought autoregressive generation,
    parses out the final numerical value using regex, and checks for Exact Match.
    """
    logger.info(f"--- Running GSM8K Math Benchmarking ({len(GSM8K_SAMPLES)} questions) ---")
    correct_count = 0
    
    for idx, sample in enumerate(GSM8K_SAMPLES):
        prompt = f"Question: {sample['question']}\nLet's think step by step. Show your calculations and state the final answer clearly."
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generate completion autoregressively (max 256 tokens)
        generated_ids = list(input_ids)
        for _ in range(256):
            logits, _ = model(torch.tensor([generated_ids], dtype=torch.long, device=device))
            last_logits = logits[0, -1, :]
            next_token = torch.argmax(last_logits).item()
            generated_ids.append(next_token)
            
            # Break on EOS or ChatML boundary
            if next_token in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]:
                break
                
        completion = tokenizer.decode(generated_ids[len(input_ids):], skip_special_tokens=True)
        
        # Extract numerical digits from the completion text
        # In GSM8K, the final answer is usually the last number in the reasoning text
        numbers = re.findall(r'\b\d+\b', completion)
        predicted_answer = numbers[-1] if numbers else "NONE"
        
        is_correct = predicted_answer == sample["answer"]
        if is_correct:
            correct_count += 1
            
        logger.info(
            f"Math Q{idx+1}: Predicted={predicted_answer} | Correct={sample['answer']} | "
            f"Result={'✅ CORRECT' if is_correct else '❌ WRONG'}\n"
            f"   [CoT]: \"{completion.strip()[:120]}...\""
        )
        
    accuracy = correct_count / len(GSM8K_SAMPLES)
    return accuracy

# ==============================================================================
# Main Orchestrated Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="nano-llm: Automated Leaderboard Benchmark Evaluator")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved model .pt checkpoint file")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading checkpoint state from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model_config = checkpoint["config"]
    
    # Instantiate Model
    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B") 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        
    # 1. Run MMLU
    mmlu_acc = evaluate_mmlu(model, tokenizer, device=device)
    
    # 2. Run GSM8K
    gsm_acc = evaluate_gsm8k(model, tokenizer, device=device)
    
    logger.info("=======================================================================")
    logger.info("📊 Benchmark Leaderboard Report:")
    logger.info("-----------------------------------------------------------------------")
    logger.info(f"🏆 MMLU Multiple Choice Accuracy: {mmlu_acc*100:.2f}%")
    logger.info(f"🏆 GSM8K Grade School Math Accuracy: {gsm_acc*100:.2f}%")
    logger.info(f"🏆 Consolidated Leaderboard Index: {((mmlu_acc + gsm_acc)/2)*100:.2f}%")
    logger.info("=======================================================================")
    
    # Save a JSON metric file for our Web Dashboard to read!
    import json
    report = {
        "mmlu_accuracy": mmlu_acc,
        "gsm8k_accuracy": gsm_acc,
        "consolidated_score": (mmlu_acc + gsm_acc) / 2
    }
    with open("./outputs/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
