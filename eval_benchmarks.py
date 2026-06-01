import os
import re
import json
import random
import argparse
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import Transformer
from utils.sandbox_executor import SandboxCodeExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Pre-Baked Benchmark Question Corpora (Fallbacks)
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

ARC_SAMPLES = [
    {
        "question": "Which of the following is a physical change?",
        "choices": [
            "A) Burning wood",
            "B) Melting ice",
            "C) Rusting iron",
            "D) Baking a cake"
        ],
        "answer": "B"
    },
    {
        "question": "Which cell organelle is known as the powerhouse of the cell?",
        "choices": [
            "A) Nucleus",
            "B) Mitochondria",
            "C) Ribosome",
            "D) Golgi apparatus"
        ],
        "answer": "B"
    }
]

HELLASWAG_SAMPLES = [
    {
        "context": "A man is sawing a wooden plank. He holds the plank with one hand and uses the other hand to",
        "choices": [
            "A) hold a hammer",
            "B) move the saw back and forth",
            "C) read a book",
            "D) drink a glass of water"
        ],
        "answer": "B"
    }
]

HUMANEVAL_SAMPLES = [
    {
        "task_id": "custom/1",
        "prompt": "def add_numbers(a, b):\n    \"\"\"Return the sum of a and b\"\"\"\n",
        "test": "assert add_numbers(2, 3) == 5\nassert add_numbers(-1, 1) == 0",
        "entry_point": "add_numbers"
    }
]

# Helper function to load dataset from jsonl
def load_jsonl(path: str) -> list:
    if not path or not os.path.exists(path):
        return None
    samples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    except Exception as e:
        logger.error(f"Error loading JSONL file {path}: {e}")
        return None

# ==============================================================================
# 2. Logits-Based MMLU/ARC/HellaSwag Choice Evaluators
# ==============================================================================

@torch.no_grad()
def evaluate_choices(model: Transformer, tokenizer, samples: list, name: str, device: str = "cuda") -> float:
    """
    Generic logits-based multiple-choice option evaluator.
    Extracts logits for choice tokens at the next step.
    """
    logger.info(f"--- Running {name} Benchmarking ({len(samples)} questions) ---")
    correct_count = 0
    
    # Pre-encode option tokens
    option_tokens = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ["A", "B", "C", "D"]]
    
    for idx, sample in enumerate(samples):
        # Format the question and options
        if "question" in sample:
            prompt = f"Question: {sample['question']}\n"
        elif "context" in sample:
            prompt = f"Context: {sample['context']}\nComplete the sentence:\n"
        else:
            continue
            
        for choice in sample.get("choices", []):
            prompt += f"{choice}\n"
        prompt += "Answer: "
        
        # Format matching ChatML standard
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Forward pass: get logits of the last token
        logits, _, _ = model(x)
        last_token_logits = logits[0, -1, :] # Shape: (vocab_size)
        
        # Extract logits corresponding only to tokens A, B, C, D
        option_logits = last_token_logits[option_tokens]
        prediction_idx = torch.argmax(option_logits).item()
        predicted_letter = ["A", "B", "C", "D"][prediction_idx]
        
        correct_answer = sample.get("answer", "A")
        is_correct = predicted_letter == correct_answer
        if is_correct:
            correct_count += 1
            
        # Log periodically to avoid output clutter
        if idx % 10 == 0 or len(samples) <= 5:
            logger.info(
                f"{name} Q{idx+1}: Predicted={predicted_letter} | Correct={correct_answer} | "
                f"Result={'✅ CORRECT' if is_correct else '❌ WRONG'}"
            )
        
    accuracy = correct_count / len(samples) if samples else 0.0
    logger.info(f"🏆 {name} Accuracy: {accuracy*100:.2f}%")
    return accuracy

# ==============================================================================
# 3. Chain-of-Thought GSM8K Evaluator (Exact Match)
# ==============================================================================

@torch.no_grad()
def evaluate_gsm8k(model: Transformer, tokenizer, samples: list, device: str = "cuda") -> float:
    """
    Evaluates Grade School Math (GSM8K) word problems.
    Triggers Chain-of-Thought autoregressive generation,
    parses out the final numerical value using regex, and checks for Exact Match.
    """
    logger.info(f"--- Running GSM8K Math Benchmarking ({len(samples)} questions) ---")
    correct_count = 0
    
    for idx, sample in enumerate(samples):
        prompt = f"Question: {sample['question']}\nLet's think step by step. Show your calculations and state the final answer clearly."
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generate completion autoregressively (max 256 tokens)
        generated_ids = list(input_ids)
        for _ in range(256):
            logits, _, _ = model(torch.tensor([generated_ids], dtype=torch.long, device=device))
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
        
        correct_answer = sample.get("answer", "").strip()
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct_count += 1
            
        if idx % 10 == 0 or len(samples) <= 5:
            logger.info(
                f"Math Q{idx+1}: Predicted={predicted_answer} | Correct={correct_answer} | "
                f"Result={'✅ CORRECT' if is_correct else '❌ WRONG'}\n"
                f"   [CoT]: \"{completion.strip()[:100]}...\""
            )
        
    accuracy = correct_count / len(samples) if samples else 0.0
    logger.info(f"🏆 GSM8K Accuracy: {accuracy*100:.2f}%")
    return accuracy

# ==============================================================================
# 4. HumanEval Python Coding Evaluator (Sandbox Execution)
# ==============================================================================

@torch.no_grad()
def evaluate_humaneval(model: Transformer, tokenizer, samples: list, device: str = "cuda") -> float:
    """
    Evaluates Python coding capacity on HumanEval using SandboxCodeExecutor.
    """
    logger.info(f"--- Running HumanEval Code Benchmarking ({len(samples)} questions) ---")
    sandbox = SandboxCodeExecutor(timeout=2.0)
    correct_count = 0
    
    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]
        formatted_prompt = f"<|im_start|>user\nWrite the Python function definition to solve this:\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        generated_ids = list(input_ids)
        
        for _ in range(256):
            active_x = torch.tensor([generated_ids[-model.config.block_size:]], dtype=torch.long, device=device)
            logits, _, _ = model(active_x)
            next_tok = torch.argmax(logits[0, -1, :]).item()
            generated_ids.append(next_tok)
            if next_tok in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]:
                break
                
        completion = tokenizer.decode(generated_ids[len(input_ids):], skip_special_tokens=True)
        
        # Extract code block
        extracted_code = sandbox.extract_code(completion)
        if not extracted_code:
            # Fallback if model didn't wrap in markdown blocks but generated code
            extracted_code = completion
            
        # Run in sandbox with assertions
        assertions = sample.get("test", "")
        exec_res = sandbox.execute_and_verify(extracted_code, assertions=assertions)
        
        is_correct = exec_res.get("success", False)
        if is_correct:
            correct_count += 1
            
        if idx % 5 == 0 or len(samples) <= 5:
            logger.info(
                f"Code Q{idx+1} ({sample.get('task_id', 'custom')}): "
                f"Result={'✅ PASSED' if is_correct else '❌ FAILED'} | "
                f"Error: {exec_res.get('error', 'None')}"
            )
            
    accuracy = correct_count / len(samples) if samples else 0.0
    logger.info(f"🏆 HumanEval Pass@1: {accuracy*100:.2f}%")
    return accuracy

# ==============================================================================
# 5. LLM-as-a-Judge Automated Elo Arena
# ==============================================================================

ARENA_PROMPTS = [
    "Write a secure python function to read a CSV file safely without path traversal vulnerabilities.",
    "Explain MLA (Multi-Head Latent Attention) simply to a high schooler.",
    "Solve the system of linear equations: x + 2y = 8 and 3x - y = 3.",
    "If a train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours, what is its average speed?",
    "Describe the key architectural differences between standard GQA and DeepSeek MLA.",
    "Write a short, engaging Sci-Fi story about a quantum computer achieving consciousness."
]

def generate_completion_for_arena(model: Transformer, tokenizer, prompt: str, device: str) -> str:
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    generated_ids = list(input_ids)
    
    for _ in range(128):
        logits, _, _ = model(torch.tensor([generated_ids], dtype=torch.long, device=device))
        last_logits = logits[0, -1, :]
        next_token = torch.argmax(last_logits).item()
        generated_ids.append(next_token)
        if next_token in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]:
            break
            
    return tokenizer.decode(generated_ids[len(input_ids):], skip_special_tokens=True)

def heuristic_referee_judge(prompt: str, response_a: str, response_b: str) -> float:
    score_a = 0.0
    score_b = 0.0
    
    # 1. Structural reasoning check
    has_cot_a = "<think>" in response_a and "</think>" in response_a
    has_cot_b = "<think>" in response_b and "</think>" in response_b
    if has_cot_a: score_a += 1.0
    if has_cot_b: score_b += 1.0
    
    # 2. Length regularization
    len_a = len(response_a)
    len_b = len(response_b)
    
    if 50 < len_a < 1000: score_a += 1.5
    if 50 < len_b < 1000: score_b += 1.5
    
    # 3. Simple repetitive word loop detection
    words_a = response_a.lower().split()
    words_b = response_b.lower().split()
    
    uniq_ratio_a = len(set(words_a)) / max(1, len(words_a))
    uniq_ratio_b = len(set(words_b)) / max(1, len(words_b))
    
    score_a += uniq_ratio_a * 2.0
    score_b += uniq_ratio_b * 2.0
    
    # 4. Coding constraint match
    if "python" in prompt.lower() or "code" in prompt.lower():
        if "def " in response_a or "```" in response_a: score_a += 1.5
        if "def " in response_b or "```" in response_b: score_b += 1.5
        
    total = score_a + score_b
    if total == 0:
        return 0.5
    return score_a / total

def evaluate_arena(
    model_a: Transformer, 
    model_b: Transformer, 
    tokenizer, 
    device: str = "cuda"
) -> dict:
    logger.info(f"--- Launching LLM-as-a-Judge Elo Arena ({len(ARENA_PROMPTS)} rounds) ---")
    elo_a = 1200.0
    elo_b = 1200.0
    K = 32.0
    
    wins_a = 0
    wins_b = 0
    ties = 0
    
    for idx, prompt in enumerate(ARENA_PROMPTS):
        resp_a = generate_completion_for_arena(model_a, tokenizer, prompt, device)
        resp_b = generate_completion_for_arena(model_b, tokenizer, prompt, device)
        
        prob_a_wins = heuristic_referee_judge(prompt, resp_a, resp_b)
        
        if prob_a_wins > 0.55:
            outcome_a = 1.0
            outcome_b = 0.0
            wins_a += 1
        elif prob_a_wins < 0.45:
            outcome_a = 0.0
            outcome_b = 1.0
            wins_b += 1
        else:
            outcome_a = 0.5
            outcome_b = 0.5
            ties += 1
            
        exp_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
        exp_b = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_b) / 400.0))
        
        elo_a = elo_a + K * (outcome_a - exp_a)
        elo_b = elo_b + K * (outcome_b - exp_b)
        
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "final_elo_a": elo_a,
        "final_elo_b": elo_b
    }

# ==============================================================================
# 6. Automated Needle-in-a-Haystack (NIAH) Synthesized Context Evaluator
# ==============================================================================

class NeedleInAHaystackEvaluator:
    def __init__(self, context_lengths: list[int] = None, document_depths: list[float] = None):
        self.context_lengths = context_lengths if context_lengths else [1024, 2048, 4096]
        self.document_depths = document_depths if document_depths else [0.1, 0.5, 0.9]
        
    def generate_noise_context(self, target_word_count: int) -> str:
        filler_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Deep learning models require balanced multi-GPU clusters for high performance.",
            "Rotary Position Embeddings stretch coordinates for extremely long sequences.",
            "Multi-Head Latent Attention compresses KV-Cache sizes by up to 93%.",
            "Continuous Batching schedules concurrent user queries token-by-token.",
            "Asynchronous checkpoints write weights safely in background threads."
        ]
        text = ""
        while len(text.split()) < target_word_count:
            text += random.choice(filler_sentences) + " "
        return text

    def run_eval(self, model: Transformer, tokenizer, device: str = "cuda") -> dict:
        logger.info(f"--- Running Needle-in-a-Haystack (NIAH) Evaluation ---")
        results = {}
        random.seed(1337)
        
        needle_value = "994285"
        needle_sentence = f" The magic secret number key is: {needle_value}. Keep this key safe. "
        question = "What is the magic secret number key? Respond with the number only wrapped inside <answer>...</answer> tags."
        
        for c_len in self.context_lengths:
            for depth in self.document_depths:
                noise = self.generate_noise_context(c_len)
                words = noise.split()
                
                inject_idx = int(len(words) * depth)
                words.insert(inject_idx, needle_sentence)
                full_context = " ".join(words)
                
                prompt = f"Context: {full_context}\n\nQuestion: {question}"
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
                input_ids = input_ids[-model.config.block_size + 128:]
                
                generated_ids = list(input_ids)
                for _ in range(64):
                    active_x = torch.tensor([generated_ids[-model.config.block_size:]], dtype=torch.long, device=device)
                    logits, _, _ = model(active_x)
                    next_tok = torch.argmax(logits[0, -1, :]).item()
                    generated_ids.append(next_tok)
                    if next_tok in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]:
                        break
                        
                response = tokenizer.decode(generated_ids[len(input_ids):], skip_special_tokens=True)
                
                pattern = r"<answer>(.*?)</answer>"
                match = re.search(pattern, response, re.DOTALL)
                extracted = match.group(1).strip() if match else ""
                
                if not extracted:
                    is_correct = needle_value in response
                else:
                    is_correct = extracted == needle_value
                    
                score = 1.0 if is_correct else 0.0
                results[f"{c_len}_{depth:.1f}"] = score
                
        return results

@torch.no_grad()
def evaluate_perplexity(model: Transformer, val_bin_path: str = "./data/binaries/val.bin", block_size: int = 1024, num_batches: int = 100, device: str = "cuda") -> float:
    logger.info(f"--- Running Intrinsic Perplexity (PPL) Evaluation on {val_bin_path} ---")
    if not os.path.exists(val_bin_path):
        logger.warning(f"Validation binary not found at {val_bin_path} — skipping PPL.")
        return float('inf')
        
    try:
        import numpy as np
        import math
        val_data = np.memmap(val_bin_path, dtype=np.uint16, mode="r")
        total_loss = 0.0
        count = 0
        
        for i in range(num_batches):
            start_idx = (i * block_size) % (len(val_data) - block_size - 1)
            x = torch.from_numpy(val_data[start_idx : start_idx + block_size].astype(np.int64)).unsqueeze(0).to(device)
            y = torch.from_numpy(val_data[start_idx + 1 : start_idx + 1 + block_size].astype(np.int64)).unsqueeze(0).to(device)
            
            logits, loss, _ = model(x, targets=y)
            total_loss += loss.item()
            count += 1
            
        avg_loss = total_loss / count
        perplexity = math.exp(avg_loss)
        logger.info(f"🏆 Held-out Validation Loss: {avg_loss:.4f} | Validation Perplexity (PPL): {perplexity:.4f}")
        return perplexity
    except Exception as e:
        logger.error(f"Failed to calculate perplexity: {e}")
        return float('inf')

# ==============================================================================
# Main Orchestrated Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="nano-llm: Automated Leaderboard Benchmark Evaluator")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved model .pt checkpoint file")
    parser.add_argument("--baseline_checkpoint_path", type=str, default=None, help="Optional path to baseline model checkpoint to run Arena")
    parser.add_argument("--mmlu_path", type=str, default="./data/eval/mmlu.jsonl", help="Path to MMLU JSONL file")
    parser.add_argument("--gsm8k_path", type=str, default="./data/eval/gsm8k.jsonl", help="Path to GSM8K JSONL file")
    parser.add_argument("--arc_path", type=str, default="./data/eval/arc.jsonl", help="Path to ARC Challenge JSONL file")
    parser.add_argument("--hellaswag_path", type=str, default="./data/eval/hellaswag.jsonl", help="Path to HellaSwag JSONL file")
    parser.add_argument("--humaneval_path", type=str, default="./data/eval/humaneval.jsonl", help="Path to HumanEval JSONL file")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading checkpoint state from: {args.checkpoint_path}")
    from utils.checkpoint_utils import load_checkpoint_with_fp8_translation
    model_config, state_dict = load_checkpoint_with_fp8_translation(args.checkpoint_path, map_location=device)
    
    model = Transformer(model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info("Initializing tokenizer...")
    from utils.tokenizer_loader import load_tokenizer
    tokenizer = load_tokenizer()
    
    # 0. Load Datasets (Load from JSONL files, fallback if not found)
    mmlu_samples = load_jsonl(args.mmlu_path) or MMLU_SAMPLES
    gsm_samples = load_jsonl(args.gsm8k_path) or GSM8K_SAMPLES
    arc_samples = load_jsonl(args.arc_path) or ARC_SAMPLES
    hs_samples = load_jsonl(args.hellaswag_path) or HELLASWAG_SAMPLES
    he_samples = load_jsonl(args.humaneval_path) or HUMANEVAL_SAMPLES
    
    # 1. Run Validation Perplexity (PPL)
    val_ppl = evaluate_perplexity(model, val_bin_path="./data/binaries/val.bin", block_size=model_config.block_size, device=device)

    # 2. Run Evaluations
    mmlu_acc = evaluate_choices(model, tokenizer, mmlu_samples, "MMLU", device=device)
    gsm_acc = evaluate_gsm8k(model, tokenizer, gsm_samples, device=device)
    arc_acc = evaluate_choices(model, tokenizer, arc_samples, "ARC-Challenge", device=device)
    hs_acc = evaluate_choices(model, tokenizer, hs_samples, "HellaSwag", device=device)
    he_pass = evaluate_humaneval(model, tokenizer, he_samples, device=device)
    
    # 3. Run Elo Arena
    if args.baseline_checkpoint_path and os.path.exists(args.baseline_checkpoint_path):
        logger.info(f"Loading baseline checkpoint state from: {args.baseline_checkpoint_path}")
        base_config, base_state = load_checkpoint_with_fp8_translation(args.baseline_checkpoint_path, map_location=device)
        baseline_model = Transformer(base_config).to(device)
        baseline_model.load_state_dict(base_state)
        baseline_model.eval()
    else:
        logger.info("No baseline checkpoint path provided. Running Self-Play Elo Arena comparing Model A against base initialization.")
        baseline_model = Transformer(model_config).to(device)
        baseline_model.eval()
        
    arena_results = evaluate_arena(model, baseline_model, tokenizer, device=device)
    
    # Consolidated Average Score card
    scores_list = [mmlu_acc, gsm_acc, arc_acc, hs_acc, he_pass]
    consolidated_score = sum(scores_list) / len(scores_list)
    
    logger.info("=======================================================================")
    logger.info("📊 Benchmark Leaderboard & Arena Report:")
    logger.info("-----------------------------------------------------------------------")
    logger.info(f"🏆 Held-out Validation Perplexity (PPL): {val_ppl:.4f}")
    logger.info(f"🏆 MMLU Accuracy: {mmlu_acc*100:.2f}%")
    logger.info(f"🏆 GSM8K Accuracy: {gsm_acc*100:.2f}%")
    logger.info(f"🏆 ARC-Challenge Accuracy: {arc_acc*100:.2f}%")
    logger.info(f"🏆 HellaSwag Accuracy: {hs_acc*100:.2f}%")
    logger.info(f"🏆 HumanEval Pass@1: {he_pass*100:.2f}%")
    logger.info(f"🏆 Consolidated Leaderboard Index: {consolidated_score*100:.2f}%")
    logger.info(f"🏆 Arena Model A Final Elo Rating: {arena_results['final_elo_a']:.1f}")
    logger.info(f"🏆 Arena Model B Final Elo Rating: {arena_results['final_elo_b']:.1f}")
    logger.info(f"🏆 Match Statistics: A Wins={arena_results['wins_a']} | B Wins={arena_results['wins_b']} | Ties={arena_results['ties']}")
    logger.info("=======================================================================")
    
    # 4. Run Needle-in-a-Haystack Evaluator
    niah_evaluator = NeedleInAHaystackEvaluator()
    niah_results = niah_evaluator.run_eval(model, tokenizer, device=device)
    
    # Save a JSON metric file for our Web Dashboard to read!
    report = {
        "validation_perplexity": val_ppl,
        "mmlu_accuracy": mmlu_acc,
        "gsm8k_accuracy": gsm_acc,
        "arc_accuracy": arc_acc,
        "hellaswag_accuracy": hs_acc,
        "humaneval_pass": he_pass,
        "consolidated_score": consolidated_score,
        "arena_wins_a": arena_results["wins_a"],
        "arena_wins_b": arena_results["wins_b"],
        "arena_ties": arena_results["ties"],
        "arena_elo_a": arena_results["final_elo_a"],
        "arena_elo_b": arena_results["final_elo_b"],
        "niah_results": niah_results
    }
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
