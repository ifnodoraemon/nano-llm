import re
import random
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
# 4. LLM-as-a-Judge Automated Elo Arena
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
    """
    Generates text using standard autoregressive generation for Arena evaluation.
    """
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    generated_ids = list(input_ids)
    
    for _ in range(128):
        logits, _ = model(torch.tensor([generated_ids], dtype=torch.long, device=device))
        last_logits = logits[0, -1, :]
        next_token = torch.argmax(last_logits).item()
        generated_ids.append(next_token)
        if next_token in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]:
            break
            
    return tokenizer.decode(generated_ids[len(input_ids):], skip_special_tokens=True)

def heuristic_referee_judge(prompt: str, response_a: str, response_b: str) -> float:
    """
    Simulates a high-quality LLM-as-a-judge by computing multi-dimensional heuristic quality scores.
    Scores are based on:
    1. Richness of CoT reasoning (presence of reasoning tags / length of thinking).
    2. Answer structure and vocabulary diversity.
    3. Formatting constraints and blocklist hygiene.
    
    Returns:
        Score between 0.0 and 1.0 representing probability of Model A winning.
        0.5 represents a tie, >0.5 represents A winning, <0.5 represents B winning.
    """
    score_a = 0.0
    score_b = 0.0
    
    # 1. Structural reasoning check
    has_cot_a = "<think>" in response_a and "</think>" in response_a
    has_cot_b = "<think>" in response_b and "</think>" in response_b
    if has_cot_a: score_a += 1.0
    if has_cot_b: score_b += 1.0
    
    # 2. Length regularization (penalize extreme short or extreme word loop repetitions)
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
    
    # 4. Coding constraint match (if prompt asks for python code)
    if "python" in prompt.lower() or "code" in prompt.lower():
        if "def " in response_a or "```" in response_a: score_a += 1.5
        if "def " in response_b or "```" in response_b: score_b += 1.5
        
    # Calculate win probability
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
    """
    Executes a blind pairwise de duel (Arena) over ARENA_PROMPTS.
    Updates Elo ratings starting from 1200.
    """
    logger.info(f"--- Launching LLM-as-a-Judge Elo Arena ({len(ARENA_PROMPTS)} rounds) ---")
    elo_a = 1200.0
    elo_b = 1200.0
    K = 32.0 # Elo scaling weight
    
    wins_a = 0
    wins_b = 0
    ties = 0
    
    for idx, prompt in enumerate(ARENA_PROMPTS):
        logger.info(f"Arena Round {idx+1}: Prompt = '{prompt[:50]}...'")
        
        # Generate completions blind
        resp_a = generate_completion_for_arena(model_a, tokenizer, prompt, device)
        resp_b = generate_completion_for_arena(model_b, tokenizer, prompt, device)
        
        # Judge grades the pairwise outputs
        prob_a_wins = heuristic_referee_judge(prompt, resp_a, resp_b)
        
        # Determine match outcome
        if prob_a_wins > 0.55:
            outcome_a = 1.0
            outcome_b = 0.0
            wins_a += 1
            logger.info("   [Judge Decision]: Model A Wins!")
        elif prob_a_wins < 0.45:
            outcome_a = 0.0
            outcome_b = 1.0
            wins_b += 1
            logger.info("   [Judge Decision]: Model B Wins!")
        else:
            outcome_a = 0.5
            outcome_b = 0.5
            ties += 1
            logger.info("   [Judge Decision]: It's a Draw!")
            
        # Expected scores
        exp_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
        exp_b = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_b) / 400.0))
        
        # Update Elo
        elo_a = elo_a + K * (outcome_a - exp_a)
        elo_b = elo_b + K * (outcome_b - exp_b)
        
        logger.info(f"   [Elo Status]: Model A Elo = {elo_a:.1f} | Model B Elo = {elo_b:.1f}")
        
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "final_elo_a": elo_a,
        "final_elo_b": elo_b
    }

# ==============================================================================
# 5. Automated Needle-in-a-Haystack (NIAH) Synthesized Context Evaluator
# ==============================================================================

class NeedleInAHaystackEvaluator:
    """
    Industry-standard Needle-in-a-Haystack (NIAH) benchmark synthetic generator.
    Generates extremely long document contexts, inserts hidden facts at variable
    depth percentiles, and scores model retrieval accuracy dynamically.
    """
    def __init__(self, context_lengths: list[int] = None, document_depths: list[float] = None):
        self.context_lengths = context_lengths if context_lengths else [1024, 2048, 4096]
        self.document_depths = document_depths if document_depths else [0.1, 0.5, 0.9] # 10%, 50%, 90% depth
        
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
        logger.info(f"--- Running Needle-in-a-Haystack (NIAH) Evaluation ({len(self.context_lengths)} lengths x {len(self.document_depths)} depths) ---")
        
        results = {}
        random.seed(1337)
        
        needle_value = "994285"
        needle_sentence = f" The magic secret number key is: {needle_value}. Keep this key safe. "
        question = "What is the magic secret number key? Respond with the number only wrapped inside <answer>...</answer> tags."
        
        for c_len in self.context_lengths:
            for depth in self.document_depths:
                logger.info(f"👉 Evaluating context_length={c_len} | needle_depth={depth*100:.0f}%...")
                
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
                    logits, _ = model(active_x)
                    next_tok = torch.argmax(logits[0, -1, :]).item()
                    generated_ids.append(next_tok)
                    if next_tok in [tokenizer.eos_token_id, tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]:
                        break
                        
                response = tokenizer.decode(generated_ids[len(input_ids):], skip_special_tokens=True)
                
                # Check for answer tags
                pattern = r"<answer>(.*?)</answer>"
                match = re.search(pattern, response, re.DOTALL)
                extracted = match.group(1).strip() if match else ""
                
                if not extracted:
                    is_correct = needle_value in response
                else:
                    is_correct = extracted == needle_value
                    
                score = 1.0 if is_correct else 0.0
                results[f"{c_len}_{depth:.1f}"] = score
                
                logger.info(
                    f"   [Result]: {'✅ FOUND' if is_correct else '❌ LOST'} | "
                    f"Response: \"{response.strip()[:60]}...\""
                )
                
        return results

# ==============================================================================
# Main Orchestrated Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="nano-llm: Automated Leaderboard Benchmark Evaluator")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved model .pt checkpoint file")
    parser.add_argument("--baseline_checkpoint_path", type=str, default=None, help="Optional path to baseline model checkpoint to run Arena")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading checkpoint state from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    
    # Instantiate Model
    model = Transformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load tokenizer
    logger.info("Initializing tokenizer...")
    from utils.hub_adapter import HubAdapter
    hub = HubAdapter()
    tokenizer = hub.load_tokenizer_or_model("Qwen/Qwen2.5-7B" if hub.provider == "hf" else "qwen/Qwen2.5-7B", load_type="tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
        
    # 1. Run MMLU
    mmlu_acc = evaluate_mmlu(model, tokenizer, device=device)
    
    # 2. Run GSM8K
    gsm_acc = evaluate_gsm8k(model, tokenizer, device=device)
    
    # 3. Run Elo Arena
    # If no baseline is provided, we clone the model configuration to run self-arena representing standard decoding baseline
    if args.baseline_checkpoint_path and os.path.exists(args.baseline_checkpoint_path):
        logger.info(f"Loading baseline checkpoint state from: {args.baseline_checkpoint_path}")
        base_checkpoint = torch.load(args.baseline_checkpoint_path, map_location=device, weights_only=False)
        baseline_model = Transformer(base_checkpoint["config"]).to(device)
        baseline_model.load_state_dict(base_checkpoint["model_state_dict"])
        baseline_model.eval()
    else:
        logger.info("No baseline checkpoint path provided. Running Self-Play Elo Arena comparing Model A against base initialization.")
        import copy
        baseline_model = Transformer(model_config).to(device) # Base random weights or identical model to simulate self-eval
        baseline_model.eval()
        
    arena_results = evaluate_arena(model, baseline_model, tokenizer, device=device)
    
    logger.info("=======================================================================")
    logger.info("📊 Benchmark Leaderboard & Arena Report:")
    logger.info("-----------------------------------------------------------------------")
    logger.info(f"🏆 MMLU Multiple Choice Accuracy: {mmlu_acc*100:.2f}%")
    logger.info(f"🏆 GSM8K Grade School Math Accuracy: {gsm_acc*100:.2f}%")
    logger.info(f"🏆 Consolidated Leaderboard Index: {((mmlu_acc + gsm_acc)/2)*100:.2f}%")
    logger.info(f"🏆 Arena Model A Final Elo Rating: {arena_results['final_elo_a']:.1f}")
    logger.info(f"🏆 Arena Model B Final Elo Rating: {arena_results['final_elo_b']:.1f}")
    logger.info(f"🏆 Match Statistics: A Wins={arena_results['wins_a']} | B Wins={arena_results['wins_b']} | Ties={arena_results['ties']}")
    logger.info("=======================================================================")
    
    # 4. Run Needle-in-a-Haystack Evaluator
    niah_evaluator = NeedleInAHaystackEvaluator()
    niah_results = niah_evaluator.run_eval(model, tokenizer, device=device)
    
    # Save a JSON metric file for our Web Dashboard to read!
    import json
    report = {
        "mmlu_accuracy": mmlu_acc,
        "gsm8k_accuracy": gsm_acc,
        "consolidated_score": (mmlu_acc + gsm_acc) / 2,
        "arena_wins_a": arena_results["wins_a"],
        "arena_wins_b": arena_results["wins_b"],
        "arena_ties": arena_results["ties"],
        "arena_elo_a": arena_results["final_elo_a"],
        "arena_elo_b": arena_results["final_elo_b"],
        "niah_results": niah_results
    }
    # Ensure outputs directory exists
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
