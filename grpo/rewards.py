"""GRPO reward evaluation, answer extraction, and adaptive KL tuning."""

import re
import math
import logging
import json
import asyncio
import aiohttp
import torch
import torch.nn.functional as F
from utils.sandbox_executor import SandboxCodeExecutor

logger = logging.getLogger(__name__)

async def async_evaluate_single_completion(session, prompt, completion, judge_url="http://localhost:11434/v1/chat/completions"):
    system_prompt = (
        "You are an objective judge assessing the quality of assistant responses. "
        "Review the prompt and the response. Provide an integer score between 1 and 5 "
        "where 5 is exceptionally correct and clear, and 1 is incorrect or completely unhelpful. "
        "You must return your output strictly in JSON format as follows: {\"score\": <int>}"
    )
    user_content = f"### Prompt:\n{prompt}\n\n### Assistant Response:\n{completion}"
    
    payload = {
        "model": "pump-assistant:latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.0,
        "max_tokens": 80,
        "response_format": {"type": "json_object"}
    }
    
    try:
        async with session.post(judge_url, json=payload, timeout=30.0) as response:
            if response.status == 200:
                res_data = await response.json()
                content = res_data["choices"][0]["message"]["content"]
                
                # Robust regex extraction of score from JSON format (handles reasoning chains output before JSON)
                score_match = re.search(r'"score"\s*:\s*(\d)', content)
                if score_match:
                    return float(score_match.group(1)) / 5.0
                    
                # Direct JSON load fallback
                score_data = json.loads(content)
                return float(score_data.get("score", 3)) / 5.0
            else:
                try:
                    status_text = await response.text()
                except Exception:
                    status_text = "Could not retrieve error text"
                logger.warning(f"Judge API call returned status {response.status}: {status_text}")
    except Exception as e:
        logger.warning(f"Judge API call failed: {type(e).__name__} - {e}")
    return 0.6  # Default fallback neutral score

async def get_judge_rewards_async(prompts, completions):
    async with aiohttp.ClientSession() as session:
        tasks = [async_evaluate_single_completion(session, p, c) for p, c in zip(prompts, completions)]
        return await asyncio.gather(*tasks)

def get_judge_rewards(prompts, completions):
    """Synchronous wrapper for integration with GRPO step calculations."""
    try:
        return torch.tensor(asyncio.run(get_judge_rewards_async(prompts, completions)), dtype=torch.float32)
    except Exception as e:
        logger.error(f"Failed to get LLM judge rewards: {e}")
        return torch.full((len(completions),), 0.6, dtype=torch.float32)


def extract_answer(completion_text: str) -> str:
    """Extract answer from <answer></answer> tags in completion text."""
    match = re.search(r"<answer>(.*?)</answer>", completion_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ==============================================================================
# Process-Supervised Step Rewards (PRM) & Budget-Constrained Penalties (§3.4)
# ==============================================================================

# Known tool names for logical continuity checking. Tool calls whose name
# appears in this set (or is a reasonable snake_case identifier) receive a
# bonus; otherwise they are treated as plausible but unverified.
_KNOWN_TOOL_NAMES = {
    "get_weather", "create_event", "web_search", "summarize", "read_file",
    "write_file", "calculator", "translate", "database_query", "send_email",
    "generate_image", "search_flights", "book_flight", "execute_code",
    "get_stock_price", "product_search", "fetch_emails", "run_analysis",
    "search_hotels",
}

# Regex for extracting tool_call blocks (fenced code blocks or XML tags)
_TOOL_CALL_FENCED_RE = re.compile(r'```tool_call\s*\n(.*?)```', re.DOTALL)
_TOOL_CALL_XML_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)


def _extract_tool_call_blocks(completion_text: str) -> list[str]:
    """Return raw JSON strings from all tool_call blocks in the completion."""
    blocks = _TOOL_CALL_FENCED_RE.findall(completion_text)
    blocks += _TOOL_CALL_XML_RE.findall(completion_text)
    return [b.strip() for b in blocks if b.strip()]


def compute_step_rewards(completion_text: str, gamma: float = 0.95) -> float:
    """
    Process-supervised step reward scorer for tool_call blocks.

    For each tool_call block found in *completion_text*, assigns:
      +0.3  if the JSON is parseable and has required 'name' + 'arguments' fields
      +0.2  if the tool name looks logically reasonable (known name or valid identifier)
      -0.5  if the JSON is malformed / unparseable

    All step rewards are discounted by gamma^t where t is the 0-indexed step.

    Args:
        completion_text: The full completion string from the model.
        gamma: Discount factor applied per step (default 0.95).

    Returns:
        Total discounted step reward (float).
    """
    blocks = _extract_tool_call_blocks(completion_text)
    if not blocks:
        return 0.0

    total_reward = 0.0
    for t, block in enumerate(blocks):
        discount = gamma ** t
        step_reward = 0.0

        try:
            data = json.loads(block)
        except (json.JSONDecodeError, TypeError):
            # Malformed JSON: penalise this step
            step_reward = -0.5
            total_reward += discount * step_reward
            continue

        # Check required fields
        has_name = "name" in data or "tool_name" in data
        has_args = "arguments" in data or "params" in data or "parameters" in data
        if has_name and has_args:
            step_reward += 0.3  # Valid format reward
        else:
            step_reward -= 0.5  # Missing required structure
            total_reward += discount * step_reward
            continue

        # Logical continuity: check tool name reasonableness
        tool_name = str(data.get("name", data.get("tool_name", ""))).lower().strip()
        if tool_name in _KNOWN_TOOL_NAMES:
            step_reward += 0.2  # Known tool
        elif re.match(r'^[a-z][a-z0-9_]{1,40}$', tool_name):
            step_reward += 0.1  # Plausible snake_case identifier (partial credit)

        total_reward += discount * step_reward

    return total_reward


def compute_budget_penalty(
    completion_text: str,
    max_steps: int = 5,
    max_tokens: int = 1500,
) -> float:
    """
    Budget-constrained penalty for tool_call usage.

    Penalties:
      -3.0  if the number of tool_call invocations exceeds *max_steps*
      -1.5  if the character-estimated token count exceeds *max_tokens*
      -5.0  if an infinite loop is detected (3+ consecutive near-identical calls)
       0.0  otherwise

    Only the **first** triggered penalty is returned (most severe first).

    Args:
        completion_text: The full completion string.
        max_steps: Maximum allowed tool_call invocations.
        max_tokens: Maximum allowed token-equivalent length.

    Returns:
        Penalty value (float, <= 0).
    """
    blocks = _extract_tool_call_blocks(completion_text)
    num_calls = len(blocks)

    # Check infinite loop: 3+ consecutive near-identical tool calls
    if num_calls >= 3:
        for i in range(num_calls - 2):
            b1, b2, b3 = blocks[i], blocks[i + 1], blocks[i + 2]
            # Near-identical: strip whitespace and compare
            if _blocks_near_identical(b1, b2) and _blocks_near_identical(b2, b3):
                return -5.0

    # Check step count budget
    if num_calls > max_steps:
        return -3.0

    # Check token budget (approximate: 1 token ≈ 4 chars for English text)
    estimated_tokens = len(completion_text) // 4
    if estimated_tokens > max_tokens:
        return -1.5

    return 0.0


def _blocks_near_identical(a: str, b: str) -> bool:
    """
    Returns True if two tool_call block strings are near-identical.
    Uses JSON-level structural comparison: same tool name AND same argument
    keys AND same argument values. Falls back to whitespace-normalized exact
    match for non-JSON blocks.

    判断两个 tool_call 块是否近似相同.
    使用 JSON 级结构比较: 相同工具名 + 相同参数键 + 相同参数值.
    """
    a_clean = re.sub(r'\s+', '', a)
    b_clean = re.sub(r'\s+', '', b)
    if a_clean == b_clean:
        return True
    # Try JSON-level comparison (JSON 级比较)
    try:
        da = json.loads(a.strip())
        db = json.loads(b.strip())
        name_a = str(da.get("name", da.get("tool_name", ""))).lower()
        name_b = str(db.get("name", db.get("tool_name", ""))).lower()
        if not name_a and not name_b:
            return False  # Neither has tool name — not a valid tool_call comparison
        args_a = da.get("arguments", da.get("params", {}))
        args_b = db.get("arguments", db.get("params", {}))
        if isinstance(args_a, str):
            args_a = json.loads(args_a)
        if isinstance(args_b, str):
            args_b = json.loads(args_b)
        return name_a == name_b and args_a == args_b
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return False


def evaluate_completion_rewards(*args, **kwargs):
    """
    Calculates rewards for a group of completions.
    Supports both:
    1. (prompts, completions, ground_truths)
    2. (completions, ground_truths, tokenizer=None, use_sandbox=False)
    
    Format:
    1. Format Reward: checks if output contains <think>...</think> and <answer>...</answer>
    2. Math Reward: checks if the extracted answer matches the ground truth
    3. Code Sandbox Reward: extracts and executes generated python code in a sandbox,
       adding rewards for successful compilation and correct logic, or penalizing errors.
    4. Length Regularization: penalizes infinite repetitive generations
    5. Process-Supervised Reward Model (PRM) & Reward Clipping: scores intermediate
       reasoning steps and penalizes repetitive loops (reward hacking), clamping final scores.
    """
    prompts = None
    completions = None
    ground_truths = None
    
    if "completions" in kwargs:
        completions = kwargs["completions"]
    if "ground_truths" in kwargs:
        ground_truths = kwargs["ground_truths"]
    if "prompts" in kwargs:
        prompts = kwargs["prompts"]

    if completions is None or ground_truths is None:
        if len(args) >= 3:
            prompts, completions, ground_truths = args[:3]
        elif len(args) == 2:
            completions, ground_truths = args[:2]
            prompts = [""] * len(completions)
        else:
            raise TypeError("evaluate_completion_rewards expects at least 2 or 3 positional arguments.")

    if prompts is None:
        prompts = [""] * len(completions)

    # Fetch LLM Judge scores concurrently in one async batch
    judge_scores = get_judge_rewards(prompts, completions)
    
    rewards = []
    sandbox = SandboxCodeExecutor(timeout=2.0)
    
    for idx, (prompt, completion, gt) in enumerate(zip(prompts, completions, ground_truths)):
        reward = 0.0
        
        # A. Format Reward: Reward correct thinking and answer tags
        has_think = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        
        if has_think and has_answer:
            reward += 1.0  # Proper reasoning structure
        elif has_think or has_answer:
            reward += 0.3  # Partial formatting
            
        # B. Math Accuracy Reward: Check if answer matches ground truth
        extracted = extract_answer(completion)
        if gt and extracted:
            if extracted == gt:
                reward += 2.0  # Correct answer match
            elif extracted in gt or gt in extracted:
                reward += 0.5  # Substring partial match
                
        # C. Code Sandbox Reward: If completion contains code, execute and verify it
        extracted_code = sandbox.extract_code(completion)
        if extracted_code:
            # Execute in sandbox
            res = sandbox.execute_and_verify(extracted_code)
            if res["success"]:
                reward += 1.0  # Code successfully compiled and exited 0!
                
                # Check if code output matches ground truth (for coding questions)
                stdout = res["stdout"].strip()
                if gt and stdout == gt.strip():
                    reward += 1.5  # Code executed and matched the ground truth answer!
                elif gt and (gt.strip() in stdout or stdout in gt.strip()):
                    reward += 0.5  # Substring match on execution output
            else:
                # Syntax error, runtime error, security exception, or timeout
                reward -= 0.5  # Code failed execution or violated security bounds
                
        # D. Repetitive/Length Penalty: Penalize excessively verbose or empty generations
        comp_len = len(completion)
        if comp_len > 1500 or comp_len < 10:
            reward -= 0.5
            
        # E. Process-Supervised Reward Model (PRM) Step-Scoring & Reward Hacking Prevention
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if think_match:
            think_text = think_match.group(1).strip()
            
            # Analyze reasoning steps and transitions (e.g. Step 1, Step 2, Therefore, So, etc.)
            steps = re.findall(r"(?:Step\s*\d+|therefore|so,|hence|thus|because|firstly|secondly|let's|wait|check)", think_text, re.IGNORECASE)
            num_steps = len(steps)
            if num_steps >= 3:
                reward += 0.5  # Strong step-by-step thinking transitions
            elif num_steps >= 1:
                reward += 0.2  # Simple reasoning indicators
                
            # Repetitive / Chaotic reasoning loop check
            words = think_text.lower().split()
            if len(words) > 10:
                unique_words = set(words)
                repetition_ratio = 1.0 - (len(unique_words) / len(words))
                if repetition_ratio > 0.4:
                    reward -= 0.8  # Penalize high repetition within thinking block (preventing repetitive loop hacking)
                    
                # Check for line-level repetitive patterns
                lines = [line.strip() for line in think_text.split('\n') if line.strip()]
                if len(lines) > 2:
                    consecutive_dups = sum(1 for idx in range(len(lines) - 1) if lines[idx] == lines[idx + 1])
                    if consecutive_dups >= 2:
                        reward -= 1.0  # Penalize duplicate line cycles

            # 4-gram repetition check
            if len(words) >= 4:
                grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
                unique_grams = set(grams)
                gram_rep_ratio = 1.0 - (len(unique_grams) / len(grams))
                if gram_rep_ratio > 0.30:
                    reward = -1.0  # Force reward to negative baseline for repetitive patterns

            # Thinking chain length constraints: 100~800 is normal; > 1500 or < 20 applies quadratic decay penalty
            think_tokens = len(words)
            if think_tokens < 20:
                reward -= 0.5 * ((20.0 - think_tokens) / 20.0) ** 2
            elif think_tokens > 1500:
                reward -= 1.5 * ((think_tokens - 1500.0) / 500.0) ** 2

            # Sentence loop check (3 consecutive sentences with similarity > 0.95)
            sentences = [s.strip() for s in re.split(r'[.!?。！？\n]', think_text) if s.strip()]
            if len(sentences) >= 3:
                for i in range(len(sentences) - 2):
                    s1, s2, s3 = sentences[i], sentences[i+1], sentences[i+2]
                    w1, w2, w3 = set(s1.lower().split()), set(s2.lower().split()), set(s3.lower().split())
                    if w1 and w2 and w3:
                        sim12 = len(w1.intersection(w2)) / len(w1.union(w2))
                        sim23 = len(w2.intersection(w3)) / len(w2.union(w3))
                        if sim12 > 0.95 and sim23 > 0.95:
                            reward = -1.5  # Force negative score for looping
                            break
                        
            # F. Visual Multimodal Reasoning Reward
            if any(w in prompt.lower() for w in ["image", "chart", "graph", "screenshot", "picture", "figure", "look"]):
                visual_keywords = ["image", "pixels", "coordinate", "plot", "axis", "color", "shape", "visual", "figure", "box", "center", "left", "right", "top", "bottom", "label"]
                hits = sum(1 for w in visual_keywords if w in completion.lower())
                if hits >= 2:
                    reward += 0.5  # Reward visual descriptor utilization
                        
        # G. Add LLM-as-a-Judge Reward (scale normalized [0.2, 1.0] to [0.4, 2.0])
        judge_score = judge_scores[idx].item()
        reward += judge_score * 2.0

        # H. Process-Supervised Step Rewards (PRM) for tool_call blocks (§3.4)
        step_reward = compute_step_rewards(completion)
        reward += step_reward

        # I. Budget-Constrained Penalty for tool_call usage (§3.4)
        budget_penalty = compute_budget_penalty(completion)
        reward += budget_penalty
        
        rewards.append(reward)
        
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    return torch.clamp(rewards_tensor, min=-3.0, max=3.0)


class GRPORewardScaler:
    """Normalizes rewards within a group for stable GRPO training."""

    def __init__(self, momentum: float = 0.9, eps: float = 1e-8):
        self.momentum = momentum
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update_and_scale(self, rewards: torch.Tensor) -> torch.Tensor:
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item()
        if torch.isnan(torch.tensor(batch_var)) or batch_var == 0:
            batch_var = 1.0
        
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
            self.var = self.momentum * self.var + (1.0 - self.momentum) * batch_var
            
        self.count += 1
        return (rewards - self.mean) / (math.sqrt(self.var) + self.eps)

    def normalize(self, rewards: list[float]) -> list[float]:
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean = rewards_t.mean()
        std = rewards_t.std()
        if std < self.eps:
            return [0.0] * len(rewards)
        normalized = (rewards_t - mean) / (std + self.eps)
        return normalized.tolist()


class AdaptiveKLTuner:
    """Dynamically scales the KL penalty coefficient beta based on target KL divergence bounds."""

    def __init__(self, target_kl: float = 0.2, beta_min: float = 0.005, beta_max: float = 0.5, adjustment_rate: float = 0.01):
        self.target_kl = target_kl
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.adjustment_rate = adjustment_rate
        self.current_beta = 0.04

    def update(self, observed_kl: float) -> float:
        if observed_kl > self.target_kl * 1.5:
            self.current_beta *= (1 + self.adjustment_rate)
        elif observed_kl < self.target_kl * 0.5:
            self.current_beta *= (1 - self.adjustment_rate)
        self.current_beta = max(self.beta_min, min(self.current_beta, self.beta_max))
        return self.current_beta

    def tune_beta(self, current_kl: float, current_beta: float) -> float:
        if current_kl > self.target_kl * 1.5:
            new_beta = min(self.beta_max, current_beta * 1.2)
        elif current_kl < self.target_kl * 0.5:
            new_beta = max(self.beta_min, current_beta / 1.2)
        else:
            new_beta = current_beta
        return new_beta
