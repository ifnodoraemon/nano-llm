"""GRPO reward evaluation, answer extraction, and adaptive KL tuning."""

import re
import math
import logging
import torch
import torch.nn.functional as F
from utils.sandbox_executor import SandboxCodeExecutor

logger = logging.getLogger(__name__)


def extract_answer(completion_text: str) -> str:
    """Extract answer from <answer></answer> tags in completion text."""
    match = re.search(r"<answer>(.*?)</answer>", completion_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


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

    rewards = []
    sandbox = SandboxCodeExecutor(timeout=2.0)
    
    for prompt, completion, gt in zip(prompts, completions, ground_truths):
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
                        
            # F. Visual Multimodal Reasoning Reward
            if any(w in prompt.lower() for w in ["image", "chart", "graph", "screenshot", "picture", "figure", "look"]):
                visual_keywords = ["image", "pixels", "coordinate", "plot", "axis", "color", "shape", "visual", "figure", "box", "center", "left", "right", "top", "bottom", "label"]
                hits = sum(1 for w in visual_keywords if w in completion.lower())
                if hits >= 2:
                    reward += 0.5  # Reward visual descriptor utilization
                        
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
