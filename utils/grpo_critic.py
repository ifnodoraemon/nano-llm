import os
import json
import logging
import re
from typing import List, Dict, Any
import torch

from utils.self_instruct import ExternalAPIClient
from grpo import evaluate_completion_rewards

logger = logging.getLogger(__name__)

# ==============================================================================
# GRPO Hybrid Critic: Combining Rule verifiers & LLM-as-a-Judge Rewards
# ==============================================================================

class GRPOCritic:
    """
    GRPOCritic combines strict, high-efficiency rule-based verifiers (for math/code accuracy)
    with a model-based LLM-as-a-Judge critic (via an external OpenAI-compatible API)
    to reward open-ended reasoning quality, conversational safety, and style adherence.
    """
    def __init__(self, api_key: str = "", base_url: str = "https://api.openai.com/v1", model: str = "gpt-4-turbo"):
        self.client = ExternalAPIClient(api_key=api_key, base_url=base_url)
        self.model = model
        
        # Balance weights between rule-based correctness and model-based style/reasoning
        self.rule_weight = 0.6
        self.critic_weight = 0.4
        
        logger.info(f"Initialized GRPOCritic targeting model: {self.model}")

    def evaluate_hybrid_rewards(
        self, 
        prompts: List[str], 
        completions: List[str], 
        ground_truths: List[str]
    ) -> torch.Tensor:
        """
        Computes blended hybrid rewards:
        Blended Reward = Rule_Weight * Rule_Reward + Critic_Weight * Model_Reward
        """
        group_size = len(prompts)
        
        # 1. Compute strict rule-based format/math rewards (zero VRAM overhead)
        # rule_rewards shape: [Group_size]
        rule_rewards = evaluate_completion_rewards(prompts, completions, ground_truths)
        
        # 2. Query external Judge Critic API to evaluate subjective reasoning depth
        critic_rewards = []
        for i in range(group_size):
            p = prompts[i]
            c = completions[i]
            
            # extract thinking block
            think_match = re.search(r"<think>(.*?)</think>", c, re.DOTALL)
            think_content = think_match.group(1).strip() if think_match else "None"
            
            system = (
                "You are an expert RLHF Reward Model Critic. Analyze the following user prompt, "
                "the model's internal thinking process, and the final response. Score the reasoning depth, "
                "logical flow, helpfulness, and safety. "
                "Output exactly a JSON: {\"reasoning_score\": float (0.0-1.0), \"helpfulness_score\": float (0.0-1.0), \"total_score\": float (0.0-5.0)}"
            )
            user = (
                f"Prompt: {p}\n"
                f"Thinking Block: {think_content}\n"
                f"Full Output: {c}\n"
                f"Output your JSON evaluation."
            )
            
            # Handle mock offline fallback
            if self.client.api_key == "MOCK_KEY":
                # Simulated high-quality judge score
                score = 3.5 if len(think_content) > 30 else 1.0
                critic_rewards.append(score)
                continue
                
            try:
                raw_response = self.client.query_completion(system, user, model=self.model)
                match = re.search(r"({.*})", raw_response, re.DOTALL)
                if match:
                    raw_response = match.group(1)
                data = json.loads(raw_response)
                critic_rewards.append(data.get("total_score", 0.0))
            except Exception as e:
                logger.error(f"GRPOCritic query failed for sample {i+1}: {e}. Fallback to simulated score.")
                critic_rewards.append(2.0)
                
        critic_rewards_tensor = torch.tensor(critic_rewards, dtype=torch.float32)
        
        # 3. Blend rule-based and model-based rewards
        hybrid_rewards = self.rule_weight * rule_rewards + self.critic_weight * critic_rewards_tensor
        
        logger.info(f"Hybrid rewards computed | Rules Avg: {rule_rewards.mean().item():.2f} | Critic Avg: {critic_rewards_tensor.mean().item():.2f}")
        return hybrid_rewards
