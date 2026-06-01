import unittest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import ModelConfig, Transformer
from grpo import (
    evaluate_completion_rewards,
    generate_completions,
    compute_action_logprobs,
    extract_answer
)

class TestGRPOMathAndFlow(unittest.TestCase):
    def setUp(self):
        # Set up a small mock model configuration
        self.config = ModelConfig(
            n_embd=128,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            vocab_size=1000,
            block_size=128,
            use_mla=True,
            use_moe=False
        )
        self.model = Transformer(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_answer_extraction(self):
        """
        Verify that our tag extractor retrieves values inside <answer>...</answer> tags.
        """
        text_with_answer = "Thinking process here... <think>let's add 2 and 3</think> and <answer>5</answer> is the output."
        self.assertEqual(extract_answer(text_with_answer), "5")
        
        text_no_answer = "Just some text without tags."
        self.assertEqual(extract_answer(text_no_answer), "")

    def test_reward_evaluation(self):
        """
        Ensure rewards are computed accurately according to DeepSeek-R1 rule verifiers.
        """
        prompts = ["Compute 2+3"] * 3
        completions = [
            "<think>Adding 2 and 3 yields 5</think> <answer>5</answer>",  # Perfect format + Perfect answer -> Score = 3.0
            "<think>Let's guess</think> <answer>6</answer>",            # Perfect format + Wrong answer -> Score = 1.0
            "Just number 5"                                           # No tags -> Score = 0.0
        ]
        ground_truths = ["5"] * 3
        
        rewards = evaluate_completion_rewards(prompts, completions, ground_truths)
        self.assertEqual(rewards.shape, (3,))
        self.assertGreater(rewards[0].item(), rewards[1].item())
        self.assertGreater(rewards[1].item(), rewards[2].item())

    def test_generation_and_logprobs(self):
        """
        Tests multi-step autoregressive completion generation and action logprobs calculations.
        """
        batch_size = 2
        prompt_len = 8
        max_gen_len = 16
        
        # Mock input prompt IDs (e.g. padded with 0s)
        prompt_ids = torch.randint(1, self.config.vocab_size, (batch_size, prompt_len), device=self.device)
        
        # Generation rollout
        full_seqs, gen_mask = generate_completions(
            self.model,
            prompt_ids,
            max_gen_len=max_gen_len,
            temperature=0.8,
            top_p=0.9
        )
        
        # Verify shapes
        self.assertEqual(full_seqs.shape, (batch_size, prompt_len + max_gen_len))
        self.assertEqual(gen_mask.shape, (batch_size, prompt_len + max_gen_len))
        self.assertTrue(torch.all(gen_mask[:, :prompt_len] == 0.0))
        self.assertTrue(torch.all(gen_mask[:, prompt_len:] == 1.0))
        
        # Compute forward pass for action log probabilities
        logits, _, _ = self.model(full_seqs)
        logprobs = compute_action_logprobs(logits, full_seqs, gen_mask)
        
        # Logprob per prompt element
        self.assertEqual(logprobs.shape, (batch_size,))
        self.assertTrue(torch.all(torch.isfinite(logprobs)))
        
        # Test backpropagation to ensure gradients flow cleanly
        loss = -logprobs.mean()
        loss.backward()
        
        # Verify gradients populated in trainable parameters
        has_grads = any(p.grad is not None for p in self.model.parameters() if p.requires_grad)
        self.assertTrue(has_grads)

if __name__ == "__main__":
    unittest.main()
