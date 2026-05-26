import unittest
import torch
import torch.nn as nn
from utils.mcts_engine import MCTSNode, MCTSEngine
from model import ModelConfig, Transformer

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 50256

    def encode(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        # Simple dummy encoding
        return torch.zeros(1, 4, dtype=torch.long)

    def decode(self, token_ids: list, skip_special_tokens: bool = True) -> str:
        # Mock decode output
        return "This is a <think>step by step thinking</think> <answer>42</answer>"

class TestMCTSEngine(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = ModelConfig(
            block_size=128,
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            n_embd=32,
            vision_dim=None
        )
        self.model = Transformer(self.config).to(self.device)
        self.tokenizer = MockTokenizer()

    def test_mcts_node_statistics(self):
        # Create a node
        token_ids = torch.zeros(1, 4, dtype=torch.long)
        node = MCTSNode(step_text="Initial step", token_ids=token_ids)
        
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.q_value, 0.0)
        self.assertFalse(node.is_terminal())
        
        # Test backpropagation
        parent = MCTSNode(step_text="Root", token_ids=token_ids)
        child = MCTSNode(step_text="Step 1", token_ids=token_ids, parent=parent)
        parent.children.append(child)
        
        engine = MCTSEngine(self.model, self.tokenizer)
        engine.backpropagate(child, value=2.0)
        
        self.assertEqual(child.visit_count, 1)
        self.assertEqual(child.total_value, 2.0)
        self.assertEqual(child.q_value, 2.0)
        
        self.assertEqual(parent.visit_count, 1)
        self.assertEqual(parent.total_value, 2.0)

    def test_mcts_selection_and_expansion(self):
        engine = MCTSEngine(self.model, self.tokenizer)
        root = MCTSNode(step_text="Root", token_ids=torch.zeros(1, 4, dtype=torch.long))
        
        # Expand
        children = engine.expand(root, num_branches=2)
        self.assertEqual(len(children), 2)
        self.assertEqual(len(root.children), 2)
        
        # Select unvisited first
        selected = engine.select(root)
        self.assertEqual(selected.visit_count, 0)

    def test_mcts_evaluation(self):
        engine = MCTSEngine(self.model, self.tokenizer)
        node = MCTSNode(step_text="This is a <think>thinking step</think> <answer>42</answer>", token_ids=torch.zeros(1, 4, dtype=torch.long))
        
        # Evaluate against correct ground truth
        score = engine.evaluate(node, ground_truth="42")
        self.assertGreater(score, 1.0) # Should have format + truth rewards

if __name__ == "__main__":
    unittest.main()
