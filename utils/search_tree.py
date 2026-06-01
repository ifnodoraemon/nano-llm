import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from model import Transformer
from grpo import extract_answer

# ==============================================================================
# reasoning Search Tree & Backtracking Decoder (OpenAI o1 / o3 Style)
# ==============================================================================

class ReasoningNode:
    """
    Represents a single node (reasoning step) in the Search Tree.
    """
    def __init__(self, step_text: str, token_ids: torch.Tensor, score: float = 0.0, parent: 'ReasoningNode' = None):
        self.step_text = step_text          # Generated text of the step
        self.token_ids = token_ids          # Token IDs of the step
        self.score = score                  # Evaluation score of this path segment
        self.parent = parent
        self.children: List[ReasoningNode] = []

    def get_full_sequence(self) -> torch.Tensor:
        """Trace back to the root to compile the full token sequence."""
        nodes = []
        curr = self
        while curr is not None:
            nodes.insert(0, curr.token_ids)
            curr = curr.parent
        return torch.cat(nodes, dim=-1)


class SearchTreeDecoder:
    """
    Search Tree & Backtracking Decoder.
    Directs the Transformer model to generate multiple reasoning branches (rollouts),
    scores them dynamically using rule-based/model-based reward critics, and
    backtracks or self-corrects to select the optimal path (DeepSeek-R1 / o1-style).
    """
    def __init__(
        self, 
        model: Transformer, 
        tokenizer,
        max_branches: int = 3, 
        max_steps: int = 4
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_branches = max_branches
        self.max_steps = max_steps
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def search_optimal_path(
        self, 
        prompt_ids: torch.Tensor, 
        ground_truth: str = "",
        temperature: float = 0.8
    ) -> str:
        """
        Runs Monte Carlo Tree Search / Backtracking decoding over reasoning paths.
        Returns the optimal generated completion string.
        """
        self.model.eval()
        root = ReasoningNode("", prompt_ids, score=0.0)
        
        active_leaves = [root]
        
        for step in range(self.max_steps):
            next_leaves = []
            
            for leaf in active_leaves:
                current_seq = leaf.get_full_sequence()
                
                # Generate multiple alternative reasoning branches (Rollout)
                for b in range(self.max_branches):
                    # Generate a short reasoning chunk (e.g. 32 tokens)
                    branch_tokens = self._generate_step_chunk(current_seq, chunk_len=32, temp=temperature)
                    branch_text = self.tokenizer.decode(branch_tokens[0], skip_special_tokens=True)
                    
                    # Score current segment
                    score = self._evaluate_segment_score(branch_text, ground_truth)
                    
                    # Create child node
                    child = ReasoningNode(branch_text, branch_tokens, score=score, parent=leaf)
                    leaf.children.append(child)
                    next_leaves.append(child)
            
            # Backtracking: Prune paths and keep only the top leaves (Beam search over tree)
            # Sort leaves by score and keep best branches
            next_leaves.sort(key=lambda node: node.score, reverse=True)
            active_leaves = next_leaves[:self.max_branches]
            
            # Early stopping: if a leaf contains </answer> tag (reached final answer)
            complete_nodes = [node for node in active_leaves if "</answer>" in self.tokenizer.decode(node.get_full_sequence()[0], skip_special_tokens=True)]
            if complete_nodes:
                best_complete = max(complete_nodes, key=lambda node: node.score)
                return self.tokenizer.decode(best_complete.get_full_sequence()[0], skip_special_tokens=True)
                
        # Return best overall leaf
        best_leaf = max(active_leaves, key=lambda node: node.score)
        return self.tokenizer.decode(best_leaf.get_full_sequence()[0], skip_special_tokens=True)

    def _generate_step_chunk(self, seq: torch.Tensor, chunk_len: int = 32, temp: float = 0.8) -> torch.Tensor:
        """Generates a small chunk of tokens autoregressively."""
        batch_size = seq.size(0)
        generated = []
        
        curr_seq = seq.clone()
        for _ in range(chunk_len):
            logits, _, _ = self.model(curr_seq)
            next_logits = logits[:, -1, :] / max(temp, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token)
            curr_seq = torch.cat([curr_seq, next_token], dim=-1)
            
        return torch.cat(generated, dim=-1)

    def _evaluate_segment_score(self, segment_text: str, ground_truth: str) -> float:
        """
        Dynamically grades the segment quality.
        - Boosts score if reasoning format `<think>` is initiated.
        - Heavily rewards correct answer inside `<answer>` matches.
        - Penalizes nonsense repetitions.
        """
        score = 0.0
        
        # Format checks
        if "<think>" in segment_text:
            score += 0.5
        if "</think>" in segment_text:
            score += 0.5
            
        # Semantic checks
        extracted = extract_answer(segment_text)
        if ground_truth and extracted:
            if extracted == ground_truth:
                score += 3.0  # Big hit!
            elif extracted in ground_truth or ground_truth in extracted:
                score += 0.8
                
        # Verbosity guard (avoid infinite loops)
        if "the the the" in segment_text or "and and and" in segment_text:
            score -= 2.0
            
        return score
