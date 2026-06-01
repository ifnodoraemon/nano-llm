import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple

class MCTSNode:
    """
    MCTS Node representing a reasoning step / state in the search tree.
    """
    def __init__(self, step_text: str, token_ids: torch.Tensor, parent: Optional['MCTSNode'] = None):
        self.step_text = step_text
        self.token_ids = token_ids  # Tensor of shape [1, seq_len] representing this node's generated tokens
        self.parent = parent
        self.children: List['MCTSNode'] = []
        
        # MCTS Statistics
        self.visit_count = 0
        self.total_value = 0.0  # Cumulative Q-value

    @property
    def q_value(self) -> float:
        """Returns the mean value of this node (Q)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def get_full_sequence(self) -> torch.Tensor:
        """Trace back to the root to compile the full token sequence."""
        nodes = []
        curr = self
        while curr is not None:
            nodes.insert(0, curr.token_ids)
            curr = curr.parent
        device = nodes[0].device
        return torch.cat([n.to(device) for n in nodes], dim=-1)

    def is_terminal(self) -> bool:
        """Returns True if the node represents a final answer (contains </answer>)."""
        # Simply check if the token_ids contain a terminal sequence or if step_text suggests completion
        return "</answer>" in self.step_text or "final answer" in self.step_text.lower()


class MCTSEngine:
    """
    Monte Carlo Tree Search (MCTS) Engine for LLM reasoning.
    Leverages UCB1 to scaling compute at test-time (Test-time compute scaling).
    """
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer: Any,
        exploration_constant: float = 1.414,
        max_depth: int = 8,
        num_simulations: int = 16
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.device = next(model.parameters()).device

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        UCB1 Selection: Selects the child node that maximizes the Upper Confidence Bound.
        UCB1 = Q/N + c * sqrt(ln(N_parent) / N)
        """
        best_node = None
        best_ucb = -float('inf')
        
        for child in node.children:
            if child.visit_count == 0:
                # Always explore unvisited nodes first (infinity bound)
                return child
                
            ucb = child.q_value + self.exploration_constant * math.sqrt(
                math.log(node.visit_count) / child.visit_count
            )
            if ucb > best_ucb:
                best_ucb = ucb
                best_node = child
                
        return best_node if best_node is not None else node

    def expand(self, node: MCTSNode, num_branches: int = 3) -> List[MCTSNode]:
        """
        Node Expansion: Generates next reasoning step options (branches).
        """
        if node.is_terminal():
            return []
            
        self.model.eval()
        seq = node.get_full_sequence()
        
        # Generate several reasoning path candidates (micro-step rollout)
        for _ in range(num_branches):
            branch_tokens = self._generate_step_chunk(seq, max_tokens=32)
            branch_text = self.tokenizer.decode(branch_tokens[0].tolist(), skip_special_tokens=True)
            
            child = MCTSNode(
                step_text=branch_text,
                token_ids=branch_tokens,
                parent=node
            )
            node.children.append(child)
            
        return node.children

    def evaluate(self, node: MCTSNode, ground_truth: str = "") -> float:
        """
        Evaluation: Estimates the quality of the current reasoning state.
        Uses process-supervised PRM heuristics or rule-based matching.
        """
        full_text = self.tokenizer.decode(node.get_full_sequence()[0].tolist(), skip_special_tokens=True)
        score = 0.0
        
        # Heuristics mimicking Process-Supervised Reward Models (PRMs)
        if "<think>" in full_text:
            score += 0.5
        if "</think>" in full_text:
            score += 0.5
            
        # Parse output answer
        if "</answer>" in full_text:
            score += 1.0
            
        # Reward accuracy against ground truth if present
        if ground_truth:
            # Simple substring matching
            if ground_truth in full_text:
                score += 2.0
            else:
                score -= 0.5
                
        # Repetition penalty
        if "the the the" in full_text or "so so so" in full_text:
            score -= 1.0
            
        return score

    def backpropagate(self, node: Optional[MCTSNode], value: float):
        """
        Backpropagation: Propagates the evaluation score back to the root of the tree.
        """
        curr = node
        while curr is not None:
            curr.visit_count += 1
            curr.total_value += value
            curr = curr.parent

    @torch.no_grad()
    def search(self, prompt: str, ground_truth: str = "") -> str:
        """
        Executes the Monte Carlo Tree Search loop.
        Returns the optimal generated response sequence.
        """
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        root = MCTSNode(step_text="", token_ids=prompt_ids)
        
        for _ in range(self.num_simulations):
            # 1. Selection
            curr = root
            depth = 0
            while curr.children and depth < self.max_depth:
                curr = self.select(curr)
                depth += 1
                
            # 2. Expansion
            if not curr.is_terminal() and depth < self.max_depth:
                children = self.expand(curr)
                if children:
                    curr = children[0]  # Dive into the first new expansion node
                    
            # 3. Evaluation
            value = self.evaluate(curr, ground_truth)
            
            # 4. Backpropagation
            self.backpropagate(curr, value)
            
        # Compile the best path
        best_path_node = root
        while best_path_node.children:
            # Pick the child with the highest visit count (robust MCTS choice)
            best_path_node = max(best_path_node.children, key=lambda node: node.visit_count)
            
        full_tokens = best_path_node.get_full_sequence()
        return self.tokenizer.decode(full_tokens[0].tolist(), skip_special_tokens=True)

    def _generate_step_chunk(self, seq: torch.Tensor, max_tokens: int = 32) -> torch.Tensor:
        """Generates a small reasoning chunk autoregressively."""
        generated = []
        device = next(self.model.parameters()).device
        curr_seq = seq.clone().to(device)
        
        for _ in range(max_tokens):
            logits, _, _ = self.model(curr_seq)
            next_token_logits = logits[:, -1, :] / 0.8
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token)
            curr_seq = torch.cat([curr_seq, next_token], dim=-1)
            
            # Stop if we hit a newline (end of reasoning step) or terminal token
            if next_token.item() == self.tokenizer.eos_token_id or next_token.item() == 13: # 13 is often '\n'
                break
                
        return torch.cat(generated, dim=-1).cpu()
