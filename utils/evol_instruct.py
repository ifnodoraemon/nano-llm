import random
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class EvolInstructEngine:
    """
    Online Instruction Evolution & Mutation Engine (Evol-Instruct).
    Dynamically mutates seed prompts during training loops to climb instruction complexity gradients.
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
        # Prefabricated mutators representing state-of-the-art instruction evolution dimensions
        self.constraints = [
            "Ensure the output is strictly structured as JSON. Do not include conversational preambles.",
            "Avoid standard conditional branches or recursive loops in your solution if possible.",
            "Write the algorithm with O(1) auxiliary space complexity constraints.",
            "The final solution must not use any third-party external library imports."
        ]
        
        self.reasoning_demands = [
            "Analyze and elaborate on the exact computational time complexity (Big O notation) of this approach.",
            "Detail the hardware resource footprint trade-offs (CPU cache, VRAM allocations) of your design.",
            "Explain step-by-step how your proposed solution prevents race conditions and memory bus bottlenecks.",
            "Compare this approach comprehensively against three alternative algorithms, proving why this is superior."
        ]
        
        self.formats = [
            "You must follow a strict tagging schema: encapsulate all reasoning steps inside <think></think> and the final clean solution inside <answer></answer>.",
            "Format the entire output as a single, valid, raw markdown code block without trailing greetings.",
            "Present your steps as a rigorous, numbered logical derivation chain.",
            "Wrap key entities and math formulas in strictly matching LaTeX equations ($...$)."
        ]
        
        self.context_enrichments = [
            "Incorporate comprehensive validation checks for extreme inputs, null references, and edge cases.",
            "Design the implementation specifically to resist potential path traversal or prompt injection attacks.",
            "Provide detailed error-handling blocks with meaningful custom error messages for the caller.",
            "Frame the answer targeting highly restricted low-power embedded environments."
        ]

    def mutate_add_constraint(self, prompt: str) -> str:
        constraint = self.rng.choice(self.constraints)
        return f"{prompt}\n\n[Constraint Upgrade]: {constraint}"

    def mutate_deepen_reasoning(self, prompt: str) -> str:
        demand = self.rng.choice(self.reasoning_demands)
        return f"{prompt}\n\n[Reasoning Depth Upgrade]: {demand}"

    def mutate_concretize_format(self, prompt: str) -> str:
        fmt = self.rng.choice(self.formats)
        return f"{prompt}\n\n[Format Concretization]: {fmt}"

    def mutate_in_context_enrichment(self, prompt: str) -> str:
        enrichment = self.rng.choice(self.context_enrichments)
        return f"{prompt}\n\n[Context Enrichment]: {enrichment}"

    def mutate(self, prompt: str) -> str:
        """
        Randomly selects a mutation operator to evolve the input prompt.
        """
        mutators = [
            self.mutate_add_constraint,
            self.mutate_deepen_reasoning,
            self.mutate_concretize_format,
            self.mutate_in_context_enrichment
        ]
        chosen_mutator = self.rng.choice(mutators)
        evolved_prompt = chosen_mutator(prompt)
        logger.debug(f"Mutated prompt: '{prompt[:40]}...' -> '{evolved_prompt[:60]}...'")
        return evolved_prompt

    def batch_mutate(self, prompts: List[str], mutation_rate: float = 0.5) -> List[str]:
        """
        Applies mutations on a batch of prompts depending on the mutation rate.
        """
        results = []
        for prompt in prompts:
            if self.rng.random() < mutation_rate:
                results.append(self.mutate(prompt))
            else:
                results.append(prompt)
        return results
