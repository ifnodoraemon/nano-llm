import unittest
from utils.evol_instruct import EvolInstructEngine

class TestEvolInstruct(unittest.TestCase):
    def setUp(self):
        self.engine = EvolInstructEngine(seed=1337)

    def test_individual_mutations(self):
        prompt = "Write a binary search algorithm."
        
        # Test constraint addition
        mutated_constraint = self.engine.mutate_add_constraint(prompt)
        self.assertIn("[Constraint Upgrade]:", mutated_constraint)
        self.assertTrue(mutated_constraint.startswith(prompt))
        
        # Test reasoning deepening
        mutated_reasoning = self.engine.mutate_deepen_reasoning(prompt)
        self.assertIn("[Reasoning Depth Upgrade]:", mutated_reasoning)
        
        # Test format concretization
        mutated_fmt = self.engine.mutate_concretize_format(prompt)
        self.assertIn("[Format Concretization]:", mutated_fmt)
        
        # Test context enrichment
        mutated_ctx = self.engine.mutate_in_context_enrichment(prompt)
        self.assertIn("[Context Enrichment]:", mutated_ctx)

    def test_random_mutation(self):
        prompt = "Explain quantum computing."
        mutated = self.engine.mutate(prompt)
        self.assertNotEqual(prompt, mutated)
        self.assertTrue(any(tag in mutated for tag in ["[Constraint Upgrade]", "[Reasoning Depth Upgrade]", "[Format Concretization]", "[Context Enrichment]"]))

    def test_batch_mutation(self):
        prompts = ["Question 1", "Question 2", "Question 3"]
        mutated_batch = self.engine.batch_mutate(prompts, mutation_rate=1.0)
        self.assertEqual(len(mutated_batch), 3)
        for i, original in enumerate(prompts):
            self.assertNotEqual(original, mutated_batch[i])

if __name__ == "__main__":
    unittest.main()
