import os
import shutil
import tempfile
import unittest
from utils.data_mixer import DynamicDataMixer

class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Extremely simple mock tokenizer returning token ids as length of word
        return [len(word) for word in text.split()]

class TestDataMixer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.source_a = os.path.join(self.temp_dir, "source_a")
        self.source_b = os.path.join(self.temp_dir, "source_b")
        
        os.makedirs(self.source_a)
        os.makedirs(self.source_b)
        
        # Write dummy txt files
        with open(os.path.join(self.source_a, "file1.txt"), "w", encoding="utf-8") as f:
            f.write("hello world hello world\n" * 10)
            
        with open(os.path.join(self.source_b, "file2.txt"), "w", encoding="utf-8") as f:
            f.write("code python compile debug\n" * 10)
            
        self.tokenizer = MockTokenizer()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dynamic_mixer(self):
        sources = {
            self.source_a: 0.6,
            self.source_b: 0.4
        }
        
        # Instantiate mixer
        mixer = DynamicDataMixer(
            sources=sources,
            tokenizer=self.tokenizer,
            block_size=8,
            buffer_size=10,
            seed=42
        )
        
        # Draw some mixed batches
        iterator = iter(mixer)
        batches = []
        for _ in range(5):
            batch = next(iterator)
            batches.append(batch)
            
        self.assertEqual(len(batches), 5)
        for batch in batches:
            self.assertIn("input_ids", batch)
            self.assertIn("labels", batch)
            self.assertEqual(batch["input_ids"].shape[0], 8)
            self.assertEqual(batch["labels"].shape[0], 8)
            
if __name__ == "__main__":
    unittest.main()
