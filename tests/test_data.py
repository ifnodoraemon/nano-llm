import unittest
import os
import json
import torch
from data import ChatFormatter, SFTDataset, SequencePackingCollator, DPODataset, MultimodalSFTDataset, MultimodalSequenceCollator
from train_tokenizer import CustomBPETokenizer

class TestDataModules(unittest.TestCase):
    def setUp(self):
        self.tmp_sft_path = "./tests/tmp_sft.jsonl"
        self.tmp_dpo_path = "./tests/tmp_dpo.jsonl"
        self.tmp_tokenizer_path = "./tests/tmp_tokenizer.json"
        os.makedirs("./tests", exist_ok=True)
        
        # 1. Setup mock tokenizer rules
        self.tokenizer = CustomBPETokenizer()
        self.tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        # Add special token segments
        self.tokenizer.vocab[256] = b"<|im_start|>"
        self.tokenizer.vocab[257] = b"<|im_end|>"
        self.tokenizer.vocab_size = 258
        
        # 2. Write SFT mock conversations
        sft_data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]},
            {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
        ]
        with open(self.tmp_sft_path, "w", encoding="utf-8") as f:
            for item in sft_data:
                f.write(json.dumps(item) + "\n")
                
        # 3. Write SFT multimodal mock conversations
        multimodal_data = [
            {"messages": [{"role": "user", "content": "<image> Describe"}, {"role": "assistant", "content": "It is a cat"}], "image": "mock.jpg"},
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello"}]} # No image item
        ]
        self.tmp_multimodal_path = "./tests/tmp_multimodal.jsonl"
        with open(self.tmp_multimodal_path, "w", encoding="utf-8") as f:
            for item in multimodal_data:
                f.write(json.dumps(item) + "\n")

        # 4. Write DPO mock pairs
        dpo_data = [
            {
                "prompt": [{"role": "user", "content": "Write a python function"}],
                "chosen": {"role": "assistant", "content": "def f(): return 1"},
                "rejected": {"role": "assistant", "content": "def f() return 1"}
            }
        ]
        with open(self.tmp_dpo_path, "w", encoding="utf-8") as f:
            for item in dpo_data:
                f.write(json.dumps(item) + "\n")

    def tearDown(self):
        for p in [self.tmp_sft_path, self.tmp_dpo_path, self.tmp_tokenizer_path, self.tmp_multimodal_path]:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists("./tests"):
            try:
                os.rmdir("./tests")
            except OSError:
                pass

    def test_chat_formatter(self):
        formatter = ChatFormatter()
        msg = {"role": "user", "content": "Hello"}
        formatted = formatter.format_message(msg)
        self.assertEqual(formatted, "<|im_start|>user\nHello<|im_end|>\n")

    def test_sft_dataset_loading(self):
        dataset = SFTDataset(self.tmp_sft_path, self.tokenizer, max_length=128)
        self.assertEqual(len(dataset), 2)
        
        sample = dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("labels", sample)
        self.assertEqual(sample["input_ids"].ndim, 1)
        self.assertEqual(sample["labels"].ndim, 1)
        
        # User input parts of labels must be masked with -100
        user_tokens_len = len(self.tokenizer.encode("<|im_start|>user\nHello<|im_end|>\n", add_special_tokens=False))
        self.assertEqual(sample["labels"][:user_tokens_len].tolist(), [-100] * user_tokens_len)

    def test_sequence_packing_collator(self):
        dataset = SFTDataset(self.tmp_sft_path, self.tokenizer, max_length=128)
        collator = SequencePackingCollator(pad_token_id=0, max_length=64)
        
        batch = collator([dataset[0], dataset[1]])
        self.assertEqual(batch["input_ids"].shape, (1, 64)) # Packed into 1 sequence of length 64
        self.assertEqual(batch["labels"].shape, (1, 64))
        self.assertEqual(batch["position_ids"].shape, (1, 64))

    def test_dpo_dataset_loading(self):
        dataset = DPODataset(self.tmp_dpo_path, self.tokenizer, max_prompt_length=64, max_length=128)
        self.assertEqual(len(dataset), 1)
        
        sample = dataset[0]
        self.assertIn("chosen_input_ids", sample)
        self.assertIn("chosen_labels", sample)
        self.assertIn("rejected_input_ids", sample)
        self.assertIn("rejected_labels", sample)
        
        # Ensure correct prompt mask
        self.assertEqual(sample["chosen_labels"][:10].tolist(), [-100] * 10)

    def test_multimodal_sft_dataset_and_collator(self):
        # 1. Test Multimodal SFT Dataset Ingestion
        dataset = MultimodalSFTDataset(
            self.tmp_multimodal_path, 
            self.tokenizer, 
            max_length=128, 
            vision_dim=32, 
            num_patches=4
        )
        self.assertEqual(len(dataset), 2)
        
        sample_img = dataset[0] # Has image
        sample_no_img = dataset[1] # No image
        
        self.assertIsNotNone(sample_img["pixel_values"])
        self.assertEqual(sample_img["pixel_values"].shape, (4, 32))
        self.assertIsNone(sample_no_img["pixel_values"])
        
        # 2. Test Multimodal Sequence Collator
        collator = MultimodalSequenceCollator(pad_token_id=0, vision_dim=32, num_patches=4)
        batch = collator([sample_img, sample_no_img])
        
        self.assertIn("pixel_values", batch)
        # Batch size 2, 4 patches of dimension 32
        self.assertEqual(batch["pixel_values"].shape, (2, 4, 32))
        # First item has real random features, second item was padded with zeros
        self.assertEqual(batch["pixel_values"][1].sum().item(), 0.0)

if __name__ == "__main__":
    unittest.main()
