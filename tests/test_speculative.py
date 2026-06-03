import unittest
import os
import torch
from unittest.mock import patch, MagicMock
from serve.speculative import generate_with_speculative_decoding
from model import ModelConfig, Transformer
from train_tokenizer import CustomBPETokenizer

class TestSpeculativeDecoding(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Setup mock tokenizer rules
        self.tokenizer = CustomBPETokenizer()
        self.tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        self.tokenizer.vocab[256] = b"<|im_start|>"
        self.tokenizer.vocab[257] = b"<|im_end|>"
        self.tokenizer.vocab_size = 258
        self.tokenizer.eos_token_id = 257
        self.tokenizer.bos_token_id = 256
        self.tokenizer.special_tokens = {
            "<|im_start|>": 256,
            "<|im_end|>": 257,
        }
        
    def test_speculative_decoding_mla(self):
        """
        Verify speculative decoding runs correctly when MLA is active (latent caching).
        """
        config = ModelConfig(
            n_embd=64,
            n_layer=4,
            n_head=2,
            block_size=128,
            vocab_size=258,
            use_mla=True,
            use_moe=False
        )
        model = Transformer(config).to(self.device)
        model.eval()
        
        with patch('builtins.print'):
            output = generate_with_speculative_decoding(
                model=model,
                tokenizer=self.tokenizer,
                prompt="Translate deep learning to Chinese.",
                max_new_tokens=5,
                device=str(self.device)
            )
        self.assertIsInstance(output, str)

    def test_speculative_decoding_standard(self):
        """
        Verify speculative decoding runs correctly when MLA is disabled (standard caching).
        """
        config = ModelConfig(
            n_embd=64,
            n_layer=4,
            n_head=2,
            block_size=128,
            vocab_size=258,
            use_mla=False,
            use_moe=False
        )
        model = Transformer(config).to(self.device)
        model.eval()
        
        with patch('builtins.print'):
            output = generate_with_speculative_decoding(
                model=model,
                tokenizer=self.tokenizer,
                prompt="Translate deep learning to Chinese.",
                max_new_tokens=5,
                device=str(self.device)
            )
        self.assertIsInstance(output, str)

if __name__ == "__main__":
    unittest.main()
