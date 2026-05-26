import unittest
import torch
import torch.nn as nn
from utils.triton_kernels import triton_rms_norm, triton_swiglu, triton_mla_flash_attn

class TestTritonKernels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_triton_rms_norm_fallback(self):
        x = torch.randn(2, 4, 8, device=self.device, requires_grad=True)
        weight = torch.ones(8, device=self.device, requires_grad=True)
        
        # Test forward pass
        out = triton_rms_norm(x, weight, eps=1e-5)
        self.assertEqual(out.shape, x.shape)
        
        # Test backward pass
        out.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)

    def test_triton_swiglu_fallback(self):
        gate = torch.randn(2, 4, 8, device=self.device)
        up = torch.randn(2, 4, 8, device=self.device)
        
        out = triton_swiglu(gate, up)
        self.assertEqual(out.shape, gate.shape)

    def test_triton_mla_flash_attn_fallback(self):
        q = torch.randn(1, 2, 8, 16, device=self.device)
        k = torch.randn(1, 2, 8, 16, device=self.device)
        v = torch.randn(1, 2, 8, 16, device=self.device)
        
        # Expected outputs shape transposed matching: [Batch, Seq_len, Num_heads, Head_dim]
        # Transposed in from [Batch, Num_heads, Seq_len, Head_dim]
        q_transposed = q.transpose(1, 2)
        k_transposed = k.transpose(1, 2)
        v_transposed = v.transpose(1, 2)
        
        out = triton_mla_flash_attn(q_transposed, k_transposed, v_transposed, scale=0.25, attn_scale_multiplier=1.0)
        self.assertEqual(out.shape, (1, 8, 2, 16))

if __name__ == "__main__":
    unittest.main()
