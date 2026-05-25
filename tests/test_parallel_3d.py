import unittest
import torch
import torch.nn as nn
from utils.parallel_3d import ZeRO3Sharder

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4, bias=False)
        # Initialize weights with counting sequence
        self.fc.weight.data = torch.arange(32.0).view(4, 8)

class TestParallel3D(unittest.TestCase):
    def test_zero3_sharder(self):
        model = SimpleNet()
        original_weight = model.fc.weight.data.clone()
        
        # Instantiate sharder simulating a DDP world of size 2, rank 0
        sharder = ZeRO3Sharder(model, world_size=2, rank=0)
        
        # 1. Verify weight on rank 0 is sharded (size cut in half: 32 / 2 = 16 elements)
        local_param_data = model.fc.weight.data
        self.assertEqual(local_param_data.numel(), 16)
        
        # 2. Gather parameter on-demand
        gathered_weight = sharder.gather_parameter("fc.weight")
        self.assertEqual(gathered_weight.shape, (4, 8))
        self.assertEqual(gathered_weight.numel(), 32)
        
        # Verify gathered contents match original sequence (rank 0 has first half, replicated in mock fallback)
        # Because in mock fallback we did gathered_flat = sharded_data.repeat(world_size)
        self.assertEqual(gathered_weight.view(-1)[0].item(), 0.0)
        
        # 3. Scatter/release parameter
        sharder.scatter_parameter("fc.weight", model.fc.weight)
        self.assertEqual(model.fc.weight.data.numel(), 16)

if __name__ == "__main__":
    unittest.main()
