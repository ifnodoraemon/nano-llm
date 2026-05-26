import unittest
import torch
import torch.nn as nn
from utils.parallel_3d import ZeRO3Sharder
from utils.tensor_parallel import ColumnParallelLinear, RowParallelLinear, init_tp_process_group
from utils.pipeline_parallel import PipelineStage, OneFOneBScheduler, init_pp_process_group
from utils.expert_parallel import ExpertParallelRouter, init_ep_process_group

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
        self.assertEqual(gathered_weight.view(-1)[0].item(), 0.0)
        
        # 3. Scatter/release parameter
        sharder.scatter_parameter("fc.weight", model.fc.weight)
        self.assertEqual(model.fc.weight.data.numel(), 16)

    def test_tensor_parallel_mock(self):
        # Test TP group initialization in mock mode
        init_tp_process_group(tp_size=1)
        
        # ColumnParallelLinear
        col_layer = ColumnParallelLinear(in_features=8, out_features=16, bias=False)
        self.assertEqual(col_layer.local_out_features, 16)
        
        # RowParallelLinear
        row_layer = RowParallelLinear(in_features=16, out_features=4, bias=False)
        self.assertEqual(row_layer.local_in_features, 16)
        
        x = torch.randn(2, 8)
        y = col_layer(x)
        self.assertEqual(y.shape, (2, 16))
        
        z = row_layer(y)
        self.assertEqual(z.shape, (2, 4))
        
        # Backward pass check
        loss = z.sum()
        loss.backward()
        self.assertIsNotNone(col_layer.weight.grad)
        self.assertIsNotNone(row_layer.weight.grad)

    def test_pipeline_parallel_1f1b_mock(self):
        init_pp_process_group(pp_size=1)
        
        layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
        embedding = nn.Embedding(10, 8)
        head = nn.Linear(8, 2)
        
        stage = PipelineStage(layers, embedding=embedding, head=head)
        scheduler = OneFOneBScheduler(stage, num_microbatches=2, d_model=8)
        
        # Mock inputs and targets
        micro_batches = [torch.randint(0, 10, (2, 4)) for _ in range(2)]
        targets = [torch.randint(0, 2, (2, 4)) for _ in range(2)]
        
        def loss_fn(pred, target):
            return F.cross_entropy(pred.transpose(1, 2), target)
            
        import torch.nn.functional as F
        losses = scheduler.run_1f1b(
            micro_batches=micro_batches,
            targets=targets,
            loss_fn=loss_fn,
            device=torch.device("cpu")
        )
        
        self.assertEqual(len(losses), 2)
        self.assertGreater(losses[0].item(), 0.0)

    def test_expert_parallel_router_mock(self):
        init_ep_process_group(ep_size=1)
        
        router = ExpertParallelRouter(num_experts=4)
        self.assertEqual(router.experts_per_rank, 4)
        
        experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        
        tokens = torch.randn(6, 8)
        gate_weights = torch.ones(6, 2) * 0.5
        expert_indices = torch.tensor([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 2],
            [1, 3]
        ])
        
        out = router(tokens, gate_weights, expert_indices, experts)
        self.assertEqual(out.shape, (6, 8))

if __name__ == "__main__":
    unittest.main()
