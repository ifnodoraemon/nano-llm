import os
import tempfile
import shutil
import unittest
import torch
import torch.nn as nn
from utils.stability import TrainingTelemetry

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

class TestStability(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, "checkpoint.pt")
        
        # Instantiate simple model and save it
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {}
        }, self.checkpoint_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_stability_telemetry(self):
        telemetry = TrainingTelemetry(
            ema_alpha=0.9,
            spike_threshold=3.0,
            loss_spike_factor=1.5,
            min_steps_before_checking=3
        )
        
        # 1. Update with regular steady steps to establish EMA
        for _ in range(5):
            telemetry.update(loss=1.0, grad_norm=0.5)
            
        # 2. Assert no anomalies on normal steady metrics
        res = telemetry.check_anomaly(loss=1.0, grad_norm=0.5)
        self.assertFalse(res["is_anomaly"])
        
        # 3. Simulate gradient spike anomaly
        res = telemetry.check_anomaly(loss=1.0, grad_norm=5.0)
        self.assertTrue(res["is_anomaly"])
        self.assertIn("Gradient norm spike", res["reason"])
        
        # 4. Simulate NaN anomaly
        res = telemetry.check_anomaly(loss=float("nan"), grad_norm=0.5)
        self.assertTrue(res["is_anomaly"])
        self.assertIn("NaN or Inf loss", res["reason"])

    def test_rollback_execution(self):
        telemetry = TrainingTelemetry(
            ema_alpha=0.9,
            spike_threshold=3.0,
            loss_spike_factor=1.5,
            min_steps_before_checking=3
        )
        
        # Establish stable telemetry
        for _ in range(5):
            telemetry.update(loss=1.0, grad_norm=0.5)
            
        # Corrupt weights to check if they restore from checkpoint
        with torch.no_grad():
            self.model.fc.weight.fill_(999.0)
            
        # Trigger an anomaly checking step
        res = telemetry.check_and_rollback(
            model=self.model,
            optimizer=self.optimizer,
            loss=10.0,  # Sudden severe loss spike
            grad_norm=0.5,
            checkpoint_path=self.checkpoint_path,
            current_lr=1e-3,
            lr_decay_factor=0.5
        )
        
        self.assertTrue(res["rolled_back"])
        self.assertTrue(res["rollback_success"])
        self.assertEqual(res["new_lr"], 5e-4)
        
        # Verify weights rolled back successfully (no longer 999.0)
        self.assertNotEqual(self.model.fc.weight[0][0].item(), 999.0)

if __name__ == "__main__":
    unittest.main()
