import unittest
import sys
import os
from utils.system_monitor import SystemMonitor

class TestSystemMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = SystemMonitor()

    def test_cpu_utilization(self):
        # First call sets initial counters
        self.monitor.get_cpu_utilization()
        # Sleep briefly or call again to compute delta
        import time
        time.sleep(0.01)
        cpu_val = self.monitor.get_cpu_utilization()
        self.assertIsInstance(cpu_val, float)
        self.assertTrue(0.0 <= cpu_val <= 100.0)

    def test_ram_utilization(self):
        ram_val = self.monitor.get_ram_utilization()
        self.assertIsInstance(ram_val, float)
        self.assertTrue(0.0 <= ram_val <= 100.0)

    def test_network_bandwidth(self):
        # First call sets initial counters
        self.monitor.get_network_bandwidth()
        import time
        time.sleep(0.01)
        rx, tx = self.monitor.get_network_bandwidth()
        self.assertIsInstance(rx, float)
        self.assertIsInstance(tx, float)
        self.assertTrue(rx >= 0.0)
        self.assertTrue(tx >= 0.0)

    def test_gpu_telemetry(self):
        gpu_data = self.monitor.get_gpu_telemetry()
        self.assertIsInstance(gpu_data, dict)
        self.assertIn("gpu_util", gpu_data)
        self.assertIn("vram_used", gpu_data)
        self.assertIn("vram_total", gpu_data)
        self.assertIn("gpu_count", gpu_data)
        self.assertIn("name", gpu_data)
        
        self.assertTrue(0.0 <= gpu_data["gpu_util"] <= 100.0)
        self.assertTrue(gpu_data["vram_used"] >= 0.0)
        self.assertTrue(gpu_data["vram_total"] >= 0.0)
        self.assertTrue(gpu_data["gpu_count"] >= 0)

    def test_telemetry_report_and_formatted(self):
        report = self.monitor.get_telemetry_report()
        self.assertIsInstance(report, dict)
        self.assertIn("cpu", report)
        self.assertIn("ram", report)
        self.assertIn("gpu_util", report)
        self.assertIn("vram_used", report)
        self.assertIn("net_rx", report)
        self.assertIn("net_tx", report)

        formatted_str = self.monitor.get_formatted_telemetry()
        self.assertIsInstance(formatted_str, str)
        self.assertTrue(len(formatted_str) > 0)
        self.assertIn("CPU:", formatted_str)
        self.assertIn("RAM:", formatted_str)
        self.assertIn("GPU:", formatted_str)
        self.assertIn("VRAM:", formatted_str)

    def test_print_dashboard(self):
        # Just ensure printing dashboard runs to completion without raising exceptions
        try:
            self.monitor.print_dashboard()
            success = True
        except Exception as e:
            success = False
            print(f"Failed to print dashboard: {e}")
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()
