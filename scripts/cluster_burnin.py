import os
import sys
import time
import json
import subprocess
import logging
import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_gpu_telemetry():
    """
    Calls nvidia-smi to query temperature, power draw, and memory usage of all GPUs.
    """
    try:
        cmd = "nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        gpu_metrics = []
        for line in output.split("\n"):
            parts = line.split(",")
            if len(parts) >= 6:
                gpu_metrics.append({
                    "index": int(parts[0].strip()),
                    "temp_c": float(parts[1].strip()),
                    "power_w": float(parts[2].strip()),
                    "mem_used_mb": float(parts[3].strip()),
                    "mem_total_mb": float(parts[4].strip()),
                    "util_pct": float(parts[5].strip())
                })
        return gpu_metrics
    except Exception as e:
        logger.warning(f"Could not fetch nvidia-smi telemetry: {e}")
        return []

def main():
    # 1. Initialize Distributed Environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        logger.info("Not running in DDP mode. Launching single-GPU simulation.")
        rank = 0
        world_size = 1
        local_rank = 0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    is_master = (rank == 0)
    
    if is_master:
        logger.info("==============================================================")
        logger.info("🔥 Starting NCCL Autotune Cluster Burn-in Stress Test")
        logger.info("==============================================================")
        logger.info(f"Nodes size: {world_size}")
        
    # Duration configuration (default 30 mins)
    duration_sec = 30 * 60
    if len(sys.argv) > 1:
        duration_sec = int(sys.argv[1])
        
    # 2. Allocate stress payload (large tensors to saturate NVLink/PCIe and GPU memory)
    # 128MB tensor for All-Reduce stress
    tensor_size = 32 * 1024 * 1024 # 32M float32 = 128MB
    if "cuda" in device:
        payload = torch.randn(tensor_size, device=device, dtype=torch.float32)
        # Allocate static cache to occupy VRAM (stress VRAM controller)
        # Occupy ~2GB VRAM
        vram_occupant = torch.randn(1024 * 1024 * 1024, device=device, dtype=torch.float16)
    else:
        payload = torch.randn(1000)
        vram_occupant = None
        
    start_time = time.time()
    step = 0
    log_interval = 100
    
    all_telemetry = []
    
    # 3. Stress Loop
    while time.time() - start_time < duration_sec:
        if "cuda" in device and world_size > 1:
            # NCCL All-Reduce call
            t0 = time.perf_counter()
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            # Calculate bandwidth (each GPU sends/receives tensor_size * 4 bytes)
            bytes_transferred = tensor_size * 4
            elapsed = t1 - t0
            # Algorithmic bandwidth formula for ring all-reduce: 2 * (N-1)/N * Size / Time
            bandwidth_gbps = (2 * (world_size - 1) / world_size) * (bytes_transferred / elapsed) / 1e9 * 8
        else:
            # Single GPU stress fallback (Matrix Multiplication)
            t0 = time.perf_counter()
            if "cuda" in device:
                _ = torch.matmul(payload[:1000].unsqueeze(1), payload[:1000].unsqueeze(0))
                torch.cuda.synchronize()
            else:
                _ = torch.matmul(payload[:100].unsqueeze(1), payload[:100].unsqueeze(0))
            t1 = time.perf_counter()
            elapsed = t1 - t0
            bandwidth_gbps = 0.0
            
        step += 1
        
        if step % log_interval == 0:
            elapsed_total = time.time() - start_time
            gpus = get_gpu_telemetry()
            
            step_info = {
                "step": step,
                "elapsed_total_sec": elapsed_total,
                "nccl_bandwidth_gbps": bandwidth_gbps,
                "gpu_telemetry": gpus
            }
            
            if is_master:
                all_telemetry.append(step_info)
                # Print logs
                logger.info(f"Progress: {elapsed_total:.1f}/{duration_sec}s | NCCL Bandwidth: {bandwidth_gbps:.2f} Gbps")
                for g in gpus:
                    logger.info(f"  GPU {g['index']} | Temp: {g['temp_c']}°C | Power: {g['power_w']}W | VRAM: {g['mem_used_mb']}/{g['mem_total_mb']}MB | Util: {g['util_pct']}%")
                    if g['temp_c'] > 85:
                        logger.warning(f"  ⚠️ HIGH TEMPERATURE WARNING ON GPU {g['index']}: {g['temp_c']}°C")
                        
    # 4. Generate report
    if is_master:
        report_path = "./outputs/cluster_burnin_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        max_temp = 0.0
        avg_bandwidth = 0.0
        bandwidths = [s["nccl_bandwidth_gbps"] for s in all_telemetry if s["nccl_bandwidth_gbps"] > 0]
        if bandwidths:
            avg_bandwidth = sum(bandwidths) / len(bandwidths)
            
        for s in all_telemetry:
            for g in s["gpu_telemetry"]:
                if g["temp_c"] > max_temp:
                    max_temp = g["temp_c"]
                    
        status = "HEALTHY"
        reasons = []
        if max_temp > 85:
            status = "WARNING_THROTTLING_RISK"
            reasons.append(f"Peak temperature reached {max_temp}°C (critical threshold is 85°C).")
        if avg_bandwidth > 0 and avg_bandwidth < 10.0:
            status = "DEGRADED_NETWORK"
            reasons.append(f"Average NCCL Ring bandwidth was low: {avg_bandwidth:.2f} Gbps.")
            
        summary_report = {
            "status": status,
            "duration_tested_sec": duration_sec,
            "avg_nccl_bandwidth_gbps": avg_bandwidth,
            "peak_gpu_temp_c": max_temp,
            "warnings": reasons,
            "telemetry_log": all_telemetry
        }
        
        with open(report_path, "w") as f:
            json.dump(summary_report, f, indent=2)
            
        logger.info("==============================================================")
        logger.info("🎉 NCCL Stress Test Completed successfully!")
        logger.info(f"📊 Summary Status: {status}")
        logger.info(f"📊 Report saved to {report_path}")
        logger.info("==============================================================")
        
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
