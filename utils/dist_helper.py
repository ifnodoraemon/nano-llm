import os
import logging

logger = logging.getLogger(__name__)

def autotune_nccl():
    """
    Dynamically autotunes PyTorch Distributed NCCL environment variables based on 
    active hardware topology, network interfaces, and RoCE/InfiniBand support.
    """
    logger.info("📡 Running Topology-Aware NCCL Autotuner...")
    
    # 1. Detect active net interfaces in /sys/class/net
    interfaces = []
    try:
        if os.path.exists("/sys/class/net"):
            interfaces = os.listdir("/sys/class/net")
    except Exception:
        pass
        
    active_interfaces = []
    for iface in interfaces:
        try:
            with open(f"/sys/class/net/{iface}/operstate", "r") as f:
                if f.read().strip() == "up":
                    active_interfaces.append(iface)
        except Exception:
            active_interfaces.append(iface)
            
    has_ib = any("ib" in iface for iface in active_interfaces)
    has_roce = any("roce" in iface or "mlx" in iface for iface in active_interfaces)
    eth_interfaces = [iface for iface in active_interfaces if iface.startswith(("eth", "en", "wl", "bond"))]
    
    # Autotuned environment variables mapping
    tuned_vars = {
        "NCCL_DEBUG": "WARN",
        "NCCL_DEBUG_SUBSYS": "INIT",
        "NCCL_BUFFSIZE": "4194304", # Set 4MB socket cache buffer to saturate high bandwidth links
        "OMP_NUM_THREADS": "8"
    }
    
    # Enable RoCE / InfiniBand Direct Memory Access if present
    if has_ib or has_roce:
        tuned_vars["NCCL_IB_DISABLE"] = "0"
        tuned_vars["NCCL_IB_CUDA_SUPPORT"] = "1"
        tuned_vars["NCCL_NET_GDR_LEVEL"] = "5" # Enable GPUDirect RDMA Level 5 (PCIe direct pass-through)
        
        # Pick MLC/IB interfaces
        ib_iface = next((iface for iface in interfaces if "ib" in iface or "mlx" in iface), None)
        if ib_iface:
            tuned_vars["NCCL_IB_HCA"] = ib_iface
        
        # GID Index 3 is widely standard for RoCE v2 routed packet headers
        if has_roce:
            tuned_vars["NCCL_IB_GID_INDEX"] = "3"
            logger.info("🔥 High-Performance RoCE (v2) network interfaces detected!")
        else:
            logger.info("🔥 Dedicated InfiniBand (IB) link interfaces detected!")
    else:
        tuned_vars["NCCL_IB_DISABLE"] = "1" # Disable IB to force fallback on standard sockets
        logger.info("🌐 Falling back on TCP Sockets (No hardware RoCE/IB detected).")
        
    # Configure interface binding
    if eth_interfaces:
        # e.g. "eth0,enp3s0"
        tuned_vars["NCCL_SOCKET_IFNAME"] = ",".join(eth_interfaces)
        
    # Inject variables safely into environment
    applied_vars = []
    for var_name, value in tuned_vars.items():
        if var_name not in os.environ:
            os.environ[var_name] = value
            applied_vars.append(f"{var_name}={value}")
            
    # Visual premium confirmation panel
    if applied_vars:
        border = "=" * 60
        logger.info(border)
        logger.info("🚀 TOPOLOGY-AWARE NCCL CONFIGURATIONS INJECTED SUCCESSFULLY:")
        for av in applied_vars:
            logger.info(f"  🔹 {av}")
        logger.info(border)
    else:
        logger.info("✅ Active NCCL environment variables already customized by user. Skipping overrides.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    autotune_nccl()
