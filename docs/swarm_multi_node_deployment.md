# Docker Swarm Multi-Node GPU cluster Deployment Blueprint

```mermaid
graph TB
    subgraph Legend
        SwarmNode[Swarm / Docker Orchestration]:::swarm
        NetworkNode[Network Interface / NCCL]:::net
        ComputeNode[Multi-GPU CUDA Host]:::cuda
        classDef swarm fill:#ffaa00,stroke:#8c5d00,stroke-width:1.5px,color:#000;
        classDef net fill:#b927fc,stroke:#4a0072,stroke-width:2px,color:#fff;
        classDef cuda fill:#42e695,stroke:#005c1e,stroke-width:1.5px,color:#000;
    end
```

---

## 👁️ 1. Multi-Node distributed Topology (Swarm Manager vs. Workers)

```mermaid
graph TD
    classDef manager fill:#ffaa00,stroke:#000,stroke-width:2px,color:#000;
    classDef worker fill:#e2e2e2,stroke:#000,stroke-width:1.5px,color:#000;
    classDef net fill:#b927fc,stroke:#4a0072,stroke-width:2px,color:#fff;

    %% Nodes
    Master["Master Node 0 (Swarm Manager) <br> IP: 192.168.1.100 <br> 8x H800 (Ranks 0-7)"]:::manager
    Worker1["Worker Node 1 (Swarm Worker) <br> IP: 192.168.1.101 <br> 8x H800 (Ranks 8-15)"]:::worker
    
    %% Communication
    Master -->|1. Swarm Overlay Network Control| Worker1
    
    %% NCCL Data Flow
    Master <-->|2. High-Speed InfiniBand / RoCEv2 <br> (Host Network Mode bypasses virtualization!)| NetPipe["NCCL Multi-Node P2P Ring"]:::net
    Worker1 <-->|2. High-Speed InfiniBand / RoCEv2| NetPipe
```

---

## 🚀 2. Step-by-Step Deployment Protocol

```mermaid
sequenceDiagram
    autonumber
    actor Admin as Cluster Administrator
    participant M as Master Node 0 (Manager)
    participant W as Worker Node 1
    participant Swarm as Docker Swarm Stack
    
    %% Setup Swarm
    Admin->>M: run: docker swarm init --advertise-addr 192.168.1.100
    M-->>Admin: Output Swarm Join Token
    Admin->>W: run: docker swarm join --token <TOKEN> 192.168.1.100:2377
    W-->>M: Join cluster successfully!
    
    %% Node labels
    Admin->>M: run: docker node update --label-add type=worker1 <node-id-1>
    M-->>Admin: Node label updated!
    
    %% Build image
    Admin->>M: run: docker compose -f docker-compose-swarm.yml build
    Admin->>M: run: docker compose -f docker-compose-swarm.yml push
    
    %% Stack Deploy
    Admin->>M: run: docker stack deploy -c docker-compose-swarm.yml nano-llm-cluster
    M->>Swarm: Spawn Master container on Manager & Worker container on Node 1
    Swarm->>Swarm: Initialize multi-node torchrun handshakes (Ranks 0-15)
    Swarm-->>Admin: NCCL distributed training begins successfully!
```

---

## ⚙️ 3. Critical Network & VRAM Configurations

```mermaid
graph TD
    classDef config fill:#f3f3f3,stroke:#333,stroke-width:1px,color:#000;
    classDef risk fill:#ff4d4d,stroke:#990000,stroke-width:1.5px,color:#fff;
    classDef safe fill:#42e695,stroke:#005c1e,stroke-width:2px,color:#000;

    Configuration["Multi-Node Swarm Configuration Parameters"]:::config
    
    %% Network Mode
    Configuration --> NetMode["network_mode: host"]:::config
    NetMode -->|Why?| Bypass["Bypasses Docker virtual network overlays, maps containers directly to host NICs"]:::safe
    Bypass -->|Result| NCCL_Max["NCCL saturates raw 100G/200G InfiniBand/RoCEv2 bandwidth!"]:::safe
    
    %% Shared Memory
    Configuration --> SharedMem["ipc: host / volumes: /dev/shm"]:::config
    SharedMem -->|Why?| Prevents["Prevents multi-GPU inter-process shared memory bus blocks"]:::safe
    Prevents -->|Result| NoNCCLTimeout["Zero NCCL timeout errors during FSDP sharding!"]:::safe
    
    %% GPU Device reservations
    Configuration --> GPUReserve["deploy.resources.reservations.devices"]:::config
    GPUReserve -->|Why?| Pass["Direct Nvidia GPU pass-through in Swarm Mode"]:::safe
    Pass -->|Result| GPUSaturate["Allocates all 8 local GPU cores natively to container"]:::safe
```
