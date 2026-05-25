# ==============================================================================
# nano-llm: High-Performance 8xH800 Deep Learning GPU Docker Container
# ==============================================================================

# Base deep learning image pre-packaged with PyTorch, CUDA Toolkit, and cuDNN
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for high-performance DDP & NCCL communication
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"

# Install critical system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set standard working directory
WORKDIR /workspace

# Copy and install python dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI Dashboard port
EXPOSE 8000

# Default command to launch the autopilot control panel server
CMD ["python3", "web/server.py"]
