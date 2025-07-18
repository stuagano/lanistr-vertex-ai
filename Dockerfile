# Use NVIDIA CUDA base image with PyTorch
FROM nvidia/cuda:11.3-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements files
COPY requirements_vertex_ai.txt .
COPY requirements-prod.txt .
COPY requirements-dev.txt .

# Install PyTorch with CUDA 11.3
RUN pip3 install --no-cache-dir \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install production requirements
RUN pip3 install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY lanistr/ ./lanistr/
COPY setup.py .

# Install LANISTR in development mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/output_dir /workspace/logs /workspace/checkpoints

# Create a non-root user for security
RUN useradd -m -s /bin/bash lanistr && \
    chown -R lanistr:lanistr /workspace

# Switch to non-root user
USER lanistr

# Set the entrypoint for Vertex AI
ENTRYPOINT ["python3", "lanistr/main_vertex_ai.py"] 