# Use NVIDIA CUDA base image with PyTorch
FROM nvidia/cuda:11.3-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY setup.py .
COPY lanistr/ ./lanistr/

# Install PyTorch with CUDA 11.3
RUN pip3 install --no-cache-dir \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
RUN pip3 install --no-cache-dir \
    omegaconf==2.3.0 \
    transformers==4.26.0 \
    torchmetrics==0.9.3 \
    pytz==2021.3 \
    pandas==1.3.5 \
    scikit-learn==1.3.2 \
    google-cloud-storage \
    google-cloud-aiplatform

# Install LANISTR in development mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/output_dir

# Set the entrypoint
ENTRYPOINT ["python3"] 