# Use Ubuntu base image (compatible with Cloud Build)
FROM ubuntu:20.04

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

# Install minimal requirements for cloud deployment
RUN pip3 install --no-cache-dir \
    torch>=2.0.0,<3.0.0 \
    torchvision>=0.15.0,<1.0.0 \
    torchaudio>=2.0.0,<3.0.0 \
    transformers==4.26.0 \
    omegaconf==2.3.0 \
    google-cloud-storage>=2.0.0,<3.0.0 \
    google-cloud-aiplatform>=1.25.0,<2.0.0 \
    google-auth>=2.0.0,<3.0.0 \
    tqdm>=4.62.0,<5.0.0 \
    numpy>=1.21.0,<2.0.0 \
    pandas>=1.3.5,<2.0.0

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