# LANISTR Container Build and Push Guide

This guide shows you how to build and push the LANISTR Docker container to Google Container Registry (GCR) for use with Vertex AI.

## Prerequisites

1. **Docker**: Install Docker Desktop from [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
2. **Google Cloud SDK**: Install from [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
3. **Google Cloud Project**: You need a Google Cloud project with billing enabled

## Quick Start

### 1. Update Configuration

Edit `build_and_push_container.sh` and update the PROJECT_ID:

```bash
PROJECT_ID="your-actual-project-id"  # Replace with your actual project ID
```

### 2. Run the Build Script

```bash
./build_and_push_container.sh
```

The script will:
- ✅ Check prerequisites (Docker, gcloud, authentication)
- ✅ Enable required APIs
- ✅ Configure Docker for GCR
- ✅ Build the Docker image
- ✅ Push to Google Container Registry
- ✅ Verify the image

## Manual Steps (Alternative)

If you prefer to run steps manually:

### 1. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs

```bash
gcloud services enable containerregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### 3. Configure Docker for GCR

```bash
gcloud auth configure-docker
```

### 4. Build the Image

```bash
docker build -t gcr.io/YOUR_PROJECT_ID/lanistr:latest .
```

### 5. Push to GCR

```bash
docker push gcr.io/YOUR_PROJECT_ID/lanistr:latest
```

## Container Details

### Base Image
- **CUDA**: 11.3 with cuDNN8
- **OS**: Ubuntu 20.04
- **Python**: 3.8

### Key Features
- ✅ PyTorch 2.0+ with CUDA support
- ✅ All LANISTR dependencies pre-installed
- ✅ Vertex AI optimized entry point
- ✅ Non-root user for security
- ✅ Proper working directory setup

### Container Structure
```
/workspace/
├── lanistr/           # LANISTR source code
├── data/              # Data directory (mounted)
├── output_dir/        # Output directory (mounted)
├── logs/              # Logs directory
└── checkpoints/       # Checkpoints directory
```

## Using the Container

### Container URI Format
```
gcr.io/YOUR_PROJECT_ID/lanistr:latest
```

### Example Vertex AI Job Configuration
```python
job_config = {
    "display_name": "lanistr-training",
    "job_spec": {
        "worker_pool_specs": [{
            "machine_spec": {
                "machine_type": "n1-standard-8",
                "accelerator_type": "NVIDIA_TESLA_V100",
                "accelerator_count": 1
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "gcr.io/YOUR_PROJECT_ID/lanistr:latest",
                "args": [
                    "python", "lanistr/main_vertex_ai.py",
                    "--config", "gs://your-bucket/config.yaml"
                ]
            }
        }]
    }
}
```

## Troubleshooting

### Common Issues

1. **Docker not found**
   ```bash
   # Install Docker Desktop
   # macOS: https://docs.docker.com/desktop/install/mac-install/
   # Linux: https://docs.docker.com/engine/install/
   ```

2. **Authentication failed**
   ```bash
   gcloud auth login
   gcloud auth configure-docker
   ```

3. **Permission denied**
   ```bash
   # Make sure you have the necessary IAM roles:
   # - Storage Admin
   # - AI Platform Developer
   # - Container Registry Service Agent
   ```

4. **Build fails**
   ```bash
   # Check Dockerfile syntax
   docker build --no-cache -t test-image .
   ```

5. **Push fails**
   ```bash
   # Verify authentication
   gcloud auth list
   # Check project permissions
   gcloud projects describe YOUR_PROJECT_ID
   ```

### Build Optimization

For faster builds:

1. **Use BuildKit**:
   ```bash
   export DOCKER_BUILDKIT=1
   docker build -t gcr.io/YOUR_PROJECT_ID/lanistr:latest .
   ```

2. **Multi-stage builds** (if needed):
   ```dockerfile
   # Build stage
   FROM python:3.8 as builder
   COPY requirements*.txt ./
   RUN pip install --user -r requirements.txt
   
   # Runtime stage
   FROM nvidia/cuda:11.3-cudnn8-devel-ubuntu20.04
   COPY --from=builder /root/.local /root/.local
   ```

## Cost Optimization

### Container Registry Pricing
- **Storage**: $0.026 per GB per month
- **Network egress**: $0.12 per GB (after 1GB free)

### Tips
1. **Use specific tags** instead of `latest`
2. **Clean up old images** regularly
3. **Use multi-arch builds** if needed
4. **Optimize image size** by removing unnecessary files

## Next Steps

After building and pushing the container:

1. **Update your notebook** with the correct PROJECT_ID
2. **Upload your data** to Google Cloud Storage
3. **Submit training jobs** using the notebook
4. **Monitor progress** in the Vertex AI console

## Support

If you encounter issues:

1. Check the [Google Cloud documentation](https://cloud.google.com/ai-platform/docs)
2. Review the [Docker documentation](https://docs.docker.com/)
3. Check the [LANISTR repository](https://github.com/your-repo/lanistr) for updates

---

**Happy cloud training! ☁️🚀** 