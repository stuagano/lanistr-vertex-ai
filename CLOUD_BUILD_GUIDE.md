# LANISTR Cloud Build Guide

This guide shows you how to build and push the LANISTR Docker container using Google Cloud Build, which is faster and more reliable than building locally.

## Why Cloud Build?

1. **No Local Resources**: Builds happen on Google Cloud's powerful infrastructure
2. **Faster Builds**: High-CPU machines with more memory and disk space
3. **Consistent Environment**: Same build environment every time
4. **No Docker Installation**: No need to install Docker locally
5. **Automatic Pushing**: Container is automatically pushed to Container Registry

## Quick Start

### 1. Update Configuration

Edit `trigger_cloud_build.sh` and update the PROJECT_ID:

```bash
PROJECT_ID="mgm-digitalconcierge"  # Your actual project ID
```

### 2. Run the Cloud Build Script

```bash
./trigger_cloud_build.sh
```

The script will:
- ✅ Check prerequisites (gcloud, authentication)
- ✅ Enable required APIs
- ✅ Trigger Cloud Build
- ✅ Stream build logs to your terminal
- ✅ Provide build status and container URI

## Manual Steps (Alternative)

If you prefer manual control:

### 1. Enable Required APIs

```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### 2. Trigger Cloud Build

```bash
gcloud builds submit --config=cloudbuild.yaml .
```

### 3. Monitor Build Progress

```bash
# List recent builds
gcloud builds list --limit=5

# Get build details
gcloud builds describe BUILD_ID

# Stream logs
gcloud builds log BUILD_ID
```

## Cloud Build Configuration

The `cloudbuild.yaml` file defines the build process:

### Build Steps

1. **Build Docker Image**: Creates the LANISTR container
2. **Push to Registry**: Uploads to Google Container Registry

### Build Options

- **Machine Type**: `E2_HIGHCPU_8` (8 vCPUs, 8GB RAM)
- **Disk Size**: 100GB (for PyTorch dependencies)
- **Timeout**: 1 hour total, 30 minutes per step

### Output Images

- `gcr.io/PROJECT_ID/lanistr:latest` - Latest version
- `gcr.io/PROJECT_ID/lanistr:COMMIT_SHA` - Versioned by commit

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
gcr.io/mgm-digitalconcierge/lanistr:latest
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
                "image_uri": "gcr.io/mgm-digitalconcierge/lanistr:latest",
                "args": [
                    "python", "lanistr/main_vertex_ai.py",
                    "--config", "gs://your-bucket/config.yaml"
                ]
            }
        }]
    }
}
```

## Cost Optimization

### Cloud Build Pricing
- **Build Time**: $0.003 per minute (first 120 minutes free per day)
- **Storage**: $0.026 per GB per month (Container Registry)

### Tips
1. **Use specific tags** instead of `latest`
2. **Clean up old images** regularly
3. **Optimize Dockerfile** to reduce build time
4. **Use build caching** for faster rebuilds

## Troubleshooting

### Common Issues

1. **Build fails with timeout**
   ```bash
   # Increase timeout in cloudbuild.yaml
   timeout: '7200s'  # 2 hours
   ```

2. **Out of disk space**
   ```bash
   # Increase disk size in cloudbuild.yaml
   options:
     diskSizeGb: '200'
   ```

3. **Permission denied**
   ```bash
   # Check IAM roles
   gcloud projects get-iam-policy PROJECT_ID
   # Need: Cloud Build Service Account, Storage Admin
   ```

4. **API not enabled**
   ```bash
   # Enable required APIs
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

### Build Optimization

1. **Use multi-stage builds** (if needed):
   ```dockerfile
   # Build stage
   FROM python:3.8 as builder
   COPY requirements*.txt ./
   RUN pip install --user -r requirements.txt
   
   # Runtime stage
   FROM nvidia/cuda:11.3-cudnn8-devel-ubuntu20.04
   COPY --from=builder /root/.local /root/.local
   ```

2. **Optimize layer caching**:
   ```dockerfile
   # Copy requirements first (changes less often)
   COPY requirements*.txt ./
   RUN pip install -r requirements.txt
   
   # Copy source code last (changes more often)
   COPY . .
   ```

## Monitoring and Debugging

### View Build History
```bash
# List recent builds
gcloud builds list --limit=10

# Get build details
gcloud builds describe BUILD_ID

# View build logs
gcloud builds log BUILD_ID
```

### Cloud Console
- **Build History**: https://console.cloud.google.com/cloud-build/builds
- **Container Registry**: https://console.cloud.google.com/gcr/images
- **Build Logs**: Real-time streaming in terminal

## Next Steps

After Cloud Build completes:

1. **Verify the container**:
   ```bash
   gcloud container images list-tags gcr.io/PROJECT_ID/lanistr
   ```

2. **Update your notebook** with the container URI:
   ```python
   container_uri = "gcr.io/mgm-digitalconcierge/lanistr:latest"
   ```

3. **Submit training jobs** using the notebook functions

4. **Monitor training** in Vertex AI console

## Support

If you encounter issues:

1. Check the [Cloud Build documentation](https://cloud.google.com/build/docs)
2. Review the [Container Registry documentation](https://cloud.google.com/container-registry/docs)
3. Check build logs for specific error messages
4. Verify IAM permissions and API enablement

---

**Happy cloud building! ☁️🚀** 