# LANISTR Vertex AI Distributed Training Deployment Guide

This guide provides comprehensive instructions for deploying LANISTR on Google Cloud Vertex AI for distributed training with enterprise-grade features.

## Overview

LANISTR is a multimodal learning framework that can process language, image, and structured data. This deployment setup enables:

- **Distributed Training**: Multi-GPU and multi-node training
- **Enterprise Features**: Security, monitoring, logging, and compliance
- **Scalability**: Automatic scaling based on workload
- **Cost Optimization**: Spot instances and resource management
- **Production Ready**: Error handling, checkpointing, and recovery

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Cloud Vertex AI                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Node 1    │  │   Node 2    │  │   Node N    │         │
│  │ 8x V100 GPU │  │ 8x V100 GPU │  │ 8x V100 GPU │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│              Google Cloud Storage (Data)                   │
├─────────────────────────────────────────────────────────────┤
│              Google Cloud Logging & Monitoring             │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Google Cloud Setup

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Required APIs

The deployment script automatically enables these APIs:
- `aiplatform.googleapis.com` - Vertex AI
- `storage.googleapis.com` - Cloud Storage
- `logging.googleapis.com` - Cloud Logging
- `monitoring.googleapis.com` - Cloud Monitoring
- `errorreporting.googleapis.com` - Error Reporting
- `containerregistry.googleapis.com` - Container Registry

### 3. IAM Permissions

Ensure your account has these roles:
- `roles/aiplatform.user`
- `roles/storage.admin`
- `roles/logging.admin`
- `roles/monitoring.admin`

## Quick Start

### 1. One-Command Deployment

```bash
# Set your project ID
export PROJECT_ID=your-project-id

# Run the deployment script
./deploy_vertex_ai.sh
```

This script will:
- ✅ Check prerequisites
- ✅ Authenticate with Google Cloud
- ✅ Enable required APIs
- ✅ Create GCS bucket
- ✅ Upload data (if available)
- ✅ Build and push Docker image
- ✅ Create training scripts
- ✅ Setup monitoring

### 2. Manual Deployment

If you prefer manual deployment:

```bash
# 1. Build and deploy Docker image
./build_and_deploy.sh

# 2. Upload data to GCS
python3 upload_data_to_gcs.py \
    --local-data-dir ./lanistr/data \
    --bucket-name your-bucket-name \
    --project-id your-project-id \
    --dataset mimic \
    --create-bucket

# 3. Run training
python3 vertex_ai_setup.py \
    --project-id your-project-id \
    --job-name "lanistr-mimic-pretrain" \
    --config-file "lanistr/configs/mimic_pretrain.yaml" \
    --machine-type "n1-standard-4" \
    --accelerator-type "NVIDIA_TESLA_V100" \
    --accelerator-count 8 \
    --replica-count 1 \
    --base-output-dir "gs://your-bucket/lanistr-output" \
    --base-data-dir "gs://your-bucket/lanistr-data" \
    --image-uri "gcr.io/your-project/lanistr-training:latest"
```

## Training Scripts

After deployment, you'll have these training scripts:

### Single Node Training

```bash
# MIMIC-IV pretraining
./run_mimic_pretrain.sh

# Amazon pretraining
./run_amazon_pretrain.sh
```

### Multi-Node Training

```bash
# 2 nodes x 8 GPUs each
./run_multi_node_training.sh 2 8

# 4 nodes x 4 GPUs each
./run_multi_node_training.sh 4 4
```

### Custom Training

```bash
python3 vertex_ai_setup.py \
    --project-id your-project-id \
    --job-name "custom-training" \
    --config-file "lanistr/configs/your_config.yaml" \
    --machine-type "n1-standard-8" \
    --accelerator-type "NVIDIA_A100" \
    --accelerator-count 4 \
    --replica-count 2 \
    --base-output-dir "gs://your-bucket/output" \
    --base-data-dir "gs://your-bucket/data" \
    --image-uri "gcr.io/your-project/lanistr-training:latest"
```

## Monitoring and Management

### 1. Monitor Training Jobs

```bash
# List all jobs
./monitor_training.sh

# Get details for specific job
./monitor_training.sh job_name
```

### 2. View Logs

```bash
# View recent logs
gcloud logging read "resource.type=ml_job" --limit=20

# View logs for specific job
gcloud logging read "resource.type=ml_job AND resource.labels.job_id=JOB_ID"
```

### 3. Cloud Console

Visit the [Vertex AI Console](https://console.cloud.google.com/vertex-ai/training/custom-jobs) for:
- Real-time job status
- Resource utilization
- Logs and metrics
- Cost tracking

## Enterprise Features

### 1. Security

- **Container Security**: Non-root user, minimal base image
- **Network Security**: VPC, firewall rules, SSL/TLS
- **Data Encryption**: At-rest and in-transit encryption
- **Access Control**: IAM roles and service accounts

### 2. Monitoring

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Metrics Collection**: Prometheus metrics for training progress
- **Resource Monitoring**: CPU, GPU, memory usage
- **Error Reporting**: Automatic error tracking and alerting

### 3. Reliability

- **Checkpointing**: Automatic model checkpointing to GCS
- **Error Recovery**: Graceful error handling and cleanup
- **Signal Handling**: Proper shutdown on interruption
- **Resource Cleanup**: Automatic cleanup of distributed resources

### 4. Compliance

- **Audit Logging**: All actions logged for compliance
- **Data Governance**: Data lineage and access tracking
- **Security Scanning**: Automated vulnerability scanning
- **Backup and Recovery**: Automated backup procedures

## Configuration Files

### 1. Requirements Files

- `requirements_vertex_ai.txt` - Full enterprise requirements
- `requirements-prod.txt` - Minimal production requirements
- `requirements-dev.txt` - Development requirements

### 2. Security Configuration

- `security-config.yaml` - Security policies and settings

### 3. Training Configurations

- `lanistr/configs/mimic_pretrain.yaml` - MIMIC-IV pretraining
- `lanistr/configs/amazon_pretrain_office.yaml` - Amazon pretraining

## Performance Optimization

### 1. Resource Selection

```bash
# High-performance training
--machine-type "n1-standard-8" \
--accelerator-type "NVIDIA_A100" \
--accelerator-count 8

# Cost-effective training
--machine-type "n1-standard-4" \
--accelerator-type "NVIDIA_TESLA_V100" \
--accelerator-count 4
```

### 2. Batch Size Optimization

```yaml
# In your config file
train_batch_size: 256  # Adjust based on GPU memory
eval_batch_size: 128
test_batch_size: 128
```

### 3. Data Loading Optimization

```python
# In your data loader configuration
num_workers: 4
pin_memory: true
prefetch_factor: 2
```

## Cost Management

### 1. Spot Instances

```bash
# Use spot instances for cost savings
gcloud ai custom-jobs create \
    --enable-spot \
    # ... other parameters
```

### 2. Resource Monitoring

```bash
# Monitor costs
gcloud billing budgets list

# Set up billing alerts
gcloud billing budgets create \
    --billing-account=ACCOUNT_ID \
    --budget-amount=1000USD \
    --threshold-rule=percent=0.5
```

### 3. Cleanup

```bash
# Clean up completed jobs
./cleanup_vertex_ai.sh --force

# Clean up old Docker images
docker system prune -f
```

## Troubleshooting

### 1. Common Issues

#### Authentication Errors
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

#### Permission Errors
```bash
# Check IAM roles
gcloud projects get-iam-policy YOUR_PROJECT_ID

# Add required roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="user:your-email@domain.com" \
    --role="roles/aiplatform.user"
```

#### Out of Memory Errors
```bash
# Reduce batch size in config
train_batch_size: 128  # Instead of 256

# Use gradient accumulation
gradient_accumulation_steps: 2
```

#### Data Loading Issues
```bash
# Check GCS permissions
gsutil ls gs://your-bucket/

# Verify data paths
gsutil ls gs://your-bucket/MIMIC-IV-V2.2/
```

### 2. Debug Commands

```bash
# Check container logs
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Download logs
gsutil cp gs://your-bucket/lanistr-output/job_name/logs/* ./local_logs/

# Check resource usage
gcloud compute instances describe INSTANCE_NAME --zone=ZONE
```

### 3. Performance Debugging

```bash
# Monitor GPU usage
nvidia-smi

# Monitor system resources
htop

# Profile memory usage
python -m memory_profiler your_script.py
```

## Advanced Configuration

### 1. Custom Docker Image

```dockerfile
# Extend the base image
FROM gcr.io/your-project/lanistr-training:latest

# Add custom dependencies
RUN pip install your-custom-package

# Copy custom code
COPY your-custom-code/ /workspace/your-custom-code/
```

### 2. Hyperparameter Tuning

```bash
# Use Vertex AI Hyperparameter Tuning
gcloud ai hyperparameter-tuning-jobs create \
    --display-name="lanistr-hpt" \
    --max-trial-count=10 \
    --parallel-trial-count=2 \
    --config=hyperparameter_tuning_config.yaml
```

### 3. Pipeline Integration

```python
# Use Vertex AI Pipelines
from kfp import dsl
from kfp.v2 import compiler

@dsl.pipeline(
    name="lanistr-training-pipeline",
    pipeline_root="gs://your-bucket/pipeline_root"
)
def lanistr_pipeline():
    # Define your pipeline steps
    pass
```

## Support and Maintenance

### 1. Regular Maintenance

- **Weekly**: Update security patches
- **Monthly**: Review and update dependencies
- **Quarterly**: Performance benchmarking
- **Annually**: Major version upgrades

### 2. Monitoring Checklist

- [ ] Application logs are being collected
- [ ] Metrics are being exported to monitoring system
- [ ] Error reporting is configured
- [ ] Security scans are automated
- [ ] Backup and recovery procedures are tested
- [ ] Performance baselines are established

### 3. Support Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [LANISTR Paper](https://arxiv.org/pdf/2305.16556.pdf)
- [Google Cloud Support](https://cloud.google.com/support)

## License

This deployment setup follows the same Apache 2.0 license as the original LANISTR project. 