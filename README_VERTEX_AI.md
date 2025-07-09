# LANISTR on Google Cloud Vertex AI

This guide explains how to run LANISTR distributed training on Google Cloud Vertex AI.

## Overview

LANISTR is a multimodal learning framework that can process language, image, and structured data. This setup enables distributed training on Google Cloud Vertex AI with multiple GPUs and nodes.

## Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with billing enabled
2. **Google Cloud SDK**: Install and configure the Google Cloud SDK
3. **Docker**: Install Docker for building the container image
4. **Python Dependencies**: Install required Python packages

### Install Google Cloud SDK

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Install Python Dependencies

```bash
pip install google-cloud-storage google-cloud-aiplatform
```

## Setup Steps

### 1. Prepare Your Data

First, upload your data to Google Cloud Storage:

```bash
# Upload MIMIC-IV data
python3 upload_data_to_gcs.py \
    --local-data-dir ./lanistr/data \
    --bucket-name your-bucket-name \
    --project-id your-project-id \
    --dataset mimic \
    --create-bucket

# Upload Amazon data
python3 upload_data_to_gcs.py \
    --local-data-dir ./lanistr/data \
    --bucket-name your-bucket-name \
    --project-id your-project-id \
    --dataset amazon
```

### 2. Build and Deploy Docker Image

```bash
# Set your project ID
export PROJECT_ID=your-project-id

# Build and deploy the Docker image
./build_and_deploy.sh
```

This will:
- Build the Docker image with all LANISTR dependencies
- Push it to Google Container Registry
- Create a training script

### 3. Update Configuration

Edit the generated `run_vertex_training.sh` script to update:
- Your bucket names for data and outputs
- Machine types and GPU configurations
- Training parameters

### 4. Run Training

```bash
# Run MIMIC-IV pretraining
./run_vertex_training.sh
```

## Configuration Options

### Machine Types

Vertex AI supports various machine types with different GPU configurations:

- **n1-standard-4**: 4 vCPUs, 15 GB memory
- **n1-standard-8**: 8 vCPUs, 30 GB memory
- **n1-standard-16**: 16 vCPUs, 60 GB memory
- **n1-standard-32**: 32 vCPUs, 120 GB memory

### GPU Accelerators

Available GPU types:
- **NVIDIA_TESLA_V100**: 16 GB memory
- **NVIDIA_TESLA_P100**: 16 GB memory
- **NVIDIA_TESLA_K80**: 12 GB memory
- **NVIDIA_TESLA_T4**: 16 GB memory
- **NVIDIA_A100**: 40 GB memory

### Multi-Node Training

For multi-node training, set `--replica-count` to the number of nodes:

```bash
python3 vertex_ai_setup.py \
    --project-id your-project-id \
    --job-name "lanistr-multi-node" \
    --config-file "vertex_ai_configs/mimic_pretrain_vertex.yaml" \
    --replica-count 4 \
    --accelerator-count 8
```

## Training Scripts

### MIMIC-IV Pretraining

```bash
python3 vertex_ai_setup.py \
    --project-id your-project-id \
    --location us-central1 \
    --job-name "lanistr-mimic-pretrain" \
    --config-file "vertex_ai_configs/mimic_pretrain_vertex.yaml" \
    --machine-type "n1-standard-4" \
    --accelerator-type "NVIDIA_TESLA_V100" \
    --accelerator-count 8 \
    --replica-count 1 \
    --base-output-dir "gs://your-bucket/lanistr-output" \
    --base-data-dir "gs://your-bucket/lanistr-data" \
    --image-uri "gcr.io/your-project/lanistr-training:latest"
```

### Amazon Product Review Pretraining

```bash
python3 vertex_ai_setup.py \
    --project-id your-project-id \
    --location us-central1 \
    --job-name "lanistr-amazon-pretrain" \
    --config-file "vertex_ai_configs/amazon_pretrain_vertex.yaml" \
    --machine-type "n1-standard-4" \
    --accelerator-type "NVIDIA_TESLA_V100" \
    --accelerator-count 8 \
    --replica-count 1 \
    --base-output-dir "gs://your-bucket/lanistr-output" \
    --base-data-dir "gs://your-bucket/lanistr-data" \
    --image-uri "gcr.io/your-project/lanistr-training:latest"
```

## Monitoring and Logs

### View Training Progress

1. **Vertex AI Console**: Visit the [Vertex AI Training Jobs](https://console.cloud.google.com/vertex-ai/training/custom-jobs) page
2. **Cloud Logging**: View detailed logs in Cloud Logging
3. **TensorBoard**: If enabled, view TensorBoard logs in the output directory

### Check Job Status

```bash
# List all training jobs
gcloud ai custom-jobs list --region=us-central1

# Get job details
gcloud ai custom-jobs describe JOB_ID --region=us-central1
```

## Cost Optimization

### Tips for Reducing Costs

1. **Use Spot Instances**: Add `--enable-spot` flag for cost savings
2. **Right-size Machines**: Choose appropriate machine types
3. **Monitor Usage**: Use Cloud Monitoring to track resource usage
4. **Clean Up**: Delete completed jobs and unused resources

### Cost Estimation

Example cost for 8xV100 training:
- Machine: n1-standard-4 with 8xV100
- Duration: 24 hours
- Estimated cost: ~$1,200-1,500

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   # Ensure proper IAM roles
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
       --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
       --role="roles/aiplatform.user"
   ```

2. **Out of Memory Errors**
   - Reduce batch size in config
   - Use larger machine types
   - Enable gradient accumulation

3. **Data Loading Issues**
   - Verify GCS bucket permissions
   - Check data paths in config
   - Ensure data is properly uploaded

### Debug Commands

```bash
# Check container logs
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Download logs
gsutil cp gs://your-bucket/lanistr-output/job_name/logs/* ./local_logs/
```

## Advanced Configuration

### Custom Training Scripts

You can create custom training scripts for specific use cases:

```python
# Example: Custom training with hyperparameter tuning
from vertex_ai_setup import create_vertex_ai_job

create_vertex_ai_job(
    project_id="your-project-id",
    job_name="lanistr-hyperparameter-tuning",
    config_file="custom_config.yaml",
    machine_type="n1-standard-8",
    accelerator_count=4,
    replica_count=2
)
```

### Integration with Vertex AI Pipelines

For more complex workflows, consider using Vertex AI Pipelines:

```python
from kfp import dsl
from kfp.v2 import compiler

@dsl.pipeline(
    name="lanistr-training-pipeline",
    pipeline_root="gs://your-bucket/pipeline_root"
)
def lanistr_pipeline():
    # Define your pipeline steps here
    pass

# Compile and run
compiler.Compiler().compile(
    pipeline_func=lanistr_pipeline,
    package_path="lanistr_pipeline.json"
)
```

## Support

For issues and questions:
1. Check the [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs)
2. Review [LANISTR paper](https://arxiv.org/pdf/2305.16556.pdf)
3. Open an issue in the repository

## License

This setup follows the same Apache 2.0 license as the original LANISTR project. 