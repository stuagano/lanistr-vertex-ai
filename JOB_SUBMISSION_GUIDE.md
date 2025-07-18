# LANISTR Job Submission Guide

This guide walks you through the complete process of submitting a LANISTR training job to Google Cloud Vertex AI, from data validation to job submission and monitoring.

## 🎯 Overview

The job submission process involves these key steps:
1. **Data Validation** - Ensure your dataset is ready
2. **Environment Setup** - Configure Google Cloud and build container
3. **Data Upload** - Transfer data to Google Cloud Storage
4. **Job Submission** - Submit training job to Vertex AI
5. **Monitoring** - Track training progress

## 📋 Prerequisites

### Required Tools
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Install Docker
# (Follow instructions for your OS: https://docs.docker.com/get-docker/)

# Install Python dependencies
pip install -r requirements_vertex_ai.txt
```

### Google Cloud Setup
```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## 🚀 Quick Start: Automated Deployment

The easiest way to submit a job is using the automated deployment script:

### 1. Configure Environment Variables
```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export BUCKET_NAME="lanistr-data-bucket"
```

### 2. Run Automated Deployment
```bash
# Make script executable
chmod +x deploy_vertex_ai.sh

# Run deployment
./deploy_vertex_ai.sh
```

This script will:
- ✅ Check prerequisites
- ✅ Authenticate with Google Cloud
- ✅ Enable required APIs
- ✅ Create GCS bucket
- ✅ Upload your data
- ✅ Build and push Docker image
- ✅ Generate training scripts

### 3. Submit Training Job
After deployment, you'll have generated scripts:

```bash
# MIMIC-IV pretraining
./run_mimic_pretrain.sh

# Amazon pretraining
./run_amazon_pretrain.sh

# Multi-node training
./run_multi_node_training.sh
```

## 🔧 Manual Job Submission

If you prefer manual control, here's the step-by-step process:

### Step 1: Validate Your Dataset

```bash
# Validate Amazon dataset
python validate_dataset.py \
  --dataset amazon \
  --jsonl-file ./data/amazon.jsonl \
  --data-dir ./data \
  --gcs-bucket your-bucket-name

# Validate MIMIC-IV dataset
python validate_dataset.py \
  --dataset mimic-iv \
  --jsonl-file ./data/mimic.jsonl \
  --data-dir ./data \
  --gcs-bucket your-bucket-name
```

**Expected Output:**
```
Overall Status: ✅ PASSED
💡 RECOMMENDATIONS:
  • Dataset is ready for training
```

### Step 2: Upload Data to Google Cloud Storage

```bash
# Create bucket (if not exists)
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l us-central1 gs://your-bucket-name

# Upload data
gsutil -m cp -r ./data/ gs://your-bucket-name/

# Verify upload
gsutil ls gs://your-bucket-name/
```

### Step 3: Build and Push Docker Image

```bash
# Build Docker image
docker build -t lanistr-training:latest .

# Tag for Google Container Registry
docker tag lanistr-training:latest gcr.io/YOUR_PROJECT_ID/lanistr-training:latest

# Push to GCR
docker push gcr.io/YOUR_PROJECT_ID/lanistr-training:latest
```

### Step 4: Submit Training Job

#### For MIMIC-IV Pretraining:
```bash
python vertex_ai_setup.py \
  --project-id YOUR_PROJECT_ID \
  --location us-central1 \
  --job-name "lanistr-mimic-pretrain" \
  --config-file "lanistr/configs/mimic_pretrain.yaml" \
  --machine-type "n1-standard-4" \
  --accelerator-type "NVIDIA_TESLA_V100" \
  --accelerator-count 8 \
  --replica-count 1 \
  --base-output-dir "gs://your-bucket-name/lanistr-output" \
  --base-data-dir "gs://your-bucket-name" \
  --image-uri "gcr.io/YOUR_PROJECT_ID/lanistr-training:latest"
```

#### For Amazon Pretraining:
```bash
python vertex_ai_setup.py \
  --project-id YOUR_PROJECT_ID \
  --location us-central1 \
  --job-name "lanistr-amazon-pretrain" \
  --config-file "lanistr/configs/amazon_pretrain_office.yaml" \
  --machine-type "n1-standard-4" \
  --accelerator-type "NVIDIA_TESLA_V100" \
  --accelerator-count 8 \
  --replica-count 1 \
  --base-output-dir "gs://your-bucket-name/lanistr-output" \
  --base-data-dir "gs://your-bucket-name" \
  --image-uri "gcr.io/YOUR_PROJECT_ID/lanistr-training:latest"
```

## 📊 Job Configuration Options

### Machine Types
```bash
# Single GPU (development/testing)
--machine-type "n1-standard-4" --accelerator-count 1

# Multi-GPU (production)
--machine-type "n1-standard-4" --accelerator-count 8

# High-memory (large datasets)
--machine-type "n1-standard-8" --accelerator-count 8
```

### GPU Types
```bash
# V100 (good performance/cost ratio)
--accelerator-type "NVIDIA_TESLA_V100"

# A100 (best performance)
--accelerator-type "NVIDIA_TESLA_A100"

# T4 (cost-effective)
--accelerator-type "NVIDIA_TESLA_T4"
```

### Multi-Node Training
```bash
# 2 nodes, 8 GPUs each
--replica-count 2 --accelerator-count 8

# 4 nodes, 4 GPUs each
--replica-count 4 --accelerator-count 4
```

## 🔍 Monitoring Your Job

### 1. Console Monitoring
```bash
# Get job URL from submission output
echo "Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs/JOB_ID"
```

### 2. Command Line Monitoring
```bash
# List all jobs
gcloud ai custom-jobs list

# Get job details
gcloud ai custom-jobs describe JOB_ID

# Get job logs
gcloud ai custom-jobs describe JOB_ID --format="value(status)"
```

### 3. Automated Monitoring
```bash
# Use the generated monitoring script
./monitor_training.sh
```

## 📁 Configuration Files

### MIMIC-IV Configuration
```yaml
# lanistr/configs/mimic_pretrain.yaml
seed: 2022
dataset_name: mimic-iv
task: pretrain

# Modalities
image: true
text: true
time: true
tab: false

# Training parameters
train_batch_size: 128
scheduler:
  num_epochs: 40
  warmup_epochs: 5

# Data paths (will be overridden by Vertex AI)
root_data_dir: ./data/MIMIC-IV-V2.2/physionet.org/files
image_data_dir: ./data/MIMIC-IV-V2.2/physionet.org/files/mimic-cxr-jpg/2.0.0
```

### Amazon Configuration
```yaml
# lanistr/configs/amazon_pretrain_office.yaml
seed: 2022
dataset_name: amazon
category: Office_Products
task: pretrain

# Modalities
image: true
text: true
tab: true
time: false

# Training parameters
train_batch_size: 128
scheduler:
  num_epochs: 20
  warmup_epochs: 5

# Data paths (will be overridden by Vertex AI)
data_dir: ./data/APR2018/Office_Products
image_data_dir: ./data/APR2018/Office_Products/images
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Authentication Errors
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

#### 2. Permission Errors
```bash
# Check IAM roles
gcloud projects get-iam-policy YOUR_PROJECT_ID

# Add required roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:your-email@domain.com" \
  --role="roles/aiplatform.user"
```

#### 3. Docker Build Errors
```bash
# Check Docker is running
docker ps

# Clean Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t lanistr-training:latest .
```

#### 4. Data Upload Errors
```bash
# Check bucket permissions
gsutil iam get gs://your-bucket-name

# Set bucket permissions
gsutil iam ch allUsers:objectViewer gs://your-bucket-name
```

#### 5. Job Submission Errors
```bash
# Check API is enabled
gcloud services list --enabled --filter="name:aiplatform.googleapis.com"

# Enable if needed
gcloud services enable aiplatform.googleapis.com
```

## 📈 Cost Optimization

### Cost-Effective Training
```bash
# Use T4 GPUs for development
--accelerator-type "NVIDIA_TESLA_T4" --accelerator-count 1

# Use spot instances (cheaper but can be preempted)
# (Add to vertex_ai_setup.py configuration)

# Use smaller machine types for testing
--machine-type "n1-standard-2" --accelerator-count 1
```

### Resource Monitoring
```bash
# Monitor costs
gcloud billing budgets list

# Set up billing alerts
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="LANISTR Training Budget" \
  --budget-amount=100USD \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90
```

## 🔄 Complete Workflow Example

Here's a complete example for MIMIC-IV pretraining:

```bash
#!/bin/bash
# Complete MIMIC-IV Training Workflow

# 1. Set environment
export PROJECT_ID="your-project-id"
export BUCKET_NAME="lanistr-data-bucket"
export REGION="us-central1"

# 2. Validate dataset
python validate_dataset.py \
  --dataset mimic-iv \
  --jsonl-file ./data/mimic.jsonl \
  --data-dir ./data \
  --gcs-bucket $BUCKET_NAME

# 3. Upload data
gsutil -m cp -r ./data/ gs://$BUCKET_NAME/

# 4. Build and push image
docker build -t lanistr-training:latest .
docker tag lanistr-training:latest gcr.io/$PROJECT_ID/lanistr-training:latest
docker push gcr.io/$PROJECT_ID/lanistr-training:latest

# 5. Submit job
python vertex_ai_setup.py \
  --project-id $PROJECT_ID \
  --location $REGION \
  --job-name "lanistr-mimic-pretrain-$(date +%Y%m%d-%H%M%S)" \
  --config-file "lanistr/configs/mimic_pretrain.yaml" \
  --machine-type "n1-standard-4" \
  --accelerator-type "NVIDIA_TESLA_V100" \
  --accelerator-count 8 \
  --replica-count 1 \
  --base-output-dir "gs://$BUCKET_NAME/lanistr-output" \
  --base-data-dir "gs://$BUCKET_NAME" \
  --image-uri "gcr.io/$PROJECT_ID/lanistr-training:latest"

# 6. Monitor
echo "Job submitted! Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
```

## 📞 Support

- **Documentation**: See `DATASET_REQUIREMENTS.md` for data specifications
- **Validation**: Use `validate_dataset.py` for pre-submission validation
- **Examples**: Use `generate_sample_data.py` to create test datasets
- **Monitoring**: Use `monitor_training.sh` for job tracking

This framework provides a complete, production-ready solution for submitting LANISTR training jobs to Vertex AI with proper validation, monitoring, and error handling. 