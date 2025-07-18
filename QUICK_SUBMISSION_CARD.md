# 🚀 LANISTR Job Submission Quick Reference

## 📋 Essential Commands

### 1. **Validate Your Dataset** (Required First Step)
```bash
# Amazon dataset
python validate_dataset.py --dataset amazon --jsonl-file ./data/amazon.jsonl --data-dir ./data

# MIMIC-IV dataset  
python validate_dataset.py --dataset mimic-iv --jsonl-file ./data/mimic.jsonl --data-dir ./data
```

### 2. **Quick Start (Automated)**
```bash
# Set your project
export PROJECT_ID="your-project-id"
export BUCKET_NAME="lanistr-data-bucket"

# Run automated deployment
./deploy_vertex_ai.sh

# Submit job
./run_mimic_pretrain.sh    # For MIMIC-IV
./run_amazon_pretrain.sh   # For Amazon
```

### 3. **Manual Submission**
```bash
# Upload data
gsutil -m cp -r ./data/ gs://your-bucket-name/

# Build & push image
docker build -t lanistr-training:latest .
docker tag lanistr-training:latest gcr.io/YOUR_PROJECT_ID/lanistr-training:latest
docker push gcr.io/YOUR_PROJECT_ID/lanistr-training:latest

# Submit job
python vertex_ai_setup.py \
  --project-id YOUR_PROJECT_ID \
  --location us-central1 \
  --job-name "lanistr-training" \
  --config-file "lanistr/configs/mimic_pretrain.yaml" \
  --machine-type "n1-standard-4" \
  --accelerator-type "NVIDIA_TESLA_V100" \
  --accelerator-count 8 \
  --base-output-dir "gs://your-bucket-name/output" \
  --base-data-dir "gs://your-bucket-name" \
  --image-uri "gcr.io/YOUR_PROJECT_ID/lanistr-training:latest"
```

## 🎯 Configuration Options

### **Machine Types**
```bash
--machine-type "n1-standard-4"    # 4 vCPU, 15GB RAM
--machine-type "n1-standard-8"    # 8 vCPU, 30GB RAM
--machine-type "n1-standard-16"   # 16 vCPU, 60GB RAM
```

### **GPU Types**
```bash
--accelerator-type "NVIDIA_TESLA_T4"    # Cost-effective
--accelerator-type "NVIDIA_TESLA_V100"  # Good performance
--accelerator-type "NVIDIA_TESLA_A100"  # Best performance
```

### **GPU Count**
```bash
--accelerator-count 1    # Single GPU
--accelerator-count 4    # 4 GPUs
--accelerator-count 8    # 8 GPUs
```

### **Multi-Node Training**
```bash
--replica-count 2 --accelerator-count 4    # 2 nodes, 4 GPUs each
--replica-count 4 --accelerator-count 2    # 4 nodes, 2 GPUs each
```

## 📊 Monitoring Commands

```bash
# List all jobs
gcloud ai custom-jobs list

# Get job details
gcloud ai custom-jobs describe JOB_ID

# Get job logs
gcloud ai custom-jobs describe JOB_ID --format="value(status)"

# Monitor with script
./monitor_training.sh
```

## 🔧 Configuration Files

### **MIMIC-IV**
```yaml
# lanistr/configs/mimic_pretrain.yaml
dataset_name: mimic-iv
task: pretrain
image: true
text: true
time: true
train_batch_size: 128
```

### **Amazon**
```yaml
# lanistr/configs/amazon_pretrain_office.yaml
dataset_name: amazon
category: Office_Products
task: pretrain
image: true
text: true
tab: true
train_batch_size: 128
```

## 🚨 Common Issues

### **Authentication Error**
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### **Permission Error**
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:your-email@domain.com" \
  --role="roles/aiplatform.user"
```

### **API Not Enabled**
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
```

## 💰 Cost Optimization

### **Development/Testing**
```bash
--machine-type "n1-standard-2" --accelerator-type "NVIDIA_TESLA_T4" --accelerator-count 1
```

### **Production**
```bash
--machine-type "n1-standard-4" --accelerator-type "NVIDIA_TESLA_V100" --accelerator-count 8
```

## 📁 Data Requirements

### **Amazon Dataset**
- **Required Fields**: `Review`, `ImageFileName`, `reviewerID`, `verified`, `asin`, `year`, `vote`, `unixReviewTime`
- **Format**: JSONL with GCS paths
- **Min Examples**: 1,000 recommended

### **MIMIC-IV Dataset**
- **Required Fields**: `text`, `image`, `timeseries`
- **Optional Fields**: `y_true` (for finetuning)
- **Format**: JSONL with GCS paths
- **Min Examples**: 100 recommended

## 🎯 Complete Example

```bash
#!/bin/bash
# Complete job submission example

# 1. Validate
python validate_dataset.py --dataset mimic-iv --jsonl-file ./data/mimic.jsonl --data-dir ./data

# 2. Upload data
gsutil -m cp -r ./data/ gs://your-bucket-name/

# 3. Build image
docker build -t lanistr-training:latest .
docker tag lanistr-training:latest gcr.io/YOUR_PROJECT_ID/lanistr-training:latest
docker push gcr.io/YOUR_PROJECT_ID/lanistr-training:latest

# 4. Submit job
python vertex_ai_setup.py \
  --project-id YOUR_PROJECT_ID \
  --job-name "lanistr-mimic-$(date +%Y%m%d-%H%M%S)" \
  --config-file "lanistr/configs/mimic_pretrain.yaml" \
  --machine-type "n1-standard-4" \
  --accelerator-type "NVIDIA_TESLA_V100" \
  --accelerator-count 8 \
  --base-output-dir "gs://your-bucket-name/output" \
  --base-data-dir "gs://your-bucket-name" \
  --image-uri "gcr.io/YOUR_PROJECT_ID/lanistr-training:latest"

# 5. Monitor
echo "Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
```

## 📞 Support Files

- **Full Guide**: `JOB_SUBMISSION_GUIDE.md`
- **Data Requirements**: `DATASET_REQUIREMENTS.md`
- **Validation Tool**: `validate_dataset.py`
- **Example Script**: `submit_job_example.sh`
- **Deployment Script**: `deploy_vertex_ai.sh` 