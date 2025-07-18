# 🚀 LANISTR - Easy Start Guide

**Get LANISTR training on Vertex AI in under 10 minutes!**

## ⚡ **Quick Start (3 Commands)**

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd lanistr

# 2. Run one-click setup
./one_click_setup.sh

# 3. Submit your first training job
./quick_submit.sh
```

That's it! Your training job will be running on Google Cloud Vertex AI.

## 🎯 **What Just Happened?**

The one-click setup automatically:
- ✅ Sets up Google Cloud authentication
- ✅ Enables required APIs
- ✅ Creates storage bucket
- ✅ Generates sample data
- ✅ Validates your dataset
- ✅ Builds and pushes Docker image
- ✅ Creates quick submit scripts

## 📊 **Monitor Your Job**

```bash
# Check job status
gcloud ai custom-jobs list

# Or visit the console
open https://console.cloud.google.com/vertex-ai/training/custom-jobs
```

## 💰 **Cost Optimization**

### **Development Mode (Cheapest)**
```bash
# Use for testing - costs ~$0.45/hour
./quick_submit_dev.sh
```

### **Production Mode (Full Power)**
```bash
# Use for final training - costs ~$20+/hour
./quick_submit.sh
```

## 📁 **Add Your Own Data**

```bash
# Replace sample data with your own
cp your_data.jsonl ./data/mimic-iv/mimic-iv.jsonl

# Validate your data
python validate_dataset.py --dataset mimic-iv --jsonl-file ./data/mimic-iv/mimic-iv.jsonl

# Submit training
./quick_submit.sh
```

## 🔧 **Customization Options**

### **Different Datasets**
```bash
# MIMIC-IV (medical)
./one_click_setup.sh --dataset mimic-iv

# Amazon (product reviews)
./one_click_setup.sh --dataset amazon
```

### **Different Environments**
```bash
# Development (cheaper)
./one_click_setup.sh --environment dev

# Production (faster)
./one_click_setup.sh --environment prod
```

### **Custom Project**
```bash
./one_click_setup.sh --project-id my-project-id
```

## 🛠️ **Troubleshooting**

### **Common Issues**

**"Project ID not found"**
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

**"Permission denied"**
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:your-email@domain.com" \
  --role="roles/aiplatform.user"
```

**"Docker build failed"**
```bash
docker system prune -a
docker build --no-cache -t lanistr-training:latest .
```

## 📚 **More Options**

### **Interactive Setup**
```bash
python setup_wizard.py
```

### **Simple CLI**
```bash
python lanistr-cli.py --data-dir ./data
```

### **Manual Control**
```bash
python vertex_ai_setup.py --help
```

## 🎉 **Success!**

You now have:
- ✅ A working LANISTR training setup
- ✅ Quick submit scripts for easy training
- ✅ Cost-optimized development and production modes
- ✅ Data validation tools
- ✅ Monitoring and management tools

## 📖 **Next Steps**

1. **Read the full guide**: `JOB_SUBMISSION_GUIDE.md`
2. **Check data requirements**: `DATASET_REQUIREMENTS.md`
3. **Explore CLI options**: `python lanistr-cli.py --help`
4. **Set up monitoring**: `./monitor_training.sh`

## 🆘 **Need Help?**

- **Complete Documentation**: `JOB_SUBMISSION_GUIDE.md`
- **Quick Reference**: `QUICK_SUBMISSION_CARD.md`
- **Data Validation**: `python validate_dataset.py --help`
- **CLI Help**: `python lanistr-cli.py --help`

---

**Happy Training! 🚀** 