# 🚀 LANISTR - Multimodal Learning Framework

A comprehensive multimodal learning framework for training and deploying AI models on Google Cloud Platform with Vertex AI.

## 🌟 Features

- **Multimodal Learning**: Support for text, image, and tabular data
- **Cloud-Native**: Built for Google Cloud Platform and Vertex AI
- **Production Ready**: Complete CI/CD pipeline with Cloud Build
- **Easy Setup**: Automated virtual environment and dependency management
- **Comprehensive Testing**: Full test suite with pytest
- **Interactive Tutorials**: Jupyter notebooks for learning and experimentation

## 📋 Quick Start

### 1. Setup Virtual Environment

```bash
# Run the automated setup script
./setup_venv.sh

# Or manually create and activate virtual environment
python -m venv lanistr_env
source lanistr_env/bin/activate  # On Windows: lanistr_env\Scripts\activate
pip install -r requirements-cloud.txt
```

### 2. Cloud Deployment (Recommended)

```bash
# Build and push container to Google Cloud
./build_minimal.sh

# Or use Cloud Build
./trigger_cloud_build.sh
```

### 3. Run Training Jobs

```bash
# Open the cloud tutorial notebook
jupyter notebook lanistr_cloud_tutorial.ipynb
```

## 🏗️ Project Structure

```
lanistr/
├── lanistr/                    # Core LANISTR package
│   ├── main.py                # Main entry point
│   ├── model/                 # Model architectures
│   ├── dataset/               # Data loading utilities
│   └── utils/                 # Utility functions
├── configs/                   # Local configuration files
├── vertex_ai_configs/         # Vertex AI deployment configs
├── tests/                     # Test suite
├── web_interface/             # Web-based interface
├── Dockerfile.minimal         # Minimal container for cloud
├── cloudbuild.yaml           # Cloud Build configuration
├── setup_venv.sh             # Automated setup script
└── lanistr_cloud_tutorial.ipynb  # Interactive tutorial
```

## 🐳 Container Deployment

### Quick Container Build

```bash
# Build minimal container (~5-10 minutes)
./build_minimal.sh

# Container URI: gcr.io/mgm-digitalconcierge/lanistr:latest
```

### Manual Container Build

```bash
# Build locally
docker build -f Dockerfile.minimal -t lanistr:latest .

# Push to Google Container Registry
docker tag lanistr:latest gcr.io/PROJECT_ID/lanistr:latest
docker push gcr.io/PROJECT_ID/lanistr:latest
```

## ☁️ Cloud Deployment

### Prerequisites

1. **Google Cloud Project**: Set up with billing enabled
2. **Required APIs**: Cloud Build, Container Registry, Vertex AI
3. **Service Account**: With appropriate permissions
4. **Storage Bucket**: For data and model artifacts

### Automated Deployment

```bash
# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Build and deploy
./build_minimal.sh
```

## 📊 Supported Datasets

- **MIMIC-IV**: Medical imaging and clinical data
- **Amazon Reviews**: Multimodal product reviews
- **Custom Datasets**: Extensible framework for any multimodal data

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_data_validator.py
python -m pytest tests/test_training_pipeline.py
```

## 📚 Documentation

- **[Cloud Build Guide](CLOUD_BUILD_GUIDE.md)**: Complete Cloud Build documentation
- **[Container Build Guide](CONTAINER_BUILD_GUIDE.md)**: Container build instructions
- **[Virtual Environment Setup](venv_setup_guide.md)**: Environment setup guide
- **[Test Report](TEST_REPORT.md)**: Testing framework and results

## 🎯 Tutorials

### Interactive Notebooks

- **`lanistr_cloud_tutorial.ipynb`**: Complete cloud deployment tutorial
  - Google Cloud setup
  - Data preparation and upload
  - Job configuration and submission
  - Monitoring and results download

## 🔧 Configuration

### Environment Variables

```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export BUCKET_NAME="your-bucket-name"
```

### Vertex AI Configuration

Configuration files are located in `vertex_ai_configs/`:
- `mimic_pretrain_vertex.yaml`: MIMIC-IV pre-training
- `mimic_finetune_vertex.yaml`: MIMIC-IV fine-tuning
- `amazon_pretrain_vertex.yaml`: Amazon reviews pre-training

## 🚀 Performance

- **Container Size**: ~2.8GB (minimal build)
- **Build Time**: ~5-10 minutes (Cloud Build)
- **Memory Usage**: Configurable via Vertex AI settings
- **GPU Support**: Automatic GPU detection and utilization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the guides in the `docs/` directory
- **Tutorials**: Follow the Jupyter notebook tutorials

## 🔗 Links

- **Repository**: https://github.com/stuagano/lanistr-vertex-ai
- **Container Registry**: gcr.io/mgm-digitalconcierge/lanistr:latest
- **Documentation**: See guides in the repository

---

**Happy Training! 🎉**

