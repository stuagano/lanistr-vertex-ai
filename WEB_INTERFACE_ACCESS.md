# 🌐 LANISTR Web Interface - Access Guide

## ✅ **Services Running Successfully!**

Your LANISTR web interface is now running and ready to use! Here's how to access it:

## 🔗 **Access URLs**

### **Frontend (Streamlit)**
- **URL**: http://localhost:8501
- **Purpose**: User-friendly web interface for job management
- **Features**: Dashboard, job submission, monitoring, data validation

### **Backend API (FastAPI)**
- **URL**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Purpose**: Programmatic access and REST API

## 🎯 **What You Can Do**

### **1. Dashboard & System Status**
- ✅ Check system prerequisites
- ✅ Verify Google Cloud authentication
- ✅ Monitor API status
- ✅ View project configuration

### **2. Job Submission**
- 🚀 Submit training jobs with point-and-click interface
- ⚙️ Configure machine types and GPUs
- 📊 Set environment (dev/prod)
- 📁 Upload and validate datasets

### **3. Job Monitoring**
- 📈 Real-time job status tracking
- 📋 View job logs and metrics
- 🗑️ Manage running jobs
- 📊 Performance monitoring

### **4. Data Management**
- 📁 Upload JSONL datasets
- ✅ Validate data format and structure
- 📊 View validation statistics
- 🔍 Check data quality metrics

### **5. Configuration**
- ⚙️ Set up Google Cloud project
- 🔧 Configure regions and buckets
- 🔑 Manage authentication
- 📋 API enablement status

## 🚀 **Quick Start Steps**

### **Step 1: Access the Frontend**
1. Open your browser
2. Go to: **http://localhost:8501**
3. You'll see the LANISTR dashboard

### **Step 2: Configure Your Project**
1. Click on **"Configuration"** in the sidebar
2. Enter your Google Cloud Project ID
3. Select your preferred region
4. Click **"Setup Project"**

### **Step 3: Submit Your First Job**
1. Go to **"Submit Job"** page
2. Select dataset type (MIMIC-IV or Amazon)
3. Choose environment (dev for testing, prod for production)
4. Configure machine settings
5. Click **"Submit Job"**

### **Step 4: Monitor Progress**
1. Go to **"Job Monitoring"** page
2. View real-time job status
3. Check logs and metrics
4. Monitor training progress

## 🔌 **API Usage Examples**

### **Check System Status**
```bash
curl http://localhost:8000/status
```

### **Submit a Training Job**
```bash
curl -X POST "http://localhost:8000/jobs/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "your-project-id",
    "dataset_type": "mimic-iv",
    "environment": "dev",
    "job_name": "my-training-job"
  }'
```

### **List All Jobs**
```bash
curl http://localhost:8000/jobs
```

### **Validate Dataset**
```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_type": "mimic-iv",
    "data_dir": "./data/mimic-iv",
    "jsonl_file": "./data/mimic-iv/mimic-iv.jsonl"
  }'
```

## 🛠️ **Troubleshooting**

### **If Services Aren't Running**
```bash
# Check if services are running
lsof -i :8501 -i :8000

# Restart services
cd /Users/stuartgano/lanistr/web_interface
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
streamlit run app.py --server.port 8501 --server.address localhost &
```

### **If You Get Import Errors**
```bash
# Install missing dependencies
pip install -r web_interface/requirements.txt
pip install omegaconf
```

### **If Ports Are Busy**
```bash
# Kill existing processes
pkill -f "streamlit"
pkill -f "uvicorn"

# Start with different ports
streamlit run app.py --server.port 8502 --server.address localhost &
python -m uvicorn api:app --host 0.0.0.0 --port 8001 --reload &
```

## 📊 **Features Overview**

### **Frontend Features**
- 📊 **Interactive Dashboard**: Real-time system status
- ⚙️ **Configuration Management**: Easy project setup
- 🚀 **Job Submission**: Point-and-click job creation
- 📈 **Job Monitoring**: Real-time progress tracking
- 📁 **Data Management**: Upload and validation tools
- 📋 **Logs Viewer**: Real-time log streaming

### **API Features**
- 🔌 **REST API**: Full programmatic access
- 📚 **Auto-documentation**: Interactive API docs
- 🔄 **Async Operations**: Non-blocking job submission
- 📊 **Real-time Status**: Live job monitoring
- 🔒 **Validation**: Comprehensive data validation
- 📁 **File Upload**: Direct file upload support

## 🎯 **Use Cases**

### **For Researchers**
1. **Quick Setup**: Use web interface for easy configuration
2. **Data Validation**: Validate datasets before training
3. **Cost Control**: Use development mode for testing
4. **Monitoring**: Track job progress in real-time

### **For Development Teams**
1. **API Integration**: Use REST API in your workflows
2. **Automation**: Integrate with CI/CD pipelines
3. **Testing**: Use development mode for quick iterations
4. **Production**: Use production mode for final training

### **For Production Teams**
1. **Scalability**: Use API for programmatic access
2. **Monitoring**: Real-time job monitoring and alerts
3. **Management**: Job lifecycle management
4. **Integration**: Integrate with existing systems

## 🔧 **Configuration Options**

### **Environment Variables**
```bash
# Ports
export STREAMLIT_PORT=8501
export FASTAPI_PORT=8000

# Hosts
export STREAMLIT_HOST="localhost"
export FASTAPI_HOST="0.0.0.0"

# Google Cloud
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

## 📞 **Support**

If you encounter any issues:
1. Check the logs in the web interface
2. Use the API health endpoint: http://localhost:8000/health
3. Verify your Google Cloud configuration
4. Check the troubleshooting section above

---

**🎉 Your LANISTR web interface is ready! Access it at http://localhost:8501** 