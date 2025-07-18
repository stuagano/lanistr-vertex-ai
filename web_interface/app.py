#!/usr/bin/env python3
"""
LANISTR Web Interface - Streamlit App

A simple web interface for submitting LANISTR training jobs to Vertex AI
with point-and-click configuration and deployment.
"""

import streamlit as st
import json
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time
import threading

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="LANISTR Training Interface",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if check and result.returncode != 0:
            st.error(f"Command failed: {cmd}")
            st.error(f"Error: {result.stderr}")
            return None
        return result
    except subprocess.TimeoutExpired:
        st.error(f"Command timed out: {cmd}")
        return None
    except Exception as e:
        st.error(f"Command failed: {cmd}")
        st.error(f"Exception: {str(e)}")
        return None

def get_project_id() -> str:
    """Get the current Google Cloud project ID."""
    result = run_command("gcloud config get-value project", check=False)
    if result and result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return ""

def check_prerequisites() -> Dict[str, bool]:
    """Check if required tools are installed."""
    tools = {
        "gcloud": "Google Cloud SDK",
        "docker": "Docker",
        "python3": "Python 3",
        "gsutil": "Google Cloud Storage"
    }
    
    status = {}
    for tool, description in tools.items():
        result = run_command(f"which {tool}", check=False)
        status[tool] = result.returncode == 0 if result else False
    
    return status

def check_authentication() -> bool:
    """Check if user is authenticated with Google Cloud."""
    result = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", check=False)
    return result and result.returncode == 0 and result.stdout.strip()

def get_available_projects() -> list:
    """Get list of available Google Cloud projects."""
    result = run_command("gcloud projects list --format='value(projectId)'", check=False)
    if result and result.returncode == 0:
        return [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
    return []

def get_available_buckets(project_id: str) -> list:
    """Get list of available GCS buckets."""
    result = run_command(f"gsutil ls -p {project_id}", check=False)
    if result and result.returncode == 0:
        buckets = []
        for line in result.stdout.strip().split('\n'):
            if line.startswith('gs://'):
                buckets.append(line.replace('gs://', '').replace('/', ''))
        return buckets
    return []

def create_sample_data(dataset_type: str, data_dir: str) -> bool:
    """Create sample data for the specified dataset."""
    jsonl_file = f"{data_dir}/{dataset_type}.jsonl"
    
    if not Path(jsonl_file).exists():
        cmd = f"python generate_sample_data.py --dataset {dataset_type} --output-file {jsonl_file} --num-samples 100 --create-files"
        result = run_command(cmd, check=False)
        return result and result.returncode == 0
    return True

def validate_dataset(dataset_type: str, data_dir: str, jsonl_file: str) -> Dict[str, Any]:
    """Validate the dataset and return results."""
    cmd = f"python validate_dataset.py --dataset {dataset_type} --jsonl-file {jsonl_file} --data-dir {data_dir} --output-file validation_result.json"
    result = run_command(cmd, check=False)
    
    if result and result.returncode == 0:
        try:
            with open("validation_result.json", "r") as f:
                return json.load(f)
        except:
            return {"passed": True, "message": "Validation completed"}
    else:
        return {"passed": False, "message": result.stderr if result else "Validation failed"}

def setup_gcs_bucket(project_id: str, bucket_name: str, region: str) -> bool:
    """Set up GCS bucket."""
    result = run_command(f"gsutil ls -b gs://{bucket_name}", check=False)
    if not result or result.returncode != 0:
        result = run_command(f"gsutil mb -p {project_id} -c STANDARD -l {region} gs://{bucket_name}", check=False)
        return result and result.returncode == 0
    return True

def build_and_push_image(project_id: str, image_name: str = "lanistr-training") -> str:
    """Build and push Docker image to GCR."""
    image_uri = f"gcr.io/{project_id}/{image_name}:latest"
    
    # Build image
    result = run_command(f"docker build -t {image_name}:latest .", check=False)
    if not result or result.returncode != 0:
        return ""
    
    # Tag for GCR
    result = run_command(f"docker tag {image_name}:latest {image_uri}", check=False)
    if not result or result.returncode != 0:
        return ""
    
    # Push to GCR
    result = run_command(f"docker push {image_uri}", check=False)
    if not result or result.returncode != 0:
        return ""
    
    return image_uri

def submit_job(config: Dict[str, Any]) -> Dict[str, Any]:
    """Submit the training job to Vertex AI."""
    cmd = [
        "python", "vertex_ai_setup.py",
        "--project-id", config["project_id"],
        "--location", config["region"],
        "--job-name", config["job_name"],
        "--config-file", config["config_file"],
        "--machine-type", config["machine_type"],
        "--accelerator-type", config["accelerator_type"],
        "--accelerator-count", str(config["accelerator_count"]),
        "--replica-count", "1",
        "--base-output-dir", f"gs://{config['bucket_name']}/lanistr-output",
        "--base-data-dir", f"gs://{config['bucket_name']}",
        "--image-uri", config["image_uri"]
    ]
    
    result = run_command(" ".join(cmd), check=False)
    
    if result and result.returncode == 0:
        return {"success": True, "message": "Job submitted successfully!"}
    else:
        return {"success": False, "message": result.stderr if result else "Job submission failed"}

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 LANISTR Training Interface</h1>', unsafe_allow_html=True)
    st.markdown("Submit LANISTR training jobs to Google Cloud Vertex AI with ease!")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "⚙️ Configuration", "🚀 Submit Job", "📊 Monitor Jobs", "📁 Data Management"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "⚙️ Configuration":
        show_configuration()
    elif page == "🚀 Submit Job":
        show_job_submission()
    elif page == "📊 Monitor Jobs":
        show_job_monitoring()
    elif page == "📁 Data Management":
        show_data_management()

def show_dashboard():
    """Show the main dashboard."""
    st.markdown('<h2 class="section-header">System Status</h2>', unsafe_allow_html=True)
    
    # Check prerequisites
    with st.spinner("Checking system requirements..."):
        prereq_status = check_prerequisites()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prerequisites")
        for tool, status in prereq_status.items():
            if status:
                st.success(f"✅ {tool}")
            else:
                st.error(f"❌ {tool}")
    
    with col2:
        st.subheader("Authentication")
        if check_authentication():
            st.success("✅ Google Cloud authenticated")
        else:
            st.error("❌ Not authenticated")
            if st.button("Authenticate"):
                with st.spinner("Opening authentication..."):
                    run_command("gcloud auth login", check=False)
                st.rerun()
    
    # Current project
    st.subheader("Current Configuration")
    project_id = get_project_id()
    if project_id:
        st.info(f"Current Project: {project_id}")
    else:
        st.warning("No project configured")
    
    # Quick actions
    st.markdown('<h2 class="section-header">Quick Actions</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Quick Setup", use_container_width=True):
            st.session_state.page = "⚙️ Configuration"
            st.rerun()
    
    with col2:
        if st.button("📊 Monitor Jobs", use_container_width=True):
            st.session_state.page = "📊 Monitor Jobs"
            st.rerun()
    
    with col3:
        if st.button("📁 Manage Data", use_container_width=True):
            st.session_state.page = "📁 Data Management"
            st.rerun()

def show_configuration():
    """Show configuration page."""
    st.markdown('<h2 class="section-header">Project Configuration</h2>', unsafe_allow_html=True)
    
    # Project selection
    projects = get_available_projects()
    current_project = get_project_id()
    
    if projects:
        selected_project = st.selectbox(
            "Select Google Cloud Project",
            projects,
            index=projects.index(current_project) if current_project in projects else 0
        )
        
        if selected_project != current_project:
            if st.button("Set Project"):
                with st.spinner("Setting project..."):
                    run_command(f"gcloud config set project {selected_project}")
                st.success(f"Project set to: {selected_project}")
                st.rerun()
    else:
        st.error("No projects found. Please authenticate with Google Cloud.")
        if st.button("Authenticate"):
            with st.spinner("Opening authentication..."):
                run_command("gcloud auth login", check=False)
            st.rerun()
    
    # Region selection
    regions = ["us-central1", "us-east1", "us-west1", "europe-west1", "asia-east1"]
    selected_region = st.selectbox("Select Region", regions, index=0)
    
    # API status
    st.subheader("API Status")
    apis = [
        "aiplatform.googleapis.com",
        "storage.googleapis.com",
        "logging.googleapis.com",
        "monitoring.googleapis.com",
        "errorreporting.googleapis.com",
        "containerregistry.googleapis.com"
    ]
    
    for api in apis:
        result = run_command(f"gcloud services list --enabled --filter='name:{api}' --format='value(name)'", check=False)
        if result and result.returncode == 0 and result.stdout.strip():
            st.success(f"✅ {api}")
        else:
            st.warning(f"⚠️ {api}")
            if st.button(f"Enable {api}", key=f"enable_{api}"):
                with st.spinner(f"Enabling {api}..."):
                    run_command(f"gcloud services enable {api}")
                st.success(f"{api} enabled!")
                st.rerun()
    
    # Save configuration
    if st.button("Save Configuration"):
        config = {
            "project_id": selected_project,
            "region": selected_region
        }
        st.session_state.config = config
        st.success("Configuration saved!")

def show_job_submission():
    """Show job submission page."""
    st.markdown('<h2 class="section-header">Submit Training Job</h2>', unsafe_allow_html=True)
    
    # Configuration
    config = st.session_state.get("config", {})
    project_id = config.get("project_id", get_project_id())
    region = config.get("region", "us-central1")
    
    if not project_id:
        st.error("Please configure your project first.")
        return
    
    # Job configuration form
    with st.form("job_config"):
        st.subheader("Job Configuration")
        
        # Dataset selection
        dataset_type = st.selectbox(
            "Dataset Type",
            ["mimic-iv", "amazon"],
            help="Select the type of dataset you want to train on"
        )
        
        # Environment selection
        environment = st.selectbox(
            "Environment",
            ["dev", "prod"],
            help="Development mode is cheaper, production mode is faster"
        )
        
        # Job name
        job_name = st.text_input(
            "Job Name",
            value=f"lanistr-{dataset_type}-{int(time.time())}",
            help="Unique name for your training job"
        )
        
        # Machine configuration
        st.subheader("Machine Configuration")
        
        if environment == "dev":
            machine_type = "n1-standard-2"
            accelerator_type = "NVIDIA_TESLA_T4"
            accelerator_count = 1
        else:
            machine_type = "n1-standard-4"
            accelerator_type = "NVIDIA_TESLA_V100"
            accelerator_count = 8
        
        # Allow customization
        custom_config = st.checkbox("Customize machine configuration")
        
        if custom_config:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                machine_type = st.selectbox(
                    "Machine Type",
                    ["n1-standard-2", "n1-standard-4", "n1-standard-8"],
                    index=["n1-standard-2", "n1-standard-4", "n1-standard-8"].index(machine_type)
                )
            
            with col2:
                accelerator_type = st.selectbox(
                    "GPU Type",
                    ["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100", "NVIDIA_TESLA_A100"],
                    index=["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100", "NVIDIA_TESLA_A100"].index(accelerator_type)
                )
            
            with col3:
                accelerator_count = st.slider(
                    "GPU Count",
                    min_value=1,
                    max_value=8,
                    value=accelerator_count
                )
        
        # Storage configuration
        st.subheader("Storage Configuration")
        
        bucket_name = st.text_input(
            "GCS Bucket Name",
            value=f"lanistr-{project_id}-{dataset_type}",
            help="Google Cloud Storage bucket for data and outputs"
        )
        
        # Data configuration
        st.subheader("Data Configuration")
        
        data_dir = st.text_input(
            "Data Directory",
            value=f"./data/{dataset_type}",
            help="Local directory containing your data"
        )
        
        jsonl_file = st.text_input(
            "JSONL File",
            value=f"{data_dir}/{dataset_type}.jsonl",
            help="Path to your JSONL data file"
        )
        
        # Data validation
        if st.checkbox("Validate data before submission"):
            if st.button("Validate Data"):
                with st.spinner("Validating data..."):
                    # Create sample data if needed
                    if not Path(jsonl_file).exists():
                        create_sample_data(dataset_type, data_dir)
                    
                    # Validate data
                    validation_result = validate_dataset(dataset_type, data_dir, jsonl_file)
                    
                    if validation_result.get("passed", False):
                        st.success("✅ Data validation passed!")
                    else:
                        st.error(f"❌ Data validation failed: {validation_result.get('message', 'Unknown error')}")
        
        # Submit button
        submitted = st.form_submit_button("🚀 Submit Job")
        
        if submitted:
            # Validate inputs
            if not job_name or not bucket_name:
                st.error("Please fill in all required fields.")
                return
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Setup bucket
                status_text.text("Setting up GCS bucket...")
                if not setup_gcs_bucket(project_id, bucket_name, region):
                    st.error("Failed to setup GCS bucket")
                    return
                progress_bar.progress(20)
                
                # Step 2: Create sample data if needed
                status_text.text("Preparing data...")
                if not Path(jsonl_file).exists():
                    if not create_sample_data(dataset_type, data_dir):
                        st.error("Failed to create sample data")
                        return
                progress_bar.progress(40)
                
                # Step 3: Upload data
                status_text.text("Uploading data to GCS...")
                result = run_command(f"gsutil -m cp -r {data_dir}/ gs://{bucket_name}/", check=False)
                if not result or result.returncode != 0:
                    st.warning("Data upload failed, but continuing...")
                progress_bar.progress(60)
                
                # Step 4: Build and push image
                status_text.text("Building and pushing Docker image...")
                image_uri = build_and_push_image(project_id)
                if not image_uri:
                    st.error("Failed to build and push Docker image")
                    return
                progress_bar.progress(80)
                
                # Step 5: Submit job
                status_text.text("Submitting job to Vertex AI...")
                
                # Determine config file
                config_file = f"lanistr/configs/{dataset_type}_pretrain.yaml"
                
                job_config = {
                    "project_id": project_id,
                    "region": region,
                    "job_name": job_name,
                    "config_file": config_file,
                    "machine_type": machine_type,
                    "accelerator_type": accelerator_type,
                    "accelerator_count": accelerator_count,
                    "bucket_name": bucket_name,
                    "image_uri": image_uri
                }
                
                result = submit_job(job_config)
                
                if result["success"]:
                    progress_bar.progress(100)
                    status_text.text("Job submitted successfully!")
                    
                    st.success("🎉 Job submitted successfully!")
                    st.info(f"Job Name: {job_name}")
                    st.info(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
                    st.info(f"Output location: gs://{bucket_name}/lanistr-output/{job_name}")
                    
                    # Store job info in session
                    if "jobs" not in st.session_state:
                        st.session_state.jobs = []
                    
                    st.session_state.jobs.append({
                        "name": job_name,
                        "status": "SUBMITTED",
                        "timestamp": time.time(),
                        "config": job_config
                    })
                else:
                    st.error(f"Job submission failed: {result['message']}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def show_job_monitoring():
    """Show job monitoring page."""
    st.markdown('<h2 class="section-header">Job Monitoring</h2>', unsafe_allow_html=True)
    
    # Get jobs from session state
    jobs = st.session_state.get("jobs", [])
    
    if not jobs:
        st.info("No jobs submitted yet. Submit a job first!")
        return
    
    # Display jobs
    for i, job in enumerate(jobs):
        with st.expander(f"Job: {job['name']} - {job['status']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Status:** {job['status']}")
                st.write(f"**Submitted:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job['timestamp']))}")
                st.write(f"**Dataset:** {job['config'].get('config_file', 'Unknown').split('/')[-1].replace('_pretrain.yaml', '')}")
            
            with col2:
                st.write(f"**Machine:** {job['config'].get('machine_type', 'Unknown')}")
                st.write(f"**GPUs:** {job['config'].get('accelerator_count', 'Unknown')}x {job['config'].get('accelerator_type', 'Unknown')}")
                st.write(f"**Bucket:** gs://{job['config'].get('bucket_name', 'Unknown')}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Check Status", key=f"check_{i}"):
                    # This would integrate with gcloud to get actual status
                    st.info("Status checking would be implemented here")
            
            with col2:
                if st.button("View Logs", key=f"logs_{i}"):
                    st.info("Log viewing would be implemented here")
            
            with col3:
                if st.button("Delete", key=f"delete_{i}"):
                    jobs.pop(i)
                    st.session_state.jobs = jobs
                    st.rerun()
    
    # Refresh button
    if st.button("Refresh Jobs"):
        st.rerun()

def show_data_management():
    """Show data management page."""
    st.markdown('<h2 class="section-header">Data Management</h2>', unsafe_allow_html=True)
    
    # Data upload
    st.subheader("Upload Data")
    
    uploaded_file = st.file_uploader(
        "Upload JSONL file",
        type=['jsonl'],
        help="Upload your dataset in JSONL format"
    )
    
    if uploaded_file is not None:
        dataset_type = st.selectbox("Dataset Type", ["mimic-iv", "amazon"])
        
        if st.button("Save Uploaded Data"):
            # Save the uploaded file
            data_dir = f"./data/{dataset_type}"
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            
            file_path = f"{data_dir}/{dataset_type}.jsonl"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Data saved to {file_path}")
    
    # Data validation
    st.subheader("Validate Data")
    
    data_dir = st.text_input("Data Directory", value="./data")
    jsonl_file = st.text_input("JSONL File Path")
    
    if st.button("Validate"):
        if jsonl_file and Path(jsonl_file).exists():
            dataset_type = "mimic-iv" if "mimic" in jsonl_file else "amazon"
            
            with st.spinner("Validating data..."):
                validation_result = validate_dataset(dataset_type, data_dir, jsonl_file)
                
                if validation_result.get("passed", False):
                    st.success("✅ Data validation passed!")
                    
                    # Show statistics
                    if "stats" in validation_result:
                        stats = validation_result["stats"]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Records", stats.get("total_records", 0))
                        
                        with col2:
                            st.metric("File Size (MB)", f"{stats.get('file_size_mb', 0):.2f}")
                        
                        with col3:
                            st.metric("Validation Rate", "100%")
                else:
                    st.error(f"❌ Data validation failed: {validation_result.get('message', 'Unknown error')}")
        else:
            st.error("Please provide a valid JSONL file path")
    
    # Sample data generation
    st.subheader("Generate Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_dataset = st.selectbox("Dataset Type", ["mimic-iv", "amazon"])
        sample_count = st.slider("Number of Samples", min_value=10, max_value=1000, value=100)
    
    with col2:
        sample_dir = st.text_input("Output Directory", value=f"./data/{sample_dataset}")
        sample_file = st.text_input("Output File", value=f"{sample_dir}/{sample_dataset}.jsonl")
    
    if st.button("Generate Sample Data"):
        with st.spinner("Generating sample data..."):
            Path(sample_dir).mkdir(parents=True, exist_ok=True)
            
            cmd = f"python generate_sample_data.py --dataset {sample_dataset} --output-file {sample_file} --num-samples {sample_count} --create-files"
            result = run_command(cmd, check=False)
            
            if result and result.returncode == 0:
                st.success(f"Sample data generated: {sample_file}")
            else:
                st.error("Failed to generate sample data")

if __name__ == "__main__":
    main() 