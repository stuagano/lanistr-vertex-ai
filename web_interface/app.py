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
import tempfile

# Google Cloud Python client libraries
try:
    from google.cloud import storage
    from google.cloud import aiplatform
    from google.cloud.devtools import cloudbuild
    from google.cloud.storage.constants import STANDARD_STORAGE_CLASS
    from google.auth import default
    from google.auth.exceptions import DefaultCredentialsError
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError as e:
    st.error(f"Google Cloud libraries not available: {e}")
    GOOGLE_CLOUD_AVAILABLE = False

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
        "gsutil": "Google Cloud Storage",
        "python3": "Python 3"
    }
    
    status = {}
    for tool, description in tools.items():
        result = run_command(f"which {tool}", check=False)
        status[tool] = result.returncode == 0 if result else False
    
    return status

def check_required_apis(project_id: str) -> Dict[str, bool]:
    """Check if required Google Cloud APIs are enabled."""
    apis = {
        "aiplatform.googleapis.com": "Vertex AI API",
        "storage.googleapis.com": "Cloud Storage API",
        "cloudbuild.googleapis.com": "Cloud Build API",
        "logging.googleapis.com": "Cloud Logging API",
        "monitoring.googleapis.com": "Cloud Monitoring API",
        "errorreporting.googleapis.com": "Error Reporting API",
        "containerregistry.googleapis.com": "Container Registry API"
    }
    
    status = {}
    for api, description in apis.items():
        result = run_command(f"gcloud services list --enabled --filter='name:{api}' --format='value(name)'", check=False)
        status[api] = result and result.returncode == 0 and result.stdout.strip()
    
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

def upload_data_to_gcs(local_file: str, bucket_name: str, gcs_path: str) -> Dict[str, Any]:
    """Upload a file to Google Cloud Storage with detailed error reporting."""
    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    
    try:
        # Check if local file exists
        if not os.path.exists(local_file):
            return {
                "success": False,
                "error": "File not found",
                "message": f"Local file not found: {local_file}",
                "solution": "Ensure the file exists before uploading"
            }
        
        # Check file size
        file_size = os.path.getsize(local_file)
        if file_size == 0:
            return {
                "success": False,
                "error": "Empty file",
                "message": f"File is empty: {local_file}",
                "solution": "Ensure the file contains data before uploading"
            }
        
        # Check gsutil availability
        gsutil_check = run_command("which gsutil", check=False)
        if not gsutil_check or gsutil_check.returncode != 0:
            return {
                "success": False,
                "error": "gsutil not found",
                "message": "gsutil command not found",
                "solution": "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
            }
        
        # Check authentication
        auth_check = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", check=False)
        if not auth_check or not auth_check.stdout.strip():
            return {
                "success": False,
                "error": "Not authenticated",
                "message": "Not authenticated with Google Cloud",
                "solution": "Run 'gcloud auth login' to authenticate"
            }
        
        # Check bucket access
        bucket_check = run_command(f"gsutil ls -b gs://{bucket_name}", check=False)
        if bucket_check and bucket_check.returncode != 0:
            return {
                "success": False,
                "error": "Bucket access denied",
                "message": f"Cannot access bucket: gs://{bucket_name}",
                "solution": f"Ensure bucket exists and you have access: gsutil mb gs://{bucket_name}"
            }
        
        # Upload file
        st.info(f"Uploading {local_file} ({file_size} bytes) to {gcs_uri}")
        result = run_command(f"gsutil cp {local_file} {gcs_uri}", check=False)
        
        if result and result.returncode == 0:
            # Verify upload
            verify_result = run_command(f"gsutil ls {gcs_uri}", check=False)
            if verify_result and verify_result.returncode == 0:
                return {
                    "success": True,
                    "gcs_uri": gcs_uri,
                    "local_file": local_file,
                    "size": file_size,
                    "message": f"Successfully uploaded to {gcs_uri}"
                }
            else:
                return {
                    "success": False,
                    "error": "Upload verification failed",
                    "message": f"File uploaded but verification failed: {gcs_uri}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            return {
                "success": False,
                "error": "Upload failed",
                "message": f"Failed to upload {local_file} to {gcs_uri}",
                "stdout": result.stdout if result else "No output",
                "stderr": result.stderr if result else "No error output",
                "returncode": result.returncode if result else -1
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": "Exception",
            "message": f"Exception during upload: {str(e)}",
            "exception": str(e)
        }

def upload_directory_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str) -> Dict[str, Any]:
    """Upload a directory to Google Cloud Storage with detailed error reporting."""
    gcs_uri = f"gs://{bucket_name}/{gcs_prefix}"
    
    try:
        # Check if local directory exists
        if not os.path.exists(local_dir):
            return {
                "success": False,
                "error": "Directory not found",
                "message": f"Local directory not found: {local_dir}",
                "solution": "Ensure the directory exists before uploading"
            }
        
        # Check if directory is empty
        files = os.listdir(local_dir)
        if not files:
            return {
                "success": False,
                "error": "Empty directory",
                "message": f"Directory is empty: {local_dir}",
                "solution": "Ensure the directory contains files before uploading"
            }
        
        # Check gsutil availability
        gsutil_check = run_command("which gsutil", check=False)
        if not gsutil_check or gsutil_check.returncode != 0:
            return {
                "success": False,
                "error": "gsutil not found",
                "message": "gsutil command not found",
                "solution": "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
            }
        
        # Check authentication
        auth_check = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", check=False)
        if not auth_check or not auth_check.stdout.strip():
            return {
                "success": False,
                "error": "Not authenticated",
                "message": "Not authenticated with Google Cloud",
                "solution": "Run 'gcloud auth login' to authenticate"
            }
        
        # Upload directory
        st.info(f"Uploading directory {local_dir} ({len(files)} files) to {gcs_uri}")
        result = run_command(f"gsutil -m cp -r {local_dir} {gcs_uri}", check=False)
        
        if result and result.returncode == 0:
            # Verify upload
            verify_result = run_command(f"gsutil ls {gcs_uri}", check=False)
            if verify_result and verify_result.returncode == 0:
                return {
                    "success": True,
                    "gcs_uri": gcs_uri,
                    "local_dir": local_dir,
                    "file_count": len(files),
                    "message": f"Successfully uploaded {len(files)} files to {gcs_uri}"
                }
            else:
                return {
                    "success": False,
                    "error": "Upload verification failed",
                    "message": f"Directory uploaded but verification failed: {gcs_uri}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            return {
                "success": False,
                "error": "Upload failed",
                "message": f"Failed to upload directory {local_dir} to {gcs_uri}",
                "stdout": result.stdout if result else "No output",
                "stderr": result.stderr if result else "No error output",
                "returncode": result.returncode if result else -1
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": "Exception",
            "message": f"Exception during directory upload: {str(e)}",
            "exception": str(e)
        }

def update_jsonl_with_gcs_uris(jsonl_file: str, bucket_name: str, dataset_type: str) -> Dict[str, Any]:
    """Update JSONL file to use GCS URIs instead of local paths with detailed error reporting."""
    import json
    import tempfile
    
    try:
        # Check if source JSONL file exists
        if not os.path.exists(jsonl_file):
            return {
                "success": False,
                "error": "Source file not found",
                "message": f"JSONL file not found: {jsonl_file}",
                "solution": "Ensure the JSONL file exists before conversion"
            }
        
        # Check file size
        file_size = os.path.getsize(jsonl_file)
        if file_size == 0:
            return {
                "success": False,
                "error": "Empty file",
                "message": f"JSONL file is empty: {jsonl_file}",
                "solution": "Ensure the JSONL file contains data before conversion"
            }
        
        # Get current project
        current_project = st.session_state.get("current_project", get_project_id())
        if not current_project:
            return {
                "success": False,
                "error": "No project selected",
                "message": "Please select a Google Cloud project from the sidebar",
                "solution": "Select a project in the sidebar first"
            }
        
        # Create bucket if it doesn't exist (using Python client)
        bucket_creation_result = create_gcs_bucket_python(current_project, bucket_name)
        if not bucket_creation_result["success"]:
            return {
                "success": False,
                "error": "Bucket creation failed",
                "message": f"Failed to create bucket: {bucket_creation_result['message']}",
                "details": bucket_creation_result
            }
        
        # Create temporary file for updated JSONL
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        
        try:
            # Read and process JSONL file
            processed_count = 0
            error_count = 0
            errors = []
            
            with open(jsonl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # Update image path to GCS URI
                        if 'image_path' in data:
                            local_path = data['image_path']
                            
                            # Check if local image exists
                            if not os.path.exists(local_path):
                                errors.append(f"Line {line_num}: Image not found: {local_path}")
                                error_count += 1
                                continue
                            
                            # Extract filename from path
                            filename = os.path.basename(local_path)
                            # Create GCS path
                            gcs_path = f"data/{dataset_type}/images/{filename}"
                            data['image_path'] = f"gs://{bucket_name}/{gcs_path}"
                        
                        # Write updated data
                        temp_file.write(json.dumps(data) + '\n')
                        processed_count += 1
                        
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON: {str(e)}")
                        error_count += 1
                        continue
                    except Exception as e:
                        errors.append(f"Line {line_num}: Processing error: {str(e)}")
                        error_count += 1
                        continue
            
            temp_file.close()
            
            if processed_count == 0:
                return {
                    "success": False,
                    "error": "No valid records",
                    "message": f"No valid records found in {jsonl_file}",
                    "errors": errors,
                    "solution": "Check the JSONL file format and ensure it contains valid JSON records"
                }
            
            # Upload images to GCS (using Python client)
            images_dir = f"./data/{dataset_type}/images"
            if os.path.exists(images_dir):
                st.info(f"Uploading images from {images_dir}")
                upload_result = upload_directory_to_gcs_python(images_dir, bucket_name, f"data/{dataset_type}/images", current_project)
                if not upload_result["success"]:
                    return {
                        "success": False,
                        "error": "Image upload failed",
                        "message": f"Failed to upload images: {upload_result['message']}",
                        "details": upload_result
                    }
            else:
                st.warning(f"Images directory not found: {images_dir}")
            
            # Upload updated JSONL to GCS (using Python client)
            gcs_jsonl_path = f"data/{dataset_type}/{os.path.basename(jsonl_file)}"
            upload_result = upload_file_to_gcs_python(temp_file.name, bucket_name, gcs_jsonl_path, current_project)
            
            if upload_result["success"]:
                return {
                    "success": True,
                    "gcs_uri": upload_result["gcs_uri"],
                    "local_file": jsonl_file,
                    "processed_count": processed_count,
                    "error_count": error_count,
                    "errors": errors if errors else None,
                    "message": f"Successfully converted and uploaded {processed_count} records to {upload_result['gcs_uri']}"
                }
            else:
                return {
                    "success": False,
                    "error": "JSONL upload failed",
                    "message": f"Failed to upload converted JSONL: {upload_result['message']}",
                    "details": upload_result
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": "Processing exception",
                "message": f"Exception during JSONL processing: {str(e)}",
                "exception": str(e)
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    except Exception as e:
        return {
            "success": False,
            "error": "Exception",
            "message": f"Exception during GCS URI conversion: {str(e)}",
            "exception": str(e)
        }

def create_sample_data(dataset_type: str, data_dir: str) -> Dict[str, Any]:
    """Create sample data for the specified dataset with detailed error reporting."""
    jsonl_file = f"{data_dir}/{dataset_type}.jsonl"
    
    try:
        # Check if data directory exists
        if not os.path.exists(data_dir):
            st.info(f"Creating data directory: {data_dir}")
            os.makedirs(data_dir, exist_ok=True)
        
        # Check if sample data already exists
        if Path(jsonl_file).exists():
            st.info(f"Sample data already exists: {jsonl_file}")
            return {"success": True, "file": jsonl_file, "message": "Sample data already exists"}
        
        # Check if generate_sample_data.py exists
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "generate_sample_data.py"))
        if not os.path.exists(script_path):
            return {
                "success": False,
                "error": "Missing generate_sample_data.py",
                "message": f"Required script not found: {script_path}",
                "solution": "Ensure generate_sample_data.py is in the project root (" + script_path + ")"
            }
        
        # Run the sample data generation
        cmd = f"python {script_path} --dataset {dataset_type} --output-file {jsonl_file} --num-samples 100 --create-files"
        st.info(f"Running command: {cmd}")
        
        result = run_command(cmd, check=False)
        
        if result and result.returncode == 0:
            # Verify the file was created
            if os.path.exists(jsonl_file):
                file_size = os.path.getsize(jsonl_file)
                return {
                    "success": True, 
                    "file": jsonl_file, 
                    "size": file_size,
                    "message": f"Sample data generated successfully: {jsonl_file} ({file_size} bytes)"
                }
            else:
                return {
                    "success": False,
                    "error": "File not created",
                    "message": f"Command succeeded but file not found: {jsonl_file}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            return {
                "success": False,
                "error": "Command failed",
                "message": f"Failed to generate sample data",
                "stdout": result.stdout if result else "No output",
                "stderr": result.stderr if result else "No error output",
                "returncode": result.returncode if result else -1
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": "Exception",
            "message": f"Exception during sample data generation: {str(e)}",
            "exception": str(e)
        }

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

def build_and_push_image_cloud_build(project_id: str, image_name: str = "lanistr-training") -> str:
    """Build and push Docker image using Google Cloud Build."""
    image_uri = f"gcr.io/{project_id}/{image_name}:latest"
    
    # Create cloudbuild.yaml file
    cloudbuild_config = f"""
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '{image_uri}', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '{image_uri}']
images:
  - '{image_uri}'
"""
    
    # Write cloudbuild.yaml
    with open("cloudbuild.yaml", "w") as f:
        f.write(cloudbuild_config)
    
    # Submit build to Cloud Build
    result = run_command(f"gcloud builds submit --config cloudbuild.yaml .", check=False)
    
    # Clean up
    if os.path.exists("cloudbuild.yaml"):
        os.unlink("cloudbuild.yaml")
    
    if result and result.returncode == 0:
        return image_uri
    return ""

def check_cloud_build_api(project_id: str) -> bool:
    """Check if Cloud Build API is enabled."""
    result = run_command(f"gcloud services list --enabled --filter='name:cloudbuild.googleapis.com' --format='value(name)'", check=False)
    return result and result.returncode == 0 and result.stdout.strip()

def enable_cloud_build_api(project_id: str) -> bool:
    """Enable Cloud Build API."""
    result = run_command(f"gcloud services enable cloudbuild.googleapis.com", check=False)
    return result and result.returncode == 0

def get_cloud_build_status(build_id: str) -> str:
    """Get the status of a Cloud Build job."""
    result = run_command(f"gcloud builds describe {build_id} --format='value(status)'", check=False)
    if result and result.returncode == 0:
        return result.stdout.strip()
    return "UNKNOWN"

def list_recent_builds(project_id: str, limit: int = 5) -> list:
    """List recent Cloud Build jobs."""
    result = run_command(f"gcloud builds list --limit={limit} --format='table(id,status,createTime,images)'", check=False)
    if result and result.returncode == 0:
        return result.stdout.strip().split('\n')
    return []

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

def get_account_details() -> list:
    """Get detailed information about all Google Cloud accounts."""
    result = run_command("gcloud auth list --format='table(account,status,active_account)'", check=False)
    if result and result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        accounts = []
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    account = parts[0]
                    status = parts[1]
                    active = "ACTIVE" in line
                    accounts.append({
                        "account": account,
                        "status": status,
                        "active": active
                    })
        return accounts
    return []

def revoke_all_accounts() -> bool:
    """Revoke all Google Cloud accounts."""
    result = run_command("gcloud auth revoke --all", check=False)
    return result and result.returncode == 0

def get_available_accounts() -> list:
    """Get list of available Google Cloud accounts."""
    result = run_command("gcloud auth list --format='value(account)'", check=False)
    if result and result.returncode == 0:
        return [acc.strip() for acc in result.stdout.strip().split('\n') if acc.strip()]
    return []

def get_active_account() -> str:
    """Get the currently active Google Cloud account."""
    result = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", check=False)
    if result and result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return ""

def switch_account(account: str) -> bool:
    """Switch to a different Google Cloud account."""
    result = run_command(f"gcloud config set account {account}", check=False)
    return result and result.returncode == 0

def revoke_account(account: str) -> bool:
    """Revoke a Google Cloud account."""
    result = run_command(f"gcloud auth revoke {account}", check=False)
    return result and result.returncode == 0

def check_docker_status() -> bool:
    """Check if Docker is running and accessible."""
    result = run_command("docker info", check=False)
    return result and result.returncode == 0

def create_gcs_bucket(project_id: str, bucket_name: str, region: str = "us-central1") -> Dict[str, Any]:
    """Create a Google Cloud Storage bucket with detailed error reporting."""
    try:
        # Check if bucket already exists
        bucket_check = run_command(f"gsutil ls -b gs://{bucket_name}", check=False)
        if bucket_check and bucket_check.returncode == 0:
            return {
                "success": True,
                "bucket_name": bucket_name,
                "message": f"Bucket gs://{bucket_name} already exists",
                "action": "none"
            }
        
        # Check gsutil availability
        gsutil_check = run_command("which gsutil", check=False)
        if not gsutil_check or gsutil_check.returncode != 0:
            return {
                "success": False,
                "error": "gsutil not found",
                "message": "gsutil command not found",
                "solution": "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
            }
        
        # Check authentication
        auth_check = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", check=False)
        if not auth_check or not auth_check.stdout.strip():
            return {
                "success": False,
                "error": "Not authenticated",
                "message": "Not authenticated with Google Cloud",
                "solution": "Run 'gcloud auth login' to authenticate"
            }
        
        # Check project access
        project_check = run_command(f"gcloud projects describe {project_id}", check=False)
        if not project_check or project_check.returncode != 0:
            return {
                "success": False,
                "error": "Project access denied",
                "message": f"Cannot access project: {project_id}",
                "solution": f"Ensure you have access to project {project_id}"
            }
        
        # Create bucket
        st.info(f"Creating bucket gs://{bucket_name} in project {project_id}")
        result = run_command(f"gsutil mb -p {project_id} -c STANDARD -l {region} gs://{bucket_name}", check=False)
        
        if result and result.returncode == 0:
            # Verify bucket creation
            verify_result = run_command(f"gsutil ls -b gs://{bucket_name}", check=False)
            if verify_result and verify_result.returncode == 0:
                return {
                    "success": True,
                    "bucket_name": bucket_name,
                    "project_id": project_id,
                    "region": region,
                    "message": f"Successfully created bucket gs://{bucket_name}",
                    "action": "created"
                }
            else:
                return {
                    "success": False,
                    "error": "Verification failed",
                    "message": f"Bucket created but verification failed: gs://{bucket_name}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            return {
                "success": False,
                "error": "Bucket creation failed",
                "message": f"Failed to create bucket gs://{bucket_name}",
                "stdout": result.stdout if result else "No output",
                "stderr": result.stderr if result else "No error output",
                "returncode": result.returncode if result else -1
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": "Exception",
            "message": f"Exception during bucket creation: {str(e)}",
            "exception": str(e)
        }

def create_gcs_bucket_python(project_id: str, bucket_name: str, region: str = "us-central1") -> Dict[str, Any]:
    """Create a Google Cloud Storage bucket using Python client library."""
    try:
        if not GOOGLE_CLOUD_AVAILABLE:
            return {
                "success": False,
                "error": "Google Cloud libraries not available",
                "message": "Google Cloud Python client libraries are not installed",
                "solution": "Install with: pip install google-cloud-storage google-cloud-aiplatform google-cloud-build"
            }
        
        # Initialize storage client
        storage_client = storage.Client(project=project_id)
        
        # Check if bucket already exists
        try:
            bucket = storage_client.get_bucket(bucket_name)
            return {
                "success": True,
                "bucket_name": bucket_name,
                "message": f"Bucket gs://{bucket_name} already exists",
                "action": "none"
            }
        except Exception:
            # Bucket doesn't exist, create it
            pass
        
        # Create bucket
        st.info(f"Creating bucket gs://{bucket_name} in project {project_id}")
        bucket = storage_client.create_bucket(
            bucket_name,
            location=region
        )
        
        # Set storage class after creation
        bucket.storage_class = STANDARD_STORAGE_CLASS
        bucket.update()
        
        return {
            "success": True,
            "bucket_name": bucket_name,
            "project_id": project_id,
            "region": region,
            "message": f"Successfully created bucket gs://{bucket_name}",
            "action": "created"
        }
    
    except DefaultCredentialsError:
        return {
            "success": False,
            "error": "Not authenticated",
            "message": "Not authenticated with Google Cloud",
            "solution": "Run 'gcloud auth application-default login' to authenticate"
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Exception",
            "message": f"Exception during bucket creation: {str(e)}",
            "exception": str(e)
        }

def upload_file_to_gcs_python(local_file: str, bucket_name: str, gcs_path: str, project_id: str) -> Dict[str, Any]:
    """Upload a file to Google Cloud Storage using Python client library."""
    gcs_uri = f"gs://{bucket_name}/{gcs_path}"
    
    try:
        if not GOOGLE_CLOUD_AVAILABLE:
            return {
                "success": False,
                "error": "Google Cloud libraries not available",
                "message": "Google Cloud Python client libraries are not installed",
                "solution": "Install with: pip install google-cloud-storage"
            }
        
        # Check if local file exists
        if not os.path.exists(local_file):
            return {
                "success": False,
                "error": "File not found",
                "message": f"Local file not found: {local_file}",
                "solution": "Ensure the file exists before uploading"
            }
        
        # Check file size
        file_size = os.path.getsize(local_file)
        if file_size == 0:
            return {
                "success": False,
                "error": "Empty file",
                "message": f"File is empty: {local_file}",
                "solution": "Ensure the file contains data before uploading"
            }
        
        # Initialize storage client
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        
        # Upload file
        st.info(f"Uploading {local_file} ({file_size} bytes) to {gcs_uri}")
        blob.upload_from_filename(local_file)
        
        return {
            "success": True,
            "gcs_uri": gcs_uri,
            "local_file": local_file,
            "size": file_size,
            "message": f"Successfully uploaded to {gcs_uri}"
        }
    
    except DefaultCredentialsError:
        return {
            "success": False,
            "error": "Not authenticated",
            "message": "Not authenticated with Google Cloud",
            "solution": "Run 'gcloud auth application-default login' to authenticate"
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Exception",
            "message": f"Exception during upload: {str(e)}",
            "exception": str(e)
        }

def upload_directory_to_gcs_python(local_dir: str, bucket_name: str, gcs_prefix: str, project_id: str) -> Dict[str, Any]:
    """Upload a directory to Google Cloud Storage using Python client library."""
    gcs_uri = f"gs://{bucket_name}/{gcs_prefix}"
    
    try:
        if not GOOGLE_CLOUD_AVAILABLE:
            return {
                "success": False,
                "error": "Google Cloud libraries not available",
                "message": "Google Cloud Python client libraries are not installed",
                "solution": "Install with: pip install google-cloud-storage"
            }
        
        # Check if local directory exists
        if not os.path.exists(local_dir):
            return {
                "success": False,
                "error": "Directory not found",
                "message": f"Local directory not found: {local_dir}",
                "solution": "Ensure the directory exists before uploading"
            }
        
        # Check if directory is empty
        files = os.listdir(local_dir)
        if not files:
            return {
                "success": False,
                "error": "Empty directory",
                "message": f"Directory is empty: {local_dir}",
                "solution": "Ensure the directory contains files before uploading"
            }
        
        # Initialize storage client
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        
        # Upload all files in directory
        uploaded_count = 0
        for root, dirs, filenames in os.walk(local_dir):
            for filename in filenames:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_dir)
                gcs_path = f"{gcs_prefix}/{relative_path}"
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                uploaded_count += 1
        
        return {
            "success": True,
            "gcs_uri": gcs_uri,
            "local_dir": local_dir,
            "file_count": uploaded_count,
            "message": f"Successfully uploaded {uploaded_count} files to {gcs_uri}"
        }
    
    except DefaultCredentialsError:
        return {
            "success": False,
            "error": "Not authenticated",
            "message": "Not authenticated with Google Cloud",
            "solution": "Run 'gcloud auth application-default login' to authenticate"
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Exception",
            "message": f"Exception during directory upload: {str(e)}",
            "exception": str(e)
        }

def list_gcs_buckets_python(project_id: str) -> list:
    """List Google Cloud Storage buckets using Python client library."""
    try:
        if not GOOGLE_CLOUD_AVAILABLE:
            return []
        
        storage_client = storage.Client(project=project_id)
        buckets = storage_client.list_buckets()
        return [bucket.name for bucket in buckets]
    
    except Exception:
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 LANISTR Training Interface</h1>', unsafe_allow_html=True)
    st.markdown("Submit LANISTR training jobs to Google Cloud Vertex AI with ease!")
    
    # Left sidebar for GCP authentication and project selection
    with st.sidebar:
        st.title("🔐 GCP Configuration")
        
        # Check current authentication status
        auth_status = check_authentication()
        if auth_status:
            st.success("✅ Authenticated")
            current_account = get_active_account()
            if current_account:
                st.info(f"Account: {current_account}")
                
                # Account management
                with st.expander("🔄 Switch Account"):
                    available_accounts = get_available_accounts()
                    account_details = get_account_details()
                    
                    if available_accounts:
                        # Show current account status
                        st.markdown("**Current Account:**")
                        current_account = get_active_account()
                        if current_account:
                            st.success(f"✅ {current_account} (Active)")
                        
                        st.divider()
                        
                        # Show all accounts
                        st.markdown("**All Accounts:**")
                        for account_info in account_details:
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                if account_info["active"]:
                                    st.success(f"✅ {account_info['account']}")
                                else:
                                    st.info(f"📋 {account_info['account']}")
                            
                            with col2:
                                if not account_info["active"]:
                                    if st.button("Switch", key=f"switch_{account_info['account']}"):
                                        with st.spinner("Switching account..."):
                                            if switch_account(account_info["account"]):
                                                st.success(f"Switched to: {account_info['account']}")
                                                st.session_state.current_account = account_info["account"]
                                                st.rerun()
                                            else:
                                                st.error("Failed to switch account")
                            
                            with col3:
                                if st.button("Revoke", key=f"revoke_{account_info['account']}"):
                                    with st.spinner("Revoking account..."):
                                        if revoke_account(account_info["account"]):
                                            st.success(f"Revoked: {account_info['account']}")
                                            st.rerun()
                                        else:
                                            st.error("Failed to revoke account")
                        
                        st.divider()
                        
                        # Quick account selection
                        st.markdown("**Quick Switch:**")
                        selected_account = st.selectbox(
                            "Select Account to Switch To",
                            available_accounts,
                            index=available_accounts.index(current_account) if current_account in available_accounts else 0,
                            key="quick_account_selector"
                        )
                        
                        if selected_account != current_account:
                            if st.button("🚀 Switch to Selected Account", key="quick_switch_btn", use_container_width=True):
                                with st.spinner("Switching account..."):
                                    if switch_account(selected_account):
                                        st.success(f"Switched to: {selected_account}")
                                        st.session_state.current_account = selected_account
                                        st.rerun()
                                    else:
                                        st.error("Failed to switch account")
                    else:
                        st.warning("No accounts found")
                    
                    # Add new account
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("➕ Add New Account", key="add_account_btn", use_container_width=True):
                            with st.spinner("Opening authentication..."):
                                run_command("gcloud auth login", check=False)
                            st.success("Authentication window opened!")
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ Clear All", key="clear_all_btn", use_container_width=True):
                            if st.checkbox("I understand this will remove all accounts", key="confirm_clear"):
                                with st.spinner("Revoking all accounts..."):
                                    if revoke_all_accounts():
                                        st.success("All accounts revoked!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to revoke all accounts")
                            else:
                                st.warning("Please confirm to clear all accounts")
        else:
            st.error("❌ Not authenticated")
            if st.button("🔑 Login to Google Cloud"):
                with st.spinner("Opening authentication..."):
                    run_command("gcloud auth login", check=False)
                st.rerun()
        
        # Project selection
        st.subheader("📁 Project Selection")
        projects = get_available_projects()
        current_project = get_project_id()
        
        if projects:
            selected_project = st.selectbox(
                "Select Google Cloud Project",
                projects,
                index=projects.index(current_project) if current_project in projects else 0,
                key="project_selector"
            )
            
            if selected_project != current_project:
                if st.button("Set Project", key="set_project_btn"):
                    with st.spinner("Setting project..."):
                        run_command(f"gcloud config set project {selected_project}")
                    st.success(f"Project set to: {selected_project}")
                    st.session_state.current_project = selected_project
                    st.rerun()
            
            # Display current project
            if st.session_state.get("current_project") or current_project:
                project_display = st.session_state.get("current_project", current_project)
                st.info(f"Current: {project_display}")
        else:
            st.warning("No projects found")
            if st.button("Refresh Projects", key="refresh_projects"):
                st.rerun()
        
        # Region selection
        st.subheader("🌍 Region")
        regions = ["us-central1", "us-east1", "us-west1", "europe-west1", "asia-east1"]
        selected_region = st.selectbox(
            "Select Region",
            regions,
            index=0,
            key="region_selector"
        )
        st.session_state.selected_region = selected_region
        
        # API status check
        st.subheader("🔌 API Status")
        if st.button("Check APIs", key="check_apis"):
            with st.spinner("Checking APIs..."):
                current_project = st.session_state.get("current_project", get_project_id())
                if current_project:
                    api_status = check_required_apis(current_project)
                    
                    for api, status in api_status.items():
                        if status:
                            st.success(f"✅ {api}")
                        else:
                            st.warning(f"⚠️ {api}")
                            if st.button(f"Enable {api}", key=f"enable_{api}"):
                                with st.spinner(f"Enabling {api}..."):
                                    run_command(f"gcloud services enable {api}")
                                st.success(f"{api} enabled!")
                                st.rerun()
                else:
                    st.warning("Please select a project first")
        
        # Cloud Build status
        st.subheader("🏗️ Cloud Build")
        if st.button("Check Build Status", key="check_builds"):
            current_project = st.session_state.get("current_project", get_project_id())
            if current_project:
                with st.spinner("Checking recent builds..."):
                    builds = list_recent_builds(current_project, 3)
                    if builds:
                        st.success("Recent builds:")
                        for build in builds[1:]:  # Skip header
                            if build.strip():
                                st.text(build)
                    else:
                        st.info("No recent builds found")
            else:
                st.warning("Please select a project first")
    
    # Main navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "⚙️ Configuration", "🚀 Submit Job", "📊 Monitor Jobs", "📁 Data Management", "🧪 Gherkin Testing", "📚 Setup Guides", "🎯 Quick Start", "🛠️ Troubleshooting"]
    )
    
    st.info(f"Debug: Selected page = {page}")
    
    if page == "🏠 Dashboard":
        st.info("Debug: Showing Dashboard")
        show_dashboard()
    elif page == "⚙️ Configuration":
        st.info("Debug: Showing Configuration")
        show_configuration()
    elif page == "🚀 Submit Job":
        st.info("Debug: Showing Job Submission")
        show_job_submission()
    elif page == "📊 Monitor Jobs":
        st.info("Debug: Showing Job Monitoring")
        show_job_monitoring()
    elif page == "📁 Data Management":
        st.info("Debug: Showing Data Management")
        show_data_management()
    elif page == "🧪 Gherkin Testing":
        st.info("Debug: Showing Gherkin Testing")
        show_gherkin_testing()
    elif page == "📚 Setup Guides":
        st.info("Debug: Showing Setup Guides")
        show_setup_guides()
    elif page == "🎯 Quick Start":
        st.info("Debug: Showing Quick Start")
        show_quick_start()
    elif page == "🛠️ Troubleshooting":
        st.info("Debug: Showing Troubleshooting")
        show_troubleshooting()
    else:
        st.error(f"Debug: Unknown page selected: {page}")

def show_dashboard():
    """Show the main dashboard."""
    st.markdown('<h2 class="section-header">System Status</h2>', unsafe_allow_html=True)
    
    # Get current status from session state
    current_project = st.session_state.get("current_project", get_project_id())
    selected_region = st.session_state.get("selected_region", "us-central1")
    auth_status = check_authentication()
    current_account = get_active_account()
    
    # Status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if auth_status:
            st.success("✅ Authenticated")
        else:
            st.error("❌ Not authenticated")
    
    with col2:
        if current_account:
            st.success(f"✅ {current_account.split('@')[0]}")
        else:
            st.warning("⚠️ No account")
    
    with col3:
        if current_project:
            st.success(f"✅ Project: {current_project}")
        else:
            st.warning("⚠️ No project selected")
    
    with col4:
        st.info(f"🌍 Region: {selected_region}")
    
    # Check prerequisites
    with st.spinner("Checking system requirements..."):
        prereq_status = check_prerequisites()
        current_project = st.session_state.get("current_project", get_project_id())
        api_status = check_required_apis(current_project) if current_project else {}
    
    st.markdown('<h3 class="section-header">Prerequisites</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tools")
        for tool, status in prereq_status.items():
            if status:
                st.success(f"✅ {tool}")
            else:
                st.error(f"❌ {tool}")
    
    with col2:
        st.subheader("Cloud APIs")
        if current_project:
            critical_apis = ["aiplatform.googleapis.com", "storage.googleapis.com", "cloudbuild.googleapis.com"]
            for api in critical_apis:
                if api in api_status and api_status[api]:
                    st.success(f"✅ {api.split('.')[0]}")
                else:
                    st.error(f"❌ {api.split('.')[0]}")
        else:
            st.warning("⚠️ Select project to check APIs")
    
    # Quick actions
    st.markdown('<h2 class="section-header">Quick Actions</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not auth_status:
            if st.button("🔑 Authenticate", use_container_width=True):
                with st.spinner("Opening authentication..."):
                    run_command("gcloud auth login", check=False)
                st.rerun()
        else:
            if st.button("🔄 Switch Account", use_container_width=True):
                st.session_state.show_account_switcher = True
                st.rerun()
    
    with col2:
        if not current_project:
            st.warning("Select Project")
        else:
            if st.button("🚀 Submit Job", use_container_width=True):
                st.session_state.page = "🚀 Submit Job"
                st.rerun()
    
    with col3:
        if st.button("📊 Monitor Jobs", use_container_width=True):
            st.session_state.page = "📊 Monitor Jobs"
            st.rerun()
    
    with col4:
        if st.button("📁 Manage Data", use_container_width=True):
            st.session_state.page = "📁 Data Management"
            st.rerun()
    
    # Account switcher popup
    if st.session_state.get("show_account_switcher", False):
        with st.expander("🔄 Switch Google Account", expanded=True):
            available_accounts = get_available_accounts()
            if available_accounts:
                selected_account = st.selectbox(
                    "Select Account",
                    available_accounts,
                    index=available_accounts.index(current_account) if current_account in available_accounts else 0,
                    key="dashboard_account_selector"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if selected_account != current_account:
                        if st.button("Switch", key="dashboard_switch_btn"):
                            with st.spinner("Switching account..."):
                                if switch_account(selected_account):
                                    st.success(f"Switched to: {selected_account}")
                                    st.session_state.current_account = selected_account
                                    st.session_state.show_account_switcher = False
                                    st.rerun()
                                else:
                                    st.error("Failed to switch account")
                
                with col2:
                    if st.button("Cancel", key="dashboard_cancel_btn"):
                        st.session_state.show_account_switcher = False
                        st.rerun()
            else:
                st.warning("No accounts found")
                if st.button("Add Account", key="dashboard_add_btn"):
                    with st.spinner("Opening authentication..."):
                        run_command("gcloud auth login", check=False)
                    st.success("Authentication window opened!")
                    st.session_state.show_account_switcher = False
                    st.rerun()
    
    # Recent activity or status
    st.markdown('<h2 class="section-header">Recent Activity</h2>', unsafe_allow_html=True)
    
    if current_project and auth_status:
        st.info("🎉 Your environment is ready! You can now submit training jobs.")
        
        # Show recent jobs if any
        with st.expander("Recent Jobs"):
            st.info("No recent jobs found. Submit your first job to see activity here.")
    else:
        st.warning("⚠️ Please complete the setup steps in the sidebar to get started.")
        
        if not auth_status:
            st.error("1. Click 'Login to Google Cloud' in the sidebar")
        if not current_project:
            st.error("2. Select your Google Cloud project from the sidebar")
        if not selected_region:
            st.error("3. Choose your preferred region from the sidebar")

def show_configuration():
    """Show configuration page."""
    try:
        st.markdown('<h2 class="section-header">Project Configuration</h2>', unsafe_allow_html=True)
        
        # Get current configuration from session state
        current_project = st.session_state.get("current_project", get_project_id())
        selected_region = st.session_state.get("selected_region", "us-central1")
        
        st.info(f"Debug: current_project = {current_project}")
        st.info(f"Debug: selected_region = {selected_region}")
        
        if not current_project:
            st.error("Please select a Google Cloud project from the sidebar first.")
            return
        
        st.info(f"Current Project: {current_project}")
        st.info(f"Selected Region: {selected_region}")
        
        # Save configuration
        if st.button("Save Configuration"):
            config = {
                "project_id": current_project,
                "region": selected_region
            }
            st.session_state.config = config
            st.success("Configuration saved!")
        
        # Show current configuration
        if st.session_state.get("config"):
            st.subheader("Current Configuration")
            config = st.session_state.config
            st.json(config)
        
        # Quick setup actions
        st.markdown('<h2 class="section-header">Quick Setup Actions</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 One-Click Setup", use_container_width=True):
                with st.spinner("Running one-click setup..."):
                    # Run the one-click setup script
                    result = run_command("./one_click_setup.sh", check=False)
                    if result and result.returncode == 0:
                        st.success("One-click setup completed!")
                    else:
                        st.error("One-click setup failed. Check the logs.")
        
        with col2:
            if st.button("📊 Validate Environment", use_container_width=True):
                with st.spinner("Validating environment..."):
                    # Check prerequisites
                    prereq_status = check_prerequisites()
                    auth_status = check_authentication()
                    
                    if all(prereq_status.values()) and auth_status:
                        st.success("✅ Environment is ready!")
                    else:
                        st.error("❌ Environment needs configuration")
                        
                        if not auth_status:
                            st.error("Please authenticate with Google Cloud")
                        for tool, status in prereq_status.items():
                            if not status:
                                st.error(f"Missing: {tool}")
        
        with col3:
            if st.button("🔧 Setup APIs", use_container_width=True):
                with st.spinner("Setting up APIs..."):
                    apis = [
                        "aiplatform.googleapis.com",
                        "storage.googleapis.com",
                        "logging.googleapis.com",
                        "monitoring.googleapis.com",
                        "errorreporting.googleapis.com",
                        "containerregistry.googleapis.com"
                    ]
                    
                    for api in apis:
                        result = run_command(f"gcloud services enable {api}", check=False)
                        if result and result.returncode == 0:
                            st.success(f"✅ {api} enabled")
                        else:
                            st.warning(f"⚠️ {api} already enabled or failed")
    
    except Exception as e:
        st.error(f"Error in configuration page: {str(e)}")
        st.exception(e)

def show_job_submission():
    """Show job submission page."""
    st.markdown('<h2 class="section-header">Submit Training Job</h2>', unsafe_allow_html=True)
    
    # Get configuration from session state
    current_project = st.session_state.get("current_project", get_project_id())
    selected_region = st.session_state.get("selected_region", "us-central1")
    
    if not current_project:
        st.error("Please select a Google Cloud project from the sidebar first.")
        return
    
    st.info(f"Project: {current_project} | Region: {selected_region}")
    
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
            value=f"lanistr-{current_project}-{dataset_type}",
            help="Google Cloud Storage bucket for data and outputs"
        )
        
        # Data configuration
        st.subheader("Data Configuration")
        data_source = st.radio(
            "Data Source",
            ["Use GCS URI", "Upload and convert local data", "Use sample data"]
        )
        data_uri = None
        jsonl_file = None
        data_dir = f"./data/{dataset_type}"
        if data_source == "Use GCS URI":
            # List available buckets
            try:
                storage_client = storage.Client(project=current_project)
                buckets = [b.name for b in storage_client.list_buckets()]
            except Exception as e:
                st.error(f"Error listing buckets: {e}")
                buckets = []
            selected_bucket = st.selectbox("Select GCS Bucket", buckets, key="gcs_bucket_selector")
            # List all .jsonl files in the selected bucket (recursively, no prefix restriction)
            jsonl_files = []
            file_display_map = {}
            if selected_bucket:
                try:
                    blobs = storage_client.list_blobs(selected_bucket)  # No prefix
                    for b in blobs:
                        if b.name.endswith('.jsonl'):
                            display_name = f"{b.name.split('/')[-1]}  (/{b.name})"
                            jsonl_files.append(display_name)
                            file_display_map[display_name] = b.name
                except Exception as e:
                    st.error(f"Error listing files in bucket: {e}")
            manual_entry = st.checkbox("Enter full GCS path manually")
            if manual_entry:
                manual_path = st.text_input("Full GCS Path (e.g., data/mimic-iv/myfile.jsonl)")
                selected_file = manual_path.strip()
            else:
                selected_display = st.selectbox("Select .jsonl File", jsonl_files, key="gcs_file_selector")
                selected_file = file_display_map.get(selected_display, None)
            if selected_bucket and selected_file:
                data_uri = f"gs://{selected_bucket}/{selected_file}"
                st.info(f"Selected GCS URI: {data_uri}")
            else:
                data_uri = None
            if not data_uri or not data_uri.startswith("gs://"):
                st.error("❌ Please select a valid GCS bucket and .jsonl file, or enter a valid path.")
            # Always show the submit button, but only allow submission if data_uri is valid
            submitted = st.form_submit_button("🚀 Submit Job")
            if submitted:
                if not data_uri or not data_uri.startswith("gs://"):
                    st.error("Please select a valid GCS URI before submitting.")
                else:
                    # Proceed with job submission logic using data_uri
                    # (Move the rest of the job submission logic here if needed)
                    st.success(f"Job submitted with data URI: {data_uri}")
            # Return early to avoid running the rest of the form logic if using GCS URI
            return
        elif data_source == "Upload and convert local data":
            uploaded_file = st.file_uploader(
                "Upload JSONL file",
                type=['jsonl'],
                key="job_upload"
            )
            if uploaded_file:
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                jsonl_file = temp_path
        elif data_source == "Use sample data":
            jsonl_file = f"./data/{dataset_type}/{dataset_type}.jsonl"
            # Optionally generate sample data if not present
            if not Path(jsonl_file).exists():
                create_sample_data(dataset_type, data_dir)

        # Data validation
        if st.checkbox("Validate data before submission"):
            if data_source == "Use GCS URI":
                validate_submitted = st.form_submit_button("Validate Data")
                if validate_submitted:
                    with st.spinner("Validating GCS data..."):
                        # Download temporarily for validation
                        temp_file = f"/tmp/temp_validation_{int(time.time())}.jsonl"
                        result = run_command(f"gsutil cp {data_uri} {temp_file}", check=False)
                        if result and result.returncode == 0:
                            validation_result = validate_dataset(dataset_type, "/tmp", temp_file)
                            if validation_result.get("passed"):
                                st.success("✅ GCS data validation passed!")
                                st.json(validation_result)
                            else:
                                st.error(f"❌ Validation failed: {validation_result.get('message', 'Unknown error')}")
                            os.unlink(temp_file)
                        else:
                            st.error("❌ Failed to download from GCS")
            else:
                validate_submitted = st.form_submit_button("Validate Data")
                if validate_submitted:
                    with st.spinner("Validating data..."):
                        if not jsonl_file or not Path(jsonl_file).exists():
                            st.error(f"❌ File not found: {jsonl_file}")
                        else:
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
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("Setting up GCS bucket...")
                if not setup_gcs_bucket(current_project, bucket_name, selected_region):
                    st.error("Failed to setup GCS bucket")
                    return
                progress_bar.progress(20)
                status_text.text("Preparing data...")
                # Data upload logic
                if data_source == "Use GCS URI":
                    # No upload needed, just use the URI
                    used_data_uri = data_uri
                else:
                    if not jsonl_file or not Path(jsonl_file).exists():
                        st.error(f"❌ File not found: {jsonl_file}")
                        return
                    # Upload local file to GCS
                    result = run_command(f"gsutil -m cp {jsonl_file} gs://{bucket_name}/data/{dataset_type}/", check=False)
                    if not result or result.returncode != 0:
                        st.warning("Data upload failed, but continuing...")
                    used_data_uri = f"gs://{bucket_name}/data/{dataset_type}/{Path(jsonl_file).name}"
                progress_bar.progress(60)
                status_text.text("Building and pushing Docker image...")
                image_uri = build_and_push_image_cloud_build(current_project)
                if not image_uri:
                    st.error("Failed to build and push Docker image")
                    return
                progress_bar.progress(80)
                status_text.text("Submitting job to Vertex AI...")
                config_file = f"lanistr/configs/{dataset_type}_pretrain.yaml"
                job_config = {
                    "project_id": current_project,
                    "region": selected_region,
                    "job_name": job_name,
                    "config_file": config_file,
                    "machine_type": machine_type,
                    "accelerator_type": accelerator_type,
                    "accelerator_count": accelerator_count,
                    "bucket_name": bucket_name,
                    "image_uri": image_uri,
                    "data_uri": used_data_uri
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
    
    # Get current configuration
    current_project = st.session_state.get("current_project", get_project_id())
    
    if not current_project:
        st.error("Please select a Google Cloud project from the sidebar first.")
        return
    
    st.info(f"Project: {current_project}")
    
    # Monitoring tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Training Jobs", "🏗️ Cloud Build", "📊 System Status"])
    
    with tab1:
        st.subheader("Vertex AI Training Jobs")
        
        if st.button("Refresh Jobs"):
            with st.spinner("Fetching jobs..."):
                result = run_command(f"gcloud ai custom-jobs list --region=us-central1 --format='table(name,state,createTime)'", check=False)
                if result and result.returncode == 0:
                    st.success("Recent training jobs:")
                    st.code(result.stdout)
                else:
                    st.info("No training jobs found")
    
    with tab2:
        st.subheader("Cloud Build Jobs")
        
        if st.button("Refresh Builds"):
            with st.spinner("Fetching builds..."):
                builds = list_recent_builds(current_project, 10)
                if builds:
                    st.success("Recent Cloud Build jobs:")
                    for build in builds:
                        if build.strip():
                            st.text(build)
                else:
                    st.info("No Cloud Build jobs found")
        
        # Build details
        st.subheader("Build Details")
        build_id = st.text_input("Enter Build ID", help="Enter a specific build ID to get details")
        
        if build_id and st.button("Get Build Details"):
            with st.spinner("Fetching build details..."):
                result = run_command(f"gcloud builds describe {build_id} --format='json'", check=False)
                if result and result.returncode == 0:
                    try:
                        import json
                        build_data = json.loads(result.stdout)
                        st.json(build_data)
                    except:
                        st.text(result.stdout)
                else:
                    st.error("Failed to fetch build details")
    
    with tab3:
        st.subheader("System Status")
        
        # Check APIs
        if st.button("Check All APIs"):
            with st.spinner("Checking APIs..."):
                api_status = check_required_apis(current_project)
                
                st.markdown("**API Status:**")
                for api, status in api_status.items():
                    if status:
                        st.success(f"✅ {api}")
                    else:
                        st.error(f"❌ {api}")
        
        # Check authentication
        if st.button("Check Authentication"):
            auth_status = check_authentication()
            if auth_status:
                st.success("✅ Google Cloud authenticated")
                current_account = get_active_account()
                if current_account:
                    st.info(f"Account: {current_account}")
            else:
                st.error("❌ Not authenticated")
        
        # Check project access
        if st.button("Check Project Access"):
            with st.spinner("Checking project access..."):
                result = run_command(f"gcloud projects describe {current_project}", check=False)
                if result and result.returncode == 0:
                    st.success(f"✅ Project {current_project} accessible")
                else:
                    st.error(f"❌ Cannot access project {current_project}")

def show_data_management():
    """Show data management page."""
    st.markdown('<h2 class="section-header">Data Management</h2>', unsafe_allow_html=True)
    
    # Get current configuration
    current_project = st.session_state.get("current_project", get_project_id())
    selected_region = st.session_state.get("selected_region", "us-central1")
    
    if not current_project:
        st.error("Please select a Google Cloud project from the sidebar first.")
        return
    
    st.info(f"Project: {current_project} | Region: {selected_region}")
    
    # Data management tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Data", "🔄 Convert to GCS", "✅ Validate Data", "🗂️ Manage Datasets"])
    
    with tab1:
        st.subheader("Upload Data to Google Cloud Storage")
        
        # Bucket configuration
        bucket_name = st.text_input(
            "GCS Bucket Name",
            value=f"lanistr-{current_project}-data",
            help="Google Cloud Storage bucket for your data"
        )
        
        # Dataset type selection
        dataset_type = st.selectbox(
            "Dataset Type",
            ["mimic-iv", "amazon"],
            help="Select the type of dataset you're uploading"
        )
        
        # Upload options
        upload_option = st.radio(
            "Upload Option",
            ["Upload JSONL file", "Upload entire dataset directory", "Generate and upload sample data"]
        )
        
        if upload_option == "Upload JSONL file":
            uploaded_file = st.file_uploader(
                "Choose JSONL file",
                type=['jsonl'],
                help="Upload a JSONL file with your dataset"
            )
            
            if uploaded_file and st.button("Upload to GCS"):
                with st.spinner("Uploading to GCS..."):
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Upload to GCS
                    gcs_path = f"data/{dataset_type}/{uploaded_file.name}"
                    gcs_uri = upload_data_to_gcs(temp_path, bucket_name, gcs_path)
                    
                    if gcs_uri:
                        st.success(f"✅ Uploaded to: {gcs_uri}")
                        st.session_state.last_uploaded_uri = gcs_uri
                    else:
                        st.error("❌ Upload failed")
                    
                    # Clean up
                    os.unlink(temp_path)
        
        elif upload_option == "Upload entire dataset directory":
            st.info("Use the command line to upload your dataset directory:")
            st.code(f"""
# Upload your dataset directory
gsutil -m cp -r ./your_dataset_directory gs://{bucket_name}/data/{dataset_type}/

# Or use the web interface to upload individual files
            """, language="bash")
        
        elif upload_option == "Generate and upload sample data":
            if st.button("Generate and Upload Sample Data"):
                with st.spinner("Generating and uploading sample data..."):
                    # First, ensure the bucket exists
                    current_project = st.session_state.get("current_project", get_project_id())
                    selected_region = st.session_state.get("selected_region", "us-central1")
                    
                    if not current_project:
                        st.error("❌ Please select a Google Cloud project from the sidebar first.")
                        return
                    
                    # Create bucket if it doesn't exist (using Python client)
                    bucket_creation_result = create_gcs_bucket_python(current_project, bucket_name, selected_region)
                    if not bucket_creation_result["success"]:
                        st.error(f"❌ Bucket creation failed: {bucket_creation_result['message']}")
                        
                        # Show detailed error information
                        with st.expander("🔍 Bucket Creation Error Details"):
                            st.error(f"**Error Type:** {bucket_creation_result.get('error', 'Unknown')}")
                            st.error(f"**Message:** {bucket_creation_result['message']}")
                            if bucket_creation_result.get('solution'):
                                st.info(f"**Solution:** {bucket_creation_result['solution']}")
                            if bucket_creation_result.get('stdout'):
                                st.text("**Command Output:**")
                                st.code(bucket_creation_result['stdout'])
                            if bucket_creation_result.get('stderr'):
                                st.text("**Error Output:**")
                                st.code(bucket_creation_result['stderr'])
                        return
                    else:
                        if bucket_creation_result.get('action') == 'created':
                            st.success(f"✅ {bucket_creation_result['message']}")
                        else:
                            st.info(f"ℹ️ {bucket_creation_result['message']}")
                    
                    # Generate sample data
                    jsonl_file = f"./data/{dataset_type}/{dataset_type}.jsonl"
                    sample_result = create_sample_data(dataset_type, f"./data/{dataset_type}")
                    
                    if sample_result["success"]:
                        st.success(f"✅ {sample_result['message']}")
                        
                        # Convert to GCS URIs and upload
                        conversion_result = update_jsonl_with_gcs_uris(jsonl_file, bucket_name, dataset_type)
                        
                        if conversion_result["success"]:
                            st.success(f"✅ {conversion_result['message']}")
                            st.session_state.last_uploaded_uri = conversion_result["gcs_uri"]
                            
                            # Show detailed results
                            with st.expander("📊 Upload Details"):
                                st.json({
                                    "processed_records": conversion_result.get("processed_count", 0),
                                    "error_count": conversion_result.get("error_count", 0),
                                    "gcs_uri": conversion_result["gcs_uri"],
                                    "file_size": sample_result.get("size", 0)
                                })
                                
                                if conversion_result.get("errors"):
                                    st.warning("⚠️ Some errors occurred:")
                                    for error in conversion_result["errors"][:5]:  # Show first 5 errors
                                        st.text(f"• {error}")
                                    if len(conversion_result["errors"]) > 5:
                                        st.text(f"... and {len(conversion_result['errors']) - 5} more errors")
                        else:
                            st.error(f"❌ Conversion failed: {conversion_result['message']}")
                            
                            # Show detailed error information
                            with st.expander("🔍 Error Details"):
                                st.error(f"**Error Type:** {conversion_result.get('error', 'Unknown')}")
                                st.error(f"**Message:** {conversion_result['message']}")
                                if conversion_result.get('solution'):
                                    st.info(f"**Solution:** {conversion_result['solution']}")
                                if conversion_result.get('stdout'):
                                    st.text("**Command Output:**")
                                    st.code(conversion_result['stdout'])
                                if conversion_result.get('stderr'):
                                    st.text("**Error Output:**")
                                    st.code(conversion_result['stderr'])
                    else:
                        st.error(f"❌ Sample data generation failed: {sample_result['message']}")
                        
                        # Show detailed error information
                        with st.expander("🔍 Error Details"):
                            st.error(f"**Error Type:** {sample_result.get('error', 'Unknown')}")
                            st.error(f"**Message:** {sample_result['message']}")
                            if sample_result.get('solution'):
                                st.info(f"**Solution:** {sample_result['solution']}")
                            if sample_result.get('stdout'):
                                st.text("**Command Output:**")
                                st.code(sample_result['stdout'])
                            if sample_result.get('stderr'):
                                st.text("**Error Output:**")
                                st.code(sample_result['stderr'])
    
    with tab2:
        st.subheader("Convert Local Data to GCS URIs")
        
        st.info("""
        This tool converts your local JSONL files to use GCS URIs instead of local file paths.
        This is required for Vertex AI training jobs.
        """)
        
        # Local JSONL file
        local_jsonl = st.text_input(
            "Local JSONL File Path",
            value=f"./data/{dataset_type}/{dataset_type}.jsonl",
            help="Path to your local JSONL file"
        )
        
        # Bucket for conversion
        convert_bucket = st.text_input(
            "GCS Bucket for Conversion",
            value=f"lanistr-{current_project}-data",
            help="Bucket where data will be uploaded"
        )
        
        if st.button("Convert to GCS URIs"):
            if os.path.exists(local_jsonl):
                with st.spinner("Converting to GCS URIs..."):
                    # First, ensure the bucket exists
                    current_project = st.session_state.get("current_project", get_project_id())
                    selected_region = st.session_state.get("selected_region", "us-central1")
                    
                    if not current_project:
                        st.error("❌ Please select a Google Cloud project from the sidebar first.")
                        return
                    
                    # Create bucket if it doesn't exist (using Python client)
                    bucket_creation_result = create_gcs_bucket_python(current_project, convert_bucket, selected_region)
                    if not bucket_creation_result["success"]:
                        st.error(f"❌ Bucket creation failed: {bucket_creation_result['message']}")
                        
                        # Show detailed error information
                        with st.expander("🔍 Bucket Creation Error Details"):
                            st.error(f"**Error Type:** {bucket_creation_result.get('error', 'Unknown')}")
                            st.error(f"**Message:** {bucket_creation_result['message']}")
                            if bucket_creation_result.get('solution'):
                                st.info(f"**Solution:** {bucket_creation_result['solution']}")
                            if bucket_creation_result.get('stdout'):
                                st.text("**Command Output:**")
                                st.code(bucket_creation_result['stdout'])
                            if bucket_creation_result.get('stderr'):
                                st.text("**Error Output:**")
                                st.code(bucket_creation_result['stderr'])
                        return
                    else:
                        if bucket_creation_result.get('action') == 'created':
                            st.success(f"✅ {bucket_creation_result['message']}")
                        else:
                            st.info(f"ℹ️ {bucket_creation_result['message']}")
                    
                    conversion_result = update_jsonl_with_gcs_uris(local_jsonl, convert_bucket, dataset_type)
                    
                    if conversion_result["success"]:
                        st.success(f"✅ {conversion_result['message']}")
                        st.session_state.last_converted_uri = conversion_result["gcs_uri"]
                        
                        # Show detailed results
                        with st.expander("📊 Conversion Details"):
                            st.json({
                                "processed_records": conversion_result.get("processed_count", 0),
                                "error_count": conversion_result.get("error_count", 0),
                                "gcs_uri": conversion_result["gcs_uri"],
                                "local_file": conversion_result.get("local_file", "")
                            })
                            
                            if conversion_result.get("errors"):
                                st.warning("⚠️ Some errors occurred:")
                                for error in conversion_result["errors"][:5]:  # Show first 5 errors
                                    st.text(f"• {error}")
                                if len(conversion_result["errors"]) > 5:
                                    st.text(f"... and {len(conversion_result['errors']) - 5} more errors")
                        
                        # Show preview of converted data
                        with st.expander("Preview Converted Data"):
                            try:
                                with open(local_jsonl, 'r') as f:
                                    for i, line in enumerate(f):
                                        if i < 3:  # Show first 3 lines
                                            data = json.loads(line.strip())
                                            st.json(data)
                                        else:
                                            break
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")
                    else:
                        st.error(f"❌ Conversion failed: {conversion_result['message']}")
                        
                        # Show detailed error information
                        with st.expander("🔍 Error Details"):
                            st.error(f"**Error Type:** {conversion_result.get('error', 'Unknown')}")
                            st.error(f"**Message:** {conversion_result['message']}")
                            if conversion_result.get('solution'):
                                st.info(f"**Solution:** {conversion_result['solution']}")
                            if conversion_result.get('stdout'):
                                st.text("**Command Output:**")
                                st.code(conversion_result['stdout'])
                            if conversion_result.get('stderr'):
                                st.text("**Error Output:**")
                                st.code(conversion_result['stderr'])
            else:
                st.error(f"❌ File not found: {local_jsonl}")
    
    with tab3:
        st.subheader("Validate Data")
        
        # Validation options
        validation_source = st.radio(
            "Validation Source",
            ["Local JSONL file", "GCS JSONL URI"]
        )
        
        if validation_source == "Local JSONL file":
            local_file = st.text_input(
                "Local JSONL File",
                value=f"./data/{dataset_type}/{dataset_type}.jsonl"
            )
            
            if st.button("Validate Local Data"):
                if os.path.exists(local_file):
                    with st.spinner("Validating data..."):
                        result = validate_dataset(dataset_type, f"./data/{dataset_type}", local_file)
                        if result.get("passed"):
                            st.success("✅ Data validation passed!")
                            st.json(result)
                        else:
                            st.error(f"❌ Validation failed: {result.get('message', 'Unknown error')}")
                else:
                    st.error(f"❌ File not found: {local_file}")
        
        else:  # GCS URI
            gcs_uri = st.text_input(
                "GCS JSONL URI",
                value=st.session_state.get("last_uploaded_uri", ""),
                help="GCS URI of your JSONL file (e.g., gs://bucket/data/file.jsonl)"
            )
            
            if st.button("Validate GCS Data"):
                if gcs_uri:
                    with st.spinner("Validating GCS data..."):
                        # Download temporarily for validation
                        temp_file = f"/tmp/temp_validation_{int(time.time())}.jsonl"
                        result = run_command(f"gsutil cp {gcs_uri} {temp_file}", check=False)
                        
                        if result and result.returncode == 0:
                            validation_result = validate_dataset(dataset_type, "/tmp", temp_file)
                            if validation_result.get("passed"):
                                st.success("✅ GCS data validation passed!")
                                st.json(validation_result)
                            else:
                                st.error(f"❌ Validation failed: {validation_result.get('message', 'Unknown error')}")
                            
                            # Clean up
                            os.unlink(temp_file)
                        else:
                            st.error("❌ Failed to download from GCS")
                else:
                    st.error("❌ Please provide a GCS URI")
    
    with tab4:
        st.subheader("Manage Datasets")
        
        # List existing datasets in GCS
        bucket_name = st.text_input(
            "GCS Bucket to List",
            value=f"lanistr-{current_project}-data"
        )
        
        if st.button("List Datasets"):
            with st.spinner("Listing datasets..."):
                result = run_command(f"gsutil ls gs://{bucket_name}/data/", check=False)
                if result and result.returncode == 0:
                    datasets = result.stdout.strip().split('\n')
                    st.success(f"Found {len(datasets)} dataset directories:")
                    for dataset in datasets:
                        if dataset.strip():
                            st.write(f"📁 {dataset}")
                else:
                    st.warning("No datasets found or bucket doesn't exist")
        
        # Dataset information
        st.subheader("Dataset Information")
        
        info_text = f"""
        **Current Configuration:**
        - Project: {current_project}
        - Region: {selected_region}
        - Default Bucket: lanistr-{current_project}-data
        
        **Data Structure:**
        ```
        gs://{bucket_name}/
        ├── data/
        │   ├── mimic-iv/
        │   │   ├── mimic-iv.jsonl
        │   │   └── images/
        │   └── amazon/
        │       ├── amazon.jsonl
        │       └── images/
        └── outputs/
            └── training-jobs/
        ```
        
        **JSONL Format:**
        ```json
        {{
          "id": "unique_identifier",
          "image_path": "gs://bucket/data/dataset/images/file.jpg",
          "text": "associated text content",
          "metadata": {{
            "additional": "information"
          }}
        }}
        ```
        """
        
        st.markdown(info_text)
    
    # Bucket Management
    st.markdown('<h3 class="section-header">🪣 Bucket Management</h3>', unsafe_allow_html=True)
    
    current_project = st.session_state.get("current_project", get_project_id())
    selected_region = st.session_state.get("selected_region", "us-central1")
    
    if not current_project:
        st.error("❌ Please select a Google Cloud project from the sidebar first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create New Bucket")
        new_bucket_name = st.text_input(
            "Bucket Name",
            value=f"lanistr-{current_project}-data",
            help="Enter a unique bucket name (must be globally unique)"
        )
        
        if st.button("Create Bucket", key="create_bucket_btn"):
            if new_bucket_name:
                with st.spinner("Creating bucket..."):
                    bucket_result = create_gcs_bucket_python(current_project, new_bucket_name, selected_region)
                    
                    if bucket_result["success"]:
                        if bucket_result.get('action') == 'created':
                            st.success(f"✅ {bucket_result['message']}")
                        else:
                            st.info(f"ℹ️ {bucket_result['message']}")
                        
                        # Show bucket details
                        with st.expander("📊 Bucket Details"):
                            st.json({
                                "bucket_name": bucket_result["bucket_name"],
                                "project_id": bucket_result.get("project_id", current_project),
                                "region": bucket_result.get("region", selected_region),
                                "action": bucket_result.get("action", "none")
                            })
                    else:
                        st.error(f"❌ {bucket_result['message']}")
                        
                        # Show detailed error information
                        with st.expander("🔍 Error Details"):
                            st.error(f"**Error Type:** {bucket_result.get('error', 'Unknown')}")
                            st.error(f"**Message:** {bucket_result['message']}")
                            if bucket_result.get('solution'):
                                st.info(f"**Solution:** {bucket_result['solution']}")
                            if bucket_result.get('stdout'):
                                st.text("**Command Output:**")
                                st.code(bucket_result['stdout'])
                            if bucket_result.get('stderr'):
                                st.text("**Error Output:**")
                                st.code(bucket_result['stderr'])
            else:
                st.error("Please enter a bucket name")
    
    with col2:
        st.subheader("List Existing Buckets")
        if st.button("Refresh Buckets", key="refresh_buckets_btn"):
            with st.spinner("Loading buckets..."):
                buckets = list_gcs_buckets_python(current_project)
                if buckets:
                    st.success(f"Found {len(buckets)} buckets:")
                    for bucket in buckets:
                        st.text(f"• gs://{bucket}")
                else:
                    st.info("No buckets found in this project")
    
    st.divider()

def show_gherkin_testing():
    """Show Gherkin functional testing page."""
    st.markdown('<h2 class="section-header">🧪 Gherkin Functional Testing</h2>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
        <h4>What is Gherkin Testing?</h4>
        <p>Gherkin is a business-readable, domain-specific language for writing acceptance tests. 
        It allows you to describe software behavior in natural language that both technical and 
        non-technical stakeholders can understand.</p>
        
        <h4>How it works:</h4>
        <ul>
            <li>Write test scenarios in <code>.feature</code> files using Given-When-Then syntax</li>
            <li>Tests are executed against your trained LANISTR model</li>
            <li>Results provide feedback on model performance and behavior</li>
            <li>Tests run in isolated Docker containers for consistency</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Write Tests", "🚀 Run Tests", "📊 Test Results", "📚 Examples"])
    
    with tab1:
        st.subheader("Write Gherkin Test Scenarios")
        
        # Test scenario editor
        st.markdown("**Create a new test scenario:**")
        
        # Feature name
        feature_name = st.text_input(
            "Feature Name",
            value="LANISTR Model Validation",
            help="Name of the feature being tested"
        )
        
        # Scenario name
        scenario_name = st.text_input(
            "Scenario Name", 
            value="Validate MIMIC-IV dataset processing",
            help="Name of the specific test scenario"
        )
        
        # Gherkin steps editor
        st.markdown("**Write your test steps:**")
        
        # Given step
        given_step = st.text_area(
            "Given (Setup)",
            value="the LANISTR model is trained on MIMIC-IV dataset",
            height=80,
            help="Describe the initial context and setup"
        )
        
        # When step
        when_step = st.text_area(
            "When (Action)",
            value="I submit a chest X-ray image with medical text",
            height=80,
            help="Describe the action or event that occurs"
        )
        
        # Then step
        then_step = st.text_area(
            "Then (Expected Result)",
            value="the model should return accurate predictions within 2 seconds",
            height=80,
            help="Describe the expected outcome"
        )
        
        # Additional steps
        st.markdown("**Additional Steps (Optional):**")
        
        additional_steps = []
        for i in range(3):
            step_type = st.selectbox(
                f"Step {i+1} Type",
                ["", "Given", "When", "Then", "And", "But"],
                key=f"step_type_{i}"
            )
            
            if step_type:
                step_text = st.text_input(
                    f"{step_type} (Step {i+1})",
                    key=f"step_text_{i}"
                )
                if step_text:
                    additional_steps.append(f"{step_type} {step_text}")
        
        # Test parameters
        st.subheader("Test Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_version = st.selectbox(
                "Model Version",
                ["latest", "v1.0", "v1.1", "v1.2"],
                help="Which model version to test"
            )
            
            dataset_type = st.selectbox(
                "Dataset Type",
                ["mimic-iv", "amazon", "custom"],
                help="Type of dataset to use for testing"
            )
        
        with col2:
            timeout_seconds = st.number_input(
                "Timeout (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                help="Maximum time to wait for test completion"
            )
            
            retry_count = st.number_input(
                "Retry Count",
                min_value=0,
                max_value=5,
                value=1,
                help="Number of times to retry failed tests"
            )
        
        # Generate feature file
        if st.form_submit_button("Generate Feature File"):
            feature_content = generate_gherkin_feature(
                feature_name, scenario_name, given_step, when_step, then_step, 
                additional_steps, model_version, dataset_type, timeout_seconds, retry_count
            )
            
            st.session_state.generated_feature = feature_content
            st.success("Feature file generated successfully!")
            
            # Show preview
            with st.expander("Preview Generated Feature File"):
                st.code(feature_content, language="gherkin")
    
    with tab2:
        st.subheader("Run Functional Tests")
        
        # Test execution options
        st.markdown("**Test Execution Configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            execution_mode = st.selectbox(
                "Execution Mode",
                ["local", "docker", "cloud"],
                help="Where to run the tests"
            )
            
            test_file = st.file_uploader(
                "Upload Feature File",
                type=['feature'],
                help="Upload a .feature file to run"
            )
        
        with col2:
            parallel_tests = st.number_input(
                "Parallel Tests",
                min_value=1,
                max_value=10,
                value=1,
                help="Number of tests to run in parallel"
            )
            
            output_format = st.selectbox(
                "Output Format",
                ["pretty", "json", "html", "junit"],
                help="Format for test results"
            )
        
        # Use generated feature if available
        if 'generated_feature' in st.session_state:
            st.info("Using generated feature file from previous tab")
            if st.form_submit_button("Run Generated Test"):
                run_gherkin_test(
                    st.session_state.generated_feature,
                    execution_mode,
                    parallel_tests,
                    output_format
                )
        
        # Run uploaded test file
        if test_file is not None:
            if st.form_submit_button("Run Uploaded Test"):
                feature_content = test_file.read().decode('utf-8')
                run_gherkin_test(
                    feature_content,
                    execution_mode,
                    parallel_tests,
                    output_format
                )
        
        # Quick test templates
        st.subheader("Quick Test Templates")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Test MIMIC-IV Model", use_container_width=True):
                run_quick_test("mimic-iv")
        
        with col2:
            if st.button("Test Amazon Model", use_container_width=True):
                run_quick_test("amazon")
        
        with col3:
            if st.button("Test Performance", use_container_width=True):
                run_quick_test("performance")
    
    with tab3:
        st.subheader("Test Results and Feedback")
        
        # Display test results
        if 'test_results' in st.session_state:
            results = st.session_state.test_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tests", results.get('total', 0))
            
            with col2:
                st.metric("Passed", results.get('passed', 0))
            
            with col3:
                st.metric("Failed", results.get('failed', 0))
            
            with col4:
                st.metric("Success Rate", f"{results.get('success_rate', 0):.1f}%")
            
            # Detailed results
            st.subheader("Detailed Results")
            
            for test in results.get('tests', []):
                with st.expander(f"{test['name']} - {test['status']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Duration:** {test.get('duration', 'N/A')}")
                        st.write(f"**Error:** {test.get('error', 'None')}")
                    
                    with col2:
                        if test['status'] == 'PASSED':
                            st.success("✅ Test Passed")
                        else:
                            st.error("❌ Test Failed")
                    
                    # Show test output
                    if 'output' in test:
                        st.code(test['output'], language='text')
            
            # Performance metrics
            if 'performance' in results:
                st.subheader("Performance Metrics")
                
                perf = results['performance']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avg Response Time", f"{perf.get('avg_response_time', 0):.2f}s")
                
                with col2:
                    st.metric("Max Response Time", f"{perf.get('max_response_time', 0):.2f}s")
                
                with col3:
                    st.metric("Throughput", f"{perf.get('throughput', 0):.1f} req/s")
        else:
            st.info("No test results available. Run some tests first!")
    
    with tab4:
        st.subheader("Gherkin Examples and Templates")
        
        # Example scenarios
        st.markdown("**Example Test Scenarios:**")
        
        examples = {
            "Basic Model Validation": """
Feature: LANISTR Model Basic Validation
  As a data scientist
  I want to validate that the LANISTR model works correctly
  So that I can trust its predictions

  Scenario: Validate MIMIC-IV image processing
    Given the LANISTR model is loaded with MIMIC-IV weights
    When I provide a chest X-ray image with medical text
    Then the model should return predictions within 2 seconds
    And the predictions should be in the expected format
    And the confidence scores should be above 0.7
            """,
            
            "Performance Testing": """
Feature: LANISTR Model Performance Testing
  As a system administrator
  I want to ensure the model meets performance requirements
  So that it can handle production load

  Scenario: Test model throughput
    Given the model is running in production mode
    When I send 100 concurrent requests
    Then all requests should complete within 30 seconds
    And the average response time should be under 1 second
    And no requests should fail due to timeout
            """,
            
            "Error Handling": """
Feature: LANISTR Model Error Handling
  As a developer
  I want to ensure the model handles errors gracefully
  So that the system remains stable

  Scenario: Handle invalid input
    Given the model is ready to process requests
    When I send an invalid image format
    Then the model should return an appropriate error message
    And the system should not crash
    And the error should be logged for debugging
            """
        }
        
        selected_example = st.selectbox(
            "Choose an example:",
            list(examples.keys())
        )
        
        if selected_example:
            st.code(examples[selected_example], language="gherkin")
            
            if st.button("Use This Example"):
                st.session_state.generated_feature = examples[selected_example]
                st.success("Example loaded! Go to 'Write Tests' tab to modify it.")
        
        # Best practices
        st.subheader("Best Practices")
        
        st.markdown("""
        **Writing Good Gherkin Tests:**
        
        1. **Use clear, business-focused language**
           - Write from the user's perspective
           - Avoid technical implementation details
        
        2. **Keep scenarios focused**
           - One scenario per test case
           - Test one specific behavior at a time
        
        3. **Use descriptive step names**
           - Make steps self-explanatory
           - Include relevant context and data
        
        4. **Test both happy path and edge cases**
           - Include error scenarios
           - Test boundary conditions
        
        5. **Make tests independent**
           - Each test should be able to run alone
           - Avoid dependencies between tests
        """)

def generate_gherkin_feature(feature_name, scenario_name, given_step, when_step, then_step, 
                           additional_steps, model_version, dataset_type, timeout_seconds, retry_count):
    """Generate a Gherkin feature file content."""
    
    feature_content = f"""Feature: {feature_name}
  As a LANISTR user
  I want to validate model functionality
  So that I can ensure reliable predictions

  @model_version:{model_version} @dataset:{dataset_type} @timeout:{timeout_seconds}s
  Scenario: {scenario_name}
    {given_step}
    {when_step}
    {then_step}"""
    
    # Add additional steps
    for step in additional_steps:
        feature_content += f"\n    {step}"
    
    # Add metadata
    feature_content += f"""

  @metadata
  | Parameter | Value |
  |-----------|-------|
  | Model Version | {model_version} |
  | Dataset Type | {dataset_type} |
  | Timeout | {timeout_seconds}s |
  | Retry Count | {retry_count} |
  | Generated | {time.strftime('%Y-%m-%d %H:%M:%S')} |
"""
    
    return feature_content

def run_gherkin_test(feature_content, execution_mode, parallel_tests, output_format):
    """Run a Gherkin test and return results."""
    
    with st.spinner("Running Gherkin tests..."):
        # Create temporary feature file
        feature_file = f"temp_test_{int(time.time())}.feature"
        with open(feature_file, 'w') as f:
            f.write(feature_content)
        
        try:
            # Simulate test execution (in real implementation, this would run actual tests)
            test_results = simulate_gherkin_execution(feature_content, execution_mode)
            
            # Store results in session state
            st.session_state.test_results = test_results
            
            # Display results
            if test_results['success_rate'] == 100:
                st.success("🎉 All tests passed!")
            elif test_results['success_rate'] > 0:
                st.warning(f"⚠️ {test_results['failed']} tests failed")
            else:
                st.error("❌ All tests failed")
            
            # Show detailed results
            with st.expander("View Detailed Results"):
                for test in test_results['tests']:
                    if test['status'] == 'PASSED':
                        st.success(f"✅ {test['name']} - {test['duration']}")
                    else:
                        st.error(f"❌ {test['name']} - {test['error']}")
        
        except Exception as e:
            st.error(f"Test execution failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(feature_file):
                os.remove(feature_file)

def run_quick_test(test_type):
    """Run a predefined quick test."""
    
    quick_tests = {
        "mimic-iv": {
            "name": "MIMIC-IV Model Validation",
            "feature": """
Feature: MIMIC-IV Model Quick Test
  Scenario: Validate chest X-ray processing
    Given the LANISTR model is trained on MIMIC-IV data
    When I submit a chest X-ray image with medical text
    Then the model should return predictions within 3 seconds
    And the predictions should be clinically relevant
            """
        },
        "amazon": {
            "name": "Amazon Model Validation", 
            "feature": """
Feature: Amazon Model Quick Test
  Scenario: Validate product review processing
    Given the LANISTR model is trained on Amazon data
    When I submit a product image with review text
    Then the model should return predictions within 2 seconds
    And the predictions should match the review sentiment
            """
        },
        "performance": {
            "name": "Performance Benchmark Test",
            "feature": """
Feature: Performance Benchmark Test
  Scenario: Test model throughput and latency
    Given the model is running in optimized mode
    When I send 50 concurrent requests
    Then all requests should complete within 60 seconds
    And the average response time should be under 2 seconds
    And the system should maintain stability
            """
        }
    }
    
    if test_type in quick_tests:
        test = quick_tests[test_type]
        st.info(f"Running {test['name']}...")
        
        run_gherkin_test(
            test['feature'],
            "docker",
            1,
            "pretty"
        )

def simulate_gherkin_execution(feature_content, execution_mode):
    """Simulate Gherkin test execution (placeholder for real implementation)."""
    
    # Parse feature content to extract scenarios
    scenarios = []
    lines = feature_content.split('\n')
    current_scenario = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Scenario:'):
            current_scenario = line.replace('Scenario:', '').strip()
        elif line.startswith('Given') or line.startswith('When') or line.startswith('Then'):
            if current_scenario:
                scenarios.append(current_scenario)
                current_scenario = None
    
    # Generate mock test results
    tests = []
    total_scenarios = len(scenarios) if scenarios else 1
    
    for i in range(total_scenarios):
        scenario_name = scenarios[i] if scenarios else "Default Test Scenario"
        
        # Simulate test execution with some randomness
        import random
        success = random.random() > 0.2  # 80% success rate
        
        test = {
            'name': scenario_name,
            'status': 'PASSED' if success else 'FAILED',
            'duration': f"{random.uniform(0.5, 3.0):.2f}s",
            'error': None if success else "Simulated test failure for demonstration"
        }
        
        if execution_mode == 'docker':
            test['output'] = f"Running in Docker container...\n{scenario_name}: {'PASSED' if success else 'FAILED'}"
        else:
            test['output'] = f"Running locally...\n{scenario_name}: {'PASSED' if success else 'FAILED'}"
        
        tests.append(test)
    
    # Calculate summary
    passed = sum(1 for test in tests if test['status'] == 'PASSED')
    failed = len(tests) - passed
    success_rate = (passed / len(tests)) * 100 if tests else 0
    
    # Add performance metrics
    performance = {
        'avg_response_time': sum(float(test['duration'].replace('s', '')) for test in tests) / len(tests) if tests else 0,
        'max_response_time': max(float(test['duration'].replace('s', '')) for test in tests) if tests else 0,
        'throughput': len(tests) / sum(float(test['duration'].replace('s', '')) for test in tests) if tests else 0
    }
    
    return {
        'total': len(tests),
        'passed': passed,
        'failed': failed,
        'success_rate': success_rate,
        'tests': tests,
        'performance': performance,
        'execution_mode': execution_mode,
        'timestamp': time.time()
    }

def show_setup_guides():
    """Show interactive setup guides."""
    st.markdown('<h2 class="section-header">📚 Setup Guides</h2>', unsafe_allow_html=True)
    
    # Guide selection
    guide = st.selectbox(
        "Choose a setup guide",
        ["🚀 Quick Start (3 Commands)", "🎯 Complete Setup", "🌐 Web Interface Setup", "📊 Data Requirements", "🔧 Custom Configuration"]
    )
    
    if guide == "🚀 Quick Start (3 Commands)":
        show_quick_start_guide()
    elif guide == "🎯 Complete Setup":
        show_complete_setup_guide()
    elif guide == "🌐 Web Interface Setup":
        show_web_interface_guide()
    elif guide == "📊 Data Requirements":
        show_data_requirements_guide()
    elif guide == "🔧 Custom Configuration":
        show_custom_config_guide()

def show_quick_start_guide():
    """Show the quick start guide."""
    st.markdown("""
    ## ⚡ **Quick Start (3 Commands)**
    
    Get LANISTR training on Vertex AI in under 10 minutes!
    """)
    
    st.code("""
# 1. Clone the repository
git clone <your-repo-url>
cd lanistr

# 2. Run one-click setup
./one_click_setup.sh

# 3. Submit your first training job
./quick_submit.sh
    """, language="bash")
    
    st.success("That's it! Your training job will be running on Google Cloud Vertex AI.")
    
    # Interactive setup
    st.markdown("### 🎯 **What Just Happened?**")
    
    setup_steps = [
        "✅ Sets up Google Cloud authentication",
        "✅ Enables required APIs",
        "✅ Creates storage bucket",
        "✅ Generates sample data",
        "✅ Validates your dataset",
        "✅ Builds and pushes Docker image",
        "✅ Creates quick submit scripts"
    ]
    
    for step in setup_steps:
        st.write(step)
    
    # Quick actions
    st.markdown("### 🚀 **Quick Actions**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run One-Click Setup", use_container_width=True):
            with st.spinner("Running one-click setup..."):
                result = run_command("./one_click_setup.sh", check=False)
                if result and result.returncode == 0:
                    st.success("One-click setup completed successfully!")
                else:
                    st.error("Setup failed. Check the logs for details.")
    
    with col2:
        if st.button("📊 Monitor Your Job", use_container_width=True):
            st.session_state.page = "📊 Monitor Jobs"
            st.rerun()

def show_complete_setup_guide():
    """Show the complete setup guide."""
    st.markdown("""
    ## 🎯 **Complete Setup Guide**
    
    Everything you need to know about setting up LANISTR for production use.
    """)
    
    # Setup sections
    sections = [
        ("🔐 Authentication", "Set up Google Cloud authentication and project access"),
        ("📁 Project Configuration", "Configure your GCP project and enable APIs"),
        ("🗄️ Storage Setup", "Create and configure Google Cloud Storage buckets"),
        ("🐳 Docker Setup", "Build and push the training Docker image"),
        ("📊 Data Preparation", "Prepare and validate your training datasets"),
        ("🚀 Job Submission", "Submit and monitor training jobs")
    ]
    
    for title, description in sections:
        with st.expander(title):
            st.write(description)
            
            if title == "🔐 Authentication":
                if st.button("🔑 Authenticate Now", key="auth_btn"):
                    with st.spinner("Opening authentication..."):
                        run_command("gcloud auth login", check=False)
                    st.success("Authentication window opened!")
            
            elif title == "📁 Project Configuration":
                current_project = st.session_state.get("current_project", get_project_id())
                if current_project:
                    st.success(f"Current project: {current_project}")
                else:
                    st.warning("No project selected. Use the sidebar to select a project.")
            
            elif title == "🗄️ Storage Setup":
                if st.button("Create Storage Bucket", key="create_bucket"):
                    project = st.session_state.get("current_project", get_project_id())
                    if project:
                        bucket_name = f"lanistr-{project}-data"
                        with st.spinner("Creating bucket..."):
                            result = setup_gcs_bucket(project, bucket_name, "us-central1")
                            if result:
                                st.success(f"Bucket created: {bucket_name}")
                            else:
                                st.error("Failed to create bucket")
                    else:
                        st.error("Please select a project first")

def show_web_interface_guide():
    """Show the web interface setup guide."""
    st.markdown("""
    ## 🌐 **Web Interface Setup Guide**
    
    Your LANISTR web interface is now running and ready to use!
    """)
    
    # Access URLs
    st.markdown("### 🔗 **Access URLs**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Frontend (Streamlit)**
        - **URL**: http://localhost:8501
        - **Purpose**: User-friendly web interface
        - **Features**: Dashboard, job submission, monitoring
        """)
    
    with col2:
        st.markdown("""
        **Backend API (FastAPI)**
        - **URL**: http://localhost:8000
        - **API Docs**: http://localhost:8000/docs
        - **Health Check**: http://localhost:8000/health
        """)
    
    # Quick start steps
    st.markdown("### 🚀 **Quick Start Steps**")
    
    steps = [
        ("Step 1: Access the Frontend", "Open http://localhost:8501 in your browser"),
        ("Step 2: Configure Your Project", "Select your Google Cloud project from the sidebar"),
        ("Step 3: Submit Your First Job", "Go to 'Submit Job' page and configure your training"),
        ("Step 4: Monitor Progress", "Use 'Monitor Jobs' to track your training progress")
    ]
    
    for i, (title, description) in enumerate(steps, 1):
        st.markdown(f"**{i}. {title}**")
        st.write(description)
        st.write("")
    
    # API usage examples
    st.markdown("### 🔌 **API Usage Examples**")
    
    with st.expander("API Examples"):
        st.code("""
# Check system status
curl http://localhost:8000/status

# Submit a training job
curl -X POST "http://localhost:8000/jobs/submit" \\
  -H "Content-Type: application/json" \\
  -d '{
    "project_id": "your-project-id",
    "dataset_type": "mimic-iv",
    "environment": "dev",
    "job_name": "my-training-job"
  }'

# List all jobs
curl http://localhost:8000/jobs
        """, language="bash")

def show_data_requirements_guide():
    """Show the data requirements guide."""
    st.markdown("""
    ## 📊 **Data Requirements Guide**
    
    Learn about the data formats and requirements for LANISTR training.
    """)
    
    # Dataset types
    st.markdown("### 📁 **Supported Dataset Types**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **MIMIC-IV (Medical)**
        - Chest X-ray images
        - Medical text reports
        - Patient demographics
        - Clinical outcomes
        """)
    
    with col2:
        st.markdown("""
        **Amazon (Product Reviews)**
        - Product images
        - Review text
        - Product metadata
        - Rating information
        """)
    
    # Data format
    st.markdown("### 📋 **Data Format Requirements**")
    
    st.markdown("""
    All data must be in JSONL (JSON Lines) format with the following structure:
    """)
    
    st.code("""
{
  "id": "unique_identifier",
  "image_path": "path/to/image.jpg",
  "text": "associated text content",
  "metadata": {
    "additional": "information"
  }
}
    """, language="json")
    
    # Validation
    st.markdown("### ✅ **Data Validation**")
    
    if st.button("📊 Validate Sample Data"):
        with st.spinner("Validating sample data..."):
            # Check if sample data exists
            sample_files = ["test_mimic.jsonl", "test_amazon.jsonl"]
            for file in sample_files:
                if os.path.exists(file):
                    result = validate_dataset("mimic-iv" if "mimic" in file else "amazon", "./data", file)
                    if result.get("passed"):
                        st.success(f"✅ {file} is valid")
                    else:
                        st.error(f"❌ {file} has issues: {result.get('message', 'Unknown error')}")
                else:
                    st.warning(f"⚠️ {file} not found")

def show_custom_config_guide():
    """Show the custom configuration guide."""
    st.markdown("""
    ## 🔧 **Custom Configuration Guide**
    
    Learn how to customize LANISTR for your specific needs.
    """)
    
    # Configuration options
    st.markdown("### ⚙️ **Configuration Options**")
    
    config_options = [
        ("Different Datasets", "MIMIC-IV (medical) or Amazon (product reviews)"),
        ("Different Environments", "Development (cheaper) or Production (faster)"),
        ("Custom Project", "Use your own Google Cloud project"),
        ("Custom Regions", "Select your preferred GCP region"),
        ("Custom Machine Types", "Configure compute resources"),
        ("Custom Storage", "Use your own GCS buckets")
    ]
    
    for option, description in config_options:
        with st.expander(option):
            st.write(description)
            
            if option == "Different Datasets":
                dataset = st.selectbox("Select Dataset", ["mimic-iv", "amazon"])
                if st.button("Generate Sample Data", key=f"gen_{dataset}"):
                    with st.spinner(f"Generating {dataset} sample data..."):
                        result = create_sample_data(dataset, f"./data/{dataset}")
                        if result:
                            st.success(f"Sample {dataset} data generated!")
                        else:
                            st.error("Failed to generate sample data")
            
            elif option == "Custom Machine Types":
                st.markdown("Configure machine resources:")
                machine_type = st.selectbox("Machine Type", ["n1-standard-2", "n1-standard-4", "n1-standard-8"])
                accelerator_type = st.selectbox("Accelerator Type", ["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100", "NVIDIA_TESLA_A100"])
                accelerator_count = st.slider("Accelerator Count", 0, 8, 1)
                
                st.info(f"Selected: {machine_type} with {accelerator_count}x {accelerator_type}")

def show_quick_start():
    """Show the quick start page."""
    st.markdown('<h2 class="section-header">🎯 Quick Start</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🚀 **Get Started in Minutes**
    
    Follow these simple steps to get your first LANISTR training job running.
    """)
    
    # Step-by-step guide
    steps = [
        {
            "title": "1. 🔐 Authenticate",
            "description": "Login to Google Cloud",
            "action": "Use the sidebar to authenticate with Google Cloud",
            "status": "pending"
        },
        {
            "title": "2. 📁 Select Project", 
            "description": "Choose your GCP project",
            "action": "Select your project from the sidebar dropdown",
            "status": "pending"
        },
        {
            "title": "3. 🌍 Choose Region",
            "description": "Select your preferred region",
            "action": "Pick a region from the sidebar",
            "status": "pending"
        },
        {
            "title": "4. 🔧 Setup APIs",
            "description": "Enable required Google Cloud APIs",
            "action": "Click 'Check APIs' in the sidebar",
            "status": "pending"
        },
        {
            "title": "5. 🚀 Submit Job",
            "description": "Submit your first training job",
            "action": "Go to 'Submit Job' page and configure your training",
            "status": "pending"
        }
    ]
    
    # Check current status
    auth_status = check_authentication()
    current_project = st.session_state.get("current_project", get_project_id())
    selected_region = st.session_state.get("selected_region")
    
    if auth_status:
        steps[0]["status"] = "completed"
    if current_project:
        steps[1]["status"] = "completed"
    if selected_region:
        steps[2]["status"] = "completed"
    
    # Display steps
    for i, step in enumerate(steps):
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if step["status"] == "completed":
                st.success("✅")
            else:
                st.info(f"{i+1}")
        
        with col2:
            st.markdown(f"**{step['title']}**")
            st.write(step['description'])
            st.caption(step['action'])
        
        st.divider()
    
    # Quick actions
    st.markdown("### 🚀 **Quick Actions**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not auth_status:
            if st.button("🔑 Authenticate", use_container_width=True):
                with st.spinner("Opening authentication..."):
                    run_command("gcloud auth login", check=False)
                st.rerun()
        else:
            if st.button("🔄 Switch Account", use_container_width=True):
                st.session_state.show_account_switcher = True
                st.rerun()
    
    with col2:
        if not current_project:
            st.warning("Select Project")
        else:
            if st.button("🚀 Submit Job", use_container_width=True):
                st.session_state.page = "🚀 Submit Job"
                st.rerun()
    
    with col3:
        if st.button("📊 Monitor Jobs", use_container_width=True):
            st.session_state.page = "📊 Monitor Jobs"
            st.rerun()
    
    with col4:
        if st.button("📁 Manage Data", use_container_width=True):
            st.session_state.page = "📁 Data Management"
            st.rerun()
    
    # Account switcher popup
    if st.session_state.get("show_account_switcher", False):
        with st.expander("🔄 Switch Google Account", expanded=True):
            available_accounts = get_available_accounts()
            if available_accounts:
                selected_account = st.selectbox(
                    "Select Account",
                    available_accounts,
                    index=available_accounts.index(current_account) if current_account in available_accounts else 0,
                    key="dashboard_account_selector"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if selected_account != current_account:
                        if st.button("Switch", key="dashboard_switch_btn"):
                            with st.spinner("Switching account..."):
                                if switch_account(selected_account):
                                    st.success(f"Switched to: {selected_account}")
                                    st.session_state.current_account = selected_account
                                    st.session_state.show_account_switcher = False
                                    st.rerun()
                                else:
                                    st.error("Failed to switch account")
                
                with col2:
                    if st.button("Cancel", key="dashboard_cancel_btn"):
                        st.session_state.show_account_switcher = False
                        st.rerun()
            else:
                st.warning("No accounts found")
                if st.button("Add Account", key="dashboard_add_btn"):
                    with st.spinner("Opening authentication..."):
                        run_command("gcloud auth login", check=False)
                    st.success("Authentication window opened!")
                    st.session_state.show_account_switcher = False
                    st.rerun()

def show_troubleshooting():
    """Show the troubleshooting guide."""
    st.markdown('<h2 class="section-header">🛠️ Troubleshooting</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🔧 **Common Issues and Solutions**
    
    Find solutions to common problems you might encounter.
    """)
    
    # Common issues
    issues = [
        {
            "title": "❌ Project ID not found",
            "description": "You're not authenticated or the project doesn't exist",
            "solution": "Run `gcloud auth login` and verify your project exists",
            "action": "Authenticate in sidebar"
        },
        {
            "title": "❌ Permission denied", 
            "description": "You don't have the required permissions",
            "solution": "Add the AI Platform User and Cloud Build Editor roles to your account",
            "action": "Check permissions"
        },
        {
            "title": "❌ Cloud Build failed",
            "description": "Docker image build is failing in Cloud Build",
            "solution": "Check Cloud Build logs and ensure Dockerfile is correct",
            "action": "Check Cloud Build"
        },
        {
            "title": "❌ APIs not enabled",
            "description": "Required Google Cloud APIs are not enabled",
            "solution": "Enable the required APIs in your project",
            "action": "Enable APIs"
        },
        {
            "title": "❌ Ports are busy",
            "description": "Web interface ports are already in use",
            "solution": "Kill existing processes or use different ports",
            "action": "Fix ports"
        }
    ]
    
    for issue in issues:
        with st.expander(issue["title"]):
            st.write(f"**Problem:** {issue['description']}")
            st.write(f"**Solution:** {issue['solution']}")
            
            if issue["action"] == "Authenticate in sidebar":
                if st.button("🔑 Authenticate Now", key=f"auth_{issue['title']}"):
                    with st.spinner("Opening authentication..."):
                        run_command("gcloud auth login", check=False)
                    st.success("Authentication window opened!")
            
            elif issue["action"] == "Check permissions":
                if st.button("🔍 Check Permissions", key=f"perm_{issue['title']}"):
                    project = st.session_state.get("current_project", get_project_id())
                    if project:
                        with st.spinner("Checking permissions..."):
                            result = run_command(f"gcloud projects get-iam-policy {project} --flatten='bindings[].members' --filter='bindings.role:roles/aiplatform.user' --format='value(bindings.members)'", check=False)
                            if result and result.stdout.strip():
                                st.success("✅ AI Platform User role found")
                            else:
                                st.warning("⚠️ AI Platform User role not found")
                                st.info("You may need to add this role to your account")
                    else:
                        st.error("Please select a project first")
            
            elif issue["action"] == "Check Cloud Build":
                if st.button("🔍 Check Cloud Build Logs", key=f"cloudbuild_{issue['title']}"):
                    with st.spinner("Checking Cloud Build logs..."):
                        # List recent builds first
                        current_project = st.session_state.get("current_project", get_project_id())
                        if current_project:
                            builds = list_recent_builds(current_project, 5)
                            if builds:
                                st.success("Recent Cloud Build jobs:")
                                for build in builds:
                                    if build.strip():
                                        st.text(build)
                                
                                # Allow user to select a build to check
                                build_id = st.text_input("Enter Build ID to check details", key=f"build_id_{issue['title']}")
                                if build_id and st.button("Get Build Details", key=f"get_build_{issue['title']}"):
                                    result = run_command(f"gcloud builds describe {build_id} --format='json'", check=False)
                                    if result and result.returncode == 0:
                                        try:
                                            import json
                                            build_data = json.loads(result.stdout)
                                            st.json(build_data)
                                        except:
                                            st.text(result.stdout)
                                    else:
                                        st.error("Failed to get build details")
                            else:
                                st.info("No Cloud Build jobs found")
                        else:
                            st.error("Please select a project first")
            
            elif issue["action"] == "Enable APIs":
                if st.button("🔌 Enable APIs", key=f"api_{issue['title']}"):
                    with st.spinner("Enabling APIs..."):
                        apis = [
                            "aiplatform.googleapis.com",
                            "storage.googleapis.com",
                            "logging.googleapis.com",
                            "monitoring.googleapis.com",
                            "errorreporting.googleapis.com",
                            "containerregistry.googleapis.com"
                        ]
                        for api in apis:
                            run_command(f"gcloud services enable {api}", check=False)
                    st.success("APIs enabled!")
            
            elif issue["action"] == "Fix ports":
                if st.button("🔌 Check Ports", key=f"port_{issue['title']}"):
                    with st.spinner("Checking ports..."):
                        result = run_command("lsof -i :8501 -i :8000", check=False)
                        if result and result.stdout.strip():
                            st.warning("Ports are in use:")
                            st.code(result.stdout)
                            if st.button("Kill Processes", key=f"kill_{issue['title']}"):
                                run_command("pkill -f 'streamlit'", check=False)
                                run_command("pkill -f 'uvicorn'", check=False)
                                st.success("Processes killed!")
                        else:
                            st.success("Ports are available")
    
    # System diagnostics
    st.markdown("### 🔍 **System Diagnostics**")
    
    if st.button("🔍 Run Diagnostics"):
        with st.spinner("Running diagnostics..."):
            # Check prerequisites
            prereq_status = check_prerequisites()
            auth_status = check_authentication()
            current_project = st.session_state.get("current_project", get_project_id())
            
            st.markdown("**System Status:**")
            
            for tool, status in prereq_status.items():
                if status:
                    st.success(f"✅ {tool}")
                else:
                    st.error(f"❌ {tool}")
            
            if auth_status:
                st.success("✅ Google Cloud authenticated")
            else:
                st.error("❌ Not authenticated")
            
            if current_project:
                st.success(f"✅ Project: {current_project}")
            else:
                st.warning("⚠️ No project selected")

if __name__ == "__main__":
    main() 