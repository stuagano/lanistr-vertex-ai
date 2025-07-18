#!/usr/bin/env python3
"""
LANISTR FastAPI Backend

REST API for LANISTR job submission, monitoring, and management.
Provides programmatic access to all LANISTR functionality.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import subprocess
import json
import os
import sys
from pathlib import Path
import time
import asyncio
from datetime import datetime
import uuid

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import LANISTR utilities
try:
    from lanistr.utils.data_validator import DataValidator, validate_amazon_dataset, validate_mimic_dataset
except ImportError:
    # Fallback if LANISTR not installed
    pass

# FastAPI app
app = FastAPI(
    title="LANISTR API",
    description="REST API for LANISTR training job submission and management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class JobConfig(BaseModel):
    project_id: str = Field(..., description="Google Cloud Project ID")
    region: str = Field(default="us-central1", description="GCP region")
    dataset_type: str = Field(..., description="Dataset type (mimic-iv or amazon)")
    environment: str = Field(default="dev", description="Environment (dev or prod)")
    job_name: Optional[str] = Field(None, description="Job name (auto-generated if not provided)")
    machine_type: Optional[str] = Field(None, description="Machine type")
    accelerator_type: Optional[str] = Field(None, description="GPU type")
    accelerator_count: Optional[int] = Field(None, description="Number of GPUs")
    bucket_name: Optional[str] = Field(None, description="GCS bucket name")
    data_dir: Optional[str] = Field(None, description="Data directory")
    jsonl_file: Optional[str] = Field(None, description="JSONL file path")
    validate_data: bool = Field(default=True, description="Validate data before submission")

class JobStatus(BaseModel):
    job_id: str
    name: str
    status: str
    created_at: datetime
    config: Dict[str, Any]
    logs: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class ValidationResult(BaseModel):
    passed: bool
    message: str
    stats: Optional[Dict[str, Any]] = None
    errors: List[str] = []
    warnings: List[str] = []

class SystemStatus(BaseModel):
    prerequisites: Dict[str, bool]
    authenticated: bool
    project_id: Optional[str] = None
    apis_enabled: Dict[str, bool]

# Global state (in production, use a proper database)
jobs_db = {}
validation_results = {}

def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if check and result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Command failed: {result.stderr}")
        return result
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail=f"Command timed out: {cmd}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command failed: {str(e)}")

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
    for tool in tools:
        result = run_command(f"which {tool}", check=False)
        status[tool] = result.returncode == 0 if result else False
    
    return status

def check_authentication() -> bool:
    """Check if user is authenticated with Google Cloud."""
    result = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", check=False)
    return result and result.returncode == 0 and result.stdout.strip()

def check_apis_enabled(project_id: str) -> Dict[str, bool]:
    """Check if required APIs are enabled."""
    apis = [
        "aiplatform.googleapis.com",
        "storage.googleapis.com",
        "logging.googleapis.com",
        "monitoring.googleapis.com",
        "errorreporting.googleapis.com",
        "containerregistry.googleapis.com"
    ]
    
    status = {}
    for api in apis:
        result = run_command(f"gcloud services list --enabled --filter='name:{api}' --format='value(name)'", check=False)
        status[api] = result and result.returncode == 0 and result.stdout.strip() == api
    
    return status

def setup_gcs_bucket(project_id: str, bucket_name: str, region: str) -> bool:
    """Set up GCS bucket."""
    result = run_command(f"gsutil ls -b gs://{bucket_name}", check=False)
    if not result or result.returncode != 0:
        result = run_command(f"gsutil mb -p {project_id} -c STANDARD -l {region} gs://{bucket_name}", check=False)
        return result and result.returncode == 0
    return True

def create_sample_data(dataset_type: str, data_dir: str) -> bool:
    """Create sample data for the specified dataset."""
    jsonl_file = f"{data_dir}/{dataset_type}.jsonl"
    
    if not Path(jsonl_file).exists():
        cmd = f"python generate_sample_data.py --dataset {dataset_type} --output-file {jsonl_file} --num-samples 100 --create-files"
        result = run_command(cmd, check=False)
        return result and result.returncode == 0
    return True

def build_and_push_image(project_id: str, image_name: str = "lanistr-training") -> str:
    """Build and push Docker image to GCR."""
    image_uri = f"gcr.io/{project_id}/{image_name}:latest"
    
    # Build image
    result = run_command(f"docker build -t {image_name}:latest .", check=False)
    if not result or result.returncode != 0:
        raise HTTPException(status_code=500, detail="Docker build failed")
    
    # Tag for GCR
    result = run_command(f"docker tag {image_name}:latest {image_uri}", check=False)
    if not result or result.returncode != 0:
        raise HTTPException(status_code=500, detail="Docker tag failed")
    
    # Push to GCR
    result = run_command(f"docker push {image_uri}", check=False)
    if not result or result.returncode != 0:
        raise HTTPException(status_code=500, detail="Docker push failed")
    
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

async def submit_job_async(job_config: JobConfig) -> str:
    """Submit job asynchronously."""
    job_id = str(uuid.uuid4())
    
    # Set defaults
    if not job_config.job_name:
        job_config.job_name = f"lanistr-{job_config.dataset_type}-{int(time.time())}"
    
    if not job_config.bucket_name:
        job_config.bucket_name = f"lanistr-{job_config.project_id}-{job_config.dataset_type}"
    
    if not job_config.data_dir:
        job_config.data_dir = f"./data/{job_config.dataset_type}"
    
    if not job_config.jsonl_file:
        job_config.jsonl_file = f"{job_config.data_dir}/{job_config.dataset_type}.jsonl"
    
    # Set machine configuration based on environment
    if job_config.environment == "dev":
        machine_type = job_config.machine_type or "n1-standard-2"
        accelerator_type = job_config.accelerator_type or "NVIDIA_TESLA_T4"
        accelerator_count = job_config.accelerator_count or 1
    else:
        machine_type = job_config.machine_type or "n1-standard-4"
        accelerator_type = job_config.accelerator_type or "NVIDIA_TESLA_V100"
        accelerator_count = job_config.accelerator_count or 8
    
    # Store job info
    jobs_db[job_id] = {
        "id": job_id,
        "name": job_config.job_name,
        "status": "PENDING",
        "created_at": datetime.now(),
        "config": job_config.dict(),
        "logs": []
    }
    
    # Run submission in background
    asyncio.create_task(run_job_submission(job_id, job_config, machine_type, accelerator_type, accelerator_count))
    
    return job_id

async def run_job_submission(job_id: str, job_config: JobConfig, machine_type: str, accelerator_type: str, accelerator_count: int):
    """Run job submission in background."""
    try:
        jobs_db[job_id]["status"] = "SETUP"
        jobs_db[job_id]["logs"].append("Setting up GCS bucket...")
        
        # Setup bucket
        if not setup_gcs_bucket(job_config.project_id, job_config.bucket_name, job_config.region):
            jobs_db[job_id]["status"] = "FAILED"
            jobs_db[job_id]["logs"].append("Failed to setup GCS bucket")
            return
        
        jobs_db[job_id]["logs"].append("Preparing data...")
        
        # Create sample data if needed
        if not Path(job_config.jsonl_file).exists():
            if not create_sample_data(job_config.dataset_type, job_config.data_dir):
                jobs_db[job_id]["status"] = "FAILED"
                jobs_db[job_id]["logs"].append("Failed to create sample data")
                return
        
        jobs_db[job_id]["logs"].append("Uploading data to GCS...")
        
        # Upload data
        result = run_command(f"gsutil -m cp -r {job_config.data_dir}/ gs://{job_config.bucket_name}/", check=False)
        if not result or result.returncode != 0:
            jobs_db[job_id]["logs"].append("Warning: Data upload failed, but continuing...")
        
        jobs_db[job_id]["logs"].append("Building and pushing Docker image...")
        
        # Build and push image
        image_uri = build_and_push_image(job_config.project_id)
        
        jobs_db[job_id]["logs"].append("Submitting job to Vertex AI...")
        
        # Submit job
        config_file = f"lanistr/configs/{job_config.dataset_type}_pretrain.yaml"
        
        job_submit_config = {
            "project_id": job_config.project_id,
            "region": job_config.region,
            "job_name": job_config.job_name,
            "config_file": config_file,
            "machine_type": machine_type,
            "accelerator_type": accelerator_type,
            "accelerator_count": accelerator_count,
            "bucket_name": job_config.bucket_name,
            "image_uri": image_uri
        }
        
        result = submit_job(job_submit_config)
        
        if result["success"]:
            jobs_db[job_id]["status"] = "SUBMITTED"
            jobs_db[job_id]["logs"].append("Job submitted successfully!")
        else:
            jobs_db[job_id]["status"] = "FAILED"
            jobs_db[job_id]["logs"].append(f"Job submission failed: {result['message']}")
    
    except Exception as e:
        jobs_db[job_id]["status"] = "FAILED"
        jobs_db[job_id]["logs"].append(f"Error: {str(e)}")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LANISTR API",
        "version": "1.0.0",
        "endpoints": {
            "status": "/status",
            "jobs": "/jobs",
            "submit": "/jobs/submit",
            "validate": "/validate",
            "upload": "/upload"
        }
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and prerequisites."""
    project_id = get_project_id()
    
    return SystemStatus(
        prerequisites=check_prerequisites(),
        authenticated=check_authentication(),
        project_id=project_id,
        apis_enabled=check_apis_enabled(project_id) if project_id else {}
    )

@app.post("/jobs/submit")
async def submit_training_job(job_config: JobConfig, background_tasks: BackgroundTasks):
    """Submit a training job."""
    # Validate project ID
    if not job_config.project_id:
        job_config.project_id = get_project_id()
        if not job_config.project_id:
            raise HTTPException(status_code=400, detail="Project ID is required")
    
    # Check authentication
    if not check_authentication():
        raise HTTPException(status_code=401, detail="Not authenticated with Google Cloud")
    
    # Submit job
    job_id = await submit_job_async(job_config)
    
    return {
        "job_id": job_id,
        "message": "Job submitted successfully",
        "status": "PENDING"
    }

@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all jobs."""
    jobs = []
    for job_id, job_data in jobs_db.items():
        jobs.append(JobStatus(
            job_id=job_id,
            name=job_data["name"],
            status=job_data["status"],
            created_at=job_data["created_at"],
            config=job_data["config"],
            logs="\n".join(job_data["logs"]) if job_data["logs"] else None
        ))
    
    return jobs

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get job details."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_db[job_id]
    return JobStatus(
        job_id=job_id,
        name=job_data["name"],
        status=job_data["status"],
        created_at=job_data["created_at"],
        config=job_data["config"],
        logs="\n".join(job_data["logs"]) if job_data["logs"] else None
    )

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs_db[job_id]
    return {"message": "Job deleted successfully"}

@app.post("/validate")
async def validate_dataset_endpoint(
    dataset_type: str,
    data_dir: str,
    jsonl_file: str,
    project_id: Optional[str] = None,
    gcs_bucket: Optional[str] = None
):
    """Validate a dataset."""
    validation_id = str(uuid.uuid4())
    
    try:
        # Create sample data if needed
        if not Path(jsonl_file).exists():
            create_sample_data(dataset_type, data_dir)
        
        # Validate dataset
        if dataset_type == "amazon":
            result = validate_amazon_dataset(
                jsonl_file=jsonl_file,
                data_dir=data_dir,
                gcs_bucket=gcs_bucket,
                project_id=project_id
            )
        elif dataset_type == "mimic-iv":
            result = validate_mimic_dataset(
                jsonl_file=jsonl_file,
                data_dir=data_dir,
                gcs_bucket=gcs_bucket,
                project_id=project_id
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid dataset type")
        
        # Store result
        validation_results[validation_id] = {
            "id": validation_id,
            "dataset_type": dataset_type,
            "result": result,
            "timestamp": datetime.now()
        }
        
        return ValidationResult(
            passed=result.get("passed", False),
            message=result.get("message", "Validation completed"),
            stats=result.get("stats"),
            errors=result.get("errors", []),
            warnings=result.get("warnings", [])
        )
    
    except Exception as e:
        return ValidationResult(
            passed=False,
            message=f"Validation failed: {str(e)}",
            errors=[str(e)]
        )

@app.get("/validate/{validation_id}")
async def get_validation_result(validation_id: str):
    """Get validation result."""
    if validation_id not in validation_results:
        raise HTTPException(status_code=404, detail="Validation result not found")
    
    validation_data = validation_results[validation_id]
    result = validation_data["result"]
    
    return ValidationResult(
        passed=result.get("passed", False),
        message=result.get("message", "Validation completed"),
        stats=result.get("stats"),
        errors=result.get("errors", []),
        warnings=result.get("warnings", [])
    )

@app.post("/upload")
async def upload_data(file: UploadFile = File(...), dataset_type: str = "mimic-iv"):
    """Upload data file."""
    if not file.filename.endswith('.jsonl'):
        raise HTTPException(status_code=400, detail="Only JSONL files are supported")
    
    # Create data directory
    data_dir = f"./data/{dataset_type}"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = f"{data_dir}/{dataset_type}.jsonl"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "message": "File uploaded successfully",
        "file_path": file_path,
        "file_size": len(content)
    }

@app.post("/setup")
async def setup_project(project_id: str, region: str = "us-central1"):
    """Setup project and enable APIs."""
    try:
        # Set project
        run_command(f"gcloud config set project {project_id}")
        
        # Enable APIs
        apis = [
            "aiplatform.googleapis.com",
            "storage.googleapis.com",
            "logging.googleapis.com",
            "monitoring.googleapis.com",
            "errorreporting.googleapis.com",
            "containerregistry.googleapis.com"
        ]
        
        for api in apis:
            run_command(f"gcloud services enable {api}")
        
        return {
            "message": "Project setup completed",
            "project_id": project_id,
            "region": region
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 