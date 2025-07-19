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


class CommandExecutor:
    """Handles command execution with consistent error handling."""
    
    @staticmethod
    def run_command(cmd: str, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run a shell command and return the result.
        
        Args:
            cmd: Command to execute
            check: Whether to raise exception on non-zero exit
            timeout: Command timeout in seconds
            
        Returns:
            CompletedProcess result
            
        Raises:
            HTTPException: On command failure or timeout
        """
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            if check and result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Command failed: {result.stderr}")
            return result
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail=f"Command timed out: {cmd}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Command failed: {str(e)}")
    
    @staticmethod
    def get_command_output(cmd: str) -> str:
        """Get command output or empty string on failure.
        
        Args:
            cmd: Command to execute
            
        Returns:
            Command output or empty string
        """
        try:
            result = CommandExecutor.run_command(cmd, check=False)
            return result.stdout.strip() if result.returncode == 0 else ""
        except:
            return ""


class GoogleCloudManager:
    """Manages Google Cloud operations and authentication."""
    
    def __init__(self):
        self.cmd_executor = CommandExecutor()
    
    def get_project_id(self) -> str:
        """Get the current Google Cloud project ID."""
        return self.cmd_executor.get_command_output("gcloud config get-value project")
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if required tools are installed."""
        tools = ["gcloud", "docker", "python3", "gsutil"]
        results = {}
        
        for tool in tools:
            try:
                result = self.cmd_executor.run_command(f"which {tool}", check=False)
                results[tool] = result.returncode == 0
            except:
                results[tool] = False
        
        return results
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated with Google Cloud."""
        try:
            result = self.cmd_executor.run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", check=False)
            return bool(result.stdout.strip())
        except:
            return False
    
    def check_apis_enabled(self, project_id: str) -> Dict[str, bool]:
        """Check if required Google Cloud APIs are enabled."""
        apis = {
            "aiplatform.googleapis.com": "Vertex AI API",
            "storage.googleapis.com": "Cloud Storage API",
            "cloudbuild.googleapis.com": "Cloud Build API",
            "containerregistry.googleapis.com": "Container Registry API"
        }
        
        results = {}
        for api, description in apis.items():
            try:
                cmd = f"gcloud services list --enabled --filter='name:{api}' --format='value(name)' --project={project_id}"
                result = self.cmd_executor.run_command(cmd, check=False)
                results[api] = bool(result.stdout.strip())
            except:
                results[api] = False
        
        return results
    
    def setup_gcs_bucket(self, project_id: str, bucket_name: str, region: str) -> bool:
        """Setup a GCS bucket if it doesn't exist."""
        try:
            # Check if bucket exists
            check_cmd = f"gsutil ls gs://{bucket_name}"
            result = self.cmd_executor.run_command(check_cmd, check=False)
            
            if result.returncode != 0:
                # Create bucket
                create_cmd = f"gsutil mb -p {project_id} -l {region} gs://{bucket_name}"
                self.cmd_executor.run_command(create_cmd)
            
            return True
        except:
            return False


class DataManager:
    """Manages data operations and validation."""
    
    def __init__(self):
        self.cmd_executor = CommandExecutor()
    
    def create_sample_data(self, dataset_type: str, data_dir: str) -> bool:
        """Create sample data for the specified dataset type."""
        try:
            cmd = f"python3 generate_sample_data.py --dataset {dataset_type} --output-dir {data_dir}"
            self.cmd_executor.run_command(cmd)
            return True
        except:
            return False
    
    def validate_dataset(self, dataset_type: str, jsonl_file: str, data_dir: str = None) -> ValidationResult:
        """Validate a dataset using LANISTR data validator."""
        try:
            # Use the appropriate validation function
            if dataset_type == "amazon":
                result = validate_amazon_dataset(jsonl_file, data_dir)
            elif dataset_type == "mimic-iv":
                result = validate_mimic_dataset(jsonl_file, data_dir)
            else:
                return ValidationResult(
                    passed=False,
                    message=f"Unknown dataset type: {dataset_type}",
                    errors=[f"Unsupported dataset type: {dataset_type}"]
                )
            
            return ValidationResult(
                passed=result.get("passed", False),
                message=result.get("message", "Validation completed"),
                stats=result.get("stats", {}),
                errors=result.get("errors", []),
                warnings=result.get("warnings", [])
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Validation failed: {str(e)}",
                errors=[str(e)]
            )


class ContainerManager:
    """Manages container operations for training jobs."""
    
    def __init__(self):
        self.cmd_executor = CommandExecutor()
    
    def build_and_push_image(self, project_id: str, image_name: str = "lanistr-training") -> str:
        """Build and push Docker image to Container Registry."""
        try:
            image_uri = f"gcr.io/{project_id}/{image_name}:latest"
            
            # Build image
            build_cmd = f"docker build -t {image_uri} ."
            self.cmd_executor.run_command(build_cmd, timeout=600)
            
            # Push image
            push_cmd = f"docker push {image_uri}"
            self.cmd_executor.run_command(push_cmd, timeout=600)
            
            return image_uri
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build/push image: {str(e)}")


class JobManager:
    """Manages training job operations."""
    
    def __init__(self):
        self.cmd_executor = CommandExecutor()
        self.gcp_manager = GoogleCloudManager()
        self.container_manager = ContainerManager()
    
    def submit_job(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a training job to Vertex AI."""
        try:
            # Build image if needed
            image_uri = self.container_manager.build_and_push_image(
                config["project_id"], 
                config.get("image_name", "lanistr-training")
            )
            
            # Generate job submission command
            job_name = config.get("job_name") or f"lanistr-{int(time.time())}"
            
            cmd_parts = [
                "python3", "vertex_ai_setup.py",
                f"--project-id={config['project_id']}",
                f"--region={config.get('region', 'us-central1')}",
                f"--job-name={job_name}",
                f"--image-uri={image_uri}",
                f"--dataset-type={config['dataset_type']}"
            ]
            
            # Add optional parameters
            optional_params = ["machine_type", "accelerator_type", "accelerator_count", "bucket_name", "data_dir"]
            for param in optional_params:
                if config.get(param):
                    cmd_parts.append(f"--{param.replace('_', '-')}={config[param]}")
            
            cmd = " ".join(cmd_parts)
            result = self.cmd_executor.run_command(cmd)
            
            return {
                "job_id": job_name,
                "status": "submitted",
                "command": cmd,
                "output": result.stdout
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")
    
    async def run_job_submission(self, job_id: str, job_config: JobConfig, 
                                machine_type: str, accelerator_type: str, accelerator_count: int):
        """Run job submission asynchronously."""
        try:
            # Store job status
            jobs_db[job_id] = JobStatus(
                job_id=job_id,
                name=job_config.job_name or job_id,
                status="preparing",
                created_at=datetime.now(),
                config=job_config.dict()
            )
            
            # Prepare configuration
            config = {
                "project_id": job_config.project_id,
                "region": job_config.region,
                "dataset_type": job_config.dataset_type,
                "job_name": job_id,
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
                "bucket_name": job_config.bucket_name,
                "data_dir": job_config.data_dir
            }
            
            # Submit job
            jobs_db[job_id].status = "submitting"
            result = self.submit_job(config)
            
            # Update status
            jobs_db[job_id].status = "running"
            jobs_db[job_id].logs = result.get("output", "")
            
        except Exception as e:
            if job_id in jobs_db:
                jobs_db[job_id].status = "failed"
                jobs_db[job_id].logs = str(e)


# Initialize managers
gcp_manager = GoogleCloudManager()
data_manager = DataManager()
job_manager = JobManager()

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LANISTR API",
        "version": "1.0.0",
        "description": "REST API for LANISTR training job submission and management",
        "endpoints": {
            "system": "/system/status",
            "jobs": "/jobs/",
            "validation": "/validate/",
            "health": "/health"
        }
    }

@app.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and prerequisites."""
    project_id = gcp_manager.get_project_id()
    prerequisites = gcp_manager.check_prerequisites()
    authenticated = gcp_manager.check_authentication()
    apis_enabled = gcp_manager.check_apis_enabled(project_id) if project_id else {}
    
    return SystemStatus(
        prerequisites=prerequisites,
        authenticated=authenticated,
        project_id=project_id,
        apis_enabled=apis_enabled
    )

@app.post("/jobs/submit")
async def submit_training_job(job_config: JobConfig, background_tasks: BackgroundTasks):
    """Submit a new training job."""
    # Generate job ID
    job_id = job_config.job_name or f"lanistr-{uuid.uuid4().hex[:8]}"
    
    # Validate configuration
    if not gcp_manager.get_project_id():
        raise HTTPException(status_code=400, detail="No Google Cloud project configured")
    
    if not gcp_manager.check_authentication():
        raise HTTPException(status_code=400, detail="Not authenticated with Google Cloud")
    
    # Set default machine configuration
    machine_type = job_config.machine_type or "n1-standard-4"
    accelerator_type = job_config.accelerator_type or "NVIDIA_TESLA_T4"
    accelerator_count = job_config.accelerator_count or 1
    
    # Add background task for job submission
    background_tasks.add_task(
        job_manager.run_job_submission, 
        job_id, job_config, machine_type, accelerator_type, accelerator_count
    )
    
    return {"job_id": job_id, "status": "accepted", "message": "Job submission started"}

@app.get("/jobs/")
async def list_jobs():
    """List all submitted jobs."""
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status,
                "created_at": job.created_at,
                "dataset_type": job.config.get("dataset_type")
            }
            for job in jobs_db.values()
        ]
    }

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get detailed information about a specific job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status,
        "created_at": job.created_at,
        "config": job.config,
        "logs": job.logs,
        "metrics": job.metrics
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from the system."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs_db[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.post("/validate/dataset")
async def validate_dataset_endpoint(
    dataset_type: str = Field(..., description="Dataset type"),
    jsonl_file: str = Field(..., description="Path to JSONL file"),
    data_dir: Optional[str] = Field(None, description="Data directory path"),
    background_tasks: BackgroundTasks = None
):
    """Validate a dataset and return results."""
    # Generate validation ID
    validation_id = f"validation-{uuid.uuid4().hex[:8]}"
    
    # Store initial validation status
    validation_results[validation_id] = {
        "status": "running",
        "started_at": datetime.now(),
        "dataset_type": dataset_type,
        "jsonl_file": jsonl_file
    }
    
    try:
        # Run validation
        result = data_manager.validate_dataset(dataset_type, jsonl_file, data_dir)
        
        # Store results
        validation_results[validation_id].update({
            "status": "completed",
            "completed_at": datetime.now(),
            "result": result
        })
        
        return {
            "validation_id": validation_id,
            "result": result
        }
        
    except Exception as e:
        validation_results[validation_id].update({
            "status": "failed",
            "completed_at": datetime.now(),
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/validate/{validation_id}")
async def get_validation_result(validation_id: str):
    """Get validation result by ID."""
    if validation_id not in validation_results:
        raise HTTPException(status_code=404, detail="Validation not found")
    
    return validation_results[validation_id]

@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...), dataset_type: str = "mimic-iv"):
    """Upload a data file for processing."""
    try:
        # Create upload directory
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "filename": file.filename,
            "file_path": str(file_path),
            "size": len(content),
            "dataset_type": dataset_type,
            "message": "File uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/setup/project")
async def setup_project(project_id: str, region: str = "us-central1"):
    """Setup project with required APIs and resources."""
    try:
        # Enable required APIs
        apis = [
            "aiplatform.googleapis.com",
            "storage.googleapis.com", 
            "cloudbuild.googleapis.com",
            "containerregistry.googleapis.com"
        ]
        
        for api in apis:
            cmd = f"gcloud services enable {api} --project={project_id}"
            gcp_manager.cmd_executor.run_command(cmd)
        
        # Create default bucket
        bucket_name = f"{project_id}-lanistr-data"
        gcp_manager.setup_gcs_bucket(project_id, bucket_name, region)
        
        return {
            "project_id": project_id,
            "region": region,
            "bucket_name": bucket_name,
            "apis_enabled": apis,
            "message": "Project setup completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project setup failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 