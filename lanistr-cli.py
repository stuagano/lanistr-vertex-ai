#!/usr/bin/env python3
"""
LANISTR CLI - Easy Job Submission Tool

A simple command-line interface for submitting LANISTR training jobs to Vertex AI
with minimal configuration and smart defaults.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Configuration templates
DEFAULT_CONFIGS = {
    "mimic-iv": {
        "config_file": "lanistr/configs/mimic_pretrain.yaml",
        "machine_type": "n1-standard-4",
        "accelerator_type": "NVIDIA_TESLA_V100",
        "accelerator_count": 8,
        "min_examples": 100
    },
    "amazon": {
        "config_file": "lanistr/configs/amazon_pretrain_office.yaml",
        "machine_type": "n1-standard-4", 
        "accelerator_type": "NVIDIA_TESLA_V100",
        "accelerator_count": 8,
        "min_examples": 1000
    }
}

# Cost-effective configurations for development
DEV_CONFIGS = {
    "mimic-iv": {
        "machine_type": "n1-standard-2",
        "accelerator_type": "NVIDIA_TESLA_T4",
        "accelerator_count": 1
    },
    "amazon": {
        "machine_type": "n1-standard-2",
        "accelerator_type": "NVIDIA_TESLA_T4", 
        "accelerator_count": 1
    }
}

def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"🔄 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result

def get_project_id() -> str:
    """Get the current Google Cloud project ID."""
    try:
        result = run_command("gcloud config get-value project", check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except:
        pass
    
    # Try to get from environment
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    if project_id:
        return project_id
    
    # Prompt user
    project_id = input("🔧 Enter your Google Cloud Project ID: ").strip()
    if not project_id:
        print("❌ Project ID is required")
        sys.exit(1)
    
    return project_id

def auto_detect_dataset_type(data_dir: str) -> Optional[str]:
    """Auto-detect dataset type from data directory."""
    data_path = Path(data_dir)
    
    # Check for MIMIC-IV files
    if (data_path / "mimic.jsonl").exists() or (data_path / "MIMIC-IV-V2.2").exists():
        return "mimic-iv"
    
    # Check for Amazon files
    if (data_path / "amazon.jsonl").exists() or (data_path / "APR2018").exists():
        return "amazon"
    
    return None

def validate_dataset(dataset_type: str, data_dir: str, jsonl_file: str) -> bool:
    """Validate the dataset using the validation script."""
    print(f"🔍 Validating {dataset_type} dataset...")
    
    cmd = f"python validate_dataset.py --dataset {dataset_type} --jsonl-file {jsonl_file} --data-dir {data_dir}"
    result = run_command(cmd, check=False)
    
    if result.returncode == 0:
        print("✅ Dataset validation passed")
        return True
    else:
        print("❌ Dataset validation failed")
        print(result.stdout)
        print(result.stderr)
        return False

def setup_gcs_bucket(project_id: str, bucket_name: str, region: str) -> str:
    """Set up GCS bucket and return the bucket name."""
    print(f"📦 Setting up GCS bucket: gs://{bucket_name}")
    
    # Check if bucket exists
    result = run_command(f"gsutil ls -b gs://{bucket_name}", check=False)
    if result.returncode != 0:
        # Create bucket
        run_command(f"gsutil mb -p {project_id} -c STANDARD -l {region} gs://{bucket_name}")
        print(f"✅ Created bucket: gs://{bucket_name}")
    else:
        print(f"✅ Bucket already exists: gs://{bucket_name}")
    
    return bucket_name

def upload_data(data_dir: str, bucket_name: str) -> bool:
    """Upload data to GCS bucket."""
    print(f"📤 Uploading data to gs://{bucket_name}/...")
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"⚠️  Data directory {data_dir} not found")
        return False
    
    # Upload data
    result = run_command(f"gsutil -m cp -r {data_dir}/ gs://{bucket_name}/", check=False)
    if result.returncode == 0:
        print("✅ Data uploaded successfully")
        return True
    else:
        print("❌ Data upload failed")
        print(result.stderr)
        return False

def build_and_push_image(project_id: str, image_name: str = "lanistr-training") -> str:
    """Build and push Docker image to GCR."""
    print("🐳 Building and pushing Docker image...")
    
    image_uri = f"gcr.io/{project_id}/{image_name}:latest"
    
    # Build image
    run_command(f"docker build -t {image_name}:latest .")
    
    # Tag for GCR
    run_command(f"docker tag {image_name}:latest {image_uri}")
    
    # Push to GCR
    run_command(f"docker push {image_uri}")
    
    print(f"✅ Image pushed: {image_uri}")
    return image_uri

def submit_job(
    project_id: str,
    job_name: str,
    dataset_type: str,
    bucket_name: str,
    image_uri: str,
    region: str = "us-central1",
    dev_mode: bool = False
) -> bool:
    """Submit the training job to Vertex AI."""
    print(f"🚀 Submitting job: {job_name}")
    
    # Get configuration
    config = DEFAULT_CONFIGS[dataset_type].copy()
    if dev_mode:
        config.update(DEV_CONFIGS[dataset_type])
    
    # Build command
    cmd = [
        "python", "vertex_ai_setup.py",
        "--project-id", project_id,
        "--location", region,
        "--job-name", job_name,
        "--config-file", config["config_file"],
        "--machine-type", config["machine_type"],
        "--accelerator-type", config["accelerator_type"],
        "--accelerator-count", str(config["accelerator_count"]),
        "--replica-count", "1",
        "--base-output-dir", f"gs://{bucket_name}/lanistr-output",
        "--base-data-dir", f"gs://{bucket_name}",
        "--image-uri", image_uri
    ]
    
    result = run_command(" ".join(cmd), check=False)
    
    if result.returncode == 0:
        print("✅ Job submitted successfully!")
        print(f"📊 Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
        return True
    else:
        print("❌ Job submission failed")
        print(result.stderr)
        return False

def create_sample_data(dataset_type: str, data_dir: str) -> str:
    """Create sample data if none exists."""
    print(f"📝 Creating sample {dataset_type} data...")
    
    jsonl_file = f"{data_dir}/{dataset_type}.jsonl"
    
    cmd = f"python generate_sample_data.py --dataset {dataset_type} --output-file {jsonl_file} --num-samples 100 --create-files"
    run_command(cmd)
    
    print(f"✅ Sample data created: {jsonl_file}")
    return jsonl_file

def main():
    parser = argparse.ArgumentParser(
        description="LANISTR CLI - Easy Job Submission Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with auto-detection
  python lanistr-cli.py --data-dir ./data

  # Specify dataset type
  python lanistr-cli.py --dataset mimic-iv --data-dir ./data

  # Development mode (cheaper)
  python lanistr-cli.py --dataset amazon --data-dir ./data --dev

  # Custom configuration
  python lanistr-cli.py --dataset mimic-iv --data-dir ./data --machine-type n1-standard-8 --gpus 4
        """
    )
    
    parser.add_argument("--dataset", choices=["mimic-iv", "amazon"], 
                       help="Dataset type (auto-detected if not specified)")
    parser.add_argument("--data-dir", default="./data", 
                       help="Data directory (default: ./data)")
    parser.add_argument("--jsonl-file", 
                       help="JSONL file path (auto-detected if not specified)")
    parser.add_argument("--project-id", 
                       help="Google Cloud Project ID (auto-detected if not specified)")
    parser.add_argument("--bucket-name", 
                       help="GCS bucket name (auto-generated if not specified)")
    parser.add_argument("--region", default="us-central1", 
                       help="GCP region (default: us-central1)")
    parser.add_argument("--job-name", 
                       help="Job name (auto-generated if not specified)")
    parser.add_argument("--machine-type", 
                       help="Machine type (default: n1-standard-4)")
    parser.add_argument("--accelerator-type", 
                       help="GPU type (default: NVIDIA_TESLA_V100)")
    parser.add_argument("--accelerator-count", type=int, 
                       help="Number of GPUs (default: 8)")
    parser.add_argument("--dev", action="store_true", 
                       help="Use development configuration (cheaper)")
    parser.add_argument("--skip-validation", action="store_true", 
                       help="Skip dataset validation")
    parser.add_argument("--skip-upload", action="store_true", 
                       help="Skip data upload (assumes data is already in GCS)")
    parser.add_argument("--skip-build", action="store_true", 
                       help="Skip Docker build (assumes image already exists)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without executing")
    
    args = parser.parse_args()
    
    print("🚀 LANISTR CLI - Easy Job Submission")
    print("=" * 50)
    
    # Auto-detect values
    project_id = args.project_id or get_project_id()
    dataset_type = args.dataset or auto_detect_dataset_type(args.data_dir)
    
    if not dataset_type:
        print("❌ Could not auto-detect dataset type. Please specify --dataset")
        sys.exit(1)
    
    # Set up paths
    data_dir = args.data_dir
    jsonl_file = args.jsonl_file or f"{data_dir}/{dataset_type}.jsonl"
    bucket_name = args.bucket_name or f"lanistr-{project_id}-{dataset_type}"
    job_name = args.job_name or f"lanistr-{dataset_type}-{os.getpid()}"
    
    # Show configuration
    print(f"📋 Configuration:")
    print(f"   Dataset: {dataset_type}")
    print(f"   Project: {project_id}")
    print(f"   Region: {args.region}")
    print(f"   Bucket: gs://{bucket_name}")
    print(f"   Job Name: {job_name}")
    print(f"   Data Dir: {data_dir}")
    print(f"   JSONL File: {jsonl_file}")
    print(f"   Dev Mode: {args.dev}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No actions will be performed")
        return
    
    # Check if JSONL file exists, create sample if not
    if not Path(jsonl_file).exists():
        print(f"⚠️  JSONL file not found: {jsonl_file}")
        create_sample = input("Create sample data? (y/n): ").lower().startswith('y')
        if create_sample:
            jsonl_file = create_sample_data(dataset_type, data_dir)
        else:
            print("❌ Please provide a valid JSONL file")
            sys.exit(1)
    
    # Validate dataset
    if not args.skip_validation:
        if not validate_dataset(dataset_type, data_dir, jsonl_file):
            print("❌ Dataset validation failed. Please fix issues before proceeding.")
            sys.exit(1)
    
    # Set up GCS bucket
    setup_gcs_bucket(project_id, bucket_name, args.region)
    
    # Upload data
    if not args.skip_upload:
        if not upload_data(data_dir, bucket_name):
            print("⚠️  Data upload failed, but continuing...")
    
    # Build and push image
    if not args.skip_build:
        image_uri = build_and_push_image(project_id)
    else:
        image_uri = f"gcr.io/{project_id}/lanistr-training:latest"
    
    # Submit job
    success = submit_job(
        project_id=project_id,
        job_name=job_name,
        dataset_type=dataset_type,
        bucket_name=bucket_name,
        image_uri=image_uri,
        region=args.region,
        dev_mode=args.dev
    )
    
    if success:
        print("\n🎉 Job submission completed successfully!")
        print(f"📊 Monitor your job: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
        print(f"📁 Output location: gs://{bucket_name}/lanistr-output/{job_name}")
    else:
        print("\n❌ Job submission failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 