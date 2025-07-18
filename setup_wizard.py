#!/usr/bin/env python3
"""
LANISTR Setup Wizard

Interactive setup wizard for configuring LANISTR for Vertex AI training.
Guides users through initial setup with validation and smart defaults.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

def print_header():
    """Print the wizard header."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🚀 LANISTR Setup Wizard                   ║
║                                                              ║
║  This wizard will help you configure LANISTR for Vertex AI  ║
║  training. We'll set up your environment and create         ║
║  configuration files for easy job submission.               ║
╚══════════════════════════════════════════════════════════════╝
""")

def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        return None
    
    return result

def check_prerequisites() -> bool:
    """Check if required tools are installed."""
    print("🔍 Checking prerequisites...")
    
    tools = {
        "gcloud": "Google Cloud SDK",
        "docker": "Docker",
        "python3": "Python 3",
        "gsutil": "Google Cloud Storage (part of gcloud)"
    }
    
    missing = []
    for tool, description in tools.items():
        result = run_command(f"which {tool}", check=False)
        if result.returncode != 0:
            missing.append(f"{tool} ({description})")
        else:
            print(f"✅ {description}")
    
    if missing:
        print(f"\n❌ Missing tools: {', '.join(missing)}")
        print("\nPlease install the missing tools:")
        print("  • Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        print("  • Docker: https://docs.docker.com/get-docker/")
        print("  • Python 3: https://www.python.org/downloads/")
        return False
    
    print("✅ All prerequisites are installed!")
    return True

def get_project_id() -> str:
    """Get or prompt for Google Cloud project ID."""
    print("\n🔧 Google Cloud Project Configuration")
    print("-" * 40)
    
    # Try to get from gcloud
    result = run_command("gcloud config get-value project", check=False)
    if result and result.returncode == 0 and result.stdout.strip():
        project_id = result.stdout.strip()
        print(f"📋 Current project: {project_id}")
        use_current = input("Use this project? (y/n): ").lower().startswith('y')
        if use_current:
            return project_id
    
    # Try environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    if project_id:
        print(f"📋 Environment project: {project_id}")
        use_env = input("Use this project? (y/n): ").lower().startswith('y')
        if use_env:
            return project_id
    
    # Prompt user
    while True:
        project_id = input("Enter your Google Cloud Project ID: ").strip()
        if project_id:
            # Validate project exists
            result = run_command(f"gcloud projects describe {project_id}", check=False)
            if result and result.returncode == 0:
                return project_id
            else:
                print(f"❌ Project '{project_id}' not found or not accessible")
                retry = input("Try again? (y/n): ").lower().startswith('y')
                if not retry:
                    sys.exit(1)
        else:
            print("❌ Project ID is required")

def setup_authentication() -> bool:
    """Set up Google Cloud authentication."""
    print("\n🔐 Google Cloud Authentication")
    print("-" * 40)
    
    # Check if already authenticated
    result = run_command("gcloud auth list --filter=status:ACTIVE --format='value(account)'", check=False)
    if result and result.returncode == 0 and result.stdout.strip():
        account = result.stdout.strip()
        print(f"✅ Already authenticated as: {account}")
        return True
    
    print("🔑 Setting up authentication...")
    
    # Login
    print("Please authenticate with Google Cloud...")
    result = run_command("gcloud auth login", check=False)
    if not result or result.returncode != 0:
        print("❌ Authentication failed")
        return False
    
    # Application default credentials
    print("Setting up application default credentials...")
    result = run_command("gcloud auth application-default login", check=False)
    if not result or result.returncode != 0:
        print("❌ Application default credentials setup failed")
        return False
    
    print("✅ Authentication setup complete!")
    return True

def enable_apis(project_id: str) -> bool:
    """Enable required Google Cloud APIs."""
    print("\n🔌 Enabling Required APIs")
    print("-" * 40)
    
    apis = [
        "aiplatform.googleapis.com",
        "storage.googleapis.com",
        "logging.googleapis.com",
        "monitoring.googleapis.com",
        "errorreporting.googleapis.com",
        "containerregistry.googleapis.com"
    ]
    
    for api in apis:
        print(f"Enabling {api}...")
        result = run_command(f"gcloud services enable {api} --project={project_id}", check=False)
        if result and result.returncode == 0:
            print(f"✅ {api}")
        else:
            print(f"⚠️  {api} (may already be enabled)")
    
    print("✅ API setup complete!")
    return True

def configure_dataset() -> Dict[str, Any]:
    """Configure dataset settings."""
    print("\n📊 Dataset Configuration")
    print("-" * 40)
    
    # Dataset type
    print("Available datasets:")
    print("  1. MIMIC-IV (medical imaging + text + timeseries)")
    print("  2. Amazon (product reviews + images + tabular)")
    
    while True:
        choice = input("Select dataset type (1/2): ").strip()
        if choice == "1":
            dataset_type = "mimic-iv"
            break
        elif choice == "2":
            dataset_type = "amazon"
            break
        else:
            print("❌ Please enter 1 or 2")
    
    # Data directory
    default_data_dir = f"./data/{dataset_type}"
    data_dir = input(f"Data directory (default: {default_data_dir}): ").strip()
    if not data_dir:
        data_dir = default_data_dir
    
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    jsonl_file = f"{data_dir}/{dataset_type}.jsonl"
    if Path(jsonl_file).exists():
        print(f"✅ Found existing data: {jsonl_file}")
    else:
        print(f"⚠️  No data found at: {jsonl_file}")
        create_sample = input("Create sample data for testing? (y/n): ").lower().startswith('y')
        if create_sample:
            print("Creating sample data...")
            cmd = f"python generate_sample_data.py --dataset {dataset_type} --output-file {jsonl_file} --num-samples 100 --create-files"
            run_command(cmd)
            print(f"✅ Sample data created: {jsonl_file}")
    
    return {
        "dataset_type": dataset_type,
        "data_dir": data_dir,
        "jsonl_file": jsonl_file
    }

def configure_compute() -> Dict[str, Any]:
    """Configure compute resources."""
    print("\n⚡ Compute Configuration")
    print("-" * 40)
    
    # Environment type
    print("Environment types:")
    print("  1. Development (cheaper, slower)")
    print("  2. Production (faster, more expensive)")
    
    while True:
        choice = input("Select environment type (1/2): ").strip()
        if choice == "1":
            env_type = "dev"
            break
        elif choice == "2":
            env_type = "prod"
            break
        else:
            print("❌ Please enter 1 or 2")
    
    # Machine type
    if env_type == "dev":
        machine_type = "n1-standard-2"
        accelerator_type = "NVIDIA_TESLA_T4"
        accelerator_count = 1
    else:
        machine_type = "n1-standard-4"
        accelerator_type = "NVIDIA_TESLA_V100"
        accelerator_count = 8
    
    # Allow customization
    print(f"\nDefault configuration for {env_type} environment:")
    print(f"  Machine Type: {machine_type}")
    print(f"  GPU Type: {accelerator_type}")
    print(f"  GPU Count: {accelerator_count}")
    
    customize = input("Customize these settings? (y/n): ").lower().startswith('y')
    
    if customize:
        # Machine type
        print("\nMachine types:")
        print("  n1-standard-2  (2 vCPU, 7.5GB RAM) - $0.095/hour")
        print("  n1-standard-4  (4 vCPU, 15GB RAM)  - $0.190/hour")
        print("  n1-standard-8  (8 vCPU, 30GB RAM)  - $0.380/hour")
        
        custom_machine = input(f"Machine type (default: {machine_type}): ").strip()
        if custom_machine:
            machine_type = custom_machine
        
        # GPU type
        print("\nGPU types:")
        print("  NVIDIA_TESLA_T4   - $0.35/hour per GPU")
        print("  NVIDIA_TESLA_V100 - $2.48/hour per GPU")
        print("  NVIDIA_TESLA_A100 - $3.67/hour per GPU")
        
        custom_gpu = input(f"GPU type (default: {accelerator_type}): ").strip()
        if custom_gpu:
            accelerator_type = custom_gpu
        
        # GPU count
        custom_count = input(f"GPU count (default: {accelerator_count}): ").strip()
        if custom_count and custom_count.isdigit():
            accelerator_count = int(custom_count)
    
    return {
        "machine_type": machine_type,
        "accelerator_type": accelerator_type,
        "accelerator_count": accelerator_count,
        "env_type": env_type
    }

def configure_storage(project_id: str) -> Dict[str, Any]:
    """Configure storage settings."""
    print("\n📦 Storage Configuration")
    print("-" * 40)
    
    # Bucket name
    default_bucket = f"lanistr-{project_id}-data"
    bucket_name = input(f"GCS bucket name (default: {default_bucket}): ").strip()
    if not bucket_name:
        bucket_name = default_bucket
    
    # Region
    default_region = "us-central1"
    region = input(f"GCP region (default: {default_region}): ").strip()
    if not region:
        region = default_region
    
    # Create bucket if it doesn't exist
    print(f"Setting up bucket: gs://{bucket_name}")
    result = run_command(f"gsutil ls -b gs://{bucket_name}", check=False)
    if not result or result.returncode != 0:
        print("Creating bucket...")
        run_command(f"gsutil mb -p {project_id} -c STANDARD -l {region} gs://{bucket_name}")
        print(f"✅ Created bucket: gs://{bucket_name}")
    else:
        print(f"✅ Bucket already exists: gs://{bucket_name}")
    
    return {
        "bucket_name": bucket_name,
        "region": region
    }

def create_config_file(config: Dict[str, Any]) -> str:
    """Create configuration file."""
    print("\n📝 Creating Configuration File")
    print("-" * 40)
    
    config_file = "lanistr_config.json"
    
    config_data = {
        "project_id": config["project_id"],
        "region": config["region"],
        "bucket_name": config["bucket_name"],
        "dataset": config["dataset"],
        "compute": config["compute"],
        "created_at": str(Path().absolute()),
        "version": "1.0"
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"✅ Configuration saved to: {config_file}")
    return config_file

def create_quick_scripts(config: Dict[str, Any]) -> None:
    """Create quick start scripts."""
    print("\n📜 Creating Quick Start Scripts")
    print("-" * 40)
    
    # Quick submit script
    quick_script = "quick_submit.sh"
    with open(quick_script, 'w') as f:
        f.write(f"""#!/bin/bash
# Quick submit script for LANISTR
# Generated by setup wizard

set -e

# Load configuration
PROJECT_ID="{config['project_id']}"
REGION="{config['region']}"
BUCKET_NAME="{config['bucket_name']}"
DATASET_TYPE="{config['dataset']['dataset_type']}"
DATA_DIR="{config['dataset']['data_dir']}"

# Submit job using CLI
python lanistr-cli.py \\
    --dataset $DATASET_TYPE \\
    --data-dir $DATA_DIR \\
    --project-id $PROJECT_ID \\
    --bucket-name $BUCKET_NAME \\
    --region $REGION \\
    --machine-type "{config['compute']['machine_type']}" \\
    --accelerator-type "{config['compute']['accelerator_type']}" \\
    --accelerator-count {config['compute']['accelerator_count']} \\
    {"--dev" if config['compute']['env_type'] == 'dev' else ""}

echo "Job submitted! Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
""")
    
    run_command(f"chmod +x {quick_script}")
    print(f"✅ Created: {quick_script}")
    
    # Development script
    dev_script = "quick_submit_dev.sh"
    with open(dev_script, 'w') as f:
        f.write(f"""#!/bin/bash
# Quick submit script for development (cheaper)
# Generated by setup wizard

set -e

# Load configuration
PROJECT_ID="{config['project_id']}"
REGION="{config['region']}"
BUCKET_NAME="{config['bucket_name']}"
DATASET_TYPE="{config['dataset']['dataset_type']}"
DATA_DIR="{config['dataset']['data_dir']}"

# Submit job using CLI in development mode
python lanistr-cli.py \\
    --dataset $DATASET_TYPE \\
    --data-dir $DATA_DIR \\
    --project-id $PROJECT_ID \\
    --bucket-name $BUCKET_NAME \\
    --region $REGION \\
    --dev

echo "Development job submitted! Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
""")
    
    run_command(f"chmod +x {dev_script}")
    print(f"✅ Created: {dev_script}")

def show_next_steps(config: Dict[str, Any]) -> None:
    """Show next steps to the user."""
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    
    print("Your LANISTR environment is now configured!")
    print()
    print("📋 Configuration Summary:")
    print(f"  Project: {config['project_id']}")
    print(f"  Region: {config['region']}")
    print(f"  Bucket: gs://{config['bucket_name']}")
    print(f"  Dataset: {config['dataset']['dataset_type']}")
    print(f"  Data Dir: {config['dataset']['data_dir']}")
    print(f"  Machine: {config['compute']['machine_type']}")
    print(f"  GPU: {config['compute']['accelerator_count']}x {config['compute']['accelerator_type']}")
    print()
    print("🚀 Next Steps:")
    print("  1. Prepare your data in the specified directory")
    print("  2. Run: ./quick_submit.sh")
    print("  3. Monitor your job in the Google Cloud Console")
    print()
    print("📚 Useful Commands:")
    print("  ./quick_submit.sh          # Submit production job")
    print("  ./quick_submit_dev.sh      # Submit development job")
    print("  python lanistr-cli.py --help  # See all CLI options")
    print("  python validate_dataset.py --help  # Validate your data")
    print()
    print("📖 Documentation:")
    print("  JOB_SUBMISSION_GUIDE.md    # Complete guide")
    print("  QUICK_SUBMISSION_CARD.md   # Quick reference")
    print("  DATASET_REQUIREMENTS.md    # Data specifications")

def main():
    """Main wizard function."""
    print_header()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Get project ID
    project_id = get_project_id()
    
    # Set up authentication
    if not setup_authentication():
        sys.exit(1)
    
    # Enable APIs
    if not enable_apis(project_id):
        sys.exit(1)
    
    # Configure dataset
    dataset_config = configure_dataset()
    
    # Configure compute
    compute_config = configure_compute()
    
    # Configure storage
    storage_config = configure_storage(project_id)
    
    # Combine configuration
    config = {
        "project_id": project_id,
        "region": storage_config["region"],
        "bucket_name": storage_config["bucket_name"],
        "dataset": dataset_config,
        "compute": compute_config
    }
    
    # Create configuration file
    config_file = create_config_file(config)
    
    # Create quick scripts
    create_quick_scripts(config)
    
    # Show next steps
    show_next_steps(config)

if __name__ == "__main__":
    main() 