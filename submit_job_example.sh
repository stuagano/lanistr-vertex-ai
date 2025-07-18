#!/bin/bash

# LANISTR Job Submission Example
# This script demonstrates how to submit a training job to Vertex AI

set -e  # Exit on any error

# Configuration - UPDATE THESE VALUES
PROJECT_ID="your-project-id"
REGION="us-central1"
BUCKET_NAME="lanistr-data-bucket"
DATASET_TYPE="mimic-iv"  # or "amazon"
JOB_NAME="lanistr-${DATASET_TYPE}-pretrain-$(date +%Y%m%d-%H%M%S)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_prerequisites() {
    print_step "1" "Checking prerequisites..."
    
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud is not installed. Please install Google Cloud SDK."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_error "docker is not installed. Please install Docker."
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        print_error "python3 is not installed."
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Authenticate with Google Cloud
authenticate_gcloud() {
    print_step "2" "Authenticating with Google Cloud..."
    
    # Check if already authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_warning "No active Google Cloud authentication found"
        gcloud auth login
        gcloud auth application-default login
    else
        print_success "Already authenticated with Google Cloud"
    fi
    
    # Set project
    gcloud config set project "$PROJECT_ID"
    print_success "Project set to: $PROJECT_ID"
}

# Validate dataset
validate_dataset() {
    print_step "3" "Validating dataset..."
    
    # Determine dataset file based on type
    if [ "$DATASET_TYPE" = "mimic-iv" ]; then
        DATASET_FILE="./data/mimic.jsonl"
        CONFIG_FILE="lanistr/configs/mimic_pretrain.yaml"
    elif [ "$DATASET_TYPE" = "amazon" ]; then
        DATASET_FILE="./data/amazon.jsonl"
        CONFIG_FILE="lanistr/configs/amazon_pretrain_office.yaml"
    else
        print_error "Invalid dataset type: $DATASET_TYPE. Use 'mimic-iv' or 'amazon'"
        exit 1
    fi
    
    # Check if dataset file exists
    if [ ! -f "$DATASET_FILE" ]; then
        print_warning "Dataset file $DATASET_FILE not found. Generating sample data..."
        python3 generate_sample_data.py --dataset "$DATASET_TYPE" --output-file "$DATASET_FILE" --num-samples 100 --create-files
    fi
    
    # Validate dataset
    if python3 validate_dataset.py --dataset "$DATASET_TYPE" --jsonl-file "$DATASET_FILE" --data-dir ./data; then
        print_success "Dataset validation passed"
    else
        print_error "Dataset validation failed. Please fix issues before proceeding."
        exit 1
    fi
}

# Create GCS bucket and upload data
setup_storage() {
    print_step "4" "Setting up Google Cloud Storage..."
    
    # Create bucket if it doesn't exist
    if ! gsutil ls -b "gs://$BUCKET_NAME" >/dev/null 2>&1; then
        print_warning "Bucket gs://$BUCKET_NAME does not exist. Creating..."
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$BUCKET_NAME"
        print_success "Bucket created: gs://$BUCKET_NAME"
    else
        print_success "Bucket already exists: gs://$BUCKET_NAME"
    fi
    
    # Upload data
    print_warning "Uploading data to GCS (this may take a while)..."
    gsutil -m cp -r ./data/ "gs://$BUCKET_NAME/" || {
        print_warning "Data upload failed or data directory is empty. Continuing with sample data..."
        # Create sample data and upload
        python3 generate_sample_data.py --dataset "$DATASET_TYPE" --output-file "./data/sample.jsonl" --num-samples 100 --create-files
        gsutil -m cp -r ./data/ "gs://$BUCKET_NAME/"
    }
    
    print_success "Data uploaded to gs://$BUCKET_NAME/"
}

# Build and push Docker image
build_image() {
    print_step "5" "Building and pushing Docker image..."
    
    IMAGE_NAME="lanistr-training"
    IMAGE_URI="gcr.io/$PROJECT_ID/$IMAGE_NAME:latest"
    
    # Build Docker image
    print_warning "Building Docker image (this may take several minutes)..."
    docker build -t "$IMAGE_NAME:latest" . || {
        print_error "Docker build failed"
        exit 1
    }
    
    # Tag for Google Container Registry
    docker tag "$IMAGE_NAME:latest" "$IMAGE_URI"
    
    # Push to GCR
    print_warning "Pushing image to Google Container Registry (this may take several minutes)..."
    docker push "$IMAGE_URI" || {
        print_error "Failed to push image to GCR"
        exit 1
    }
    
    print_success "Image pushed: $IMAGE_URI"
    echo "$IMAGE_URI" > .image_uri
}

# Submit training job
submit_job() {
    print_step "6" "Submitting training job to Vertex AI..."
    
    # Get image URI
    IMAGE_URI=$(cat .image_uri 2>/dev/null || echo "gcr.io/$PROJECT_ID/lanistr-training:latest")
    
    # Determine config file
    if [ "$DATASET_TYPE" = "mimic-iv" ]; then
        CONFIG_FILE="lanistr/configs/mimic_pretrain.yaml"
    else
        CONFIG_FILE="lanistr/configs/amazon_pretrain_office.yaml"
    fi
    
    # Submit job
    python3 vertex_ai_setup.py \
        --project-id "$PROJECT_ID" \
        --location "$REGION" \
        --job-name "$JOB_NAME" \
        --config-file "$CONFIG_FILE" \
        --machine-type "n1-standard-4" \
        --accelerator-type "NVIDIA_TESLA_V100" \
        --accelerator-count 8 \
        --replica-count 1 \
        --base-output-dir "gs://$BUCKET_NAME/lanistr-output" \
        --base-data-dir "gs://$BUCKET_NAME" \
        --image-uri "$IMAGE_URI" || {
        print_error "Job submission failed"
        exit 1
    }
    
    print_success "Job '$JOB_NAME' submitted successfully!"
}

# Show monitoring information
show_monitoring() {
    print_step "7" "Monitoring information..."
    
    echo
    echo "🎯 Job Details:"
    echo "   Job Name: $JOB_NAME"
    echo "   Dataset: $DATASET_TYPE"
    echo "   Project: $PROJECT_ID"
    echo "   Region: $REGION"
    echo "   Bucket: gs://$BUCKET_NAME"
    echo
    echo "📊 Monitor your job:"
    echo "   Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
    echo "   Command: gcloud ai custom-jobs list"
    echo
    echo "📁 Output location:"
    echo "   gs://$BUCKET_NAME/lanistr-output/$JOB_NAME"
    echo
    print_success "Job submission complete!"
}

# Main execution
main() {
    echo "🚀 LANISTR Job Submission Example"
    echo "=================================="
    echo "Dataset Type: $DATASET_TYPE"
    echo "Project ID: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Bucket: $BUCKET_NAME"
    echo "Job Name: $JOB_NAME"
    echo
    
    # Check if PROJECT_ID is set
    if [ "$PROJECT_ID" = "your-project-id" ]; then
        print_error "Please update PROJECT_ID in this script before running"
        exit 1
    fi
    
    check_prerequisites
    authenticate_gcloud
    validate_dataset
    setup_storage
    build_image
    submit_job
    show_monitoring
}

# Run main function
main "$@" 