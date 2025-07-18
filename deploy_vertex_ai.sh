#!/bin/bash

# Comprehensive Vertex AI Deployment Script for LANISTR
# This script handles the complete deployment process for distributed training

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
REGION=${REGION:-"us-central1"}
BUCKET_NAME=${BUCKET_NAME:-"lanistr-data-bucket"}
IMAGE_NAME="lanistr-training"
IMAGE_TAG="latest"
GCR_HOSTNAME="gcr.io"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command_exists gcloud; then
        missing_deps+=("gcloud")
    fi
    
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Function to authenticate with Google Cloud
authenticate_gcloud() {
    print_status "Authenticating with Google Cloud..."
    
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_warning "No active Google Cloud authentication found"
        gcloud auth login
    else
        print_success "Already authenticated with Google Cloud"
    fi
    
    # Set project
    gcloud config set project "$PROJECT_ID"
    print_success "Project set to: $PROJECT_ID"
}

# Function to enable required APIs
enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    local apis=(
        "aiplatform.googleapis.com"
        "storage.googleapis.com"
        "logging.googleapis.com"
        "monitoring.googleapis.com"
        "errorreporting.googleapis.com"
        "containerregistry.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        if ! gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
            print_status "Enabling $api..."
            gcloud services enable "$api"
        else
            print_status "$api is already enabled"
        fi
    done
    
    print_success "All required APIs are enabled"
}

# Function to create GCS bucket
create_bucket() {
    print_status "Setting up Google Cloud Storage bucket..."
    
    if ! gsutil ls -b "gs://$BUCKET_NAME" >/dev/null 2>&1; then
        print_status "Creating bucket: gs://$BUCKET_NAME"
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$BUCKET_NAME"
        
        # Set bucket permissions
        gsutil iam ch allUsers:objectViewer "gs://$BUCKET_NAME"
    else
        print_success "Bucket gs://$BUCKET_NAME already exists"
    fi
}

# Function to upload data
upload_data() {
    print_status "Uploading data to Google Cloud Storage..."
    
    local data_dir="./lanistr/data"
    
    if [ ! -d "$data_dir" ]; then
        print_warning "Data directory $data_dir not found. Skipping data upload."
        return
    fi
    
    # Upload MIMIC-IV data
    if [ -d "$data_dir/MIMIC-IV-V2.2" ]; then
        print_status "Uploading MIMIC-IV data..."
        gsutil -m cp -r "$data_dir/MIMIC-IV-V2.2" "gs://$BUCKET_NAME/"
        print_success "MIMIC-IV data uploaded"
    fi
    
    # Upload Amazon data
    if [ -d "$data_dir/amazon" ]; then
        print_status "Uploading Amazon data..."
        gsutil -m cp -r "$data_dir/amazon" "gs://$BUCKET_NAME/"
        print_success "Amazon data uploaded"
    fi
}

# Function to build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    local image_uri="$GCR_HOSTNAME/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG"
    
    # Build the Docker image
    print_status "Building Docker image..."
    docker build -t "$IMAGE_NAME:$IMAGE_TAG" .
    
    # Tag the image for Google Container Registry
    print_status "Tagging image for Google Container Registry..."
    docker tag "$IMAGE_NAME:$IMAGE_TAG" "$image_uri"
    
    # Push the image to Google Container Registry
    print_status "Pushing image to Google Container Registry..."
    docker push "$image_uri"
    
    print_success "Image pushed: $image_uri"
    echo "$image_uri" > .image_uri
}

# Function to create Vertex AI training scripts
create_training_scripts() {
    print_status "Creating training scripts..."
    
    local image_uri
    image_uri=$(cat .image_uri 2>/dev/null || echo "$GCR_HOSTNAME/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG")
    
    # Create MIMIC-IV pretraining script
    cat > run_mimic_pretrain.sh << EOF
#!/bin/bash

# MIMIC-IV Pretraining on Vertex AI
# Usage: ./run_mimic_pretrain.sh

PROJECT_ID=$PROJECT_ID
REGION=$REGION
IMAGE_URI="$image_uri"
BUCKET_NAME=$BUCKET_NAME

python3 vertex_ai_setup.py \\
    --project-id \${PROJECT_ID} \\
    --location \${REGION} \\
    --job-name "lanistr-mimic-pretrain" \\
    --config-file "lanistr/configs/mimic_pretrain.yaml" \\
    --machine-type "n1-standard-4" \\
    --accelerator-type "NVIDIA_TESLA_V100" \\
    --accelerator-count 8 \\
    --replica-count 1 \\
    --base-output-dir "gs://\${BUCKET_NAME}/lanistr-output" \\
    --base-data-dir "gs://\${BUCKET_NAME}" \\
    --image-uri \${IMAGE_URI}

echo "MIMIC-IV pretraining job submitted!"
EOF

    # Create Amazon pretraining script
    cat > run_amazon_pretrain.sh << EOF
#!/bin/bash

# Amazon Pretraining on Vertex AI
# Usage: ./run_amazon_pretrain.sh

PROJECT_ID=$PROJECT_ID
REGION=$REGION
IMAGE_URI="$image_uri"
BUCKET_NAME=$BUCKET_NAME

python3 vertex_ai_setup.py \\
    --project-id \${PROJECT_ID} \\
    --location \${REGION} \\
    --job-name "lanistr-amazon-pretrain" \\
    --config-file "lanistr/configs/amazon_pretrain_office.yaml" \\
    --machine-type "n1-standard-4" \\
    --accelerator-type "NVIDIA_TESLA_V100" \\
    --accelerator-count 8 \\
    --replica-count 1 \\
    --base-output-dir "gs://\${BUCKET_NAME}/lanistr-output" \\
    --base-data-dir "gs://\${BUCKET_NAME}" \\
    --image-uri \${IMAGE_URI}

echo "Amazon pretraining job submitted!"
EOF

    # Create multi-node training script
    cat > run_multi_node_training.sh << EOF
#!/bin/bash

# Multi-Node Training on Vertex AI
# Usage: ./run_multi_node_training.sh [num_nodes] [gpus_per_node]

NUM_NODES=\${1:-2}
GPUS_PER_NODE=\${2:-8}

PROJECT_ID=$PROJECT_ID
REGION=$REGION
IMAGE_URI="$image_uri"
BUCKET_NAME=$BUCKET_NAME

python3 vertex_ai_setup.py \\
    --project-id \${PROJECT_ID} \\
    --location \${REGION} \\
    --job-name "lanistr-multi-node-\${NUM_NODES}x\${GPUS_PER_NODE}" \\
    --config-file "lanistr/configs/mimic_pretrain.yaml" \\
    --machine-type "n1-standard-4" \\
    --accelerator-type "NVIDIA_TESLA_V100" \\
    --accelerator-count \${GPUS_PER_NODE} \\
    --replica-count \${NUM_NODES} \\
    --base-output-dir "gs://\${BUCKET_NAME}/lanistr-output" \\
    --base-data-dir "gs://\${BUCKET_NAME}" \\
    --image-uri \${IMAGE_URI}

echo "Multi-node training job submitted: \${NUM_NODES} nodes x \${GPUS_PER_NODE} GPUs!"
EOF

    # Make scripts executable
    chmod +x run_mimic_pretrain.sh run_amazon_pretrain.sh run_multi_node_training.sh
    
    print_success "Training scripts created"
}

# Function to create monitoring script
create_monitoring_script() {
    print_status "Creating monitoring script..."
    
    cat > monitor_training.sh << EOF
#!/bin/bash

# Monitor Vertex AI Training Jobs
# Usage: ./monitor_training.sh [job_name]

JOB_NAME=\${1:-""}

echo "=== Vertex AI Training Jobs ==="
gcloud ai custom-jobs list --region=$REGION --format="table(name,displayName,state,createTime)"

if [ -n "\$JOB_NAME" ]; then
    echo ""
    echo "=== Job Details: \$JOB_NAME ==="
    gcloud ai custom-jobs describe "\$JOB_NAME" --region=$REGION
fi

echo ""
echo "=== Recent Logs ==="
gcloud logging read "resource.type=ml_job" --limit=10 --format="table(timestamp,textPayload)"
EOF

    chmod +x monitor_training.sh
    print_success "Monitoring script created"
}

# Function to create cleanup script
create_cleanup_script() {
    print_status "Creating cleanup script..."
    
    cat > cleanup_vertex_ai.sh << EOF
#!/bin/bash

# Cleanup Vertex AI Resources
# Usage: ./cleanup_vertex_ai.sh [--force]

FORCE=\${1:-""}

echo "=== Cleaning up Vertex AI resources ==="

# List and optionally delete training jobs
echo "Training jobs:"
gcloud ai custom-jobs list --region=$REGION --format="table(name,displayName,state)"

if [ "\$FORCE" = "--force" ]; then
    echo "Deleting all completed training jobs..."
    gcloud ai custom-jobs list --region=$REGION --filter="state=JOB_STATE_SUCCEEDED OR state=JOB_STATE_FAILED" --format="value(name)" | xargs -I {} gcloud ai custom-jobs delete {} --region=$REGION --quiet
fi

# Clean up old Docker images
if [ "\$FORCE" = "--force" ]; then
    echo "Cleaning up old Docker images..."
    docker image prune -f
    docker system prune -f
fi

echo "Cleanup complete"
EOF

    chmod +x cleanup_vertex_ai.sh
    print_success "Cleanup script created"
}

# Function to display next steps
show_next_steps() {
    print_success "Deployment completed successfully!"
    echo ""
    echo "=== Next Steps ==="
    echo "1. Upload your data to Google Cloud Storage:"
    echo "   gsutil -m cp -r ./lanistr/data/* gs://$BUCKET_NAME/"
    echo ""
    echo "2. Run training jobs:"
    echo "   ./run_mimic_pretrain.sh     # MIMIC-IV pretraining"
    echo "   ./run_amazon_pretrain.sh    # Amazon pretraining"
    echo "   ./run_multi_node_training.sh 2 8  # 2 nodes x 8 GPUs"
    echo ""
    echo "3. Monitor training:"
    echo "   ./monitor_training.sh"
    echo ""
    echo "4. View logs in Google Cloud Console:"
    echo "   https://console.cloud.google.com/vertex-ai/training/custom-jobs"
    echo ""
    echo "5. Clean up resources when done:"
    echo "   ./cleanup_vertex_ai.sh --force"
}

# Main deployment function
main() {
    echo "=========================================="
    echo "LANISTR Vertex AI Deployment Script"
    echo "=========================================="
    echo ""
    
    # Check if PROJECT_ID is set
    if [ "$PROJECT_ID" = "your-project-id" ]; then
        print_error "Please set PROJECT_ID environment variable"
        echo "Usage: PROJECT_ID=your-project-id ./deploy_vertex_ai.sh"
        exit 1
    fi
    
    # Run deployment steps
    check_prerequisites
    authenticate_gcloud
    enable_apis
    create_bucket
    upload_data
    build_and_push_image
    create_training_scripts
    create_monitoring_script
    create_cleanup_script
    show_next_steps
}

# Run main function
main "$@" 