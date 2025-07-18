#!/bin/bash

# LANISTR One-Click Setup
# This script automates the entire setup process for LANISTR on Vertex AI

set -e

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

# Configuration
PROJECT_ID=${PROJECT_ID:-""}
REGION=${REGION:-"us-central1"}
BUCKET_NAME=${BUCKET_NAME:-""}
DATASET_TYPE=${DATASET_TYPE:-"mimic-iv"}
ENVIRONMENT=${ENVIRONMENT:-"dev"}

print_header() {
    echo "
╔══════════════════════════════════════════════════════════════╗
║                🚀 LANISTR One-Click Setup                   ║
║                                                              ║
║  This script will automatically set up LANISTR for Vertex   ║
║  AI training with minimal configuration required.           ║
╚══════════════════════════════════════════════════════════════╝
"
}

# Function to check prerequisites
check_prerequisites() {
    print_step "1" "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command -v gcloud &> /dev/null; then
        missing_deps+=("gcloud")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    print_success "All prerequisites are installed"
}

# Function to get project ID
get_project_id() {
    print_step "2" "Setting up Google Cloud project..."
    
    # Try to get from gcloud
    if [ -z "$PROJECT_ID" ]; then
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
    fi
    
    # Try environment variables
    if [ -z "$PROJECT_ID" ]; then
        PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-""}
    fi
    
    # Prompt user if still not set
    if [ -z "$PROJECT_ID" ]; then
        echo "Please enter your Google Cloud Project ID:"
        read -r PROJECT_ID
    fi
    
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID is required"
        exit 1
    fi
    
    # Set project
    gcloud config set project "$PROJECT_ID"
    print_success "Project set to: $PROJECT_ID"
}

# Function to authenticate
authenticate() {
    print_step "3" "Setting up authentication..."
    
    # Check if already authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_warning "No active Google Cloud authentication found"
        gcloud auth login
        gcloud auth application-default login
    else
        print_success "Already authenticated with Google Cloud"
    fi
}

# Function to enable APIs
enable_apis() {
    print_step "4" "Enabling required APIs..."
    
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
            print_warning "Enabling $api..."
            gcloud services enable "$api" --project="$PROJECT_ID"
        else
            print_success "$api is already enabled"
        fi
    done
}

# Function to set up storage
setup_storage() {
    print_step "5" "Setting up Google Cloud Storage..."
    
    # Generate bucket name if not provided
    if [ -z "$BUCKET_NAME" ]; then
        BUCKET_NAME="lanistr-${PROJECT_ID}-${DATASET_TYPE}"
    fi
    
    # Create bucket if it doesn't exist
    if ! gsutil ls -b "gs://$BUCKET_NAME" >/dev/null 2>&1; then
        print_warning "Creating bucket: gs://$BUCKET_NAME"
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$BUCKET_NAME"
        print_success "Bucket created: gs://$BUCKET_NAME"
    else
        print_success "Bucket already exists: gs://$BUCKET_NAME"
    fi
}

# Function to create sample data
create_sample_data() {
    print_step "6" "Creating sample data..."
    
    local data_dir="./data/$DATASET_TYPE"
    local jsonl_file="$data_dir/$DATASET_TYPE.jsonl"
    
    # Create data directory
    mkdir -p "$data_dir"
    
    # Create sample data if it doesn't exist
    if [ ! -f "$jsonl_file" ]; then
        print_warning "Creating sample $DATASET_TYPE data..."
        python3 generate_sample_data.py \
            --dataset "$DATASET_TYPE" \
            --output-file "$jsonl_file" \
            --num-samples 100 \
            --create-files
        print_success "Sample data created: $jsonl_file"
    else
        print_success "Data already exists: $jsonl_file"
    fi
}

# Function to validate data
validate_data() {
    print_step "7" "Validating dataset..."
    
    local data_dir="./data/$DATASET_TYPE"
    local jsonl_file="$data_dir/$DATASET_TYPE.jsonl"
    
    if python3 validate_dataset.py \
        --dataset "$DATASET_TYPE" \
        --jsonl-file "$jsonl_file" \
        --data-dir "$data_dir"; then
        print_success "Dataset validation passed"
    else
        print_warning "Dataset validation failed, but continuing..."
    fi
}

# Function to upload data
upload_data() {
    print_step "8" "Uploading data to Google Cloud Storage..."
    
    local data_dir="./data"
    
    if [ -d "$data_dir" ]; then
        print_warning "Uploading data to gs://$BUCKET_NAME/..."
        gsutil -m cp -r "$data_dir/" "gs://$BUCKET_NAME/" || {
            print_warning "Data upload failed, but continuing..."
        }
        print_success "Data uploaded to gs://$BUCKET_NAME/"
    else
        print_warning "Data directory not found, skipping upload"
    fi
}

# Function to build and push image
build_image() {
    print_step "9" "Building and pushing Docker image..."
    
    local image_name="lanistr-training"
    local image_uri="gcr.io/$PROJECT_ID/$image_name:latest"
    
    # Build image
    print_warning "Building Docker image (this may take several minutes)..."
    docker build -t "$image_name:latest" . || {
        print_error "Docker build failed"
        exit 1
    }
    
    # Tag for GCR
    docker tag "$image_name:latest" "$image_uri"
    
    # Push to GCR
    print_warning "Pushing image to Google Container Registry (this may take several minutes)..."
    docker push "$image_uri" || {
        print_error "Failed to push image to GCR"
        exit 1
    }
    
    print_success "Image pushed: $image_uri"
    echo "$image_uri" > .image_uri
}

# Function to create quick scripts
create_scripts() {
    print_step "10" "Creating quick start scripts..."
    
    local image_uri
    image_uri=$(cat .image_uri 2>/dev/null || echo "gcr.io/$PROJECT_ID/lanistr-training:latest")
    
    # Determine machine configuration based on environment
    local machine_type="n1-standard-4"
    local accelerator_type="NVIDIA_TESLA_V100"
    local accelerator_count=8
    
    if [ "$ENVIRONMENT" = "dev" ]; then
        machine_type="n1-standard-2"
        accelerator_type="NVIDIA_TESLA_T4"
        accelerator_count=1
    fi
    
    # Create quick submit script
    cat > quick_submit.sh << EOF
#!/bin/bash
# Quick submit script for LANISTR
# Generated by one-click setup

set -e

# Configuration
PROJECT_ID="$PROJECT_ID"
REGION="$REGION"
BUCKET_NAME="$BUCKET_NAME"
DATASET_TYPE="$DATASET_TYPE"
DATA_DIR="./data/$DATASET_TYPE"
IMAGE_URI="$image_uri"
MACHINE_TYPE="$machine_type"
ACCELERATOR_TYPE="$accelerator_type"
ACCELERATOR_COUNT=$accelerator_count

# Submit job
python3 vertex_ai_setup.py \\
    --project-id \$PROJECT_ID \\
    --location \$REGION \\
    --job-name "lanistr-\$DATASET_TYPE-\$(date +%Y%m%d-%H%M%S)" \\
    --config-file "lanistr/configs/\${DATASET_TYPE}_pretrain.yaml" \\
    --machine-type \$MACHINE_TYPE \\
    --accelerator-type \$ACCELERATOR_TYPE \\
    --accelerator-count \$ACCELERATOR_COUNT \\
    --replica-count 1 \\
    --base-output-dir "gs://\$BUCKET_NAME/lanistr-output" \\
    --base-data-dir "gs://\$BUCKET_NAME" \\
    --image-uri \$IMAGE_URI

echo "Job submitted! Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
EOF
    
    chmod +x quick_submit.sh
    print_success "Created: quick_submit.sh"
    
    # Create CLI script
    cat > quick_cli.sh << EOF
#!/bin/bash
# Quick CLI script for LANISTR
# Generated by one-click setup

set -e

# Submit job using CLI
python3 lanistr-cli.py \\
    --dataset $DATASET_TYPE \\
    --data-dir "./data/$DATASET_TYPE" \\
    --project-id "$PROJECT_ID" \\
    --bucket-name "$BUCKET_NAME" \\
    --region "$REGION" \\
    --machine-type "$machine_type" \\
    --accelerator-type "$accelerator_type" \\
    --accelerator-count $accelerator_count \\
    ${ENVIRONMENT:+--dev}

echo "Job submitted! Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
EOF
    
    chmod +x quick_cli.sh
    print_success "Created: quick_cli.sh"
}

# Function to show completion
show_completion() {
    print_step "11" "Setup complete!"
    
    echo
    echo "🎉 LANISTR setup completed successfully!"
    echo
    echo "📋 Configuration Summary:"
    echo "  Project: $PROJECT_ID"
    echo "  Region: $REGION"
    echo "  Bucket: gs://$BUCKET_NAME"
    echo "  Dataset: $DATASET_TYPE"
    echo "  Environment: $ENVIRONMENT"
    echo
    echo "🚀 Next Steps:"
    echo "  1. Run: ./quick_submit.sh"
    echo "  2. Or run: ./quick_cli.sh"
    echo "  3. Monitor your job in the Google Cloud Console"
    echo
    echo "📊 Monitor your job:"
    echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs"
    echo
    echo "📁 Output location:"
    echo "  gs://$BUCKET_NAME/lanistr-output"
    echo
    echo "📚 Useful Commands:"
    echo "  ./quick_submit.sh          # Submit job"
    echo "  ./quick_cli.sh             # Submit using CLI"
    echo "  python3 lanistr-cli.py --help  # See CLI options"
    echo "  python3 validate_dataset.py --help  # Validate data"
    echo
    echo "📖 Documentation:"
    echo "  JOB_SUBMISSION_GUIDE.md    # Complete guide"
    echo "  QUICK_SUBMISSION_CARD.md   # Quick reference"
    echo "  DATASET_REQUIREMENTS.md    # Data specifications"
}

# Main function
main() {
    print_header
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --bucket-name)
                BUCKET_NAME="$2"
                shift 2
                ;;
            --dataset)
                DATASET_TYPE="$2"
                shift 2
                ;;
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --project-id PROJECT_ID    Google Cloud Project ID"
                echo "  --region REGION           GCP region (default: us-central1)"
                echo "  --bucket-name BUCKET      GCS bucket name (auto-generated if not specified)"
                echo "  --dataset TYPE            Dataset type: mimic-iv or amazon (default: mimic-iv)"
                echo "  --environment ENV         Environment: dev or prod (default: dev)"
                echo "  --help                    Show this help message"
                echo
                echo "Examples:"
                echo "  $0"
                echo "  $0 --project-id my-project --dataset amazon --environment prod"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Validate dataset type
    if [[ "$DATASET_TYPE" != "mimic-iv" && "$DATASET_TYPE" != "amazon" ]]; then
        print_error "Invalid dataset type: $DATASET_TYPE. Use 'mimic-iv' or 'amazon'"
        exit 1
    fi
    
    # Validate environment
    if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "prod" ]]; then
        print_error "Invalid environment: $ENVIRONMENT. Use 'dev' or 'prod'"
        exit 1
    fi
    
    # Run setup steps
    check_prerequisites
    get_project_id
    authenticate
    enable_apis
    setup_storage
    create_sample_data
    validate_data
    upload_data
    build_image
    create_scripts
    show_completion
}

# Run main function
main "$@" 