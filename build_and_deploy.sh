#!/bin/bash

# Build and Deploy Script for LANISTR Vertex AI Training
# This script builds the Docker image and deploys it to Google Cloud

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
REGION=${REGION:-"us-central1"}
IMAGE_NAME="lanistr-training"
IMAGE_TAG="latest"
GCR_HOSTNAME="gcr.io"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building and deploying LANISTR for Vertex AI training...${NC}"

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo -e "${RED}Error: Please set PROJECT_ID environment variable${NC}"
    echo "Usage: PROJECT_ID=your-project-id ./build_and_deploy.sh"
    exit 1
fi

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Tag the image for Google Container Registry
echo -e "${YELLOW}Tagging image for Google Container Registry...${NC}"
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${GCR_HOSTNAME}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}

# Push the image to Google Container Registry
echo -e "${YELLOW}Pushing image to Google Container Registry...${NC}"
docker push ${GCR_HOSTNAME}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}

echo -e "${GREEN}Successfully built and deployed image: ${GCR_HOSTNAME}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}${NC}"

# Create a sample training script
cat > run_vertex_training.sh << EOF
#!/bin/bash

# Sample script to run training on Vertex AI
# Usage: ./run_vertex_training.sh

PROJECT_ID=${PROJECT_ID}
REGION=${REGION}
IMAGE_URI="${GCR_HOSTNAME}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

# Example: Run MIMIC-IV pretraining
python3 vertex_ai_setup.py \\
    --project-id \${PROJECT_ID} \\
    --location \${REGION} \\
    --job-name "lanistr-mimic-pretrain" \\
    --config-file "configs/mimic_pretrain.yaml" \\
    --machine-type "n1-standard-4" \\
    --accelerator-type "NVIDIA_TESLA_V100" \\
    --accelerator-count 8 \\
    --replica-count 1 \\
    --base-output-dir "gs://your-bucket/lanistr-output" \\
    --base-data-dir "gs://your-bucket/lanistr-data" \\
    --image-uri \${IMAGE_URI}

echo "Training job submitted! Check the Vertex AI console for progress."
EOF

chmod +x run_vertex_training.sh

echo -e "${GREEN}Created run_vertex_training.sh script${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Upload your data to Google Cloud Storage"
echo "2. Update the bucket paths in run_vertex_training.sh"
echo "3. Run: ./run_vertex_training.sh" 