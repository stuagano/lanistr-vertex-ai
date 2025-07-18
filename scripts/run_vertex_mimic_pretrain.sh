#!/bin/bash

# MIMIC-IV Pretraining on Vertex AI
# Usage: ./scripts/run_vertex_mimic_pretrain.sh

set -e

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
REGION=${REGION:-"us-central1"}
JOB_NAME="lanistr-mimic-pretrain-$(date +%Y%m%d-%H%M%S)"
CONFIG_FILE="vertex_ai_configs/mimic_pretrain_vertex.yaml"
MACHINE_TYPE="n1-standard-4"
ACCELERATOR_TYPE="NVIDIA_TESLA_V100"
ACCELERATOR_COUNT=8
REPLICA_COUNT=1
BASE_OUTPUT_DIR="gs://your-bucket/lanistr-output"
BASE_DATA_DIR="gs://your-bucket/lanistr-data"
IMAGE_URI="gcr.io/${PROJECT_ID}/lanistr-training:latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting MIMIC-IV pretraining on Vertex AI...${NC}"

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo -e "${RED}Error: Please set PROJECT_ID environment variable${NC}"
    echo "Usage: PROJECT_ID=your-project-id ./scripts/run_vertex_mimic_pretrain.sh"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Run the training job
python3 vertex_ai_setup.py \
    --project-id "$PROJECT_ID" \
    --location "$REGION" \
    --job-name "$JOB_NAME" \
    --config-file "$CONFIG_FILE" \
    --machine-type "$MACHINE_TYPE" \
    --accelerator-type "$ACCELERATOR_TYPE" \
    --accelerator-count "$ACCELERATOR_COUNT" \
    --replica-count "$REPLICA_COUNT" \
    --base-output-dir "$BASE_OUTPUT_DIR" \
    --base-data-dir "$BASE_DATA_DIR" \
    --image-uri "$IMAGE_URI"

echo -e "${GREEN}MIMIC-IV pretraining job submitted successfully!${NC}"
echo -e "${YELLOW}Job Name: $JOB_NAME${NC}"
echo -e "${YELLOW}Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs${NC}" 