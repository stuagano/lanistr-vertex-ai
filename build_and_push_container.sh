#!/bin/bash

# LANISTR Container Build and Push Script
# This script builds the LANISTR Docker container and pushes it to Google Container Registry

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - UPDATE THESE WITH YOUR VALUES
PROJECT_ID="mgm-digitalconcierge"  # Replace with your actual project ID
REGION="us-central1"  # Choose your preferred region
IMAGE_NAME="lanistr"
TAG="latest"

# Derived values
FULL_IMAGE_NAME="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"

echo -e "${BLUE}🚀 LANISTR Container Build and Push Script${NC}"
echo "=================================================="
echo ""

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo -e "${RED}❌ Error: Please update PROJECT_ID in this script${NC}"
    echo "Edit build_and_push_container.sh and set PROJECT_ID to your actual project ID"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Project ID: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Image: ${FULL_IMAGE_NAME}"
echo ""

# Step 1: Check prerequisites
echo -e "${BLUE}📋 Step 1: Checking prerequisites...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✅ Docker is installed${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Google Cloud SDK is not installed${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo -e "${GREEN}✅ Google Cloud SDK is installed${NC}"

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${RED}❌ Not authenticated with Google Cloud${NC}"
    echo "Run: gcloud auth login"
    exit 1
fi
echo -e "${GREEN}✅ Authenticated with Google Cloud${NC}"

# Check if project is set
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo -e "${YELLOW}⚠️  Setting project to ${PROJECT_ID}${NC}"
    gcloud config set project $PROJECT_ID
fi
echo -e "${GREEN}✅ Project is set to ${PROJECT_ID}${NC}"

# Step 2: Enable required APIs
echo -e "${BLUE}🔧 Step 2: Enabling required APIs...${NC}"
gcloud services enable containerregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com
echo -e "${GREEN}✅ APIs enabled${NC}"

# Step 3: Configure Docker for GCR
echo -e "${BLUE}🐳 Step 3: Configuring Docker for Google Container Registry...${NC}"
gcloud auth configure-docker
echo -e "${GREEN}✅ Docker configured for GCR${NC}"

# Step 4: Build the Docker image
echo -e "${BLUE}🔨 Step 4: Building Docker image...${NC}"
echo "This may take several minutes..."

# Build with progress output
docker build -t $FULL_IMAGE_NAME . --progress=plain

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Step 5: Push the image to GCR
echo -e "${BLUE}📤 Step 5: Pushing image to Google Container Registry...${NC}"
echo "This may take several minutes..."

docker push $FULL_IMAGE_NAME

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image pushed successfully to GCR${NC}"
else
    echo -e "${RED}❌ Failed to push image${NC}"
    exit 1
fi

# Step 6: Verify the image
echo -e "${BLUE}🔍 Step 6: Verifying the image...${NC}"
gcloud container images list-tags gcr.io/$PROJECT_ID/$IMAGE_NAME --limit=1

echo ""
echo -e "${GREEN}🎉 Success! LANISTR container is ready for Vertex AI${NC}"
echo "=================================================="
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update your notebook with the correct PROJECT_ID: ${PROJECT_ID}"
echo "2. The container URI is: ${FULL_IMAGE_NAME}"
echo "3. You can now submit training jobs using the notebook"
echo ""
echo -e "${BLUE}Example job submission:${NC}"
echo "PROJECT_ID = \"${PROJECT_ID}\""
echo "container_uri = \"${FULL_IMAGE_NAME}\""
echo ""
echo -e "${GREEN}Happy cloud training! ☁️🚀${NC}" 