#!/bin/bash

# LANISTR Cloud Build Trigger Script
# This script triggers a Cloud Build to build and push the LANISTR container

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

echo -e "${BLUE}🚀 LANISTR Cloud Build Trigger Script${NC}"
echo "=============================================="
echo ""

# Check if PROJECT_ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo -e "${RED}❌ Error: Please update PROJECT_ID in this script${NC}"
    echo "Edit trigger_cloud_build.sh and set PROJECT_ID to your actual project ID"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Project ID: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo ""

# Step 1: Check prerequisites
echo -e "${BLUE}📋 Step 1: Checking prerequisites...${NC}"

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
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com
echo -e "${GREEN}✅ APIs enabled${NC}"

# Step 3: Check if cloudbuild.yaml exists
if [ ! -f "cloudbuild.yaml" ]; then
    echo -e "${RED}❌ cloudbuild.yaml not found${NC}"
    echo "Make sure you're in the LANISTR project directory"
    exit 1
fi
echo -e "${GREEN}✅ cloudbuild.yaml found${NC}"

# Step 4: Trigger Cloud Build
echo -e "${BLUE}🚀 Step 3: Triggering Cloud Build...${NC}"
echo "This will build the container on Google Cloud and push to Container Registry"
echo "Build logs will be streamed to your terminal"
echo ""

# Trigger the build
BUILD_ID=$(gcloud builds submit --config=cloudbuild.yaml --format="value(id)" .)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Cloud Build triggered successfully!${NC}"
    echo -e "${GREEN}✅ Build ID: ${BUILD_ID}${NC}"
    echo ""
    echo -e "${YELLOW}Build Details:${NC}"
    echo "  Build ID: ${BUILD_ID}"
    echo "  Project: ${PROJECT_ID}"
    echo "  Status: https://console.cloud.google.com/cloud-build/builds/${BUILD_ID}?project=${PROJECT_ID}"
    echo ""
    echo -e "${YELLOW}Container Details:${NC}"
    echo "  Image: gcr.io/${PROJECT_ID}/lanistr:latest"
    echo "  Registry: https://console.cloud.google.com/gcr/images/${PROJECT_ID}/lanistr?project=${PROJECT_ID}"
    echo ""
    echo -e "${GREEN}🎉 Container will be available at: gcr.io/${PROJECT_ID}/lanistr:latest${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Wait for the build to complete (usually 10-20 minutes)"
    echo "2. Update your notebook with the container URI: gcr.io/${PROJECT_ID}/lanistr:latest"
    echo "3. Submit training jobs using the notebook"
    echo ""
    echo -e "${GREEN}Happy cloud training! ☁️🚀${NC}"
else
    echo -e "${RED}❌ Cloud Build failed${NC}"
    echo "Check the build logs for more details"
    exit 1
fi 