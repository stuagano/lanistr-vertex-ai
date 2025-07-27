#!/bin/bash

# Quick Minimal LANISTR Container Build
# This builds a lightweight container in ~5-10 minutes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ID="mgm-digitalconcierge"
REGION="us-central1"

echo -e "${BLUE}🚀 Quick Minimal LANISTR Container Build${NC}"
echo "=============================================="
echo ""

echo -e "${YELLOW}Configuration:${NC}"
echo "  Project ID: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Build time: ~5-10 minutes"
echo ""

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Google Cloud SDK is not installed${NC}"
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${RED}❌ Not authenticated with Google Cloud${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"

# Enable APIs
echo -e "${BLUE}🔧 Enabling APIs...${NC}"
gcloud services enable containerregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com
echo -e "${GREEN}✅ APIs enabled${NC}"

# Build the minimal container
echo -e "${BLUE}🚀 Building minimal container...${NC}"
echo "This should take ~5-10 minutes..."

gcloud builds submit \
    --config cloudbuild-minimal.yaml \
    --timeout=900s \
    --machine-type=E2_HIGHCPU_8 \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 Success! Minimal container is ready${NC}"
    echo "=============================================="
    echo ""
    echo -e "${YELLOW}Container URI:${NC}"
    echo "gcr.io/${PROJECT_ID}/lanistr:latest"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Update your notebook with this container URI"
    echo "2. You can now submit training jobs!"
    echo ""
    echo -e "${GREEN}Happy cloud training! ☁️🚀${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi 