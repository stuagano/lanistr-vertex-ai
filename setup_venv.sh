#!/bin/bash

# LANISTR Virtual Environment Setup Script
# This script automates the setup of a virtual environment for LANISTR

set -e  # Exit on any error

echo "🚀 LANISTR Virtual Environment Setup"
echo "====================================="

# Check if we're in the right directory
if [ ! -f "requirements-core.txt" ]; then
    echo "❌ Error: requirements-core.txt not found. Please run this script from the LANISTR root directory."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo "📋 Python version: $PYTHON_VERSION"

# Check Python version compatibility
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 8 ]]; then
    echo "❌ Error: Python $PYTHON_VERSION is not supported"
    echo "   LANISTR requires Python 3.8 or higher"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION is compatible with PyTorch 2.0+"

# Set environment name
ENV_NAME="lanistr_env"

# Check if virtual environment already exists
if [ -d "$ENV_NAME" ]; then
    echo "⚠️  Virtual environment '$ENV_NAME' already exists."
    read -p "Remove existing environment and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf "$ENV_NAME"
    else
        echo "Using existing virtual environment."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_NAME" ]; then
    echo "📦 Creating virtual environment '$ENV_NAME'..."
    python -m venv "$ENV_NAME"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$ENV_NAME/bin/activate"

# Verify activation
if [[ "$VIRTUAL_ENV" != *"$ENV_NAME"* ]]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Choose installation type
echo ""
echo "Choose installation type:"
echo "1) Minimal cloud setup (recommended - no compilation)"
echo "2) Cloud deployment setup (includes pandas/numpy)"
echo "3) Core setup (includes local PyTorch)"
echo "4) Full development setup (includes all tools)"
echo "5) Production setup (for Vertex AI)"
echo "6) Custom setup (manual selection)"

read -p "Enter your choice (1-6): " -n 1 -r
echo

case $REPLY in
    1)
        echo "☁️  Installing minimal cloud requirements (no compilation)..."
        pip install -r requirements-minimal.txt
        ;;
    2)
        echo "☁️  Installing cloud deployment requirements..."
        pip install -r requirements-cloud.txt
        ;;
    3)
        echo "📦 Installing core requirements..."
        pip install -r requirements-core.txt
        ;;
    4)
        echo "🛠️  Installing development requirements..."
        pip install -r requirements-dev.txt
        ;;
    5)
        echo "🏭 Installing production requirements..."
        pip install -r requirements_vertex_ai.txt
        ;;
    6)
        echo "🔧 Manual installation mode"
        echo "Please install packages manually:"
        echo "  pip install -r requirements-minimal.txt"
        echo "  # or"
        echo "  pip install -r requirements-cloud.txt"
        echo "  # or"
        echo "  pip install -r requirements-core.txt"
        echo "  # or"
        echo "  pip install -r requirements-dev.txt"
        echo "  # or"
        echo "  pip install -r requirements_vertex_ai.txt"
        ;;
    *)
        echo "❌ Invalid choice. Using minimal cloud setup."
        pip install -r requirements-minimal.txt
        ;;
esac

# Install LANISTR in development mode
echo "🔧 Installing LANISTR in development mode..."
pip install -e .

# Verify installation
echo "✅ Verifying installation..."
python -c "
import torch
import transformers
import omegaconf
import pandas as pd
import numpy as np
print('✅ All core packages imported successfully!')
print(f'📊 PyTorch version: {torch.__version__}')
print(f'🚀 CUDA available: {torch.cuda.is_available()}')
"

# Test LANISTR imports
echo "🔍 Testing LANISTR imports..."
python -c "
try:
    from lanistr.dataset.mimic_iv.load_data import load_mimic
    from lanistr.model.modeling_lanistr import LANISTRMultiModalForPreTraining
    print('✅ LANISTR modules imported successfully!')
except ImportError as e:
    print(f'❌ LANISTR import error: {e}')
    print('This might be normal if data files are not set up yet.')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: source $ENV_NAME/bin/activate"
echo "2. Read the tutorial: jupyter notebook lanistr_tutorial.ipynb"
echo "3. Prepare your data (see README.md)"
echo "4. Start training with: python lanistr/main.py --config lanistr/configs/mimic_pretrain.yaml"
echo ""
echo "🔧 Environment management:"
echo "  Activate:   source $ENV_NAME/bin/activate"
echo "  Deactivate: deactivate"
echo "  Remove:     rm -rf $ENV_NAME"
echo ""
echo "📚 For more information, see: venv_setup_guide.md" 