# LANISTR Virtual Environment Setup Guide

This guide will help you set up a proper virtual environment for LANISTR development and training.

## Prerequisites

- Python 3.8-3.10 (recommended for PyTorch compatibility)
- pip (Python package installer)
- git (for cloning the repository)

## Current System Information

- **Python Version**: 3.13.3 (⚠️ **Note**: This is newer than recommended for PyTorch 1.11.0)
- **Operating System**: macOS (darwin)
- **Working Directory**: `/Users/stuartgano/lanistr`

## ⚠️ Important Compatibility Note

You're running Python 3.13.3, but the requirements specify PyTorch 1.11.0 which has compatibility constraints. For optimal compatibility, consider using Python 3.8-3.10.

## Setup Options

### Option 1: Quick Setup (Recommended for Development)

```bash
# 1. Create a virtual environment
python -m venv lanistr_env

# 2. Activate the virtual environment
source lanistr_env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install core requirements (for local development)
pip install -r requirements-core.txt

# 5. Install LANISTR in development mode
pip install -e .
```

### Option 2: Full Development Setup

```bash
# 1. Create a virtual environment
python -m venv lanistr_env

# 2. Activate the virtual environment
source lanistr_env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install development requirements (includes all tools)
pip install -r requirements-dev.txt

# 5. Install LANISTR in development mode
pip install -e .

# 6. Set up pre-commit hooks (optional)
pre-commit install
```

### Option 3: Production Setup (for Vertex AI)

```bash
# 1. Create a virtual environment
python -m venv lanistr_env

# 2. Activate the virtual environment
source lanistr_env/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install production requirements
pip install -r requirements_vertex_ai.txt

# 5. Install LANISTR in development mode
pip install -e .
```

## Step-by-Step Setup Commands

Run these commands in your terminal:

```bash
# Navigate to the LANISTR directory
cd /Users/stuartgano/lanistr

# Create virtual environment
python -m venv lanistr_env

# Activate virtual environment
source lanistr_env/bin/activate

# Verify activation (should show lanistr_env in prompt)
which python

# Upgrade pip
pip install --upgrade pip

# Install core requirements
pip install -r requirements-core.txt

# Install LANISTR in development mode
pip install -e .

# Verify installation
python -c "import lanistr; print('LANISTR installed successfully!')"
```

## Environment Management

### Activating the Environment

```bash
# Activate virtual environment
source lanistr_env/bin/activate

# Your prompt should change to show (lanistr_env)
```

### Deactivating the Environment

```bash
# Deactivate virtual environment
deactivate
```

### Checking Installed Packages

```bash
# List installed packages
pip list

# Check specific package versions
pip show torch
pip show transformers
```

## Troubleshooting

### Python Version Issues

If you encounter compatibility issues with Python 3.13.3:

1. **Install Python 3.10** (recommended):
   ```bash
   # Using Homebrew (if available)
   brew install python@3.10
   
   # Or download from python.org
   ```

2. **Create virtual environment with specific Python version**:
   ```bash
   python3.10 -m venv lanistr_env
   ```

### PyTorch Installation Issues

If PyTorch installation fails:

```bash
# For macOS (CPU only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For macOS with M1/M2 (if available)
pip install torch torchvision torchaudio
```

### Memory Issues

If you encounter memory issues during installation:

```bash
# Install packages one by one
pip install torch
pip install transformers
pip install omegaconf
# ... continue with other packages
```

### Permission Issues

If you encounter permission errors:

```bash
# Use --user flag
pip install --user -r requirements-core.txt

# Or fix permissions
sudo chown -R $(whoami) lanistr_env/
```

## Verification

After setup, verify everything is working:

```bash
# Activate environment
source lanistr_env/bin/activate

# Test imports
python -c "
import torch
import transformers
import omegaconf
import pandas as pd
import numpy as np
print('All core packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Test LANISTR imports
python -c "
from lanistr.dataset.mimic_iv.load_data import load_mimic
from lanistr.model.modeling_lanistr import LANISTRMultiModalForPreTraining
print('LANISTR modules imported successfully!')
"
```

## Development Workflow

### Daily Usage

```bash
# 1. Activate environment
source lanistr_env/bin/activate

# 2. Work on your code
# ... your development work ...

# 3. Deactivate when done
deactivate
```

### Running Tests

```bash
# Activate environment
source lanistr_env/bin/activate

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=lanistr --cov-report=html
```

### Running Training

```bash
# Activate environment
source lanistr_env/bin/activate

# Run training
python lanistr/main.py --config lanistr/configs/mimic_pretrain.yaml
```

## Environment Variables

You might want to set these environment variables:

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export PYTHONPATH="${PYTHONPATH}:/Users/stuartgano/lanistr"
export CUDA_VISIBLE_DEVICES=0  # If using GPU
export TOKENIZERS_PARALLELISM=false  # For transformers
```

## Next Steps

After setting up your virtual environment:

1. **Read the tutorial**: Open `lanistr_tutorial.ipynb` in Jupyter
2. **Prepare data**: Follow the data preparation instructions in the README
3. **Start training**: Use the configuration files in `lanistr/configs/`
4. **Explore the codebase**: Check out the examples and documentation

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the README.md file
3. Check the requirements files for version compatibility
4. Ensure your Python version is compatible with PyTorch

Happy coding with LANISTR! 🚀 