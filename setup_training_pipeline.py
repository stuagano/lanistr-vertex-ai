#!/usr/bin/env python3
"""
LANISTR Training Pipeline Setup
This script sets up the complete training pipeline for LANISTR on macOS.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

def print_step(step: int, message: str):
    """Print a formatted step message."""
    print(f"\n{'='*60}")
    print(f"STEP {step}: {message}")
    print(f"{'='*60}")

def print_success(message: str):
    """Print a success message."""
    print(f"✅ {message}")

def print_warning(message: str):
    """Print a warning message."""
    print(f"⚠️  {message}")

def print_error(message: str):
    """Print an error message."""
    print(f"❌ {message}")

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    print_step(1, "Checking Dependencies")
    
    required_packages = [
        'torch', 'torchvision', 'torchaudio', 'torchmetrics',
        'transformers', 'omegaconf', 'numpy', 'pandas',
        'sklearn', 'PIL', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            missing_packages.append(package)
            print_error(f"{package} is missing")
    
    if missing_packages:
        print_error(f"Missing packages: {', '.join(missing_packages)}")
        print_warning("Run: pip install -r requirements-core.txt")
        return False
    
    print_success("All dependencies are installed")
    return True

def setup_directories() -> Dict[str, Path]:
    """Create necessary directories for training."""
    print_step(2, "Setting up Directories")
    
    directories = {
        'data': Path('./data'),
        'output': Path('./output_dir'),
        'logs': Path('./logs'),
        'checkpoints': Path('./checkpoints'),
        'configs': Path('./lanistr/configs'),
        'models': Path('./lanistr/model'),
        'utils': Path('./lanistr/utils'),
        'dataset': Path('./lanistr/dataset'),
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created directory: {path}")
    
    return directories

def create_sample_data(dataset_type: str = 'mimic-iv') -> bool:
    """Create sample data for training."""
    print_step(3, "Creating Sample Data")
    
    try:
        # Check if sample data already exists
        data_file = Path(f'./data/{dataset_type}/{dataset_type}.jsonl')
        if data_file.exists():
            print_success(f"Sample data already exists: {data_file}")
            return True
        
        # Create sample data using the existing script
        cmd = [
            sys.executable, 'generate_sample_data.py',
            '--dataset', dataset_type,
            '--output-file', str(data_file),
            '--num-samples', '100',
            '--create-files'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success(f"Sample data created: {data_file}")
            return True
        else:
            print_error(f"Failed to create sample data: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"Error creating sample data: {e}")
        return False

def validate_data(dataset_type: str = 'mimic-iv') -> bool:
    """Validate the dataset."""
    print_step(4, "Validating Dataset")
    
    try:
        data_file = Path(f'./data/{dataset_type}/{dataset_type}.jsonl')
        data_dir = Path(f'./data/{dataset_type}')
        
        if not data_file.exists():
            print_error(f"Data file not found: {data_file}")
            return False
        
        cmd = [
            sys.executable, 'validate_dataset.py',
            '--dataset', dataset_type,
            '--jsonl-file', str(data_file),
            '--data-dir', str(data_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success("Dataset validation passed")
            return True
        else:
            print_warning(f"Dataset validation failed: {result.stderr}")
            print_warning("Continuing with setup...")
            return True  # Continue anyway
            
    except Exception as e:
        print_error(f"Error validating dataset: {e}")
        return False

def create_training_config(dataset_type: str = 'mimic-iv', task: str = 'pretrain') -> bool:
    """Create a training configuration file."""
    print_step(5, "Creating Training Configuration")
    
    config_dir = Path('./lanistr/configs')
    config_file = config_dir / f'{dataset_type}_{task}_local.yaml'
    
    # Base configuration
    config = {
        'seed': 2022,
        'sub_samples': 0,
        'do_train': True,
        'do_test': False,
        'dataset_name': dataset_type,
        'task': task,
        'image': True,
        'text': True,
        'tab': False,
        'time': True,
        'pretrain_resume': False,
        'pretrain_initialize_from_epoch': 0,
        'image_size': 224,
        'image_crop': 224,
        'mask_patch_size': 16,
        'model_patch_size': 16,
        'image_masking_ratio': 0.5,
        'root_data_dir': f'./data/{dataset_type}',
        'image_data_dir': f'./data/{dataset_type}/images',
        'task_data_dir': f'./data/{dataset_type}/task',
        'unimodal_data_dir': f'./data/{dataset_type}/unimodal',
        'preprocessed_data_dir': f'./data/{dataset_type}/',
        'output_dir': f'./output_dir/{dataset_type}_{task}',
        'experiment_name': f'{dataset_type}_{task}',
        'test_ratio': 0.2,
        'train_batch_size': 8,  # Smaller for local training
        'eval_batch_size': 8,
        'test_batch_size': 8,
        'scheduler': {
            'num_epochs': 5,  # Fewer epochs for testing
            'warmup_epochs': 1
        },
        'optimizer': {
            'learning_rate': 0.0001,
            'weight_decay': 0.02,
            'clip_value': 5.0
        },
        'lambda_mim': 1.0,
        'lambda_mlm': 1.0,
        'lambda_mtm': 0.1,
        'lambda_mmm': 1.0,
        'mm_encoder_trainable': True,
        'mm_hidden_dim': 1024,  # Smaller for local training
        'mm_output_dim': 1024,
        'projection_type': 'SimSiam',
        'predictor_hidden_dim': 256,
        'predictor_out_dim': 1024,
        'projection_dim': 512,
        'text_encoder_name': 'bert-base-uncased',
        'text_encoder_pretrained': True,
        'text_encoder_trainable': True,
        'text_embedding_dim': 768,
        'max_token_length': 512,
        'mlm_probability': 0.15,
        'image_encoder_name': 'google/vit-base-patch16-224',
        'image_encoder_pretrained': True,
        'image_encoder_trainable': True,
        'image_embedding_dim': 768,
        'timeseries_input_dim': 76,
        'timeseries_dim_feedforward': 256,
        'timeseries_max_seq_len': 48,
        'timeseries_layers': 3,
        'timeseries_n_heads': 4,
        'timeseries_dropout': 0.1,
        'timeseries_embedding_dim': 76,
        'timeseries_activation': 'gelu',
        'timeseries_encoder_trainable': True,
        'timeseries_masking_ratio': 0.15,
        'timeseries_mean_mask_length': 3,
        'timeseries_mask_mode': 'separate',
        'timeseries_mask_distribution': 'geometric',
        'impute_strategy': 'zero',
        'start_time': 'zero',
        'timestep': 1.0,
        'multiprocessing_distributed': False,  # Disable for local training
        'dist_backend': 'nccl',
        'ngpus_per_node': 0,  # No GPUs for local training
        'world_size': 1,
        'workers': 4  # Fewer workers for local training
    }
    
    try:
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print_success(f"Training configuration created: {config_file}")
        return True
        
    except Exception as e:
        print_error(f"Error creating training configuration: {e}")
        return False

def create_training_script() -> bool:
    """Create a training script."""
    print_step(6, "Creating Training Script")
    
    script_content = '''#!/bin/bash
# LANISTR Local Training Script

set -e

# Configuration
DATASET_TYPE=${1:-"mimic-iv"}
TASK=${2:-"pretrain"}
CONFIG_FILE="./lanistr/configs/${DATASET_TYPE}_${TASK}_local.yaml"

echo "🚀 Starting LANISTR Training"
echo "Dataset: $DATASET_TYPE"
echo "Task: $TASK"
echo "Config: $CONFIG_FILE"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    echo "Run: python setup_training_pipeline.py --dataset $DATASET_TYPE --task $TASK"
    exit 1
fi

# Create output directory
mkdir -p "./output_dir/${DATASET_TYPE}_${TASK}"

# Start training
echo "📊 Starting training..."
python lanistr/main.py --config "$CONFIG_FILE"

echo "✅ Training completed!"
echo "📁 Check output directory: ./output_dir/${DATASET_TYPE}_${TASK}"
'''
    
    try:
        with open('train_local.sh', 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod('train_local.sh', 0o755)
        
        print_success("Training script created: train_local.sh")
        return True
        
    except Exception as e:
        print_error(f"Error creating training script: {e}")
        return False

def create_monitoring_script() -> bool:
    """Create a monitoring script."""
    print_step(7, "Creating Monitoring Script")
    
    script_content = '''#!/bin/bash
# LANISTR Training Monitor

set -e

# Configuration
DATASET_TYPE=${1:-"mimic-iv"}
TASK=${2:-"pretrain"}
LOG_DIR="./output_dir/${DATASET_TYPE}_${TASK}"

echo "📊 LANISTR Training Monitor"
echo "Dataset: $DATASET_TYPE"
echo "Task: $TASK"
echo "Log directory: $LOG_DIR"

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Log directory not found: $LOG_DIR"
    echo "Start training first: ./train_local.sh $DATASET_TYPE $TASK"
    exit 1
fi

# Monitor training logs
echo "🔍 Monitoring training logs..."
tail -f "$LOG_DIR/${TASK}.log" 2>/dev/null || echo "No log file found yet"

echo "✅ Monitoring completed!"
'''
    
    try:
        with open('monitor_training.sh', 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod('monitor_training.sh', 0o755)
        
        print_success("Monitoring script created: monitor_training.sh")
        return True
        
    except Exception as e:
        print_error(f"Error creating monitoring script: {e}")
        return False

def create_quick_start_guide() -> bool:
    """Create a quick start guide."""
    print_step(8, "Creating Quick Start Guide")
    
    guide_content = '''# 🚀 LANISTR Training Pipeline - Quick Start Guide

## ✅ Setup Complete!

Your LANISTR training pipeline is now ready for local development.

## 🎯 Quick Commands

### Start Training
```bash
# Train MIMIC-IV dataset (pretrain)
./train_local.sh mimic-iv pretrain

# Train Amazon dataset (pretrain)
./train_local.sh amazon pretrain

# Fine-tune MIMIC-IV
./train_local.sh mimic-iv finetune
```

### Monitor Training
```bash
# Monitor training progress
./monitor_training.sh mimic-iv pretrain
```

### Validate Data
```bash
# Validate your dataset
python validate_dataset.py --dataset mimic-iv --jsonl-file ./data/mimic-iv/mimic-iv.jsonl
```

## 📁 Directory Structure

```
lanistr/
├── data/                    # Your datasets
│   ├── mimic-iv/
│   └── amazon/
├── output_dir/              # Training outputs
├── logs/                    # Log files
├── checkpoints/             # Model checkpoints
├── lanistr/configs/         # Training configurations
└── train_local.sh          # Training script
```

## ⚙️ Configuration

- **Local Training**: Optimized for macOS with smaller batch sizes
- **No GPU Required**: Uses CPU for training
- **Sample Data**: 100 samples for testing
- **Quick Training**: 5 epochs for validation

## 🔧 Customization

### Modify Training Parameters
Edit: `./lanistr/configs/{dataset}_{task}_local.yaml`

### Add Your Own Data
1. Replace sample data in `./data/{dataset}/`
2. Update configuration file
3. Run training

### Production Training
For production training on Vertex AI:
```bash
./one_click_setup.sh --dataset mimic-iv --environment prod
```

## 🐛 Troubleshooting

### Common Issues

**"Config file not found"**
```bash
python setup_training_pipeline.py --dataset mimic-iv --task pretrain
```

**"Dependencies missing"**
```bash
pip install -r requirements-core.txt
```

**"Data validation failed"**
```bash
python generate_sample_data.py --dataset mimic-iv --create-files
```

## 📚 Next Steps

1. **Test the pipeline**: Run a quick training session
2. **Add your data**: Replace sample data with your own
3. **Tune parameters**: Adjust configuration for your needs
4. **Scale up**: Use Vertex AI for production training

## 🆘 Need Help?

- Check logs in `./output_dir/`
- Validate your data with `validate_dataset.py`
- Review configuration files in `./lanistr/configs/`

---

**Happy Training! 🎉**
'''
    
    try:
        with open('TRAINING_QUICK_START.md', 'w') as f:
            f.write(guide_content)
        
        print_success("Quick start guide created: TRAINING_QUICK_START.md")
        return True
        
    except Exception as e:
        print_error(f"Error creating quick start guide: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup LANISTR Training Pipeline')
    parser.add_argument('--dataset', default='mimic-iv', choices=['mimic-iv', 'amazon'],
                       help='Dataset type (default: mimic-iv)')
    parser.add_argument('--task', default='pretrain', choices=['pretrain', 'finetune'],
                       help='Training task (default: pretrain)')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency check')
    
    args = parser.parse_args()
    
    print("🚀 LANISTR Training Pipeline Setup")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            sys.exit(1)
    else:
        print_step(1, "Skipping Dependency Check")
    
    # Step 2: Setup directories
    directories = setup_directories()
    
    # Step 3: Create sample data
    if not create_sample_data(args.dataset):
        print_warning("Sample data creation failed, continuing...")
    
    # Step 4: Validate data
    validate_data(args.dataset)
    
    # Step 5: Create training configuration
    if not create_training_config(args.dataset, args.task):
        sys.exit(1)
    
    # Step 6: Create training script
    if not create_training_script():
        sys.exit(1)
    
    # Step 7: Create monitoring script
    if not create_monitoring_script():
        sys.exit(1)
    
    # Step 8: Create quick start guide
    create_quick_start_guide()
    
    print("\n" + "=" * 60)
    print("🎉 LANISTR Training Pipeline Setup Complete!")
    print("=" * 60)
    
    print(f"\n📋 Summary:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Task: {args.task}")
    print(f"   Config: ./lanistr/configs/{args.dataset}_{args.task}_local.yaml")
    print(f"   Training script: ./train_local.sh")
    print(f"   Monitor script: ./monitor_training.sh")
    print(f"   Quick start guide: TRAINING_QUICK_START.md")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Start training: ./train_local.sh {args.dataset} {args.task}")
    print(f"   2. Monitor progress: ./monitor_training.sh {args.dataset} {args.task}")
    print(f"   3. Check the quick start guide: cat TRAINING_QUICK_START.md")
    
    print(f"\n✅ Setup complete! Your training pipeline is ready.")

if __name__ == '__main__':
    main() 