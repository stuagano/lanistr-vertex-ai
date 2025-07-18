#!/bin/bash
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

# Activate virtual environment and start training
echo "📊 Starting training..."
source venv/bin/activate
python run_training.py --config "$CONFIG_FILE"

echo "✅ Training completed!"
echo "📁 Check output directory: ./output_dir/${DATASET_TYPE}_${TASK}"
