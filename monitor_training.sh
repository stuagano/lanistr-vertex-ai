#!/bin/bash
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
