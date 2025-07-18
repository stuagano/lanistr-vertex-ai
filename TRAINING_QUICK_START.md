# 🚀 LANISTR Training Pipeline - Quick Start Guide

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
