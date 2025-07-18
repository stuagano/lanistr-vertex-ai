# Quick Start: Data Validation for LANISTR

This guide shows you how to quickly validate your datasets before submitting to Vertex AI distributed training.

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements_vertex_ai.txt
   ```

2. **Set up Google Cloud** (for GCS validation):
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

## Quick Validation Steps

### 1. Generate Sample Data (Optional)

If you don't have data yet, generate sample datasets for testing:

```bash
# Generate Amazon sample dataset
python generate_sample_data.py --dataset amazon --output-file sample_amazon.jsonl --num-samples 100 --create-files

# Generate MIMIC-IV sample dataset  
python generate_sample_data.py --dataset mimic-iv --output-file sample_mimic.jsonl --num-samples 50 --create-files
```

### 2. Validate Your Dataset

#### For Amazon Dataset:
```bash
python validate_dataset.py \
  --dataset amazon \
  --jsonl-file your_data.jsonl \
  --data-dir ./data \
  --image-data-dir ./images
```

#### For MIMIC-IV Dataset:
```bash
python validate_dataset.py \
  --dataset mimic-iv \
  --jsonl-file your_data.jsonl \
  --data-dir ./data \
  --gcs-bucket your-bucket-name
```

#### For Local Files Only:
```bash
python validate_dataset.py \
  --dataset amazon \
  --jsonl-file your_data.jsonl \
  --data-dir ./data \
  --no-content-validation
```

### 3. Check Validation Results

The validation script will output:
- ✅ **PASSED** - Dataset is ready for training
- ❌ **FAILED** - Fix issues before proceeding
- ⚠️ **WARNINGS** - Review potential issues

## Common Validation Scenarios

### Scenario 1: Local Dataset Validation
```bash
# Quick validation without content checks
python validate_dataset.py \
  --dataset amazon \
  --jsonl-file ./data/amazon.jsonl \
  --data-dir ./data \
  --no-content-validation
```

### Scenario 2: GCS Dataset Validation
```bash
# Full validation with GCS file checks
python validate_dataset.py \
  --dataset mimic-iv \
  --jsonl-file ./data/mimic.jsonl \
  --data-dir ./data \
  --gcs-bucket lanistr-data-bucket \
  --project-id your-project-id
```

### Scenario 3: Save Validation Results
```bash
# Save detailed results to file
python validate_dataset.py \
  --dataset amazon \
  --jsonl-file ./data/amazon.jsonl \
  --data-dir ./data \
  --output-file validation_results.json
```

## Expected Output

### Successful Validation:
```
================================================================================
LANISTR DATASET VALIDATION REPORT
================================================================================
Overall Status: ✅ PASSED

📊 DATASET STATISTICS:
  total_records: 1000
  file_size_mb: 2.45
  field_statistics: {...}

💡 RECOMMENDATIONS:
  • Dataset is ready for training
================================================================================
✅ Dataset validation PASSED - ready for Vertex AI submission
```

### Failed Validation:
```
================================================================================
LANISTR DATASET VALIDATION REPORT
================================================================================
Overall Status: ❌ FAILED

❌ ERRORS:
  • Missing required fields: ['ImageFileName']
  • File not found: images/product_123.jpg

💡 RECOMMENDATIONS:
  • Fix errors before proceeding with training
================================================================================
❌ Dataset validation FAILED - fix issues before submission
```

## Troubleshooting Common Issues

### Issue 1: Missing Files
**Error**: `File not found: images/product_123.jpg`

**Solution**:
```bash
# Check if files exist
ls -la ./data/images/

# Create missing directories
mkdir -p ./data/images/

# Upload files to GCS (if using cloud storage)
gsutil cp -r ./data/images/ gs://your-bucket/
```

### Issue 2: Invalid JSON Format
**Error**: `Line 5: Invalid JSON - Expecting property name`

**Solution**:
```bash
# Check JSONL format
head -5 your_data.jsonl

# Validate JSON syntax
python -m json.tool your_data.jsonl | head -10
```

### Issue 3: GCS Access Issues
**Error**: `GCS access denied`

**Solution**:
```bash
# Check authentication
gcloud auth list

# Set project
gcloud config set project YOUR_PROJECT_ID

# Check bucket permissions
gsutil ls gs://your-bucket/
```

### Issue 4: Large File Sizes
**Warning**: `File size exceeds 50MB limit`

**Solution**:
```bash
# Compress images
for file in *.jpg; do
  convert "$file" -quality 85 "compressed_$file"
done

# Use appropriate image formats
# - JPEG for photos (smaller)
# - PNG for medical images (lossless)
```

## Data Format Examples

### Amazon Dataset Format:
```json
{
  "Review": "Great product! The quality is excellent and it arrived on time.",
  "ImageFileName": "images/electronics/product_12345.jpg",
  "reviewerID": "A1B2C3D4E5F6",
  "verified": true,
  "asin": "B08N5WRWNW",
  "year": 2023,
  "vote": 15,
  "unixReviewTime": 1672531200
}
```

### MIMIC-IV Dataset Format:
```json
{
  "text": "CHEST X-RAY REPORT: The cardiac silhouette is normal in size...",
  "image": "mimic-cxr-jpg/2.0.0/files/p10/p10000032/s50414267/image.jpg",
  "timeseries": "timeseries/patient_12345_vitals.csv",
  "y_true": 0
}
```

## Performance Tips

### For Large Datasets:
```bash
# Validate subset first
python validate_dataset.py \
  --dataset amazon \
  --jsonl-file large_dataset.jsonl \
  --data-dir ./data \
  --max-samples 1000

# Skip content validation for speed
python validate_dataset.py \
  --dataset amazon \
  --jsonl-file large_dataset.jsonl \
  --data-dir ./data \
  --no-content-validation
```

### For Production Validation:
```bash
# Full validation with verbose logging
python validate_dataset.py \
  --dataset mimic-iv \
  --jsonl-file production_data.jsonl \
  --data-dir ./data \
  --gcs-bucket production-bucket \
  --verbose \
  --output-file validation_report.json
```

## Next Steps

After successful validation:

1. **Upload to GCS** (if not already done):
   ```bash
   gsutil cp -r ./data/ gs://your-bucket/
   ```

2. **Submit to Vertex AI**:
   ```bash
   python vertex_ai_setup.py --config-file your_config.yaml
   ```

3. **Monitor Training**:
   ```bash
   ./monitor_training.sh
   ```

## Support

- **Documentation**: See `DATASET_REQUIREMENTS.md` for detailed specifications
- **Examples**: Use `generate_sample_data.py` to create test datasets
- **Validation**: Use `validate_dataset.py` for comprehensive validation
- **Issues**: Check the troubleshooting section above 