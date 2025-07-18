# LANISTR Data Format Examples

This document provides comprehensive examples of the expected JSONL data format for each modality and dataset type used in LANISTR training.

## Overview

LANISTR expects data in JSONL (JSON Lines) format, where each line contains a JSON object representing a single training example. All file paths should point to Google Cloud Storage (GCS) buckets.

## Data Format Requirements

### Required Fields
Every JSONL line must contain:
- `id`: Unique identifier for the example
- `split`: Data split ("train", "valid", "test")

### Optional Modalities
Depending on your use case, include any combination of:
- `text`: Text data (clinical notes, reviews, etc.)
- `image`: Image file paths
- `tabular`: Tabular features
- `timeseries`: Time series data paths
- `label`: Ground truth labels (for finetuning)

## MIMIC-IV Dataset Examples

### Example 1: Text + Image + Timeseries (Pretraining)

```json
{
  "id": "mimic_001",
  "split": "train",
  "text": "gs://your-bucket/mimic-iv/text/patient_001_notes.txt",
  "image": "gs://your-bucket/mimic-iv/images/patient_001_cxr.jpg",
  "timeseries": "gs://your-bucket/mimic-iv/timeseries/patient_001_vitals.csv",
  "features": [0.5, 0.3, 0.8, 0.2, 0.9]
}
```

### Example 2: Text + Image + Label (Finetuning)

```json
{
  "id": "mimic_002",
  "split": "train",
  "text": "gs://your-bucket/mimic-iv/text/patient_002_notes.txt",
  "image": "gs://your-bucket/mimic-iv/images/patient_002_cxr.jpg",
  "y_true": 1
}
```

### Example 3: Text Only

```json
{
  "id": "mimic_003",
  "split": "valid",
  "text": "gs://your-bucket/mimic-iv/text/patient_003_notes.txt"
}
```

### Example 4: Image + Timeseries

```json
{
  "id": "mimic_004",
  "split": "test",
  "image": "gs://your-bucket/mimic-iv/images/patient_004_cxr.jpg",
  "timeseries": "gs://your-bucket/mimic-iv/timeseries/patient_004_vitals.csv"
}
```

## Amazon Dataset Examples

### Example 1: Text + Image + Tabular (Pretraining)

```json
{
  "id": "amazon_001",
  "split": "train",
  "Review": "gs://your-bucket/amazon/reviews/product_001_review.txt",
  "ImageFileName": "gs://your-bucket/amazon/images/product_001.jpg",
  "reviewerID": "A1B2C3D4E5",
  "verified": true,
  "asin": "B000123456",
  "year": 2023,
  "vote": 15,
  "unixReviewTime": 1672531200
}
```

### Example 2: Text + Image + Label (Finetuning)

```json
{
  "id": "amazon_002",
  "split": "train",
  "Review": "gs://your-bucket/amazon/reviews/product_002_review.txt",
  "ImageFileName": "gs://your-bucket/amazon/images/product_002.jpg",
  "labels": 5
}
```

### Example 3: Text + Tabular

```json
{
  "id": "amazon_003",
  "split": "valid",
  "Review": "gs://your-bucket/amazon/reviews/product_003_review.txt",
  "reviewerID": "F6G7H8I9J0",
  "verified": false,
  "asin": "B000789012",
  "year": 2022,
  "vote": 8,
  "unixReviewTime": 1640995200
}
```

## File Content Examples

### Text Files

#### Clinical Notes (MIMIC-IV)
```
Patient admitted with chest pain and shortness of breath. 
ECG shows ST elevation in leads II, III, aVF. 
Troponin elevated to 2.5 ng/mL. 
Diagnosis: Acute inferior wall myocardial infarction.
```

#### Product Review (Amazon)
```
This product exceeded my expectations! The quality is outstanding and it arrived quickly. 
I would definitely recommend this to anyone looking for a reliable solution. 
The customer service was also excellent when I had questions.
```

### Timeseries CSV Files (MIMIC-IV)

```csv
Hours,Heart Rate,Blood Pressure,Systolic,Diastolic,Temperature,Oxygen Saturation
0.0,85,120/80,120,80,37.2,98
0.5,88,125/82,125,82,37.1,97
1.0,92,130/85,130,85,37.3,96
1.5,90,128/83,128,83,37.2,97
2.0,87,122/81,122,81,37.1,98
```

### Image Files

Images should be in common formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)

Recommended specifications:
- **MIMIC-IV**: Chest X-rays, typically 512x512 or 1024x1024
- **Amazon**: Product images, typically 224x224 or 512x512

## Data Organization in GCS

### Recommended Bucket Structure

```
gs://your-bucket/
├── mimic-iv/
│   ├── text/
│   │   ├── patient_001_notes.txt
│   │   ├── patient_002_notes.txt
│   │   └── ...
│   ├── images/
│   │   ├── patient_001_cxr.jpg
│   │   ├── patient_002_cxr.jpg
│   │   └── ...
│   └── timeseries/
│       ├── patient_001_vitals.csv
│       ├── patient_002_vitals.csv
│       └── ...
├── amazon/
│   ├── reviews/
│   │   ├── product_001_review.txt
│   │   ├── product_002_review.txt
│   │   └── ...
│   └── images/
│       ├── product_001.jpg
│       ├── product_002.jpg
│       └── ...
└── jsonl/
    ├── mimic_iv_train.jsonl
    ├── mimic_iv_valid.jsonl
    ├── mimic_iv_test.jsonl
    ├── amazon_train.jsonl
    ├── amazon_valid.jsonl
    └── amazon_test.jsonl
```

## Data Validation

### Required Validations

1. **JSON Format**: Each line must be valid JSON
2. **Required Fields**: `id` and `split` must be present
3. **File Paths**: All paths must be valid GCS URLs
4. **File Access**: Files must be accessible in the specified bucket
5. **Data Types**: Values must match expected types
6. **File Extensions**: Files must have correct extensions

### Validation Commands

```bash
# Validate MIMIC-IV dataset
python -m lanistr.utils.data_validator \
    --dataset-dir ./data/jsonl \
    --dataset-type mimic-iv \
    --bucket-name your-bucket \
    --project-id your-project-id \
    --output-report validation_report.txt

# Validate Amazon dataset
python -m lanistr.utils.data_validator \
    --dataset-dir ./data/jsonl \
    --dataset-type amazon \
    --bucket-name your-bucket \
    --project-id your-project-id \
    --output-report validation_report.txt
```

## Data Preparation Scripts

### Convert Existing Data to JSONL

```python
import json
import pandas as pd
from pathlib import Path

def convert_mimic_to_jsonl(
    text_dir: str,
    image_dir: str,
    timeseries_dir: str,
    output_file: str,
    bucket_name: str
):
    """Convert MIMIC-IV data to JSONL format."""
    
    # Load your existing data
    # This is a placeholder - adapt to your actual data structure
    data = pd.read_csv("your_mimic_data.csv")
    
    with open(output_file, 'w') as f:
        for idx, row in data.iterrows():
            example = {
                "id": f"mimic_{idx:06d}",
                "split": row["split"],
                "text": f"gs://{bucket_name}/mimic-iv/text/{row['text_file']}",
                "image": f"gs://{bucket_name}/mimic-iv/images/{row['image_file']}",
                "timeseries": f"gs://{bucket_name}/mimic-iv/timeseries/{row['timeseries_file']}",
                "features": row["features"].tolist() if "features" in row else None
            }
            
            # Add label for finetuning
            if "y_true" in row:
                example["y_true"] = int(row["y_true"])
            
            # Remove None values
            example = {k: v for k, v in example.items() if v is not None}
            
            f.write(json.dumps(example) + "\n")

def convert_amazon_to_jsonl(
    reviews_dir: str,
    images_dir: str,
    output_file: str,
    bucket_name: str
):
    """Convert Amazon data to JSONL format."""
    
    # Load your existing data
    data = pd.read_csv("your_amazon_data.csv")
    
    with open(output_file, 'w') as f:
        for idx, row in data.iterrows():
            example = {
                "id": f"amazon_{idx:06d}",
                "split": row["split"],
                "Review": f"gs://{bucket_name}/amazon/reviews/{row['review_file']}",
                "ImageFileName": f"gs://{bucket_name}/amazon/images/{row['image_file']}",
                "reviewerID": row["reviewerID"],
                "verified": bool(row["verified"]),
                "asin": row["asin"],
                "year": int(row["year"]),
                "vote": int(row["vote"]),
                "unixReviewTime": int(row["unixReviewTime"])
            }
            
            # Add label for finetuning
            if "labels" in row:
                example["labels"] = int(row["labels"])
            
            f.write(json.dumps(example) + "\n")
```

## Best Practices

### 1. Data Quality
- Ensure all text files are properly encoded (UTF-8)
- Validate image files are not corrupted
- Check timeseries data has consistent format
- Verify all file paths are accessible

### 2. Performance
- Use appropriate image sizes (don't store unnecessarily large images)
- Compress text files if they're very large
- Consider using parquet for large tabular datasets

### 3. Security
- Set appropriate bucket permissions
- Use service accounts with minimal required permissions
- Enable audit logging for data access

### 4. Organization
- Use consistent naming conventions
- Organize files by modality and split
- Include metadata files for dataset information

## Troubleshooting

### Common Issues

1. **Invalid JSON**: Check for missing commas, quotes, or brackets
2. **File Not Found**: Verify GCS paths and bucket permissions
3. **Wrong Data Type**: Ensure values match expected types
4. **Missing Required Fields**: Check that `id` and `split` are present
5. **Invalid File Extensions**: Use only supported file formats

### Debug Commands

```bash
# Check JSONL file format
head -n 5 your_file.jsonl | jq .

# Validate single line
echo '{"id": "test", "split": "train"}' | jq .

# Check GCS file access
gsutil ls gs://your-bucket/path/to/file.txt

# Count lines in JSONL file
wc -l your_file.jsonl
``` 