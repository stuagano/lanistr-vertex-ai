# LANISTR Dataset Requirements and Validation Guide

This document provides comprehensive information about dataset requirements, data formats, validation procedures, and testing for LANISTR multimodal training on Google Cloud Vertex AI.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Requirements](#dataset-requirements)
3. [Data Formats](#data-formats)
4. [Modality Specifications](#modality-specifications)
5. [Validation Procedures](#validation-procedures)
6. [Testing and Quality Assurance](#testing-and-quality-assurance)
7. [Data Preparation Workflow](#data-preparation-workflow)
8. [Troubleshooting](#troubleshooting)

## Overview

LANISTR is a multimodal learning framework that processes:
- **Text**: Clinical notes, product reviews, documents
- **Images**: Chest X-rays, product images, medical images
- **Tabular Data**: Patient demographics, product metadata, features
- **Time Series**: Vital signs, sensor data, temporal measurements

All data must be stored in Google Cloud Storage (GCS) and referenced via JSONL files for distributed training.

## Dataset Requirements

### Minimum Dataset Sizes

| Dataset Type | Modality | Minimum Examples | Recommended Examples |
|--------------|----------|------------------|---------------------|
| MIMIC-IV | Text + Image + Timeseries | 1,000 | 10,000+ |
| MIMIC-IV | Text + Image | 5,000 | 50,000+ |
| Amazon | Text + Image + Tabular | 10,000 | 100,000+ |
| Amazon | Text + Image | 20,000 | 200,000+ |

### Data Split Requirements

- **Training**: 70-80% of data
- **Validation**: 10-15% of data  
- **Test**: 10-15% of data

### File Size Limits

| File Type | Maximum Size | Recommended Size |
|-----------|--------------|------------------|
| Text files | 10 MB | < 1 MB |
| Image files | 50 MB | < 5 MB |
| Timeseries CSV | 10 MB | < 1 MB |
| JSONL files | 1 GB | < 100 MB |

## Data Formats

### JSONL Format Specification

Each line in a JSONL file must be a valid JSON object with the following structure:

```json
{
  "id": "unique_identifier",
  "split": "train|valid|test",
  "text": "gs://bucket/path/to/text.txt",
  "image": "gs://bucket/path/to/image.jpg",
  "timeseries": "gs://bucket/path/to/timeseries.csv",
  "features": [0.1, 0.2, 0.3],
  "y_true": 1
}
```

### Required Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Unique identifier for the example |
| `split` | string | ✅ | Data split: "train", "valid", or "test" |

### Optional Modalities

| Field | Type | Dataset | Description |
|-------|------|---------|-------------|
| `text` | string | Both | GCS path to text file |
| `image` | string | Both | GCS path to image file |
| `timeseries` | string | MIMIC-IV | GCS path to CSV timeseries file |
| `features` | array | MIMIC-IV | Tabular features array |
| `Review` | string | Amazon | GCS path to review text file |
| `ImageFileName` | string | Amazon | GCS path to product image |
| `reviewerID` | string | Amazon | Reviewer identifier |
| `verified` | boolean | Amazon | Purchase verification status |
| `asin` | string | Amazon | Product ASIN |
| `year` | integer | Amazon | Review year |
| `vote` | integer | Amazon | Helpful vote count |
| `unixReviewTime` | integer | Amazon | Unix timestamp |
| `y_true` | integer | Both | Ground truth label (finetuning) |
| `labels` | integer | Amazon | Product rating (finetuning) |

## Modality Specifications

### Text Modality

#### MIMIC-IV Text Requirements
- **Format**: Plain text (.txt) or JSON (.json)
- **Encoding**: UTF-8
- **Content**: Clinical notes, reports, discharge summaries
- **Length**: 10-10,000 characters
- **Language**: English
- **Special Characters**: Handle medical terminology and abbreviations

#### Amazon Text Requirements
- **Format**: Plain text (.txt) or JSON (.json)
- **Encoding**: UTF-8
- **Content**: Product reviews, descriptions
- **Length**: 10-5,000 characters
- **Language**: English
- **Special Characters**: Handle product names, URLs, emojis

#### Text File Content Examples

**MIMIC-IV Clinical Note:**
```
Patient admitted with chest pain and shortness of breath. 
ECG shows ST elevation in leads II, III, aVF. 
Troponin elevated to 2.5 ng/mL. 
Diagnosis: Acute inferior wall myocardial infarction.
Treatment: Aspirin, Plavix, statin, beta-blocker.
```

**Amazon Product Review:**
```
This product exceeded my expectations! The quality is outstanding and it arrived quickly. 
I would definitely recommend this to anyone looking for a reliable solution. 
The customer service was also excellent when I had questions.
```

### Image Modality

#### Supported Formats
- **JPEG**: .jpg, .jpeg
- **PNG**: .png

#### Image Requirements
- **MIMIC-IV**: Chest X-rays, medical images
  - **Recommended Size**: 512x512 or 1024x1024 pixels
  - **Color**: Grayscale or RGB
  - **Quality**: High resolution, clear visibility
- **Amazon**: Product images
  - **Recommended Size**: 224x224 or 512x512 pixels
  - **Color**: RGB
  - **Quality**: Clear product visibility

#### Image Validation
- File size < 50 MB
- Valid image format
- Readable by PIL/Pillow
- Appropriate dimensions

### Timeseries Modality (MIMIC-IV Only)

#### CSV Format Requirements
- **Format**: Comma-separated values (.csv)
- **Encoding**: UTF-8
- **Header**: First row contains column names
- **Required Column**: "Hours" (time in hours from admission)
- **Data Types**: Numeric values
- **Missing Values**: Handle with appropriate strategy

#### Timeseries CSV Example
```csv
Hours,Heart Rate,Blood Pressure,Systolic,Diastolic,Temperature,Oxygen Saturation
0.0,85,120/80,120,80,37.2,98
0.5,88,125/82,125,82,37.1,97
1.0,92,130/85,130,85,37.3,96
1.5,90,128/83,128,83,37.2,97
2.0,87,122/81,122,81,37.1,98
```

#### Timeseries Validation
- File size < 10 MB
- Valid CSV format
- Contains "Hours" column
- Numeric data values
- Consistent time intervals

### Tabular Modality

#### MIMIC-IV Tabular Features
- **Format**: Array of float values
- **Length**: 1-100 features
- **Data Type**: float32
- **Normalization**: Values typically between 0-1

#### Amazon Tabular Features
- **Format**: Object with categorical and numerical fields
- **Categorical**: reviewerID, verified, asin
- **Numerical**: year, vote, unixReviewTime
- **Required**: reviewerID, asin

#### Tabular Validation
- Correct data types
- Valid value ranges
- No missing required fields
- Consistent feature dimensions

## Validation Procedures

### 1. Data Format Validation

#### JSONL Validation
```bash
# Validate JSONL format
python -c "
import json
with open('data.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line.strip())
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
"
```

#### Required Fields Check
```bash
# Check required fields
python -c "
import json
with open('data.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line.strip())
        if 'id' not in data or 'split' not in data:
            print(f'Line {i}: Missing required fields')
        if data['split'] not in ['train', 'valid', 'test']:
            print(f'Line {i}: Invalid split value')
"
```

### 2. File Access Validation

#### GCS Path Validation
```bash
# Validate GCS paths
gsutil ls gs://your-bucket/path/to/file.txt

# Check multiple files
gsutil ls gs://your-bucket/dataset/**/*.txt | head -10
```

#### File Existence Check
```python
from google.cloud import storage

def validate_gcs_files(jsonl_file, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            for field, path in data.items():
                if isinstance(path, str) and path.startswith('gs://'):
                    # Extract bucket and blob path
                    parts = path[5:].split('/', 1)
                    if len(parts) == 2:
                        blob = bucket.blob(parts[1])
                        if not blob.exists():
                            print(f"Missing file: {path}")
```

### 3. Content Validation

#### Text Content Validation
```python
def validate_text_content(gcs_path):
    """Validate text file content."""
    # Download and check text file
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        gsutil.download(gcs_path, tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check content length
        if len(content) < 10:
            return False, "Text too short"
        if len(content) > 10000:
            return False, "Text too long"
            
        # Check encoding
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            return False, "Invalid encoding"
            
        return True, "Valid"
```

#### Image Content Validation
```python
from PIL import Image
import io

def validate_image_content(gcs_path):
    """Validate image file content."""
    # Download and check image file
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        gsutil.download(gcs_path, tmp.name)
        
        try:
            with Image.open(tmp.name) as img:
                # Check format
                if img.format not in ['JPEG', 'PNG']:
                    return False, f"Invalid format: {img.format}"
                
                # Check size
                width, height = img.size
                if width < 100 or height < 100:
                    return False, f"Image too small: {width}x{height}"
                if width > 2048 or height > 2048:
                    return False, f"Image too large: {width}x{height}"
                
                # Check file size
                file_size = os.path.getsize(tmp.name)
                if file_size > 50 * 1024 * 1024:  # 50MB
                    return False, f"File too large: {file_size} bytes"
                
                return True, "Valid"
        except Exception as e:
            return False, f"Image error: {e}"
```

### 4. Comprehensive Validation

Use the built-in validation tool:

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

## Testing and Quality Assurance

### 1. Unit Tests

#### Test Data Format
```python
import pytest
import json
from lanistr.utils.data_validator import DataValidator

def test_jsonl_format():
    """Test JSONL file format."""
    validator = DataValidator(dataset_type="mimic-iv")
    
    # Test valid JSONL
    valid_data = [
        '{"id": "test1", "split": "train", "text": "gs://bucket/test.txt"}',
        '{"id": "test2", "split": "valid", "image": "gs://bucket/test.jpg"}'
    ]
    
    with open('test.jsonl', 'w') as f:
        for line in valid_data:
            f.write(line + '\n')
    
    result = validator.validate_jsonl_file('test.jsonl')
    assert result['success'] == True
    assert result['valid_lines'] == 2

def test_required_fields():
    """Test required fields validation."""
    validator = DataValidator(dataset_type="mimic-iv")
    
    # Test missing required field
    invalid_data = [
        '{"split": "train", "text": "gs://bucket/test.txt"}',  # Missing id
        '{"id": "test2", "text": "gs://bucket/test.txt"}'      # Missing split
    ]
    
    with open('test_invalid.jsonl', 'w') as f:
        for line in invalid_data:
            f.write(line + '\n')
    
    result = validator.validate_jsonl_file('test_invalid.jsonl')
    assert result['success'] == False
    assert result['invalid_lines'] > 0
```

### 2. Integration Tests

#### Test End-to-End Pipeline
```python
def test_end_to_end_pipeline():
    """Test complete data preparation pipeline."""
    from scripts.prepare_jsonl_data import convert_mimic_iv_to_jsonl
    
    # Create test data
    test_data = pd.DataFrame({
        'id': ['test1', 'test2', 'test3'],
        'text': ['text1.txt', 'text2.txt', 'text3.txt'],
        'image': ['image1.jpg', 'image2.jpg', 'image3.jpg'],
        'timeseries': ['ts1.csv', 'ts2.csv', 'ts3.csv']
    })
    
    test_data.to_csv('test_data.csv', index=False)
    
    # Convert to JSONL
    output_files = convert_mimic_iv_to_jsonl(
        data_dir='.',
        output_dir='./output',
        bucket_name='test-bucket'
    )
    
    # Validate output
    assert 'train' in output_files
    assert 'valid' in output_files
    assert 'test' in output_files
    
    # Check file contents
    with open(output_files['train'], 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0
        
        # Check first line
        first_line = json.loads(lines[0])
        assert 'id' in first_line
        assert 'split' in first_line
        assert first_line['split'] == 'train'
```

### 3. Performance Tests

#### Test Large Dataset Handling
```python
def test_large_dataset():
    """Test handling of large datasets."""
    # Create large test dataset
    large_data = []
    for i in range(10000):
        large_data.append({
            'id': f'test_{i:06d}',
            'text': f'text_{i}.txt',
            'image': f'image_{i}.jpg',
            'timeseries': f'ts_{i}.csv'
        })
    
    large_df = pd.DataFrame(large_data)
    large_df.to_csv('large_test_data.csv', index=False)
    
    # Test conversion performance
    import time
    start_time = time.time()
    
    output_files = convert_mimic_iv_to_jsonl(
        data_dir='.',
        output_dir='./large_output',
        bucket_name='test-bucket'
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Performance requirements
    assert processing_time < 60  # Should process 10k examples in < 60 seconds
    assert len(output_files) == 3  # train, valid, test
```

## Data Preparation Workflow

### 1. Data Collection

```bash
# Create data directory structure
mkdir -p data/{mimic-iv,amazon}/{text,images,timeseries,jsonl}

# Organize your data files
# - Place text files in data/mimic-iv/text/
# - Place images in data/mimic-iv/images/
# - Place timeseries in data/mimic-iv/timeseries/
```

### 2. Data Conversion

```bash
# Convert MIMIC-IV data to JSONL
python scripts/prepare_jsonl_data.py \
    --dataset-type mimic-iv \
    --input-dir ./data/mimic-iv \
    --output-dir ./data/mimic-iv/jsonl \
    --bucket-name your-bucket \
    --project-id your-project-id \
    --validate

# Convert Amazon data to JSONL
python scripts/prepare_jsonl_data.py \
    --dataset-type amazon \
    --input-dir ./data/amazon \
    --output-dir ./data/amazon/jsonl \
    --bucket-name your-bucket \
    --project-id your-project-id \
    --validate
```

### 3. Data Upload

```bash
# Upload data to GCS
gsutil -m cp -r data/mimic-iv gs://your-bucket/
gsutil -m cp -r data/amazon gs://your-bucket/
gsutil -m cp -r data/*/jsonl gs://your-bucket/
```

### 4. Validation

```bash
# Run comprehensive validation
python -m lanistr.utils.data_validator \
    --dataset-dir ./data/mimic-iv/jsonl \
    --dataset-type mimic-iv \
    --bucket-name your-bucket \
    --project-id your-project-id \
    --output-report mimic_validation_report.txt

python -m lanistr.utils.data_validator \
    --dataset-dir ./data/amazon/jsonl \
    --dataset-type amazon \
    --bucket-name your-bucket \
    --project-id your-project-id \
    --output-report amazon_validation_report.txt
```

### 5. Pre-training Test

```bash
# Test with small subset before full training
python lanistr/main_vertex_ai.py \
    --config lanistr/configs/mimic_pretrain.yaml \
    --job_name "test-run" \
    --project_id your-project-id \
    --output_dir gs://your-bucket/test-output \
    --data_dir gs://your-bucket/mimic-iv/jsonl
```

## Troubleshooting

### Common Issues and Solutions

#### 1. JSONL Format Errors

**Problem**: Invalid JSON in JSONL file
```bash
# Error: Expecting property name enclosed in double quotes
```

**Solution**: 
```bash
# Validate JSON format
python -c "
import json
with open('data.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line.strip())
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
"
```

#### 2. Missing GCS Files

**Problem**: Files not found in GCS bucket
```bash
# Error: 404 Not Found
```

**Solution**:
```bash
# Check file existence
gsutil ls gs://your-bucket/path/to/file.txt

# Upload missing files
gsutil cp local_file.txt gs://your-bucket/path/to/file.txt

# Set bucket permissions
gsutil iam ch allUsers:objectViewer gs://your-bucket
```

#### 3. Invalid File Formats

**Problem**: Unsupported file format
```bash
# Error: Invalid file extension
```

**Solution**:
```bash
# Convert file formats
# Images: Use PIL to convert to JPEG/PNG
# Text: Ensure UTF-8 encoding
# CSV: Validate format and encoding
```

#### 4. Memory Issues

**Problem**: Out of memory during processing
```bash
# Error: MemoryError
```

**Solution**:
```bash
# Process in batches
python scripts/prepare_jsonl_data.py \
    --batch-size 1000 \
    --chunk-size 100

# Use streaming processing
# Reduce file sizes
# Use more efficient data structures
```

#### 5. Validation Failures

**Problem**: Data validation errors
```bash
# Error: Validation failed
```

**Solution**:
```bash
# Check validation report
cat validation_report.txt

# Fix specific issues
# - Correct data types
# - Fix missing required fields
# - Validate file paths
# - Check file content
```

### Debug Commands

```bash
# Check JSONL file structure
head -n 5 data.jsonl | jq .

# Count lines in JSONL file
wc -l data.jsonl

# Check file sizes
ls -lh data.jsonl

# Validate GCS access
gsutil ls gs://your-bucket/

# Test file download
gsutil cp gs://your-bucket/test.txt ./test.txt

# Check file encoding
file -i data.jsonl

# Validate JSON syntax
jq . data.jsonl > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

### Performance Optimization

```bash
# Parallel processing
gsutil -m cp -r data/ gs://your-bucket/

# Compress large files
gzip data.jsonl
gsutil cp data.jsonl.gz gs://your-bucket/

# Use efficient data formats
# - Parquet for large tabular data
# - Compressed images
# - Optimized text encoding
```

This comprehensive guide ensures your data meets all requirements for successful LANISTR training on Google Cloud Vertex AI. 