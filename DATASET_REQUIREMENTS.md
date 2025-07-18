# LANISTR Dataset Requirements

This document outlines the dataset requirements for each modality supported by LANISTR, including data formats, field specifications, and validation guidelines for Vertex AI distributed training.

## Overview

LANISTR supports four main modalities:
- **Text**: Natural language text data
- **Image**: Visual data (photos, medical images, etc.)
- **Tabular**: Structured numerical and categorical data
- **Timeseries**: Time-ordered numerical data

All datasets are expected to be in JSONL (JSON Lines) format with paths to storage buckets for file references.

## Data Format: JSONL

All datasets must be provided in JSONL format, where each line contains a valid JSON object representing one data sample.

### JSONL Structure
```json
{"field1": "value1", "field2": "value2", ...}
{"field1": "value3", "field2": "value4", ...}
...
```

### File Paths
- **Local paths**: Relative paths from the data directory
- **GCS paths**: `gs://bucket-name/path/to/file`
- **HTTP/HTTPS URLs**: `https://example.com/path/to/file`

## Dataset Types

### 1. Amazon Dataset

**Purpose**: E-commerce product reviews with images and metadata

**Required Fields**:
```json
{
  "Review": "string",           // Product review text
  "ImageFileName": "string",    // Path to product image
  "reviewerID": "string",       // Unique reviewer identifier
  "verified": "boolean",        // Whether purchase was verified
  "asin": "string",            // Amazon Standard Identification Number
  "year": "integer",           // Review year
  "vote": "integer",           // Number of helpful votes
  "unixReviewTime": "integer"  // Unix timestamp of review
}
```

**Field Specifications**:
- **Review**: Text content, typically 10-1000 characters
- **ImageFileName**: Path to JPG/PNG image file
- **reviewerID**: Unique string identifier
- **verified**: Boolean (true/false)
- **asin**: Product identifier string
- **year**: Integer year (e.g., 2020, 2021)
- **vote**: Integer count of helpful votes
- **unixReviewTime**: Unix timestamp

**Data Requirements**:
- **Minimum examples**: 1,000
- **Maximum examples**: 1,000,000
- **Image formats**: JPG, JPEG, PNG, BMP, TIFF
- **Text encoding**: UTF-8

**Example JSONL Entry**:
```json
{
  "Review": "Great product! The quality is excellent and it arrived on time. Highly recommend for anyone looking for a reliable solution.",
  "ImageFileName": "images/product_12345.jpg",
  "reviewerID": "A1B2C3D4E5F6",
  "verified": true,
  "asin": "B08N5WRWNW",
  "year": 2023,
  "vote": 15,
  "unixReviewTime": 1672531200
}
```

### 2. MIMIC-IV Dataset

**Purpose**: Medical data with clinical notes, chest X-rays, and vital signs

**Required Fields**:
```json
{
  "text": "string",        // Clinical notes text
  "image": "string",       // Path to chest X-ray image
  "timeseries": "string"   // Path to vital signs timeseries data
}
```

**Optional Fields** (for finetuning):
```json
{
  "y_true": "integer"      // Binary label (0 or 1)
}
```

**Field Specifications**:
- **text**: Clinical notes, typically 100-5000 characters
- **image**: Path to chest X-ray image file
- **timeseries**: Path to vital signs CSV/NPY file
- **y_true**: Binary classification label (0 or 1)

**Data Requirements**:
- **Minimum examples**: 100
- **Maximum examples**: 100,000
- **Image formats**: JPG, JPEG, PNG, DICOM
- **Timeseries formats**: CSV, NPY, Parquet
- **Text encoding**: UTF-8

**Example JSONL Entry**:
```json
{
  "text": "CHEST X-RAY REPORT: The cardiac silhouette is normal in size. The mediastinum is unremarkable. No evidence of pneumothorax or pleural effusion. The lungs are clear without evidence of infiltrate, mass, or consolidation.",
  "image": "mimic-cxr-jpg/2.0.0/files/p10/p10000032/s50414267/5a054c1d-5c1b4b0a-5c1b4b0a-5c1b4b0a.jpg",
  "timeseries": "timeseries/patient_12345_vitals.csv",
  "y_true": 0
}
```

## Modality-Specific Requirements

### Text Modality

**Supported Formats**:
- Plain text (UTF-8 encoded)
- Clinical notes
- Product reviews
- General natural language

**Requirements**:
- **Encoding**: UTF-8
- **Length**: 1-50,000 characters per sample
- **Content**: Human-readable text
- **Special characters**: Supported
- **Language**: English (primary)

**Validation Checks**:
- UTF-8 encoding validation
- Length bounds checking
- Content readability
- Special character handling

### Image Modality

**Supported Formats**:
- **Primary**: JPG, JPEG, PNG
- **Secondary**: BMP, TIFF, TIF
- **Medical**: DICOM (converted to standard formats)

**Requirements**:
- **Resolution**: Minimum 32x32 pixels
- **Channels**: RGB (3 channels)
- **File size**: Maximum 50MB per image
- **Quality**: Readable, non-corrupted images

**Validation Checks**:
- File format verification
- Image readability (PIL validation)
- Resolution checking
- File size limits
- Corruption detection

### Tabular Modality

**Supported Data Types**:
- **Categorical**: String labels, encoded as integers
- **Numerical**: Integer, float values
- **Boolean**: True/false values

**Requirements**:
- **Missing values**: Handled by imputation
- **Data types**: Consistent across samples
- **Encoding**: Label encoding for categorical
- **Normalization**: Z-score or min-max scaling

**Validation Checks**:
- Data type consistency
- Missing value patterns
- Categorical value counts
- Numerical value ranges
- Encoding validation

### Timeseries Modality

**Supported Formats**:
- **CSV**: Comma-separated values
- **NPY**: NumPy binary format
- **Parquet**: Columnar storage format

**Requirements**:
- **Time ordering**: Chronological sequence
- **Missing values**: Handled by imputation
- **Length**: Variable, up to configurable maximum
- **Features**: Consistent feature set

**Validation Checks**:
- File format verification
- Time ordering validation
- Feature consistency
- Missing value patterns
- Data type validation

## File Organization

### Local Storage Structure
```
data/
├── amazon/
│   ├── data.jsonl
│   └── images/
│       ├── product_1.jpg
│       └── product_2.png
└── mimic-iv/
    ├── data.jsonl
    ├── images/
    │   └── chest_xray_1.jpg
    └── timeseries/
        └── patient_1_vitals.csv
```

### GCS Storage Structure
```
gs://bucket-name/
├── amazon/
│   ├── data.jsonl
│   └── images/
│       ├── product_1.jpg
│       └── product_2.png
└── mimic-iv/
    ├── data.jsonl
    ├── images/
    │   └── chest_xray_1.jpg
    └── timeseries/
        └── patient_1_vitals.csv
```

## Data Validation

### Pre-Submission Validation

Use the provided validation script to check your dataset before submitting to Vertex AI:

```bash
# Validate Amazon dataset
python validate_dataset.py \
  --dataset amazon \
  --jsonl-file data.jsonl \
  --data-dir ./data \
  --image-data-dir ./images

# Validate MIMIC-IV dataset with GCS
python validate_dataset.py \
  --dataset mimic-iv \
  --jsonl-file data.jsonl \
  --data-dir ./data \
  --gcs-bucket my-bucket \
  --project-id my-project
```

### Validation Checks

The validation system performs the following checks:

1. **File Access**:
   - JSONL file exists and is readable
   - File size > 0 bytes
   - Valid JSON format per line

2. **Schema Compliance**:
   - Required fields present
   - Field types correct
   - No unexpected fields

3. **Data Statistics**:
   - Number of examples within bounds
   - Field-level statistics
   - Missing value analysis

4. **File References**:
   - Image files exist and accessible
   - Timeseries files exist and accessible
   - GCS paths valid (if applicable)

5. **Content Validation**:
   - Image files readable (PIL validation)
   - Timeseries files parseable
   - Text encoding valid

### Validation Output

The validation script provides:
- **Status**: PASSED/FAILED
- **Statistics**: Dataset metrics
- **Errors**: Critical issues to fix
- **Warnings**: Potential issues to review
- **Recommendations**: Next steps

## Performance Guidelines

### Dataset Size Recommendations

**Amazon Dataset**:
- **Training**: 10,000 - 500,000 examples
- **Validation**: 1,000 - 10,000 examples
- **Testing**: 1,000 - 10,000 examples

**MIMIC-IV Dataset**:
- **Training**: 1,000 - 50,000 examples
- **Validation**: 100 - 1,000 examples
- **Testing**: 100 - 1,000 examples

### Memory Considerations

- **Image size**: 224x224 pixels recommended
- **Text length**: 512 tokens maximum
- **Timeseries length**: 48 time steps maximum
- **Batch size**: Adjust based on GPU memory

### Storage Optimization

- **Image compression**: Use JPEG for photos, PNG for medical images
- **Timeseries compression**: Use NPY for numerical data
- **Text compression**: UTF-8 encoding
- **GCS optimization**: Use appropriate storage class

## Common Issues and Solutions

### 1. Missing Files
**Issue**: Referenced files not found
**Solution**: Check file paths and ensure all files are uploaded to GCS

### 2. Invalid Image Formats
**Issue**: Images not readable by PIL
**Solution**: Convert to supported formats (JPG, PNG)

### 3. Encoding Issues
**Issue**: Text contains invalid UTF-8
**Solution**: Clean text data and ensure UTF-8 encoding

### 4. Large File Sizes
**Issue**: Individual files too large
**Solution**: Compress images, optimize timeseries data

### 5. GCS Access Issues
**Issue**: Cannot access GCS files
**Solution**: Check permissions and bucket configuration

## Best Practices

1. **Data Quality**:
   - Clean and preprocess data before creating JSONL
   - Validate file references
   - Check for data consistency

2. **Performance**:
   - Use appropriate image sizes
   - Optimize file formats
   - Consider data compression

3. **Storage**:
   - Use GCS for large datasets
   - Organize files logically
   - Set appropriate permissions

4. **Validation**:
   - Always validate before submission
   - Check both local and GCS paths
   - Review validation warnings

5. **Documentation**:
   - Document data sources
   - Record preprocessing steps
   - Maintain data lineage

## Troubleshooting

### Validation Failures

1. **Check file paths**: Ensure all referenced files exist
2. **Verify permissions**: Check GCS bucket access
3. **Review data format**: Ensure JSONL format is correct
4. **Check file sizes**: Ensure files are not empty or too large

### Performance Issues

1. **Reduce image size**: Use smaller images if memory is limited
2. **Optimize batch size**: Adjust based on available GPU memory
3. **Use data compression**: Compress files where appropriate
4. **Check network**: Ensure good connectivity to GCS

### Common Error Messages

- **"File not found"**: Check file paths and GCS permissions
- **"Invalid JSON"**: Check JSONL format and encoding
- **"Image not readable"**: Convert to supported image formats
- **"Timeseries parse error"**: Check CSV/NPY file format
- **"GCS access denied"**: Verify bucket permissions and authentication 