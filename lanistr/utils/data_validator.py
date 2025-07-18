"""Data validation utilities for LANISTR multimodal datasets.

This module provides comprehensive validation for JSONL datasets containing
multimodal data (text, image, tabular, timeseries) with GCS path validation,
mimetype checks, and content validation.
"""

import json
import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings

import pandas as pd
from PIL import Image
import numpy as np
from google.cloud import storage
from google.cloud.exceptions import NotFound
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# =============================================================================
# DATA SCHEMA DEFINITIONS
# =============================================================================

# Amazon dataset schema
AMAZON_SCHEMA = {
    "required_fields": ["Review", "ImageFileName", "reviewerID", "verified", "asin", "year", "vote", "unixReviewTime"],
    "text_fields": ["Review"],
    "image_fields": ["ImageFileName"],
    "categorical_fields": ["reviewerID", "verified", "asin", "year"],
    "numerical_fields": ["vote", "unixReviewTime"],
    "field_types": {
        "Review": "string",
        "ImageFileName": "string",
        "reviewerID": "string",
        "verified": "boolean",
        "asin": "string", 
        "year": "integer",
        "vote": "integer",
        "unixReviewTime": "integer"
    },
    "min_examples": 1000,
    "max_examples": 1000000
}

# MIMIC-IV dataset schema
MIMIC_IV_SCHEMA = {
    "required_fields": ["text", "image", "timeseries"],
    "text_fields": ["text"],
    "image_fields": ["image"],
    "timeseries_fields": ["timeseries"],
    "optional_fields": ["y_true"],  # For finetuning tasks
    "field_types": {
        "text": "string",
        "image": "string",
        "timeseries": "string",
        "y_true": "integer"
    },
    "min_examples": 100,
    "max_examples": 100000
}

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Supported timeseries formats  
SUPPORTED_TIMESERIES_FORMATS = {'.csv', '.npy', '.parquet'}

# =============================================================================
# VALIDATION CLASSES
# =============================================================================

class DataValidator:
    """Comprehensive data validator for LANISTR multimodal datasets."""
    
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        image_data_dir: Optional[str] = None,
        timeseries_data_dir: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        project_id: Optional[str] = None,
        validate_content: bool = True,
        max_validation_samples: int = 1000
    ):
        """Initialize the data validator.
        
        Args:
            dataset_name: Name of the dataset ('amazon' or 'mimic-iv')
            data_dir: Directory containing the JSONL files
            image_data_dir: Directory containing image files (optional)
            timeseries_data_dir: Directory containing timeseries files (optional)
            gcs_bucket: GCS bucket name for cloud storage validation
            project_id: Google Cloud project ID
            validate_content: Whether to validate file contents
            max_validation_samples: Maximum number of samples to validate
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.image_data_dir = image_data_dir
        self.timeseries_data_dir = timeseries_data_dir
        self.gcs_bucket = gcs_bucket
        self.project_id = project_id
        self.validate_content = validate_content
        self.max_validation_samples = max_validation_samples
        
        # Get schema for dataset
        if self.dataset_name == 'amazon':
            self.schema = AMAZON_SCHEMA
        elif self.dataset_name == 'mimic-iv':
            self.schema = MIMIC_IV_SCHEMA
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Initialize GCS client if needed
        self.gcs_client = None
        if gcs_bucket:
            try:
                self.gcs_client = storage.Client(project=project_id)
                self.bucket = self.gcs_client.bucket(gcs_bucket)
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client: {e}")
                self.gcs_client = None
        
        # Validation results
        self.validation_results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
    
    def validate_dataset(self, jsonl_file: str) -> Dict[str, Any]:
        """Validate a complete JSONL dataset.
        
        Args:
            jsonl_file: Path to the JSONL file
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Starting validation of {jsonl_file}")
        
        try:
            # Reset validation results
            self.validation_results = {
                'passed': True,
                'errors': [],
                'warnings': [],
                'stats': {}
            }
            
            # Validate file exists and is readable
            if not self._validate_file_access(jsonl_file):
                return self.validation_results
            
            # Load and validate JSONL structure
            data = self._load_and_validate_jsonl(jsonl_file)
            if not data:
                return self.validation_results
            
            # Validate schema compliance
            self._validate_schema_compliance(data)
            
            # Validate data statistics
            self._validate_data_statistics(data)
            
            # Validate file references
            self._validate_file_references(data)
            
            # Validate content (if enabled)
            if self.validate_content:
                self._validate_content(data)
            
            # Generate summary
            self._generate_validation_summary()
            
        except Exception as e:
            self.validation_results['passed'] = False
            self.validation_results['errors'].append(f"Validation failed: {str(e)}")
            logger.error(f"Validation failed: {e}")
        
        return self.validation_results
    
    def _validate_file_access(self, file_path: str) -> bool:
        """Validate that the file exists and is accessible."""
        if not os.path.exists(file_path):
            self.validation_results['errors'].append(f"File not found: {file_path}")
            return False
        
        if not os.access(file_path, os.R_OK):
            self.validation_results['errors'].append(f"File not readable: {file_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            self.validation_results['errors'].append(f"File is empty: {file_path}")
            return False
        
        self.validation_results['stats']['file_size_mb'] = file_size / (1024 * 1024)
        return True
    
    def _load_and_validate_jsonl(self, jsonl_file: str) -> List[Dict[str, Any]]:
        """Load and validate JSONL structure."""
        data = []
        line_number = 0
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line_number += 1
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            self.validation_results['errors'].append(
                                f"Line {line_number}: Record is not a dictionary"
                            )
                            continue
                        
                        data.append(record)
                        
                        # Limit validation samples
                        if len(data) >= self.max_validation_samples:
                            self.validation_results['warnings'].append(
                                f"Limited validation to first {self.max_validation_samples} samples"
                            )
                            break
                    
                    except json.JSONDecodeError as e:
                        self.validation_results['errors'].append(
                            f"Line {line_number}: Invalid JSON - {str(e)}"
                        )
            
            self.validation_results['stats']['total_records'] = len(data)
            
            if len(data) == 0:
                self.validation_results['errors'].append("No valid records found in JSONL file")
                return None
            
            return data
            
        except Exception as e:
            self.validation_results['errors'].append(f"Failed to read JSONL file: {str(e)}")
            return None
    
    def _validate_schema_compliance(self, data: List[Dict[str, Any]]) -> None:
        """Validate that data complies with the schema."""
        if not data:
            return
        
        # Check required fields
        required_fields = set(self.schema['required_fields'])
        sample_record = data[0]
        missing_fields = required_fields - set(sample_record.keys())
        
        if missing_fields:
            self.validation_results['errors'].append(
                f"Missing required fields: {list(missing_fields)}"
            )
        
        # Check field types
        field_types = self.schema.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in sample_record:
                actual_type = type(sample_record[field]).__name__
                if not self._validate_field_type(sample_record[field], expected_type):
                    self.validation_results['warnings'].append(
                        f"Field '{field}' has type '{actual_type}', expected '{expected_type}'"
                    )
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate that a value matches the expected type."""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'integer':
            return isinstance(value, int)
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'float':
            return isinstance(value, (int, float))
        else:
            return True  # Unknown type, assume valid
    
    def _validate_data_statistics(self, data: List[Dict[str, Any]]) -> None:
        """Validate data statistics and quality metrics."""
        if not data:
            return
        
        # Check number of examples
        num_examples = len(data)
        min_examples = self.schema.get('min_examples', 0)
        max_examples = self.schema.get('max_examples', float('inf'))
        
        if num_examples < min_examples:
            self.validation_results['warnings'].append(
                f"Dataset has {num_examples} examples, minimum recommended is {min_examples}"
            )
        
        if num_examples > max_examples:
            self.validation_results['warnings'].append(
                f"Dataset has {num_examples} examples, maximum recommended is {max_examples}"
            )
        
        # Calculate statistics for each field
        stats = {}
        for field in self.schema['required_fields']:
            if field in data[0]:
                values = [record.get(field) for record in data]
                stats[field] = self._calculate_field_statistics(field, values)
        
        self.validation_results['stats']['field_statistics'] = stats
    
    def _calculate_field_statistics(self, field: str, values: List[Any]) -> Dict[str, Any]:
        """Calculate statistics for a field."""
        # Remove None values
        non_null_values = [v for v in values if v is not None]
        
        stats = {
            'total_count': len(values),
            'non_null_count': len(non_null_values),
            'null_count': len(values) - len(non_null_values),
            'null_percentage': (len(values) - len(non_null_values)) / len(values) * 100
        }
        
        if not non_null_values:
            return stats
        
        # Type-specific statistics
        if isinstance(non_null_values[0], str):
            stats.update({
                'min_length': min(len(v) for v in non_null_values),
                'max_length': max(len(v) for v in non_null_values),
                'avg_length': sum(len(v) for v in non_null_values) / len(non_null_values),
                'unique_count': len(set(non_null_values))
            })
        elif isinstance(non_null_values[0], (int, float)):
            stats.update({
                'min_value': min(non_null_values),
                'max_value': max(non_null_values),
                'mean_value': sum(non_null_values) / len(non_null_values)
            })
        
        return stats
    
    def _validate_file_references(self, data: List[Dict[str, Any]]) -> None:
        """Validate that referenced files exist and are accessible."""
        if not data:
            return
        
        # Validate image files
        if 'image_fields' in self.schema:
            for field in self.schema['image_fields']:
                self._validate_image_references(data, field)
        
        # Validate timeseries files
        if 'timeseries_fields' in self.schema:
            for field in self.schema['timeseries_fields']:
                self._validate_timeseries_references(data, field)
    
    def _validate_image_references(self, data: List[Dict[str, Any]], field: str) -> None:
        """Validate image file references."""
        if not data or field not in data[0]:
            return
        
        image_paths = []
        for record in data:
            if field in record and record[field]:
                image_paths.append(record[field])
        
        if not image_paths:
            return
        
        # Sample validation
        sample_size = min(100, len(image_paths))
        sample_paths = image_paths[:sample_size]
        
        valid_count = 0
        for path in sample_paths:
            if self._validate_file_path(path, SUPPORTED_IMAGE_FORMATS):
                valid_count += 1
        
        success_rate = valid_count / sample_size * 100
        self.validation_results['stats'][f'{field}_validation_rate'] = success_rate
        
        if success_rate < 95:
            self.validation_results['warnings'].append(
                f"Image validation success rate: {success_rate:.1f}% (sampled {sample_size} files)"
            )
    
    def _validate_timeseries_references(self, data: List[Dict[str, Any]], field: str) -> None:
        """Validate timeseries file references."""
        if not data or field not in data[0]:
            return
        
        timeseries_paths = []
        for record in data:
            if field in record and record[field]:
                timeseries_paths.append(record[field])
        
        if not timeseries_paths:
            return
        
        # Sample validation
        sample_size = min(100, len(timeseries_paths))
        sample_paths = timeseries_paths[:sample_size]
        
        valid_count = 0
        for path in sample_paths:
            if self._validate_file_path(path, SUPPORTED_TIMESERIES_FORMATS):
                valid_count += 1
        
        success_rate = valid_count / sample_size * 100
        self.validation_results['stats'][f'{field}_validation_rate'] = success_rate
        
        if success_rate < 95:
            self.validation_results['warnings'].append(
                f"Timeseries validation success rate: {success_rate:.1f}% (sampled {sample_size} files)"
            )
    
    def _validate_file_path(self, path: str, supported_formats: Set[str]) -> bool:
        """Validate that a file path exists and has correct format."""
        if not path or not isinstance(path, str):
            return False
        
        # Check file extension
        file_ext = os.path.splitext(path)[1].lower()
        if file_ext not in supported_formats:
            return False
        
        # Check if it's a GCS path
        if path.startswith('gs://'):
            return self._validate_gcs_file(path)
        elif path.startswith('http://') or path.startswith('https://'):
            return self._validate_http_file(path)
        else:
            return self._validate_local_file(path)
    
    def _validate_gcs_file(self, gcs_path: str) -> bool:
        """Validate GCS file exists."""
        if not self.gcs_client:
            return False
        
        try:
            # Parse GCS path
            if not gcs_path.startswith('gs://'):
                return False
            
            bucket_name = gcs_path.split('/')[2]
            blob_name = '/'.join(gcs_path.split('/')[3:])
            
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            return blob.exists()
        
        except Exception:
            return False
    
    def _validate_http_file(self, url: str) -> bool:
        """Validate HTTP file exists."""
        try:
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def _validate_local_file(self, file_path: str) -> bool:
        """Validate local file exists."""
        # Construct full path
        if self.image_data_dir and any(ext in file_path.lower() for ext in SUPPORTED_IMAGE_FORMATS):
            full_path = os.path.join(self.image_data_dir, file_path)
        elif self.timeseries_data_dir and any(ext in file_path.lower() for ext in SUPPORTED_TIMESERIES_FORMATS):
            full_path = os.path.join(self.timeseries_data_dir, file_path)
        else:
            full_path = os.path.join(self.data_dir, file_path)
        
        return os.path.exists(full_path) and os.access(full_path, os.R_OK)
    
    def _validate_content(self, data: List[Dict[str, Any]]) -> None:
        """Validate content of files (images, timeseries, etc.)."""
        if not data:
            return
        
        # Sample validation
        sample_size = min(50, len(data))
        sample_data = data[:sample_size]
        
        # Validate image content
        if 'image_fields' in self.schema:
            for field in self.schema['image_fields']:
                self._validate_image_content(sample_data, field)
        
        # Validate timeseries content
        if 'timeseries_fields' in self.schema:
            for field in self.schema['timeseries_fields']:
                self._validate_timeseries_content(sample_data, field)
        
        # Validate text content
        if 'text_fields' in self.schema:
            for field in self.schema['text_fields']:
                self._validate_text_content(sample_data, field)
    
    def _validate_image_content(self, data: List[Dict[str, Any]], field: str) -> None:
        """Validate image content."""
        valid_count = 0
        total_count = 0
        
        for record in data:
            if field in record and record[field]:
                total_count += 1
                try:
                    if self._is_valid_image(record[field]):
                        valid_count += 1
                except Exception:
                    pass
        
        if total_count > 0:
            success_rate = valid_count / total_count * 100
            self.validation_results['stats'][f'{field}_content_validation_rate'] = success_rate
            
            if success_rate < 90:
                self.validation_results['warnings'].append(
                    f"Image content validation success rate: {success_rate:.1f}%"
                )
    
    def _is_valid_image(self, image_path: str) -> bool:
        """Check if image file is valid and readable."""
        try:
            if image_path.startswith('gs://'):
                # For GCS, we'll assume it's valid if it exists
                return self._validate_gcs_file(image_path)
            else:
                # For local files, try to open with PIL
                if self.image_data_dir:
                    full_path = os.path.join(self.image_data_dir, image_path)
                else:
                    full_path = os.path.join(self.data_dir, image_path)
                
                with Image.open(full_path) as img:
                    img.verify()
                return True
        except Exception:
            return False
    
    def _validate_timeseries_content(self, data: List[Dict[str, Any]], field: str) -> None:
        """Validate timeseries content."""
        valid_count = 0
        total_count = 0
        
        for record in data:
            if field in record and record[field]:
                total_count += 1
                try:
                    if self._is_valid_timeseries(record[field]):
                        valid_count += 1
                except Exception:
                    pass
        
        if total_count > 0:
            success_rate = valid_count / total_count * 100
            self.validation_results['stats'][f'{field}_content_validation_rate'] = success_rate
            
            if success_rate < 90:
                self.validation_results['warnings'].append(
                    f"Timeseries content validation success rate: {success_rate:.1f}%"
                )
    
    def _is_valid_timeseries(self, file_path: str) -> bool:
        """Check if timeseries file is valid and readable."""
        try:
            if file_path.startswith('gs://'):
                return self._validate_gcs_file(file_path)
            else:
                if self.timeseries_data_dir:
                    full_path = os.path.join(self.timeseries_data_dir, file_path)
                else:
                    full_path = os.path.join(self.data_dir, file_path)
                
                # Try to read the file
                if file_path.endswith('.csv'):
                    pd.read_csv(full_path, nrows=5)  # Read first 5 rows
                elif file_path.endswith('.npy'):
                    np.load(full_path)
                elif file_path.endswith('.parquet'):
                    pd.read_parquet(full_path, nrows=5)
                
                return True
        except Exception:
            return False
    
    def _validate_text_content(self, data: List[Dict[str, Any]], field: str) -> None:
        """Validate text content."""
        valid_count = 0
        total_count = 0
        
        for record in data:
            if field in record and record[field]:
                total_count += 1
                try:
                    if self._is_valid_text(record[field]):
                        valid_count += 1
                except Exception:
                    pass
        
        if total_count > 0:
            success_rate = valid_count / total_count * 100
            self.validation_results['stats'][f'{field}_content_validation_rate'] = success_rate
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text is valid."""
        if not isinstance(text, str):
            return False
        
        # Check for encoding issues
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            return False
        
        # Check for reasonable length
        if len(text) < 1 or len(text) > 100000:  # 100KB limit
            return False
        
        return True
    
    def _generate_validation_summary(self) -> None:
        """Generate validation summary."""
        if self.validation_results['errors']:
            self.validation_results['passed'] = False
        
        # Add summary statistics
        self.validation_results['summary'] = {
            'total_errors': len(self.validation_results['errors']),
            'total_warnings': len(self.validation_results['warnings']),
            'validation_passed': self.validation_results['passed']
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_amazon_dataset(
    jsonl_file: str,
    data_dir: str,
    image_data_dir: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    project_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validate Amazon dataset.
    
    Args:
        jsonl_file: Path to JSONL file
        data_dir: Data directory
        image_data_dir: Image data directory
        gcs_bucket: GCS bucket name
        project_id: Google Cloud project ID
        **kwargs: Additional validator arguments
        
    Returns:
        Validation results
    """
    validator = DataValidator(
        dataset_name='amazon',
        data_dir=data_dir,
        image_data_dir=image_data_dir,
        gcs_bucket=gcs_bucket,
        project_id=project_id,
        **kwargs
    )
    return validator.validate_dataset(jsonl_file)


def validate_mimic_dataset(
    jsonl_file: str,
    data_dir: str,
    image_data_dir: Optional[str] = None,
    timeseries_data_dir: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    project_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Validate MIMIC-IV dataset.
    
    Args:
        jsonl_file: Path to JSONL file
        data_dir: Data directory
        image_data_dir: Image data directory
        timeseries_data_dir: Timeseries data directory
        gcs_bucket: GCS bucket name
        project_id: Google Cloud project ID
        **kwargs: Additional validator arguments
        
    Returns:
        Validation results
    """
    validator = DataValidator(
        dataset_name='mimic-iv',
        data_dir=data_dir,
        image_data_dir=image_data_dir,
        timeseries_data_dir=timeseries_data_dir,
        gcs_bucket=gcs_bucket,
        project_id=project_id,
        **kwargs
    )
    return validator.validate_dataset(jsonl_file)


def print_validation_report(results: Dict[str, Any]) -> None:
    """Print a formatted validation report.
    
    Args:
        results: Validation results from DataValidator
    """
    print("=" * 80)
    print("LANISTR DATASET VALIDATION REPORT")
    print("=" * 80)
    
    # Overall status
    status = "✅ PASSED" if results['passed'] else "❌ FAILED"
    print(f"Overall Status: {status}")
    print()
    
    # Statistics
    if 'stats' in results:
        print("📊 DATASET STATISTICS:")
        stats = results['stats']
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Errors
    if results['errors']:
        print("❌ ERRORS:")
        for error in results['errors']:
            print(f"  • {error}")
        print()
    
    # Warnings
    if results['warnings']:
        print("⚠️  WARNINGS:")
        for warning in results['warnings']:
            print(f"  • {warning}")
        print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if results['passed']:
        print("  • Dataset is ready for training")
        if results['warnings']:
            print("  • Review warnings before proceeding")
    else:
        print("  • Fix errors before proceeding with training")
        print("  • Review warnings for potential issues")
    
    print("=" * 80) 