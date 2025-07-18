#!/usr/bin/env python3
"""Command-line tool for validating LANISTR datasets before Vertex AI submission.

This script validates JSONL datasets for all modalities (text, image, tabular, timeseries)
with comprehensive checks including GCS path validation, mimetype verification, and
content validation.

Usage:
    python validate_dataset.py --dataset amazon --jsonl-file data.jsonl --data-dir ./data
    python validate_dataset.py --dataset mimic-iv --jsonl-file data.jsonl --data-dir ./data --gcs-bucket my-bucket
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lanistr.utils.data_validator import (
    DataValidator,
    validate_amazon_dataset,
    validate_mimic_dataset,
    print_validation_report
)

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dataset_validation.log')
        ]
    )

def validate_dataset(
    dataset_name: str,
    jsonl_file: str,
    data_dir: str,
    image_data_dir: Optional[str] = None,
    timeseries_data_dir: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    project_id: Optional[str] = None,
    validate_content: bool = True,
    max_samples: int = 1000,
    output_file: Optional[str] = None
) -> bool:
    """Validate a dataset and return success status.
    
    Args:
        dataset_name: Name of the dataset ('amazon' or 'mimic-iv')
        jsonl_file: Path to the JSONL file
        data_dir: Data directory
        image_data_dir: Image data directory
        timeseries_data_dir: Timeseries data directory
        gcs_bucket: GCS bucket name
        project_id: Google Cloud project ID
        validate_content: Whether to validate file contents
        max_samples: Maximum number of samples to validate
        output_file: Optional file to save results
        
    Returns:
        True if validation passed, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not os.path.exists(jsonl_file):
        logger.error(f"JSONL file not found: {jsonl_file}")
        return False
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    # Run validation based on dataset type
    if dataset_name.lower() == 'amazon':
        results = validate_amazon_dataset(
            jsonl_file=jsonl_file,
            data_dir=data_dir,
            image_data_dir=image_data_dir,
            gcs_bucket=gcs_bucket,
            project_id=project_id,
            validate_content=validate_content,
            max_validation_samples=max_samples
        )
    elif dataset_name.lower() == 'mimic-iv':
        results = validate_mimic_dataset(
            jsonl_file=jsonl_file,
            data_dir=data_dir,
            image_data_dir=image_data_dir,
            timeseries_data_dir=timeseries_data_dir,
            gcs_bucket=gcs_bucket,
            project_id=project_id,
            validate_content=validate_content,
            max_validation_samples=max_samples
        )
    else:
        logger.error(f"Unsupported dataset: {dataset_name}")
        return False
    
    # Print report
    print_validation_report(results)
    
    # Save results if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Validation results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    return results['passed']

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate LANISTR datasets for Vertex AI submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate Amazon dataset
  python validate_dataset.py --dataset amazon --jsonl-file data.jsonl --data-dir ./data

  # Validate MIMIC-IV dataset with GCS validation
  python validate_dataset.py --dataset mimic-iv --jsonl-file data.jsonl --data-dir ./data --gcs-bucket my-bucket

  # Validate with custom directories
  python validate_dataset.py --dataset amazon --jsonl-file data.jsonl --data-dir ./data --image-data-dir ./images

  # Quick validation (no content checks)
  python validate_dataset.py --dataset amazon --jsonl-file data.jsonl --data-dir ./data --no-content-validation

  # Save results to file
  python validate_dataset.py --dataset amazon --jsonl-file data.jsonl --data-dir ./data --output-file results.json
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['amazon', 'mimic-iv'],
        help='Dataset name (amazon or mimic-iv)'
    )
    parser.add_argument(
        '--jsonl-file',
        required=True,
        help='Path to the JSONL file to validate'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Directory containing the data files'
    )
    
    # Optional arguments
    parser.add_argument(
        '--image-data-dir',
        help='Directory containing image files (optional)'
    )
    parser.add_argument(
        '--timeseries-data-dir',
        help='Directory containing timeseries files (optional)'
    )
    parser.add_argument(
        '--gcs-bucket',
        help='GCS bucket name for cloud storage validation'
    )
    parser.add_argument(
        '--project-id',
        help='Google Cloud project ID (defaults to gcloud config)'
    )
    parser.add_argument(
        '--no-content-validation',
        action='store_true',
        help='Skip content validation (faster but less thorough)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=1000,
        help='Maximum number of samples to validate (default: 1000)'
    )
    parser.add_argument(
        '--output-file',
        help='Save validation results to JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Get project ID from gcloud if not provided
    project_id = args.project_id
    if not project_id and args.gcs_bucket:
        try:
            import subprocess
            result = subprocess.run(
                ['gcloud', 'config', 'get-value', 'project'],
                capture_output=True,
                text=True,
                check=True
            )
            project_id = result.stdout.strip()
            logger.info(f"Using project ID from gcloud: {project_id}")
        except Exception as e:
            logger.warning(f"Failed to get project ID from gcloud: {e}")
    
    # Run validation
    logger.info(f"Starting validation of {args.dataset} dataset")
    logger.info(f"JSONL file: {args.jsonl_file}")
    logger.info(f"Data directory: {args.data_dir}")
    
    success = validate_dataset(
        dataset_name=args.dataset,
        jsonl_file=args.jsonl_file,
        data_dir=args.data_dir,
        image_data_dir=args.image_data_dir,
        timeseries_data_dir=args.timeseries_data_dir,
        gcs_bucket=args.gcs_bucket,
        project_id=project_id,
        validate_content=not args.no_content_validation,
        max_samples=args.max_samples,
        output_file=args.output_file
    )
    
    # Exit with appropriate code
    if success:
        logger.info("✅ Dataset validation PASSED - ready for Vertex AI submission")
        sys.exit(0)
    else:
        logger.error("❌ Dataset validation FAILED - fix issues before submission")
        sys.exit(1)

if __name__ == "__main__":
    main() 