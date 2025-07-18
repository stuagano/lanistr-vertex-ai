#!/usr/bin/env python3
"""
Data preparation script for LANISTR JSONL format.

This script converts existing data formats to JSONL format suitable for LANISTR training
on Google Cloud Vertex AI. It supports both MIMIC-IV and Amazon datasets.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from google.cloud import storage
import structlog

logger = structlog.get_logger()

# =============================================================================
# DATA CONVERSION FUNCTIONS
# =============================================================================

def convert_mimic_iv_to_jsonl(
    data_dir: str,
    output_dir: str,
    bucket_name: str,
    splits: Optional[Dict[str, float]] = None,
    text_column: str = "text",
    image_column: str = "image",
    timeseries_column: str = "timeseries",
    features_column: str = "features",
    label_column: Optional[str] = None,
    id_column: str = "id"
) -> Dict[str, str]:
    """Convert MIMIC-IV data to JSONL format.
    
    Args:
        data_dir: Directory containing MIMIC-IV data files
        output_dir: Output directory for JSONL files
        bucket_name: GCS bucket name for file paths
        splits: Dictionary of split ratios (e.g., {"train": 0.8, "valid": 0.1, "test": 0.1})
        text_column: Column name for text data
        image_column: Column name for image paths
        timeseries_column: Column name for timeseries paths
        features_column: Column name for tabular features
        label_column: Column name for labels (optional)
        id_column: Column name for unique IDs
        
    Returns:
        Dictionary mapping split names to output file paths
    """
    logger.info(f"Converting MIMIC-IV data from {data_dir} to {output_dir}")
    
    # Load data
    data_files = list(Path(data_dir).glob("*.csv"))
    if not data_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    # Load and combine all data
    all_data = []
    for file_path in data_files:
        logger.info(f"Loading {file_path}")
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(data)} total examples")
    
    # Create splits if not provided
    if splits is None:
        splits = {"train": 0.8, "valid": 0.1, "test": 0.1}
    
    # Split data
    np.random.seed(42)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_data = {}
    start_idx = 0
    for split_name, split_ratio in splits.items():
        end_idx = start_idx + int(len(data) * split_ratio)
        split_data[split_name] = data.iloc[start_idx:end_idx]
        start_idx = end_idx
    
    # Ensure all data is used
    if start_idx < len(data):
        split_data["train"] = pd.concat([split_data["train"], data.iloc[start_idx:]], ignore_index=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each split to JSONL
    output_files = {}
    for split_name, split_df in split_data.items():
        output_file = os.path.join(output_dir, f"mimic_iv_{split_name}.jsonl")
        logger.info(f"Converting {split_name} split ({len(split_df)} examples) to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in split_df.iterrows():
                example = {
                    "id": str(row.get(id_column, f"mimic_{idx:06d}")),
                    "split": split_name
                }
                
                # Add text if available
                if text_column in row and pd.notna(row[text_column]):
                    if isinstance(row[text_column], str) and row[text_column].startswith("gs://"):
                        example["text"] = row[text_column]
                    else:
                        example["text"] = f"gs://{bucket_name}/mimic-iv/text/{row[text_column]}"
                
                # Add image if available
                if image_column in row and pd.notna(row[image_column]):
                    if isinstance(row[image_column], str) and row[image_column].startswith("gs://"):
                        example["image"] = row[image_column]
                    else:
                        example["image"] = f"gs://{bucket_name}/mimic-iv/images/{row[image_column]}"
                
                # Add timeseries if available
                if timeseries_column in row and pd.notna(row[timeseries_column]):
                    if isinstance(row[timeseries_column], str) and row[timeseries_column].startswith("gs://"):
                        example["timeseries"] = row[timeseries_column]
                    else:
                        example["timeseries"] = f"gs://{bucket_name}/mimic-iv/timeseries/{row[timeseries_column]}"
                
                # Add tabular features if available
                if features_column in row and pd.notna(row[features_column]):
                    if isinstance(row[features_column], (list, np.ndarray)):
                        example["features"] = row[features_column].tolist() if hasattr(row[features_column], 'tolist') else list(row[features_column])
                    else:
                        # Try to parse as string representation of list
                        try:
                            import ast
                            example["features"] = ast.literal_eval(str(row[features_column]))
                        except:
                            logger.warning(f"Could not parse features for example {idx}")
                
                # Add label if available
                if label_column and label_column in row and pd.notna(row[label_column]):
                    example["y_true"] = int(row[label_column])
                
                # Remove None values and write
                example = {k: v for k, v in example.items() if v is not None}
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        output_files[split_name] = output_file
        logger.info(f"Created {output_file} with {len(split_df)} examples")
    
    return output_files

def convert_amazon_to_jsonl(
    data_dir: str,
    output_dir: str,
    bucket_name: str,
    splits: Optional[Dict[str, float]] = None,
    review_column: str = "Review",
    image_column: str = "ImageFileName",
    reviewer_id_column: str = "reviewerID",
    verified_column: str = "verified",
    asin_column: str = "asin",
    year_column: str = "year",
    vote_column: str = "vote",
    unix_time_column: str = "unixReviewTime",
    label_column: Optional[str] = None,
    id_column: str = "id"
) -> Dict[str, str]:
    """Convert Amazon data to JSONL format.
    
    Args:
        data_dir: Directory containing Amazon data files
        output_dir: Output directory for JSONL files
        bucket_name: GCS bucket name for file paths
        splits: Dictionary of split ratios
        review_column: Column name for review text
        image_column: Column name for image paths
        reviewer_id_column: Column name for reviewer ID
        verified_column: Column name for verified status
        asin_column: Column name for ASIN
        year_column: Column name for year
        vote_column: Column name for vote count
        unix_time_column: Column name for Unix timestamp
        label_column: Column name for labels (optional)
        id_column: Column name for unique IDs
        
    Returns:
        Dictionary mapping split names to output file paths
    """
    logger.info(f"Converting Amazon data from {data_dir} to {output_dir}")
    
    # Load data
    data_files = list(Path(data_dir).glob("*.csv"))
    if not data_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    # Load and combine all data
    all_data = []
    for file_path in data_files:
        logger.info(f"Loading {file_path}")
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(data)} total examples")
    
    # Create splits if not provided
    if splits is None:
        splits = {"train": 0.8, "valid": 0.1, "test": 0.1}
    
    # Split data
    np.random.seed(42)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_data = {}
    start_idx = 0
    for split_name, split_ratio in splits.items():
        end_idx = start_idx + int(len(data) * split_ratio)
        split_data[split_name] = data.iloc[start_idx:end_idx]
        start_idx = end_idx
    
    # Ensure all data is used
    if start_idx < len(data):
        split_data["train"] = pd.concat([split_data["train"], data.iloc[start_idx:]], ignore_index=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each split to JSONL
    output_files = {}
    for split_name, split_df in split_data.items():
        output_file = os.path.join(output_dir, f"amazon_{split_name}.jsonl")
        logger.info(f"Converting {split_name} split ({len(split_df)} examples) to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in split_df.iterrows():
                example = {
                    "id": str(row.get(id_column, f"amazon_{idx:06d}")),
                    "split": split_name
                }
                
                # Add review if available
                if review_column in row and pd.notna(row[review_column]):
                    if isinstance(row[review_column], str) and row[review_column].startswith("gs://"):
                        example["Review"] = row[review_column]
                    else:
                        example["Review"] = f"gs://{bucket_name}/amazon/reviews/{row[review_column]}"
                
                # Add image if available
                if image_column in row and pd.notna(row[image_column]):
                    if isinstance(row[image_column], str) and row[image_column].startswith("gs://"):
                        example["ImageFileName"] = row[image_column]
                    else:
                        example["ImageFileName"] = f"gs://{bucket_name}/amazon/images/{row[image_column]}"
                
                # Add tabular features
                tabular_features = {}
                
                if reviewer_id_column in row and pd.notna(row[reviewer_id_column]):
                    tabular_features["reviewerID"] = str(row[reviewer_id_column])
                
                if verified_column in row and pd.notna(row[verified_column]):
                    tabular_features["verified"] = bool(row[verified_column])
                
                if asin_column in row and pd.notna(row[asin_column]):
                    tabular_features["asin"] = str(row[asin_column])
                
                if year_column in row and pd.notna(row[year_column]):
                    tabular_features["year"] = int(row[year_column])
                
                if vote_column in row and pd.notna(row[vote_column]):
                    tabular_features["vote"] = int(row[vote_column])
                
                if unix_time_column in row and pd.notna(row[unix_time_column]):
                    tabular_features["unixReviewTime"] = int(row[unix_time_column])
                
                # Add tabular features to example
                example.update(tabular_features)
                
                # Add label if available
                if label_column and label_column in row and pd.notna(row[label_column]):
                    example["labels"] = int(row[label_column])
                
                # Remove None values and write
                example = {k: v for k, v in example.items() if v is not None}
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        output_files[split_name] = output_file
        logger.info(f"Created {output_file} with {len(split_df)} examples")
    
    return output_files

def upload_data_to_gcs(
    local_data_dir: str,
    bucket_name: str,
    project_id: Optional[str] = None,
    dataset_type: str = "mimic-iv"
) -> None:
    """Upload data files to Google Cloud Storage.
    
    Args:
        local_data_dir: Local directory containing data files
        bucket_name: GCS bucket name
        project_id: Google Cloud project ID
        dataset_type: Type of dataset
    """
    logger.info(f"Uploading {dataset_type} data to gs://{bucket_name}")
    
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}")
        return
    
    # Upload files recursively
    local_path = Path(local_data_dir)
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            # Create GCS path
            relative_path = file_path.relative_to(local_path)
            gcs_path = f"{dataset_type}/{relative_path}"
            
            logger.info(f"Uploading {file_path} to gs://{bucket_name}/{gcs_path}")
            
            try:
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(file_path))
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")

def validate_jsonl_files(
    jsonl_dir: str,
    dataset_type: str,
    bucket_name: Optional[str] = None,
    project_id: Optional[str] = None
) -> Dict[str, Any]:
    """Validate generated JSONL files.
    
    Args:
        jsonl_dir: Directory containing JSONL files
        dataset_type: Type of dataset
        bucket_name: GCS bucket name for validation
        project_id: Google Cloud project ID
        
    Returns:
        Validation results
    """
    from lanistr.utils.data_validator import validate_dataset_files
    
    logger.info(f"Validating {dataset_type} JSONL files in {jsonl_dir}")
    
    return validate_dataset_files(
        dataset_dir=jsonl_dir,
        dataset_type=dataset_type,
        bucket_name=bucket_name,
        project_id=project_id
    )

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main function for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare data for LANISTR training")
    parser.add_argument("--dataset-type", choices=["mimic-iv", "amazon"], required=True,
                       help="Type of dataset to convert")
    parser.add_argument("--input-dir", required=True,
                       help="Input directory containing data files")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for JSONL files")
    parser.add_argument("--bucket-name", required=True,
                       help="GCS bucket name for file paths")
    parser.add_argument("--project-id",
                       help="Google Cloud project ID")
    parser.add_argument("--upload-data", action="store_true",
                       help="Upload data files to GCS after conversion")
    parser.add_argument("--validate", action="store_true",
                       help="Validate JSONL files after conversion")
    parser.add_argument("--splits", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                       help="Train/valid/test split ratios")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Validate splits
    if len(args.splits) != 3 or sum(args.splits) != 1.0:
        raise ValueError("Splits must sum to 1.0")
    
    splits = {
        "train": args.splits[0],
        "valid": args.splits[1],
        "test": args.splits[2]
    }
    
    # Convert data to JSONL
    if args.dataset_type == "mimic-iv":
        output_files = convert_mimic_iv_to_jsonl(
            data_dir=args.input_dir,
            output_dir=args.output_dir,
            bucket_name=args.bucket_name,
            splits=splits
        )
    elif args.dataset_type == "amazon":
        output_files = convert_amazon_to_jsonl(
            data_dir=args.input_dir,
            output_dir=args.output_dir,
            bucket_name=args.bucket_name,
            splits=splits
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")
    
    logger.info("Data conversion completed successfully!")
    
    # Upload data to GCS if requested
    if args.upload_data:
        upload_data_to_gcs(
            local_data_dir=args.input_dir,
            bucket_name=args.bucket_name,
            project_id=args.project_id,
            dataset_type=args.dataset_type
        )
    
    # Validate JSONL files if requested
    if args.validate:
        validation_results = validate_jsonl_files(
            jsonl_dir=args.output_dir,
            dataset_type=args.dataset_type,
            bucket_name=args.bucket_name,
            project_id=args.project_id
        )
        
        # Print validation summary
        summary = validation_results["summary"]
        logger.info(f"Validation completed:")
        logger.info(f"  Overall validity: {summary['validity_percentage']:.2f}%")
        logger.info(f"  File success rate: {summary['file_success_rate']:.2f}%")
        logger.info(f"  Files validated: {validation_results['files_validated']}")
        logger.info(f"  Total lines: {validation_results['total_lines']:,}")
        logger.info(f"  Valid lines: {validation_results['total_valid_lines']:,}")
    
    # Print output file locations
    logger.info("Generated JSONL files:")
    for split_name, file_path in output_files.items():
        logger.info(f"  {split_name}: {file_path}")

if __name__ == "__main__":
    main() 