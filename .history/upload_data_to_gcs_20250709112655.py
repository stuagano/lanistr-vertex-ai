#!/usr/bin/env python3
"""
Upload LANISTR Data to Google Cloud Storage

This script uploads the LANISTR dataset to Google Cloud Storage for use with
Vertex AI distributed training.
"""

import argparse
import os
import glob
from pathlib import Path
from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_directory_to_gcs(
    local_path: str,
    bucket_name: str,
    gcs_prefix: str,
    project_id: str = None
):
    """
    Upload a local directory to Google Cloud Storage.
    
    Args:
        local_path: Local directory path to upload
        bucket_name: GCS bucket name
        gcs_prefix: GCS prefix (folder path in bucket)
        project_id: Google Cloud project ID
    """
    
    # Initialize GCS client
    if project_id:
        client = storage.Client(project=project_id)
    else:
        client = storage.Client()
    
    bucket = client.bucket(bucket_name)
    
    # Ensure local path exists
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local path does not exist: {local_path}")
    
    # Upload files recursively
    local_path = Path(local_path)
    uploaded_count = 0
    
    for file_path in local_path.rglob('*'):
        if file_path.is_file():
            # Calculate relative path from local_path
            relative_path = file_path.relative_to(local_path)
            gcs_blob_name = f"{gcs_prefix}/{relative_path}"
            
            # Create blob and upload
            blob = bucket.blob(gcs_blob_name)
            blob.upload_from_filename(str(file_path))
            
            uploaded_count += 1
            if uploaded_count % 100 == 0:
                logger.info(f"Uploaded {uploaded_count} files...")
    
    logger.info(f"Successfully uploaded {uploaded_count} files to gs://{bucket_name}/{gcs_prefix}")


def upload_mimic_data(local_data_dir: str, bucket_name: str, project_id: str = None):
    """Upload MIMIC-IV data to GCS."""
    logger.info("Uploading MIMIC-IV data...")
    
    # Upload main data directory
    upload_directory_to_gcs(
        local_path=f"{local_data_dir}/MIMIC-IV-V2.2",
        bucket_name=bucket_name,
        gcs_prefix="lanistr-data/MIMIC-IV-V2.2",
        project_id=project_id
    )


def upload_amazon_data(local_data_dir: str, bucket_name: str, project_id: str = None):
    """Upload Amazon Product Review data to GCS."""
    logger.info("Uploading Amazon Product Review data...")
    
    # Upload main data directory
    upload_directory_to_gcs(
        local_path=f"{local_data_dir}/APR2018",
        bucket_name=bucket_name,
        gcs_prefix="lanistr-data/APR2018",
        project_id=project_id
    )


def create_bucket_if_not_exists(bucket_name: str, project_id: str = None, location: str = "US"):
    """Create GCS bucket if it doesn't exist."""
    if project_id:
        client = storage.Client(project=project_id)
    else:
        client = storage.Client()
    
    bucket = client.bucket(bucket_name)
    
    if not bucket.exists():
        logger.info(f"Creating bucket: {bucket_name}")
        bucket.create(location=location)
        logger.info(f"Bucket {bucket_name} created successfully")
    else:
        logger.info(f"Bucket {bucket_name} already exists")


def main():
    parser = argparse.ArgumentParser(description="Upload LANISTR data to Google Cloud Storage")
    parser.add_argument("--local-data-dir", required=True, help="Local directory containing data")
    parser.add_argument("--bucket-name", required=True, help="GCS bucket name")
    parser.add_argument("--project-id", help="Google Cloud project ID")
    parser.add_argument("--dataset", choices=["mimic", "amazon", "all"], default="all", 
                       help="Dataset to upload")
    parser.add_argument("--create-bucket", action="store_true", help="Create bucket if it doesn't exist")
    parser.add_argument("--bucket-location", default="US", help="Bucket location")
    
    args = parser.parse_args()
    
    # Create bucket if requested
    if args.create_bucket:
        create_bucket_if_not_exists(args.bucket_name, args.project_id, args.bucket_location)
    
    # Upload data based on dataset choice
    if args.dataset in ["mimic", "all"]:
        mimic_path = f"{args.local_data_dir}/MIMIC-IV-V2.2"
        if os.path.exists(mimic_path):
            upload_mimic_data(args.local_data_dir, args.bucket_name, args.project_id)
        else:
            logger.warning(f"MIMIC-IV data not found at {mimic_path}")
    
    if args.dataset in ["amazon", "all"]:
        amazon_path = f"{args.local_data_dir}/APR2018"
        if os.path.exists(amazon_path):
            upload_amazon_data(args.local_data_dir, args.bucket_name, args.project_id)
        else:
            logger.warning(f"Amazon data not found at {amazon_path}")
    
    logger.info("Data upload completed!")


if __name__ == "__main__":
    main() 