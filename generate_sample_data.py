#!/usr/bin/env python3
"""Generate sample JSONL datasets for LANISTR testing and validation.

This script creates sample datasets for both Amazon and MIMIC-IV formats
with realistic data for testing the validation system and training pipeline.

Usage:
    python generate_sample_data.py --dataset amazon --output-file sample_amazon.jsonl --num-samples 100
    python generate_sample_data.py --dataset mimic-iv --output-file sample_mimic.jsonl --num-samples 50
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Any
from datetime import datetime, timedelta

def generate_amazon_sample(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generate sample Amazon dataset.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of sample Amazon records
    """
    # Sample product categories and reviews
    categories = [
        "Electronics", "Books", "Home & Garden", "Sports & Outdoors",
        "Clothing", "Toys & Games", "Health & Beauty", "Automotive"
    ]
    
    # Sample review texts
    review_templates = [
        "Great product! {quality} and {delivery}. Highly recommend for anyone looking for a {solution}.",
        "I'm very satisfied with this purchase. The {quality} is excellent and it {delivery}.",
        "This item exceeded my expectations. {quality} and {delivery}. Would buy again!",
        "Good value for money. {quality} and {delivery}. Perfect for my needs.",
        "Disappointed with this product. {quality} and {delivery}. Would not recommend.",
        "Average product. {quality} and {delivery}. It's okay for the price.",
        "Excellent quality! {quality} and {delivery}. This is exactly what I was looking for.",
        "Very happy with this purchase. {quality} and {delivery}. Great customer service too."
    ]
    
    quality_phrases = [
        "quality is outstanding", "build quality is solid", "materials are durable",
        "construction is sturdy", "finish is beautiful", "design is elegant",
        "performance is reliable", "functionality is excellent"
    ]
    
    delivery_phrases = [
        "arrived on time", "shipping was fast", "delivery was prompt",
        "came well packaged", "arrived in perfect condition", "shipping was reasonable",
        "delivery took longer than expected", "arrived damaged"
    ]
    
    solution_phrases = [
        "reliable solution", "quality product", "good investment",
        "practical solution", "excellent choice", "worthwhile purchase"
    ]
    
    # Sample image filenames
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    data = []
    for i in range(num_samples):
        # Generate review text
        template = random.choice(review_templates)
        quality = random.choice(quality_phrases)
        delivery = random.choice(delivery_phrases)
        solution = random.choice(solution_phrases)
        
        review_text = template.format(
            quality=quality,
            delivery=delivery,
            solution=solution
        )
        
        # Generate image filename
        category = random.choice(categories).lower().replace(' & ', '_').replace(' ', '_')
        image_ext = random.choice(image_extensions)
        image_filename = f"images/{category}/product_{i:06d}{image_ext}"
        
        # Generate other fields
        reviewer_id = f"A{random.randint(1000000, 9999999)}"
        verified = random.choice([True, False])
        asin = f"B{random.randint(10000000, 99999999)}"
        year = random.randint(2018, 2024)
        vote = random.randint(0, 100)
        
        # Generate timestamp (within the year)
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        unix_time = int(random_date.timestamp())
        
        record = {
            "Review": review_text,
            "ImageFileName": image_filename,
            "reviewerID": reviewer_id,
            "verified": verified,
            "asin": asin,
            "year": year,
            "vote": vote,
            "unixReviewTime": unix_time
        }
        
        data.append(record)
    
    return data

def generate_mimic_sample(num_samples: int = 50) -> List[Dict[str, Any]]:
    """Generate sample MIMIC-IV dataset.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of sample MIMIC-IV records
    """
    # Sample clinical findings
    findings = [
        "normal cardiac silhouette",
        "mild cardiomegaly",
        "moderate cardiomegaly",
        "severe cardiomegaly",
        "clear lung fields",
        "bilateral infiltrates",
        "right lower lobe infiltrate",
        "left lower lobe infiltrate",
        "pleural effusion",
        "pneumothorax",
        "mediastinal widening",
        "normal mediastinum"
    ]
    
    # Sample clinical notes templates
    note_templates = [
        "CHEST X-RAY REPORT: The {cardiac} is {cardiac_size}. The {mediastinum} is {mediastinum_status}. {lung_findings}. {additional_findings}",
        "RADIOLOGY REPORT: {cardiac_size} {cardiac}. {mediastinum_status} {mediastinum}. {lung_findings}. {additional_findings}",
        "X-RAY FINDINGS: {cardiac} appears {cardiac_size}. {mediastinum} is {mediastinum_status}. {lung_findings}. {additional_findings}"
    ]
    
    cardiac_sizes = ["normal in size", "mildly enlarged", "moderately enlarged", "severely enlarged"]
    mediastinum_statuses = ["unremarkable", "normal", "slightly widened", "significantly widened"]
    lung_findings_list = [
        "The lungs are clear without evidence of infiltrate, mass, or consolidation",
        "No evidence of pneumothorax or pleural effusion",
        "Bilateral lung fields are clear",
        "No acute cardiopulmonary process",
        "Lungs are well expanded and clear",
        "No focal consolidation or mass identified"
    ]
    additional_findings_list = [
        "No acute abnormality.",
        "Clinical correlation recommended.",
        "Follow-up imaging as clinically indicated.",
        "No significant change from previous study.",
        "Impression: Normal chest x-ray.",
        "No evidence of active disease process."
    ]
    
    data = []
    for i in range(num_samples):
        # Generate clinical note
        template = random.choice(note_templates)
        cardiac = random.choice(["cardiac silhouette", "heart"])
        cardiac_size = random.choice(cardiac_sizes)
        mediastinum = random.choice(["mediastinum", "mediastinal structures"])
        mediastinum_status = random.choice(mediastinum_statuses)
        lung_findings = random.choice(lung_findings_list)
        additional_findings = random.choice(additional_findings_list)
        
        clinical_note = template.format(
            cardiac=cardiac,
            cardiac_size=cardiac_size,
            mediastinum=mediastinum,
            mediastinum_status=mediastinum_status,
            lung_findings=lung_findings,
            additional_findings=additional_findings
        )
        
        # Generate image filename (MIMIC-CXR format)
        patient_id = f"p{random.randint(10000000, 99999999)}"
        study_id = f"s{random.randint(50000000, 99999999)}"
        image_id = f"{random.randint(10000000, 99999999):08x}"
        image_filename = f"mimic-cxr-jpg/2.0.0/files/{patient_id[:2]}/{patient_id}/{study_id}/{image_id}.jpg"
        
        # Generate timeseries filename
        timeseries_filename = f"timeseries/patient_{patient_id}_vitals.csv"
        
        # Generate label (for finetuning)
        y_true = random.choice([0, 1])  # Binary classification
        
        record = {
            "text": clinical_note,
            "image": image_filename,
            "timeseries": timeseries_filename,
            "y_true": y_true
        }
        
        data.append(record)
    
    return data

def create_sample_files(data: List[Dict[str, Any]], output_file: str) -> None:
    """Create sample files referenced in the JSONL data.
    
    Args:
        data: List of data records
        output_file: Path to the JSONL file
    """
    # Create directories
    base_dir = os.path.dirname(output_file)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    
    # Create image directories
    image_dirs = set()
    for record in data:
        if "ImageFileName" in record:
            image_dir = os.path.join(base_dir, os.path.dirname(record["ImageFileName"]))
            image_dirs.add(image_dir)
        elif "image" in record:
            image_dir = os.path.join(base_dir, os.path.dirname(record["image"]))
            image_dirs.add(image_dir)
    
    for image_dir in image_dirs:
        os.makedirs(image_dir, exist_ok=True)
    
    # Create timeseries directories
    timeseries_dirs = set()
    for record in data:
        if "timeseries" in record:
            timeseries_dir = os.path.join(base_dir, os.path.dirname(record["timeseries"]))
            timeseries_dirs.add(timeseries_dir)
    
    for timeseries_dir in timeseries_dirs:
        os.makedirs(timeseries_dir, exist_ok=True)
    
    # Create placeholder files
    for record in data:
        # Create placeholder image files
        if "ImageFileName" in record:
            image_path = os.path.join(base_dir, record["ImageFileName"])
            if not os.path.exists(image_path):
                with open(image_path, 'w') as f:
                    f.write("# Placeholder image file\n")
        
        elif "image" in record:
            image_path = os.path.join(base_dir, record["image"])
            if not os.path.exists(image_path):
                with open(image_path, 'w') as f:
                    f.write("# Placeholder image file\n")
        
        # Create placeholder timeseries files
        if "timeseries" in record:
            timeseries_path = os.path.join(base_dir, record["timeseries"])
            if not os.path.exists(timeseries_path):
                # Create sample CSV with vital signs
                with open(timeseries_path, 'w') as f:
                    f.write("Hours,Heart_Rate,Blood_Pressure_Systolic,Blood_Pressure_Diastolic,Temperature,Respiratory_Rate,Oxygen_Saturation\n")
                    for hour in range(24):
                        hr = random.randint(60, 100)
                        bp_sys = random.randint(90, 140)
                        bp_dia = random.randint(60, 90)
                        temp = round(random.uniform(36.5, 37.5), 1)
                        rr = random.randint(12, 20)
                        o2 = random.randint(95, 100)
                        f.write(f"{hour},{hr},{bp_sys},{bp_dia},{temp},{rr},{o2}\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate sample JSONL datasets for LANISTR testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Amazon sample dataset
  python generate_sample_data.py --dataset amazon --output-file sample_amazon.jsonl --num-samples 100

  # Generate MIMIC-IV sample dataset
  python generate_sample_data.py --dataset mimic-iv --output-file sample_mimic.jsonl --num-samples 50

  # Generate with custom output directory
  python generate_sample_data.py --dataset amazon --output-file ./data/sample_amazon.jsonl --num-samples 200
        """
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['amazon', 'mimic-iv'],
        help='Dataset type to generate'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to generate (default: 100)'
    )
    parser.add_argument(
        '--create-files',
        action='store_true',
        help='Create placeholder files referenced in the dataset'
    )
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_samples} samples for {args.dataset} dataset...")
    
    # Generate data
    if args.dataset == 'amazon':
        data = generate_amazon_sample(args.num_samples)
    elif args.dataset == 'mimic-iv':
        data = generate_mimic_sample(args.num_samples)
    else:
        print(f"Unsupported dataset: {args.dataset}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Write JSONL file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Generated {len(data)} samples in {args.output_file}")
    
    # Create placeholder files if requested
    if args.create_files:
        print("Creating placeholder files...")
        create_sample_files(data, args.output_file)
        print("✅ Created placeholder files")
    
    # Print sample record
    print("\n📋 Sample record:")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))
    
    # Print validation command
    print(f"\n🔍 To validate this dataset, run:")
    if args.dataset == 'amazon':
        print(f"python validate_dataset.py --dataset amazon --jsonl-file {args.output_file} --data-dir {os.path.dirname(args.output_file) or '.'}")
    else:
        print(f"python validate_dataset.py --dataset mimic-iv --jsonl-file {args.output_file} --data-dir {os.path.dirname(args.output_file) or '.'}")

if __name__ == "__main__":
    main() 