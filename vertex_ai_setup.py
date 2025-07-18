#!/usr/bin/env python3
"""
Vertex AI Distributed Training Setup for LANISTR

This script sets up distributed training on Google Cloud Vertex AI for the LANISTR
multimodal learning framework.
"""

import argparse
import os
import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform import CustomJob
from google.cloud.aiplatform import WorkerPoolSpec
from google.cloud.aiplatform import ContainerSpec
from google.cloud.aiplatform import MachineSpec
from google.cloud.aiplatform import AcceleratorSpec


def create_vertex_ai_job(
    project_id: str,
    location: str,
    job_name: str,
    config_file: str,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "NVIDIA_TESLA_V100",
    accelerator_count: int = 8,
    replica_count: int = 1,
    base_output_dir: str = "gs://your-bucket/lanistr-output",
    base_data_dir: str = "gs://your-bucket/lanistr-data",
    image_uri: str = "gcr.io/your-project/lanistr-training:latest"
):
    """
    Create a Vertex AI CustomJob for distributed training.
    
    Args:
        project_id: Google Cloud project ID
        location: GCP region (e.g., 'us-central1')
        job_name: Name for the training job
        config_file: Path to the YAML config file
        machine_type: GCP machine type
        accelerator_type: GPU accelerator type
        accelerator_count: Number of GPUs per machine
        replica_count: Number of machines (for multi-node training)
        base_output_dir: GCS bucket for outputs
        base_data_dir: GCS bucket for data
        image_uri: Container image URI
    """
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Create worker pool specification
    worker_pool_specs = [
        WorkerPoolSpec(
            machine_spec=MachineSpec(
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
            ),
            replica_count=replica_count,
            container_spec=ContainerSpec(
                image_uri=image_uri,
                args=[
                    "torchrun",
                    "--nproc_per_node", str(accelerator_count),
                    "--nnodes", str(replica_count),
                    "--node_rank", "$CLOUD_ML_NODE_INDEX",
                    "--master_addr", "$CLOUD_ML_MASTER_ADDR",
                    "--master_port", "12355",
                    "lanistr/main_vertex_ai.py",
                    "--config", config_file,
                    "--job_name", job_name,
                    "--project_id", project_id,
                    f"root_data_dir={base_data_dir}/MIMIC-IV-V2.2/physionet.org/files",
                    f"image_data_dir={base_data_dir}/MIMIC-IV-V2.2/physionet.org/files/mimic-cxr-jpg/2.0.0",
                    f"task_data_dir={base_data_dir}/MIMIC-IV-V2.2/in-hospital-mortality",
                    f"unimodal_data_dir={base_data_dir}/MIMIC-IV-V2.2/in-hospital-mortality/unimodal_data",
                    f"preprocessed_data_dir={base_data_dir}/MIMIC-IV-V2.2/",
                    f"normalizer_file={base_data_dir}/MIMIC-IV-V2.2/normalizer.csv",
                    f"discretizer_config_path={base_data_dir}/MIMIC-IV-V2.2/discretizer_config.json",
                    f"output_dir={base_output_dir}/{job_name}",
                ],
                env=[
                    {"name": "PYTHONPATH", "value": "/workspace"},
                    {"name": "CUDA_VISIBLE_DEVICES", "value": "0,1,2,3,4,5,6,7"},
                ]
            ),
        )
    ]
    
    # Create the custom job
    job = CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=base_output_dir,
    )
    
    # Submit the job
    job.run()
    
    print(f"Job '{job_name}' submitted successfully!")
    print(f"Job ID: {job.name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs/{job.name}")


def main():
    parser = argparse.ArgumentParser(description="Setup Vertex AI distributed training for LANISTR")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument("--job-name", required=True, help="Name for the training job")
    parser.add_argument("--config-file", required=True, help="Path to YAML config file")
    parser.add_argument("--machine-type", default="n1-standard-4", help="GCP machine type")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_V100", help="GPU accelerator type")
    parser.add_argument("--accelerator-count", type=int, default=8, help="Number of GPUs per machine")
    parser.add_argument("--replica-count", type=int, default=1, help="Number of machines")
    parser.add_argument("--base-output-dir", required=True, help="GCS bucket for outputs")
    parser.add_argument("--base-data-dir", required=True, help="GCS bucket for data")
    parser.add_argument("--image-uri", required=True, help="Container image URI")
    
    args = parser.parse_args()
    
    create_vertex_ai_job(
        project_id=args.project_id,
        location=args.location,
        job_name=args.job_name,
        config_file=args.config_file,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        replica_count=args.replica_count,
        base_output_dir=args.base_output_dir,
        base_data_dir=args.base_data_dir,
        image_uri=args.image_uri,
    )


if __name__ == "__main__":
    main() 