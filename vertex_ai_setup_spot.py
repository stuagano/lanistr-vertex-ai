#!/usr/bin/env python3
"""
Vertex AI Distributed Training Setup with Spot Instances

This script sets up distributed training on Google Cloud Vertex AI with spot instances
for cost optimization.
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


def create_vertex_ai_spot_job(
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
    image_uri: str = "gcr.io/your-project/lanistr-training:latest",
    enable_spot: bool = True,
    max_retry_count: int = 3
):
    """
    Create a Vertex AI CustomJob with spot instances for cost optimization.
    
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
        enable_spot: Whether to use spot instances
        max_retry_count: Maximum number of retries for spot instances
    """
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Create worker pool specification with spot instances
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
                    "main.py",
                    "--config", config_file,
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
    
    # Create the custom job with spot instance configuration
    job = CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=base_output_dir,
        enable_web_access=True,  # Enable TensorBoard access
        enable_spot=enable_spot,
        max_retry_count=max_retry_count if enable_spot else 0
    )
    
    # Submit the job
    job.run()
    
    cost_savings = "60-80%" if enable_spot else "0%"
    print(f"Job '{job_name}' submitted successfully!")
    print(f"Job ID: {job.name}")
    print(f"Spot instances enabled: {enable_spot}")
    print(f"Estimated cost savings: {cost_savings}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs/{job.name}")


def estimate_costs(
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    replica_count: int,
    hours: int,
    enable_spot: bool = False
) -> dict:
    """
    Estimate training costs.
    
    Note: These are rough estimates. Actual costs may vary.
    """
    
    # Rough hourly rates (USD) - these are approximate
    machine_costs = {
        "n1-standard-4": 0.19,
        "n1-standard-8": 0.38,
        "n1-standard-16": 0.76,
        "n1-standard-32": 1.52,
    }
    
    accelerator_costs = {
        "NVIDIA_TESLA_V100": 2.48,
        "NVIDIA_TESLA_P100": 1.46,
        "NVIDIA_TESLA_K80": 0.45,
        "NVIDIA_TESLA_T4": 0.35,
        "NVIDIA_A100": 3.67,
    }
    
    machine_cost = machine_costs.get(machine_type, 0.19)
    accelerator_cost = accelerator_costs.get(accelerator_type, 2.48)
    
    total_hourly = (machine_cost + accelerator_cost * accelerator_count) * replica_count
    total_cost = total_hourly * hours
    
    if enable_spot:
        spot_cost = total_cost * 0.3  # ~70% savings
        return {
            "on_demand_cost": f"${total_cost:.2f}",
            "spot_cost": f"${spot_cost:.2f}",
            "savings": f"${total_cost - spot_cost:.2f} ({(1 - 0.3) * 100:.0f}%)"
        }
    else:
        return {
            "total_cost": f"${total_cost:.2f}",
            "hourly_rate": f"${total_hourly:.2f}"
        }


def main():
    parser = argparse.ArgumentParser(description="Setup Vertex AI distributed training with spot instances")
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
    parser.add_argument("--enable-spot", action="store_true", help="Enable spot instances for cost savings")
    parser.add_argument("--max-retry-count", type=int, default=3, help="Maximum retry count for spot instances")
    parser.add_argument("--estimate-costs", action="store_true", help="Estimate training costs")
    parser.add_argument("--hours", type=int, default=24, help="Estimated training hours for cost estimation")
    
    args = parser.parse_args()
    
    # Estimate costs if requested
    if args.estimate_costs:
        costs = estimate_costs(
            args.machine_type,
            args.accelerator_type,
            args.accelerator_count,
            args.replica_count,
            args.hours,
            args.enable_spot
        )
        print("Cost Estimation:")
        for key, value in costs.items():
            print(f"  {key}: {value}")
        print()
    
    # Create the job
    create_vertex_ai_spot_job(
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
        enable_spot=args.enable_spot,
        max_retry_count=args.max_retry_count
    )


if __name__ == "__main__":
    main() 