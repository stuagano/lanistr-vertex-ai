#!/usr/bin/env python3
"""
Vertex AI Training Job Monitor

This script provides utilities to monitor Vertex AI training jobs,
download results, and analyze training progress.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from google.cloud import aiplatform
from google.cloud import storage
import pandas as pd


class VertexAIMonitor:
    """Monitor and manage Vertex AI training jobs."""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize the monitor."""
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
        
    def list_jobs(self, filter_expr: str = None) -> List[Dict]:
        """List all custom training jobs."""
        jobs = []
        
        for job in aiplatform.CustomJob.list():
            if filter_expr and filter_expr not in job.display_name:
                continue
            jobs.append({
                'name': job.name,
                'display_name': job.display_name,
                'state': job.state.name,
                'create_time': job.create_time,
                'start_time': job.start_time,
                'end_time': job.end_time,
                'error': job.error.message if job.error else None
            })
        
        return jobs
    
    def get_job_details(self, job_name: str) -> Dict:
        """Get detailed information about a specific job."""
        job = aiplatform.CustomJob(job_name=job_name)
        return {
            'name': job.name,
            'display_name': job.display_name,
            'state': job.state.name,
            'create_time': job.create_time,
            'start_time': job.start_time,
            'end_time': job.end_time,
            'error': job.error.message if job.error else None,
            'worker_pool_specs': job.worker_pool_specs,
            'base_output_directory': job.base_output_directory
        }
    
    def wait_for_job_completion(self, job_name: str, timeout_hours: int = 24) -> str:
        """Wait for a job to complete and return the final state."""
        job = aiplatform.CustomJob(job_name=job_name)
        start_time = time.time()
        timeout_seconds = timeout_hours * 3600
        
        print(f"Waiting for job {job.display_name} to complete...")
        
        while time.time() - start_time < timeout_seconds:
            job = aiplatform.CustomJob(job_name=job_name)
            state = job.state.name
            
            print(f"Current state: {state} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if state in ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']:
                return state
            
            time.sleep(60)  # Check every minute
        
        raise TimeoutError(f"Job did not complete within {timeout_hours} hours")
    
    def download_results(self, job_name: str, local_dir: str = "./results") -> str:
        """Download training results from GCS."""
        job = aiplatform.CustomJob(job_name=job_name)
        
        if not job.base_output_directory:
            raise ValueError("Job has no output directory")
        
        gcs_path = job.base_output_directory
        local_path = Path(local_dir) / job.display_name
        
        print(f"Downloading results from {gcs_path} to {local_path}")
        
        # Download using gsutil
        os.system(f"gsutil -m cp -r {gcs_path}/* {local_path}/")
        
        return str(local_path)
    
    def analyze_training_logs(self, results_dir: str) -> Dict:
        """Analyze training logs and extract metrics."""
        results_path = Path(results_dir)
        log_files = list(results_path.rglob("*.log"))
        
        if not log_files:
            return {"error": "No log files found"}
        
        # Find the main training log
        main_log = None
        for log_file in log_files:
            if "pretrain.log" in log_file.name or "finetune.log" in log_file.name:
                main_log = log_file
                break
        
        if not main_log:
            main_log = log_files[0]  # Use first log file if no main log found
        
        # Parse log file for metrics
        metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        with open(main_log, 'r') as f:
            for line in f:
                # Extract metrics from log lines
                if 'epoch' in line.lower() and 'loss' in line.lower():
                    # Parse epoch and loss information
                    pass  # Add parsing logic based on your log format
        
        return metrics
    
    def create_job_summary(self, job_name: str) -> Dict:
        """Create a comprehensive summary of a training job."""
        details = self.get_job_details(job_name)
        
        summary = {
            'job_info': details,
            'duration': None,
            'cost_estimate': None
        }
        
        if details['start_time'] and details['end_time']:
            duration = details['end_time'] - details['start_time']
            summary['duration'] = str(duration)
            
            # Rough cost estimation (this would need to be refined)
            hours = duration.total_seconds() / 3600
            summary['cost_estimate'] = f"~${hours * 50:.2f}"  # Rough estimate
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Monitor Vertex AI training jobs")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument("--action", choices=["list", "details", "wait", "download", "summary"], 
                       required=True, help="Action to perform")
    parser.add_argument("--job-name", help="Job name for specific actions")
    parser.add_argument("--filter", help="Filter expression for listing jobs")
    parser.add_argument("--local-dir", default="./results", help="Local directory for downloads")
    parser.add_argument("--timeout-hours", type=int, default=24, help="Timeout for waiting")
    
    args = parser.parse_args()
    
    monitor = VertexAIMonitor(args.project_id, args.location)
    
    if args.action == "list":
        jobs = monitor.list_jobs(args.filter)
        print(json.dumps(jobs, indent=2, default=str))
    
    elif args.action == "details":
        if not args.job_name:
            parser.error("--job-name is required for details action")
        details = monitor.get_job_details(args.job_name)
        print(json.dumps(details, indent=2, default=str))
    
    elif args.action == "wait":
        if not args.job_name:
            parser.error("--job-name is required for wait action")
        final_state = monitor.wait_for_job_completion(args.job_name, args.timeout_hours)
        print(f"Job completed with state: {final_state}")
    
    elif args.action == "download":
        if not args.job_name:
            parser.error("--job-name is required for download action")
        local_path = monitor.download_results(args.job_name, args.local_dir)
        print(f"Results downloaded to: {local_path}")
    
    elif args.action == "summary":
        if not args.job_name:
            parser.error("--job-name is required for summary action")
        summary = monitor.create_job_summary(args.job_name)
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main() 