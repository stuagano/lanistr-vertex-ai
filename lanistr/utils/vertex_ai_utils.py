"""Vertex AI utilities for LANISTR distributed training.

This module provides utilities for running LANISTR on Google Cloud Vertex AI
with proper distributed training setup, environment detection, and monitoring.
"""

import os
import logging
import structlog
from typing import Dict, Any, Optional
import torch
import torch.distributed as dist
from google.cloud import logging as cloud_logging
from google.cloud import monitoring_v3
from prometheus_client import Counter, Histogram, start_http_server
import psutil
import GPUtil

logger = logging.getLogger(__name__)

# =============================================================================
# VERTEX AI ENVIRONMENT DETECTION
# =============================================================================

def is_vertex_ai_environment() -> bool:
    """Check if running in Vertex AI environment."""
    return (
        os.environ.get("CLOUD_ML_JOB_ID") is not None or
        os.environ.get("CLOUD_ML_PROJECT_ID") is not None or
        os.environ.get("KUBERNETES_SERVICE_HOST") is not None
    )

def get_vertex_ai_config() -> Dict[str, Any]:
    """Get Vertex AI specific configuration."""
    config = {
        "is_vertex_ai": is_vertex_ai_environment(),
        "job_id": os.environ.get("CLOUD_ML_JOB_ID"),
        "project_id": os.environ.get("CLOUD_ML_PROJECT_ID"),
        "region": os.environ.get("CLOUD_ML_REGION", "us-central1"),
        "node_index": int(os.environ.get("CLOUD_ML_NODE_INDEX", "0")),
        "master_addr": os.environ.get("CLOUD_ML_MASTER_ADDR"),
        "master_port": int(os.environ.get("CLOUD_ML_MASTER_PORT", "12355")),
        "num_nodes": int(os.environ.get("CLOUD_ML_NUM_NODES", "1")),
        "num_gpus": int(os.environ.get("CLOUD_ML_NUM_GPUS", "8")),
    }
    return config

# =============================================================================
# DISTRIBUTED TRAINING SETUP
# =============================================================================

def setup_vertex_ai_distributed_training(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> Dict[str, Any]:
    """Setup distributed training for Vertex AI environment.
    
    Args:
        backend: Distributed backend (nccl for GPU, gloo for CPU)
        init_method: Initialization method (env:// for environment variables)
        world_size: Total number of processes
        rank: Rank of current process
        
    Returns:
        Dictionary with distributed training configuration
    """
    config = get_vertex_ai_config()
    
    if not config["is_vertex_ai"]:
        logger.warning("Not running in Vertex AI environment, skipping distributed setup")
        return {"distributed": False, "config": config}
    
    # Set environment variables for torch.distributed
    if init_method is None:
        init_method = "env://"
    
    if world_size is None:
        world_size = config["num_nodes"] * config["num_gpus"]
    
    if rank is None:
        rank = config["node_index"] * config["num_gpus"]
    
    os.environ["MASTER_ADDR"] = config["master_addr"]
    os.environ["MASTER_PORT"] = str(config["master_port"])
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    
    # Initialize distributed training
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )
        logger.info(f"Initialized distributed training: rank={rank}, world_size={world_size}")
    
    return {
        "distributed": True,
        "config": config,
        "rank": rank,
        "world_size": world_size,
        "backend": backend
    }

def cleanup_distributed_training():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed training")

# =============================================================================
# MONITORING AND LOGGING
# =============================================================================

def setup_vertex_ai_logging(
    project_id: Optional[str] = None,
    log_name: str = "lanistr-training"
) -> None:
    """Setup structured logging for Vertex AI.
    
    Args:
        project_id: Google Cloud project ID
        log_name: Name of the log
    """
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup Google Cloud Logging if in Vertex AI environment
    if is_vertex_ai_environment() and project_id:
        try:
            client = cloud_logging.Client(project=project_id)
            client.setup_logging()
            logger.info("Google Cloud Logging configured")
        except Exception as e:
            logger.warning(f"Failed to setup Google Cloud Logging: {e}")

def setup_vertex_ai_monitoring(
    project_id: Optional[str] = None,
    port: int = 8000
) -> Dict[str, Any]:
    """Setup monitoring for Vertex AI training.
    
    Args:
        project_id: Google Cloud project ID
        port: Port for Prometheus metrics server
        
    Returns:
        Dictionary with monitoring metrics
    """
    # Start Prometheus metrics server
    start_http_server(port)
    
    # Define metrics
    metrics = {
        "training_steps": Counter(
            "lanistr_training_steps_total",
            "Total training steps",
            ["model", "dataset"]
        ),
        "training_time": Histogram(
            "lanistr_training_duration_seconds",
            "Training duration",
            ["model", "dataset"]
        ),
        "loss": Histogram(
            "lanistr_loss",
            "Training loss",
            ["model", "dataset", "split"]
        ),
        "gpu_memory": Histogram(
            "lanistr_gpu_memory_usage_bytes",
            "GPU memory usage",
            ["gpu_id"]
        ),
        "cpu_memory": Histogram(
            "lanistr_cpu_memory_usage_bytes",
            "CPU memory usage"
        ),
    }
    
    logger.info(f"Monitoring setup complete on port {port}")
    return metrics

# =============================================================================
# RESOURCE MONITORING
# =============================================================================

def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage."""
    resources = {
        "cpu": {
            "percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        },
        "gpu": {}
    }
    
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            resources["gpu"][f"gpu_{i}"] = {
                "memory_used_percent": gpu.memoryUtil * 100,
                "memory_used_mb": gpu.memoryUsed,
                "memory_total_mb": gpu.memoryTotal,
                "temperature": gpu.temperature,
                "load": gpu.load * 100 if gpu.load else 0,
            }
    except Exception as e:
        logger.warning(f"Failed to get GPU information: {e}")
    
    return resources

def log_resource_usage(metrics: Dict[str, Any], resources: Dict[str, Any]):
    """Log resource usage to monitoring system."""
    # Log CPU memory
    metrics["cpu_memory"].observe(resources["cpu"]["memory_available_gb"] * (1024**3))
    
    # Log GPU memory
    for gpu_id, gpu_info in resources["gpu"].items():
        metrics["gpu_memory"].labels(gpu_id=gpu_id).observe(
            gpu_info["memory_used_mb"] * (1024**2)
        )

# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def get_vertex_ai_checkpoint_path(
    output_dir: str,
    job_name: str,
    checkpoint_name: str
) -> str:
    """Get checkpoint path for Vertex AI environment.
    
    Args:
        output_dir: Base output directory
        job_name: Name of the training job
        checkpoint_name: Name of the checkpoint file
        
    Returns:
        Full path to checkpoint file
    """
    config = get_vertex_ai_config()
    
    if config["is_vertex_ai"]:
        # Use GCS path for Vertex AI
        return f"{output_dir}/{job_name}/checkpoints/{checkpoint_name}"
    else:
        # Use local path for non-Vertex AI
        return os.path.join(output_dir, "checkpoints", checkpoint_name)

def save_vertex_ai_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    output_dir: str,
    job_name: str,
    is_best: bool = False,
    **kwargs
) -> None:
    """Save checkpoint for Vertex AI training.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        output_dir: Output directory
        job_name: Job name
        is_best: Whether this is the best model so far
        **kwargs: Additional checkpoint data
    """
    checkpoint_path = get_vertex_ai_checkpoint_path(
        output_dir, job_name, f"checkpoint_epoch_{epoch}.pth"
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "is_best": is_best,
        **kwargs
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = get_vertex_ai_checkpoint_path(
            output_dir, job_name, "best_model.pth"
        )
        torch.save(checkpoint, best_path)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")

# =============================================================================
# CONFIGURATION HELPERS
# =============================================================================

def update_config_for_vertex_ai(config: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration for Vertex AI environment.
    
    Args:
        config: Original configuration
        
    Returns:
        Updated configuration
    """
    vertex_config = get_vertex_ai_config()
    
    if vertex_config["is_vertex_ai"]:
        # Update distributed training settings
        config.update({
            "distributed": True,
            "multiprocessing_distributed": True,
            "world_size": vertex_config["num_nodes"] * vertex_config["num_gpus"],
            "ngpus_per_node": vertex_config["num_gpus"],
            "backend": "nccl",
        })
        
        # Update batch sizes for distributed training
        if "train_batch_size" in config:
            config["train_batch_size"] = config["train_batch_size"] // vertex_config["num_gpus"]
        if "eval_batch_size" in config:
            config["eval_batch_size"] = config["eval_batch_size"] // vertex_config["num_gpus"]
        if "test_batch_size" in config:
            config["test_batch_size"] = config["test_batch_size"] // vertex_config["num_gpus"]
    
    return config

# =============================================================================
# ERROR HANDLING AND RECOVERY
# =============================================================================

def handle_vertex_ai_error(error: Exception, context: Dict[str, Any]) -> None:
    """Handle errors in Vertex AI environment.
    
    Args:
        error: The exception that occurred
        context: Context information about the error
    """
    logger.error(f"Vertex AI training error: {error}", extra=context)
    
    # Log to Google Cloud Error Reporting if available
    try:
        from google.cloud import error_reporting
        client = error_reporting.Client()
        client.report_exception(error)
    except Exception as e:
        logger.warning(f"Failed to report error to Google Cloud: {e}")
    
    # Cleanup distributed training
    cleanup_distributed_training()

def setup_vertex_ai_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    import signal
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_distributed_training()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) 