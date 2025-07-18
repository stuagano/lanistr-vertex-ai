"""Vertex AI entry point for LANISTR distributed training.

This module provides the main entry point for running LANISTR on Google Cloud
Vertex AI with proper distributed training setup, monitoring, and error handling.
"""

import argparse
import logging
import os
import pathlib
import random
import sys
import time
import warnings
from typing import Dict, Any

import numpy as np
import omegaconf
import torch
import transformers

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lanistr.dataset.amazon.load_data import load_amazon
from lanistr.dataset.mimic_iv.load_data import load_mimic
from trainer import Trainer
from lanistr.utils.common_utils import how_long, print_config, print_only_by_main_process, set_global_logging_level
from lanistr.utils.data_utils import generate_loaders
from lanistr.utils.model_utils import build_model
from lanistr.utils.parallelism_utils import is_main_process, setup_model
from lanistr.utils.vertex_ai_utils import (
    is_vertex_ai_environment,
    get_vertex_ai_config,
    setup_vertex_ai_distributed_training,
    setup_vertex_ai_logging,
    setup_vertex_ai_monitoring,
    get_system_resources,
    log_resource_usage,
    update_config_for_vertex_ai,
    handle_vertex_ai_error,
    setup_vertex_ai_signal_handlers,
    cleanup_distributed_training,
    save_vertex_ai_checkpoint
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
set_global_logging_level(logging.ERROR, ["transformers"])


def setup_vertex_ai_environment_config() -> Dict[str, Any]:
    """Setup Vertex AI environment configuration."""
    vertex_config = get_vertex_ai_config()
    
    if vertex_config["is_vertex_ai"]:
        logger.info("Running in Vertex AI environment")
        logger.info(f"Job ID: {vertex_config['job_id']}")
        logger.info(f"Node Index: {vertex_config['node_index']}")
        logger.info(f"Number of nodes: {vertex_config['num_nodes']}")
        logger.info(f"Number of GPUs: {vertex_config['num_gpus']}")
        
        # Setup distributed training
        dist_config = setup_vertex_ai_distributed_training()
        
        # Setup logging and monitoring
        setup_vertex_ai_logging(project_id=vertex_config["project_id"])
        metrics = setup_vertex_ai_monitoring(project_id=vertex_config["project_id"])
        
        # Setup signal handlers for graceful shutdown
        setup_vertex_ai_signal_handlers()
        
        return {
            "vertex_config": vertex_config,
            "dist_config": dist_config,
            "metrics": metrics
        }
    else:
        logger.info("Running in local environment")
        return {
            "vertex_config": vertex_config,
            "dist_config": {"distributed": False},
            "metrics": None
        }


def parse_arguments() -> omegaconf.DictConfig:
    """Parse command line arguments and configuration."""
    parser = argparse.ArgumentParser(description="LANISTR Vertex AI Distributed Training")
    parser.add_argument("--config", type=str, default="./configs/mimic_pretrain.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--job_name", type=str, default="lanistr-training")
    parser.add_argument("--project_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    
    flags = parser.parse_args()
    
    # Load configuration
    overrides = omegaconf.OmegaConf.from_cli(flags.overrides)
    config = omegaconf.OmegaConf.load(flags.config)
    args = omegaconf.OmegaConf.merge(config, overrides)
    
    # Update with command line arguments
    args.local_rank = flags.local_rank
    args.job_name = flags.job_name
    args.project_id = flags.project_id or os.environ.get("CLOUD_ML_PROJECT_ID")
    
    # Update output and data directories if provided
    if flags.output_dir:
        args.output_dir = flags.output_dir
    if flags.data_dir:
        args.root_data_dir = flags.data_dir
    
    return args


def setup_distributed_training(args: omegaconf.DictConfig, vertex_env: Dict[str, Any]) -> omegaconf.DictConfig:
    """Setup distributed training configuration."""
    vertex_config = vertex_env["vertex_config"]
    dist_config = vertex_env["dist_config"]
    
    if dist_config["distributed"]:
        # Update args for distributed training
        args.distributed = True
        args.multiprocessing_distributed = True
        args.world_size = vertex_config["num_nodes"] * vertex_config["num_gpus"]
        args.ngpus_per_node = vertex_config["num_gpus"]
        args.device = args.local_rank
        
        # Set CUDA device
        torch.cuda.set_device(args.device)
        
        logger.info(f"Distributed training setup: world_size={args.world_size}, "
                   f"ngpus_per_node={args.ngpus_per_node}, device={args.device}")
    else:
        args.distributed = False
        args.multiprocessing_distributed = False
        args.world_size = 1
        args.ngpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 0
        args.device = 0
    
    return args


def setup_dataset_loader(args: omegaconf.DictConfig):
    """Setup dataset loader based on configuration."""
    if args.dataset_name == "mimic-iv":
        return load_mimic
    elif args.dataset_name == "amazon":
        return load_amazon
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} not implemented")


def setup_seed_and_determinism(args: omegaconf.DictConfig):
    """Setup random seeds and deterministic behavior."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.distributed:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def setup_logging(args: omegaconf.DictConfig, vertex_env: Dict[str, Any]):
    """Setup logging configuration."""
    vertex_config = vertex_env["vertex_config"]
    
    # Create output directory
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup log file
    if not args.distributed or is_main_process():
        log_name = f"{args.task}.log" if not args.experiment_name else f"{args.experiment_name}.log"
        log_file = os.path.join(args.output_dir, log_name)
        
        logging.basicConfig(
            filename=log_file,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
        )
        
        # Also log to console in Vertex AI
        if vertex_config["is_vertex_ai"]:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            console_handler.setFormatter(formatter)
            logging.getLogger().addHandler(console_handler)


def main_worker(args: omegaconf.DictConfig, vertex_env: Dict[str, Any]) -> None:
    """Main worker function for training."""
    try:
        # Setup distributed training
        args = setup_distributed_training(args, vertex_env)
        
        # Setup seed and determinism
        setup_seed_and_determinism(args)
        
        # Setup logging
        setup_logging(args, vertex_env)
        
        # Log configuration
        if is_main_process():
            logger.info("Starting LANISTR training on Vertex AI")
            print_config(args)
        
        # Setup dataset loader
        load_dataset = setup_dataset_loader(args)
        
        # Load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_encoder_name)
        
        # Load dataset
        tic = time.time()
        print_only_by_main_process("Loading datasets...")
        dataset = load_dataset(args, tokenizer)
        how_long(tic)
        
        # Build model and setup parallelism
        model = build_model(args, tabular_data_information=dataset["tabular_data_information"])
        args, model = setup_model(args, model)
        
        # Create trainer and data loaders
        trainer = Trainer(model, args)
        dataloaders = generate_loaders(args, dataset)
        
        # Start training based on task
        if args.task == "pretrain":
            pretrain_start = time.time()
            trainer.pretrain(dataloaders)
            how_long(pretrain_start, f"Pre-training finished after {args.scheduler.num_epochs} epochs")
        
        elif args.task == "finetune":
            if args.do_train:
                train_start = time.time()
                trainer.train(dataloaders)
                how_long(train_start, f"Training finished after {args.scheduler.num_epochs} epochs")
            
            elif args.do_test:
                if is_main_process():
                    test_start = time.time()
                    trainer.test(dataloaders["test"])
                    how_long(test_start, "Testing finished")
        
        else:
            raise ValueError(f"Task {args.task} not implemented")
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        # Handle errors in Vertex AI environment
        context = {
            "job_name": args.job_name,
            "task": args.task,
            "dataset": args.dataset_name,
            "node_rank": args.local_rank if hasattr(args, 'local_rank') else 0
        }
        handle_vertex_ai_error(e, context)
        raise
    finally:
        # Cleanup distributed training
        if vertex_env["dist_config"]["distributed"]:
            cleanup_distributed_training()


def main() -> None:
    """Main entry point for Vertex AI training."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup Vertex AI environment
        vertex_env = setup_vertex_ai_environment_config()
        
        # Update configuration for Vertex AI
        args = omegaconf.OmegaConf.create(update_config_for_vertex_ai(omegaconf.OmegaConf.to_container(args)))
        
        # Run main worker
        main_worker(args, vertex_env)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        cleanup_distributed_training()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        cleanup_distributed_training()
        sys.exit(1)


if __name__ == "__main__":
    main() 