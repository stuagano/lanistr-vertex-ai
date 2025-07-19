"""Copyright 2024 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import collections
import datetime
import logging
import os
import re
import shutil
import time
from typing import Dict, Any, Optional, Union

import omegaconf
import pytz
import torch
import torchmetrics
from lanistr.utils.parallelism_utils import is_main_process


logger = logging.getLogger(__name__)


class TimeUtils:
    """Utility class for time-related operations."""
    
    @staticmethod
    def pretty_print(num: Union[int, float]) -> str:
        """Format number with magnitude suffixes."""
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])

    @staticmethod
    def print_time(tstart: float) -> None:
        """Print the elapsed time since start time and current time."""
        print("*" * 100)
        print("All Done! ")
        print("[Elapsed time = {:.1f} min]".format((time.time() - tstart) / (60)))

        # dd/mm/YY H:M:S
        fmt = "%d/%m/%Y %H:%M:%S"

        # Current time in UTC
        now_utc = datetime.datetime.now(pytz.timezone("UTC"))

        # Convert to US/Pacific time zone
        now_pacific = now_utc.astimezone(pytz.timezone("US/Pacific"))

        print("Job finished at =", now_pacific.strftime(fmt))
        logger.info("Job finished at %s", now_pacific.strftime(fmt))

    @staticmethod
    def how_long(tstart: float, what: str = "load data") -> None:
        """Print how long an operation took."""
        if is_main_process():
            print("*" * 100)
            print(
                " >>>>>> Elapsed time to {} = {:.1f} min ({:.1f} seconds)".format(
                    what, (time.time() - tstart) / (60), (time.time() - tstart)
                )
            )
            print("*" * 100)
            logger.info(
                " >>>>>> Elapsed time to {} = {:.1f} min".format(
                    what, (time.time() - tstart) / (60)
                )
            )


class MetricsManager:
    """Manages metrics for training and evaluation."""
    
    @staticmethod
    def get_metrics(args: omegaconf.DictConfig) -> tuple:
        """Get metrics for pretraining and finetuning.

        Args:
            args: config file

        Returns:
            metrics: dictionary of metrics
            metric_names: list of metric names
        """
        metrics = {"train": {}, "test": {}}

        # we always have loss in the metrics during both pretraining and fine tuning
        metric_names = ["Loss"]

        if args.task == "pretrain":
            metrics["train"]["Loss"] = torchmetrics.aggregation.MeanMetric().to(
                args.device
            )
            loss_names = ["MLM", "MIM", "MTM", "MFM", "MMM"]
            for loss_name in loss_names:
                metrics["train"][loss_name] = torchmetrics.aggregation.MeanMetric().to(
                    args.device
                )
            metric_names += loss_names

        elif args.task == "finetune":
            for phase in ["train", "test"]:
                metrics[phase]["Loss"] = torchmetrics.aggregation.MeanMetric().to(
                    args.device
                )

            if args.dataset_name.startswith("amazon"):
                metric_names.append("ACCURACY")
                for phase in ["train", "test"]:
                    metrics[phase]["ACCURACY"] = torchmetrics.Accuracy(
                        task="multiclass", num_classes=args.num_classes
                    ).to(args.device)

            elif args.dataset_name.startswith("mimic"):
                metric_names.append("AUROC")
                for phase in ["test", "train"]:
                    metrics[phase]["AUROC"] = torchmetrics.AUROC(
                        num_classes=args.num_classes, task="binary"
                    ).to(args.device)

        return metrics, metric_names


class ModelUtils:
    """Utility class for model-related operations."""
    
    @staticmethod
    def print_model_size(model: torch.nn.Module, model_name: str) -> None:
        """Print the number of parameters in the model.

        Args:
            model: model
            model_name: name of the model
        """
        if is_main_process():
            total_params = sum([p.numel() for p in model.parameters()])
            trainable_params = sum(
                [p.numel() for p in model.parameters() if p.requires_grad]
            )
            print(f"*********** {model_name} *********************")
            print(
                f"Number of total parameters in the model: {TimeUtils.pretty_print(total_params)}"
            )
            print(
                "Number of trainable parameters in the model:"
                f" {TimeUtils.pretty_print(trainable_params)}"
            )
            print("**************************************************")
            logger.info("*********** %s *********************", model_name)
            logger.info(
                "Number of total parameters in the model: %s",
                TimeUtils.pretty_print(total_params),
            )
            logger.info(
                "Number of trainable parameters in the model: %s",
                TimeUtils.pretty_print(trainable_params),
            )
            logger.info("**************************************************")


class PrintUtils:
    """Utility class for printing operations."""
    
    @staticmethod
    def print_only_by_main_process(to_print: str) -> None:
        """Print only if this is the main process."""
        if is_main_process():
            print("*" * 100)
            print(f"{to_print}")

    @staticmethod
    def print_df_stats(df, name: str) -> None:
        """Print dataframe statistics."""
        if is_main_process():
            print(f"{name} shape: {df.shape}")
            print(f"{name} columsn: {df.columns}")
            print("***" * 20)


class PerformancePrinter:
    """Handles performance printing for training and validation."""
    
    def __init__(self):
        self.train_template = "Epoch {}/{}, Train: "
        self.pretrain_template = "PreTraining Epoch {}/{}, Train: "
    
    def _format_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert tensor values to floats for consistent formatting."""
        formatted = {}
        for k, v in results.items():
            if torch.is_tensor(v):
                formatted[k] = v.item()
            else:
                formatted[k] = v
        return formatted
    
    def _print_metrics(self, prefix: str, results: Dict[str, Any], m_t: 'MetricsLogger', 
                      metric_prefix: str = "") -> None:
        """Print metrics with consistent formatting."""
        print(f"{prefix}: ", end="")
        for k, v in results.items():
            formatted_v = v.item() if torch.is_tensor(v) else v
            print(f"{k}: {formatted_v:.4f}", end="  ")
            m_t.update(f"{metric_prefix}{k}", formatted_v)
    
    def print_performance(self, epoch: int, num_epochs: int, m_t: 'MetricsLogger',
                         train_results: Dict[str, Any], valid_results: Dict[str, Any],
                         is_best: bool, best_perf: Union[float, torch.Tensor], 
                         metric_name: str, is_pretrain: bool = False) -> 'MetricsLogger':
        """Unified performance printing for both pretraining and finetuning.
        
        Args:
            epoch: current epoch
            num_epochs: total number of epochs
            m_t: metrics logger
            train_results: dictionary of training results
            valid_results: dictionary of validation results
            is_best: whether the current model is the best model
            best_perf: best performance of the model
            metric_name: name of the metric
            is_pretrain: whether this is pretraining
            
        Returns:
            m_t: metrics logger
        """
        if torch.is_tensor(best_perf):
            best_perf = best_perf.item()
            
        if not is_main_process():
            return m_t
        
        # Format results
        train_results = self._format_results(train_results)
        valid_results = self._format_results(valid_results)
        
        # Choose template
        template = self.pretrain_template if is_pretrain else self.train_template
        print(template.format(epoch + 1, num_epochs), end="  ")
        
        # Print training metrics
        self._print_metrics("Train", train_results, m_t, 
                           "" if is_pretrain else "train_")
        
        # Print validation metrics
        self._print_metrics("Valid", valid_results, m_t, 
                           "valid_" if not is_pretrain else "")
        
        print(f"    | Best {metric_name} :{best_perf:.4f}", end="  ")
        
        if is_best:
            print(" **")
            m_t.update("best_epoch", epoch)
            m_t.update("best_perf", best_perf)
        else:
            print()
        
        # Log detailed information for pretraining
        if is_pretrain:
            logger.info(
                "PreTraining Epoch %d/%d || Train Loss: %.6f, T-MLM: %.3f"
                " T-MIM: %.3f T-MTM: %.3f T-MFM: %.3f T-MMM: %.3f || Valid"
                " Loss: %.6f V-MLM: %.3f V-MIM: %.3f V-MTM: %.3f V-MFM: %.3f"
                " V-MMM: %.3f",
                epoch + 1, num_epochs,
                train_results["Loss"], train_results["MLM"], train_results["MIM"],
                train_results["MTM"], train_results["MFM"], train_results["MMM"],
                valid_results["Loss"], valid_results["MLM"], valid_results["MIM"],
                valid_results["MTM"], valid_results["MFM"], valid_results["MMM"],
            )
        
        return m_t


class CheckpointManager:
    """Manages model and optimizer checkpoints."""
    
    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        is_best: bool,
        file_dir: str,
        filename: str,
        best_filename: str = "model_best.pth.tar",
        **kwargs
    ) -> None:
        """Save model checkpoint.
        
        Args:
            model: model to save
            is_best: whether this is the best model
            file_dir: directory to save files
            filename: checkpoint filename
            best_filename: best model filename
            **kwargs: additional data to save
        """
        filename = os.path.join(file_dir, filename)
        checkpoint_data = {"state_dict": model.state_dict()}
        checkpoint_data.update(kwargs)
        
        torch.save(checkpoint_data, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(file_dir, best_filename))

    @staticmethod
    def save_checkpoint_optimizer(
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        is_best: bool,
        file_dir: str,
        filename: str,
        best_filename: str = "optimizer_best.pth.tar",
    ) -> None:
        """Save optimizer and scheduler checkpoint.
        
        Args:
            epoch: current epoch
            optimizer: optimizer
            scheduler: scheduler
            is_best: whether the current model is the best model
            file_dir: directory to save the model
            filename: name of the file to save the model
            best_filename: name of the file to save the best model
        """
        filename = os.path.join(file_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            filename,
        )
        if is_best:
            shutil.copyfile(filename, os.path.join(file_dir, best_filename))

    @staticmethod
    def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                       device: str) -> torch.nn.Module:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: path to checkpoint
            model: model to load state into
            device: device to load on
            
        Returns:
            model with loaded state
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @staticmethod
    def load_checkpoint_with_module(checkpoint_path: str, model: torch.nn.Module, 
                                   device: str) -> torch.nn.Module:
        """Load checkpoint with module prefix handling.
        
        Args:
            checkpoint_path: path to checkpoint
            model: model to load state into
            device: device to load on
            
        Returns:
            model with loaded state
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        return model


class MetricsLogger:
    """A class for logging and storing metrics during training."""
    
    def __init__(self):
        self.metrics = collections.defaultdict(list)
    
    def update(self, key: str, value: Union[float, torch.Tensor]) -> None:
        """Update metric with new value."""
        if torch.is_tensor(value):
            value = value.item()
        self.metrics[key].append(value)
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for a metric."""
        return self.metrics[key][-1] if self.metrics[key] else None
    
    def get_all(self, key: str) -> list:
        """Get all values for a metric."""
        return self.metrics[key]
    
    def reset(self, key: str = None) -> None:
        """Reset metrics (all if key is None)."""
        if key is None:
            self.metrics.clear()
        else:
            self.metrics[key].clear()


class ConfigUtils:
    """Utility class for configuration operations."""
    
    @staticmethod
    def print_config(config: omegaconf.DictConfig) -> None:
        """Print configuration parameters."""
        if is_main_process():
            print("*" * 100)
            print("Training parameters: ")
            logger.info("Training parameters: ")
            print("*" * 100)
            
            for k, v in config.items():
                print(f"{k} --> {v}")
                logger.info("%s --> %s", k, v)

    @staticmethod
    def set_global_logging_level(level: int, package_names: list) -> None:
        """Set global logging level for specified packages."""
        for package_name in package_names:
            logging.getLogger(package_name).setLevel(level)


# Backward compatibility - maintain old function names
def pretty_print(num): return TimeUtils.pretty_print(num)
def print_time(tstart): return TimeUtils.print_time(tstart)
def get_metrics(args): return MetricsManager.get_metrics(args)
def print_model_size(model, model_name): return ModelUtils.print_model_size(model, model_name)
def how_long(tstart, what="load data"): return TimeUtils.how_long(tstart, what)
def print_only_by_main_process(to_print): return PrintUtils.print_only_by_main_process(to_print)
def print_df_stats(df, name): return PrintUtils.print_df_stats(df, name)

# Create global instances for easy access
performance_printer = PerformancePrinter()
checkpoint_manager = CheckpointManager()

def print_pretrain_performance_by_main_process(epoch, num_epochs, m_t, train_results, 
                                             valid_results, is_best, best_perf, metric_name):
    return performance_printer.print_performance(epoch, num_epochs, m_t, train_results, 
                                               valid_results, is_best, best_perf, 
                                               metric_name, is_pretrain=True)

def print_performance_by_main_process(epoch, num_epochs, m_t, train_results, 
                                    valid_results, is_best, best_perf, metric_name):
    return performance_printer.print_performance(epoch, num_epochs, m_t, train_results, 
                                               valid_results, is_best, best_perf, 
                                               metric_name, is_pretrain=False)

def save_checkpoint(model, is_best, file_dir, filename, best_filename="model_best.pth.tar", **kwargs):
    return checkpoint_manager.save_checkpoint(model, is_best, file_dir, filename, best_filename, **kwargs)

def save_checkpoint_optimizer(epoch, optimizer, scheduler, is_best, file_dir, filename, best_filename="optimizer_best.pth.tar"):
    return checkpoint_manager.save_checkpoint_optimizer(epoch, optimizer, scheduler, is_best, file_dir, filename, best_filename)

def load_checkpoint(checkpoint_path, model, device):
    return checkpoint_manager.load_checkpoint(checkpoint_path, model, device)

def load_checkpoint_with_module(checkpoint_path, model, device):
    return checkpoint_manager.load_checkpoint_with_module(checkpoint_path, model, device)

def print_config(config):
    return ConfigUtils.print_config(config)

def set_global_logging_level(level, package_names):
    return ConfigUtils.set_global_logging_level(level, package_names)
