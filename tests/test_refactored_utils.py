"""
Comprehensive test suite for refactored LANISTR utility classes.

This test module validates the functionality of the refactored utility classes
and ensures that the DRY principle and class method conversions work correctly.
"""

import pytest
import torch
import numpy as np
import omegaconf
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import collections

# Import the refactored classes
from lanistr.utils.common_utils import (
    TimeUtils, MetricsManager, ModelUtils, PrintUtils, PerformancePrinter,
    CheckpointManager, MetricsLogger, ConfigUtils
)
from lanistr.utils.data_utils import DataLoaderManager, ImageTransformManager, MaskGenerator


class TestTimeUtils:
    """Test suite for TimeUtils class."""
    
    def test_pretty_print_formatting(self):
        """Test number formatting with magnitude suffixes."""
        assert TimeUtils.pretty_print(999) == "999.00"
        assert TimeUtils.pretty_print(1000) == "1.00K"
        assert TimeUtils.pretty_print(1500) == "1.50K"
        assert TimeUtils.pretty_print(1000000) == "1.00M"
        assert TimeUtils.pretty_print(1500000000) == "1.50G"
    
    def test_pretty_print_negative_numbers(self):
        """Test pretty print with negative numbers."""
        assert TimeUtils.pretty_print(-1500) == "-1.50K"
        assert TimeUtils.pretty_print(-1000000) == "-1.00M"
    
    def test_pretty_print_zero(self):
        """Test pretty print with zero."""
        assert TimeUtils.pretty_print(0) == "0.00"
    
    @patch('time.time')
    @patch('builtins.print')
    def test_print_time(self, mock_print, mock_time):
        """Test print_time method."""
        mock_time.return_value = 1000.0
        start_time = 940.0  # 60 seconds elapsed
        
        TimeUtils.print_time(start_time)
        
        # Check that print was called with expected format
        mock_print.assert_any_call("[Elapsed time = 1.0 min]")
    
    @patch('time.time')
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=True)
    @patch('builtins.print')
    def test_how_long(self, mock_print, mock_is_main, mock_time):
        """Test how_long method."""
        mock_time.return_value = 1000.0
        start_time = 940.0  # 60 seconds elapsed
        
        TimeUtils.how_long(start_time, "test operation")
        
        # Verify print was called with expected message
        mock_print.assert_any_call(" >>>>>> Elapsed time to test operation = 1.0 min (60.0 seconds)")
    
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=False)
    @patch('builtins.print')
    def test_how_long_not_main_process(self, mock_print, mock_is_main):
        """Test how_long doesn't print when not main process."""
        TimeUtils.how_long(0, "test")
        mock_print.assert_not_called()


class TestMetricsManager:
    """Test suite for MetricsManager class."""
    
    def create_test_config(self, task="pretrain", dataset_name="mimic"):
        """Create a test configuration."""
        config = omegaconf.DictConfig({
            "task": task,
            "dataset_name": dataset_name,
            "device": "cpu",
            "num_classes": 2
        })
        return config
    
    def test_get_metrics_pretrain(self):
        """Test metrics creation for pretraining task."""
        config = self.create_test_config(task="pretrain")
        metrics, metric_names = MetricsManager.get_metrics(config)
        
        assert "train" in metrics
        assert "test" in metrics
        assert "Loss" in metric_names
        assert "MLM" in metric_names
        assert "MIM" in metric_names
        assert "MTM" in metric_names
        assert "MFM" in metric_names
        assert "MMM" in metric_names
    
    def test_get_metrics_finetune_amazon(self):
        """Test metrics creation for Amazon finetuning task."""
        config = self.create_test_config(task="finetune", dataset_name="amazon")
        metrics, metric_names = MetricsManager.get_metrics(config)
        
        assert "Loss" in metric_names
        assert "ACCURACY" in metric_names
        assert metrics["train"]["ACCURACY"] is not None
        assert metrics["test"]["ACCURACY"] is not None
    
    def test_get_metrics_finetune_mimic(self):
        """Test metrics creation for MIMIC finetuning task."""
        config = self.create_test_config(task="finetune", dataset_name="mimic")
        metrics, metric_names = MetricsManager.get_metrics(config)
        
        assert "Loss" in metric_names
        assert "AUROC" in metric_names
        assert metrics["train"]["AUROC"] is not None
        assert metrics["test"]["AUROC"] is not None


class TestModelUtils:
    """Test suite for ModelUtils class."""
    
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=True)
    @patch('builtins.print')
    def test_print_model_size(self, mock_print, mock_is_main):
        """Test model size printing."""
        # Create a simple model
        model = torch.nn.Linear(10, 5)
        
        ModelUtils.print_model_size(model, "Test Model")
        
        # Check that print was called with model information
        assert mock_print.call_count > 0
        
        # Check for expected content in print calls
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        model_header_found = any("Test Model" in call for call in print_calls)
        assert model_header_found
    
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=False)
    @patch('builtins.print')
    def test_print_model_size_not_main_process(self, mock_print, mock_is_main):
        """Test model size printing when not main process."""
        model = torch.nn.Linear(10, 5)
        ModelUtils.print_model_size(model, "Test Model")
        mock_print.assert_not_called()


class TestPrintUtils:
    """Test suite for PrintUtils class."""
    
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=True)
    @patch('builtins.print')
    def test_print_only_by_main_process(self, mock_print, mock_is_main):
        """Test printing when main process."""
        PrintUtils.print_only_by_main_process("Test message")
        mock_print.assert_called()
    
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=False)
    @patch('builtins.print')
    def test_print_only_by_main_process_not_main(self, mock_print, mock_is_main):
        """Test no printing when not main process."""
        PrintUtils.print_only_by_main_process("Test message")
        mock_print.assert_not_called()


class TestPerformancePrinter:
    """Test suite for PerformancePrinter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.printer = PerformancePrinter()
        self.mock_logger = Mock()
    
    def test_format_results_with_tensors(self):
        """Test result formatting with tensor values."""
        results = {
            "Loss": torch.tensor(1.5),
            "Accuracy": 0.85,
            "F1": torch.tensor(0.75)
        }
        
        formatted = self.printer._format_results(results)
        
        assert formatted["Loss"] == 1.5
        assert formatted["Accuracy"] == 0.85
        assert formatted["F1"] == 0.75
    
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=True)
    @patch('builtins.print')
    def test_print_performance_pretrain(self, mock_print, mock_is_main):
        """Test performance printing for pretraining."""
        m_t = MetricsLogger()
        train_results = {"Loss": 1.5, "MLM": 0.8, "MIM": 0.7, "MTM": 0.6, "MFM": 0.5, "MMM": 0.4}
        valid_results = {"Loss": 1.2, "MLM": 0.75, "MIM": 0.65, "MTM": 0.55, "MFM": 0.45, "MMM": 0.35}
        
        self.printer.print_performance(
            epoch=0, num_epochs=10, m_t=m_t,
            train_results=train_results, valid_results=valid_results,
            is_best=True, best_perf=1.2, metric_name="Loss", is_pretrain=True
        )
        
        assert mock_print.call_count > 0
    
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=False)
    @patch('builtins.print')
    def test_print_performance_not_main_process(self, mock_print, mock_is_main):
        """Test no printing when not main process."""
        m_t = MetricsLogger()
        
        result = self.printer.print_performance(
            epoch=0, num_epochs=10, m_t=m_t,
            train_results={}, valid_results={},
            is_best=False, best_perf=1.0, metric_name="Loss"
        )
        
        mock_print.assert_not_called()
        assert result == m_t


class TestCheckpointManager:
    """Test suite for CheckpointManager class."""
    
    def test_save_checkpoint(self):
        """Test model checkpoint saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(10, 5)
            
            CheckpointManager.save_checkpoint(
                model=model,
                is_best=True,
                file_dir=temp_dir,
                filename="test_checkpoint.pth",
                best_filename="best_model.pth"
            )
            
            # Check that files exist
            assert os.path.exists(os.path.join(temp_dir, "test_checkpoint.pth"))
            assert os.path.exists(os.path.join(temp_dir, "best_model.pth"))
    
    def test_save_checkpoint_optimizer(self):
        """Test optimizer checkpoint saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(10, 5)
            optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            
            CheckpointManager.save_checkpoint_optimizer(
                epoch=5,
                optimizer=optimizer,
                scheduler=scheduler,
                is_best=True,
                file_dir=temp_dir,
                filename="optimizer.pth"
            )
            
            assert os.path.exists(os.path.join(temp_dir, "optimizer.pth"))
    
    def test_load_checkpoint(self):
        """Test loading model checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save a model
            model = torch.nn.Linear(10, 5)
            checkpoint_path = os.path.join(temp_dir, "test.pth")
            torch.save({"state_dict": model.state_dict()}, checkpoint_path)
            
            # Load into new model
            new_model = torch.nn.Linear(10, 5)
            loaded_model = CheckpointManager.load_checkpoint(checkpoint_path, new_model, "cpu")
            
            # Verify models have same state
            assert torch.equal(model.weight, loaded_model.weight)
            assert torch.equal(model.bias, loaded_model.bias)
    
    def test_load_checkpoint_with_module(self):
        """Test loading checkpoint with module prefix handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(10, 5)
            
            # Create state dict with module prefix
            state_dict = {"module." + k: v for k, v in model.state_dict().items()}
            checkpoint_path = os.path.join(temp_dir, "test_module.pth")
            torch.save({"state_dict": state_dict}, checkpoint_path)
            
            # Load into new model
            new_model = torch.nn.Linear(10, 5)
            loaded_model = CheckpointManager.load_checkpoint_with_module(checkpoint_path, new_model, "cpu")
            
            # Verify models have same state
            assert torch.equal(model.weight, loaded_model.weight)
            assert torch.equal(model.bias, loaded_model.bias)
    
    def test_load_checkpoint_not_found(self):
        """Test loading non-existent checkpoint raises error."""
        model = torch.nn.Linear(10, 5)
        
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_checkpoint("nonexistent.pth", model, "cpu")


class TestMetricsLogger:
    """Test suite for MetricsLogger class."""
    
    def test_update_and_get_latest(self):
        """Test updating metrics and getting latest values."""
        logger = MetricsLogger()
        
        logger.update("loss", 1.5)
        logger.update("loss", 1.2)
        logger.update("accuracy", torch.tensor(0.85))
        
        assert logger.get_latest("loss") == 1.2
        assert logger.get_latest("accuracy") == 0.85
        assert logger.get_latest("nonexistent") is None
    
    def test_get_all(self):
        """Test getting all values for a metric."""
        logger = MetricsLogger()
        
        logger.update("loss", 1.5)
        logger.update("loss", 1.2)
        logger.update("loss", 1.0)
        
        all_losses = logger.get_all("loss")
        assert all_losses == [1.5, 1.2, 1.0]
    
    def test_reset_specific_metric(self):
        """Test resetting a specific metric."""
        logger = MetricsLogger()
        
        logger.update("loss", 1.5)
        logger.update("accuracy", 0.85)
        
        logger.reset("loss")
        
        assert logger.get_all("loss") == []
        assert logger.get_latest("accuracy") == 0.85
    
    def test_reset_all_metrics(self):
        """Test resetting all metrics."""
        logger = MetricsLogger()
        
        logger.update("loss", 1.5)
        logger.update("accuracy", 0.85)
        
        logger.reset()
        
        assert logger.get_all("loss") == []
        assert logger.get_all("accuracy") == []


class TestDataLoaderManager:
    """Test suite for DataLoaderManager class."""
    
    def create_test_config(self, distributed=False):
        """Create test configuration."""
        return omegaconf.DictConfig({
            "task": "pretrain",
            "train_batch_size": 32,
            "eval_batch_size": 16,
            "test_batch_size": 8,
            "workers": 4,
            "distributed": distributed
        })
    
    def create_mock_dataset(self, size=100):
        """Create a mock dataset."""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=size)
        return dataset
    
    def test_init(self):
        """Test DataLoaderManager initialization."""
        config = self.create_test_config()
        manager = DataLoaderManager(config)
        
        assert manager.train_batch_size == 32
        assert manager.eval_batch_size == 16
        assert manager.test_batch_size == 8
        assert manager.workers == 4
        assert manager.distributed == False
    
    def test_create_sampler_distributed(self):
        """Test sampler creation in distributed mode."""
        config = self.create_test_config(distributed=True)
        manager = DataLoaderManager(config)
        dataset = self.create_mock_dataset()
        
        with patch('torch.utils.data.distributed.DistributedSampler') as mock_sampler:
            sampler = manager._create_sampler(dataset)
            mock_sampler.assert_called_once_with(dataset, shuffle=True, drop_last=True)
    
    def test_create_sampler_non_distributed(self):
        """Test sampler creation in non-distributed mode."""
        config = self.create_test_config(distributed=False)
        manager = DataLoaderManager(config)
        dataset = self.create_mock_dataset()
        
        sampler = manager._create_sampler(dataset)
        assert sampler is None
    
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=True)
    @patch('builtins.print')
    def test_log_dataset_info(self, mock_print, mock_is_main):
        """Test dataset info logging."""
        config = self.create_test_config()
        manager = DataLoaderManager(config)
        dataset = self.create_mock_dataset(size=1000)
        
        manager._log_dataset_info("training", dataset)
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "training" in call_args
        assert "1000" in call_args
    
    @patch('torch.utils.data.DataLoader')
    def test_generate_pretrain_loaders(self, mock_dataloader):
        """Test pretrain loader generation."""
        config = self.create_test_config()
        manager = DataLoaderManager(config)
        dataset = {"train": self.create_mock_dataset()}
        
        with patch.object(manager, '_log_dataset_info'):
            loaders = manager.generate_pretrain_loaders(dataset)
        
        assert "train" in loaders
        assert len(loaders) == 1
    
    @patch('torch.utils.data.DataLoader')
    def test_generate_finetune_loaders(self, mock_dataloader):
        """Test finetune loader generation."""
        config = self.create_test_config()
        manager = DataLoaderManager(config)
        dataset = {
            "train": self.create_mock_dataset(),
            "test": self.create_mock_dataset(),
            "valid": self.create_mock_dataset()
        }
        
        with patch.object(manager, '_log_dataset_info'):
            loaders = manager.generate_finetune_loaders(dataset)
        
        assert "train" in loaders
        assert "test" in loaders
        assert "valid" in loaders
        assert len(loaders) == 3


class TestImageTransformManager:
    """Test suite for ImageTransformManager class."""
    
    def create_test_config(self):
        """Create test configuration."""
        return omegaconf.DictConfig({
            "image_size": 224,
            "image_crop": 224,
            "image_encoder_name": "google/vit-base-patch16-224"
        })
    
    @patch('transformers.ViTImageProcessor.from_pretrained')
    def test_get_image_processor(self, mock_processor):
        """Test image processor retrieval."""
        config = self.create_test_config()
        manager = ImageTransformManager(config)
        
        manager.get_image_processor()
        
        mock_processor.assert_called_once_with("google/vit-base-patch16-224")
    
    def test_create_train_transforms(self):
        """Test training transform creation."""
        config = self.create_test_config()
        manager = ImageTransformManager(config)
        
        mock_processor = Mock()
        mock_processor.image_mean = [0.485, 0.456, 0.406]
        mock_processor.image_std = [0.229, 0.224, 0.225]
        
        transforms = manager.create_train_transforms(mock_processor)
        
        assert transforms is not None
        assert len(transforms.transforms) == 4  # RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
    
    def test_create_test_transforms(self):
        """Test test transform creation."""
        config = self.create_test_config()
        manager = ImageTransformManager(config)
        
        mock_processor = Mock()
        mock_processor.image_mean = [0.485, 0.456, 0.406]
        mock_processor.image_std = [0.229, 0.224, 0.225]
        
        transforms = manager.create_test_transforms(mock_processor)
        
        assert transforms is not None
        assert len(transforms.transforms) == 4  # Resize, CenterCrop, ToTensor, Normalize


class TestMaskGenerator:
    """Test suite for MaskGenerator class."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        generator = MaskGenerator(
            input_size=192,
            mask_patch_size=32,
            model_patch_size=4,
            mask_ratio=0.6
        )
        
        assert generator.input_size == 192
        assert generator.mask_patch_size == 32
        assert generator.model_patch_size == 4
        assert generator.mask_ratio == 0.6
        assert generator.rand_size == 6  # 192 // 32
        assert generator.scale == 8  # 32 // 4
        assert generator.token_count == 36  # 6^2
    
    def test_init_invalid_input_size(self):
        """Test initialization with invalid input size."""
        with pytest.raises(ValueError, match="Input size must be divisible by mask patch size"):
            MaskGenerator(input_size=193, mask_patch_size=32)
    
    def test_init_invalid_mask_patch_size(self):
        """Test initialization with invalid mask patch size."""
        with pytest.raises(ValueError, match="Mask patch size must be divisible by model patch size"):
            MaskGenerator(mask_patch_size=33, model_patch_size=4)
    
    def test_call_generates_valid_mask(self):
        """Test mask generation returns valid tensor."""
        generator = MaskGenerator()
        mask = generator()
        
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert mask.shape[0] == generator.rand_size * generator.scale * generator.rand_size * generator.scale
    
    def test_call_mask_ratio(self):
        """Test that generated mask respects the mask ratio."""
        generator = MaskGenerator(mask_ratio=0.5)
        mask = generator()
        
        # Check that approximately half the tokens are masked
        mask_count = mask.sum().item()
        expected_count = generator.mask_count
        
        assert mask_count == expected_count
    
    def test_get_mask_info(self):
        """Test getting mask configuration information."""
        generator = MaskGenerator(
            input_size=192,
            mask_patch_size=32,
            model_patch_size=4,
            mask_ratio=0.6
        )
        
        info = generator.get_mask_info()
        
        assert info["input_size"] == 192
        assert info["mask_patch_size"] == 32
        assert info["model_patch_size"] == 4
        assert info["mask_ratio"] == 0.6
        assert info["token_count"] == 36
        assert info["rand_size"] == 6
        assert info["scale"] == 8


class TestBackwardCompatibility:
    """Test suite for backward compatibility functions."""
    
    def test_backward_compatible_functions_exist(self):
        """Test that all backward compatibility functions exist."""
        from lanistr.utils.common_utils import (
            pretty_print, print_time, get_metrics, print_model_size,
            how_long, print_only_by_main_process, print_df_stats,
            save_checkpoint, load_checkpoint, print_config
        )
        
        # Test that functions exist and are callable
        assert callable(pretty_print)
        assert callable(print_time)
        assert callable(get_metrics)
        assert callable(print_model_size)
        assert callable(how_long)
        assert callable(print_only_by_main_process)
        assert callable(print_df_stats)
        assert callable(save_checkpoint)
        assert callable(load_checkpoint)
        assert callable(print_config)
    
    def test_backward_compatible_data_utils(self):
        """Test backward compatibility for data utils."""
        from lanistr.utils.data_utils import generate_loaders, get_image_transforms
        
        assert callable(generate_loaders)
        assert callable(get_image_transforms)


class TestIntegration:
    """Integration tests for the refactored classes."""
    
    @patch('lanistr.utils.parallelism_utils.is_main_process', return_value=True)
    def test_full_training_workflow_simulation(self, mock_is_main):
        """Test a simulated training workflow using refactored classes."""
        # Setup
        config = omegaconf.DictConfig({
            "task": "pretrain",
            "train_batch_size": 8,
            "workers": 2,
            "distributed": False,
            "device": "cpu",
            "dataset_name": "mimic"
        })
        
        # Create metrics
        metrics, metric_names = MetricsManager.get_metrics(config)
        assert "Loss" in metric_names
        
        # Create logger
        logger = MetricsLogger()
        
        # Simulate training loop
        printer = PerformancePrinter()
        
        for epoch in range(2):
            # Simulate training metrics
            train_results = {
                "Loss": torch.tensor(1.5 - epoch * 0.1),
                "MLM": torch.tensor(0.8 - epoch * 0.05),
                "MIM": torch.tensor(0.7 - epoch * 0.05),
                "MTM": torch.tensor(0.6 - epoch * 0.05),
                "MFM": torch.tensor(0.5 - epoch * 0.05),
                "MMM": torch.tensor(0.4 - epoch * 0.05)
            }
            
            valid_results = {k: v - 0.1 for k, v in train_results.items()}
            
            # Print performance
            with patch('builtins.print'):
                logger = printer.print_performance(
                    epoch=epoch, num_epochs=2, m_t=logger,
                    train_results=train_results, valid_results=valid_results,
                    is_best=(epoch == 1), best_perf=1.4, metric_name="Loss",
                    is_pretrain=True
                )
            
            # Check logger was updated
            assert logger.get_latest("Loss") is not None
        
        # Test checkpoint saving
        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(10, 5)
            
            CheckpointManager.save_checkpoint(
                model=model, is_best=True, file_dir=temp_dir,
                filename="test.pth", epoch=1, loss=1.4
            )
            
            assert os.path.exists(os.path.join(temp_dir, "test.pth"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])