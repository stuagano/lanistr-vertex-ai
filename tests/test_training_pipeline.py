"""
Unit tests for training pipeline components.
"""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lanistr.utils.common_utils import (
    print_df_stats,
    print_only_by_main_process,
    print_performance_by_main_process,
    print_pretrain_performance_by_main_process,
    how_long,
    print_config,
    set_global_logging_level,
    get_metrics,
    load_checkpoint_with_module,
    save_checkpoint,
    save_checkpoint_optimizer,
    print_model_size
)

from lanistr.utils.data_utils import (
    generate_loaders,
    get_image_transforms,
    MaskGenerator
)

from lanistr.utils.model_utils import (
    build_model,
    load_checkpoint,
    print_model_size
)

from lanistr.utils.parallelism_utils import (
    is_main_process,
    setup_model
)

from lanistr.dataset.amazon.amazon_utils import (
    load_multimodal_data,
    read_gzip,
    preprocess_amazon_tabular_features,
    encode_tabular_features,
    get_train_and_test_splits,
    drop_last,
    get_amazon_transforms
)

from lanistr.dataset.mimic_iv.mimic_utils import (
    load_mimic_data,
    preprocess_mimic_data,
    get_mimic_transforms
)


class TestCommonUtils:
    """Test cases for common utilities."""

    def test_print_df_stats(self, capsys):
        """Test DataFrame statistics printing."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        print_df_stats(df, "Test DataFrame")
        captured = capsys.readouterr()
        assert "Test DataFrame" in captured.out
        assert "5 rows" in captured.out or "5 entries" in captured.out

    @patch('lanistr.utils.common_utils.is_main_process')
    def test_print_only_by_main_process(self, mock_is_main, capsys):
        """Test printing only by main process."""
        mock_is_main.return_value = True
        print_only_by_main_process("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    @patch('lanistr.utils.common_utils.is_main_process')
    def test_print_only_by_main_process_not_main(self, mock_is_main, capsys):
        """Test printing when not main process."""
        mock_is_main.return_value = False
        print_only_by_main_process("Test message")
        captured = capsys.readouterr()
        assert "Test message" not in captured.out

    def test_how_long(self):
        """Test timing decorator."""
        import time
        
        @how_long
        def test_function():
            time.sleep(0.1)
            return "success"
        
        result = test_function()
        assert result == "success"

    def test_print_config(self, capsys, mock_omegaconf):
        """Test configuration printing."""
        print_config(mock_omegaconf)
        captured = capsys.readouterr()
        assert "Configuration" in captured.out
        assert "task" in captured.out
        assert "dataset_type" in captured.out

    def test_set_global_logging_level(self):
        """Test setting global logging level."""
        set_global_logging_level("ERROR")
        # This is mostly a smoke test since it sets global state
        assert True

    def test_get_metrics(self):
        """Test metrics calculation."""
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        targets = np.array([0, 0, 1, 1, 1])
        
        metrics = get_metrics(predictions, targets)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    @patch('torch.save')
    def test_save_checkpoint(self, mock_save):
        """Test checkpoint saving."""
        model = Mock()
        optimizer = Mock()
        epoch = 5
        loss = 0.1
        
        save_checkpoint(model, optimizer, epoch, loss, "test_checkpoint.pth")
        mock_save.assert_called_once()

    @patch('torch.save')
    def test_save_checkpoint_optimizer(self, mock_save):
        """Test optimizer checkpoint saving."""
        optimizer = Mock()
        epoch = 5
        
        save_checkpoint_optimizer(optimizer, epoch, "test_optimizer.pth")
        mock_save.assert_called_once()

    @patch('torch.load')
    def test_load_checkpoint_with_module(self, mock_load):
        """Test checkpoint loading with module."""
        mock_load.return_value = {
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'epoch': 5,
            'loss': 0.1
        }
        
        model = Mock()
        optimizer = Mock()
        
        epoch, loss = load_checkpoint_with_module(
            model, optimizer, "test_checkpoint.pth"
        )
        assert epoch == 5
        assert loss == 0.1


class TestDataUtils:
    """Test cases for data utilities."""

    def test_get_image_transforms(self):
        """Test image transforms generation."""
        transforms = get_image_transforms()
        assert transforms is not None
        # Should return a torchvision transforms.Compose object
        assert hasattr(transforms, 'transforms')

    def test_mask_generator(self):
        """Test mask generator."""
        mask_gen = MaskGenerator()
        assert mask_gen is not None

    @patch('lanistr.utils.data_utils.DataLoader')
    def test_generate_loaders(self, mock_dataloader):
        """Test data loader generation."""
        dataset = Mock()
        batch_size = 4
        num_workers = 2
        
        train_loader, val_loader = generate_loaders(
            dataset, dataset, batch_size, num_workers
        )
        
        assert mock_dataloader.call_count == 2  # Called for train and val


class TestModelUtils:
    """Test cases for model utilities."""

    @patch('lanistr.utils.model_utils.LANISTRModel')
    def test_build_model(self, mock_model_class):
        """Test model building."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        config = Mock()
        config.model_type = "lanistr"
        config.hidden_size = 768
        config.num_attention_heads = 12
        
        model = build_model(config)
        assert model is not None
        mock_model_class.assert_called_once()

    @patch('torch.load')
    def test_load_checkpoint(self, mock_load):
        """Test checkpoint loading."""
        mock_load.return_value = {
            'model_state_dict': {},
            'epoch': 5,
            'loss': 0.1
        }
        
        model = Mock()
        checkpoint_path = "test_checkpoint.pth"
        
        epoch, loss = load_checkpoint(model, checkpoint_path)
        assert epoch == 5
        assert loss == 0.1

    def test_print_model_size(self, capsys):
        """Test model size printing."""
        model = Mock()
        model.parameters.return_value = [Mock() for _ in range(10)]
        
        print_model_size(model)
        captured = capsys.readouterr()
        assert "Model size" in captured.out


class TestParallelismUtils:
    """Test cases for parallelism utilities."""

    @patch('torch.distributed.is_initialized')
    def test_is_main_process(self, mock_distributed):
        """Test main process detection."""
        mock_distributed.return_value = False
        assert is_main_process() == True

    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.get_rank')
    def test_is_main_process_distributed(self, mock_rank, mock_distributed):
        """Test main process detection in distributed setting."""
        mock_distributed.return_value = True
        mock_rank.return_value = 0
        assert is_main_process() == True
        
        mock_rank.return_value = 1
        assert is_main_process() == False

    @patch('torch.nn.parallel.DistributedDataParallel')
    def test_setup_model(self, mock_ddp):
        """Test model setup for distributed training."""
        model = Mock()
        device = Mock()
        
        setup_model(model, device)
        # Should not raise any exceptions


class TestAmazonUtils:
    """Test cases for Amazon dataset utilities."""

    def test_read_gzip(self, temp_data_dir):
        """Test gzip file reading."""
        import gzip
        
        # Create test gzip file
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        gzip_file = os.path.join(temp_data_dir, "test.json.gz")
        
        with gzip.open(gzip_file, 'wt') as f:
            f.write(json.dumps(test_data))
        
        df = read_gzip(gzip_file)
        assert isinstance(df, pd.DataFrame)
        assert "test" in df.columns
        assert "numbers" in df.columns

    def test_encode_tabular_features(self):
        """Test tabular feature encoding."""
        df = pd.DataFrame({
            'categorical': ['A', 'B', 'A', 'C'],
            'numerical': [1.0, 2.0, 3.0, 4.0]
        })
        
        categorical_cols = ['categorical']
        numerical_cols = ['numerical']
        
        encoded_df, cat_idxs, cat_dims, input_dim = encode_tabular_features(
            df, categorical_cols, numerical_cols
        )
        
        assert isinstance(encoded_df, pd.DataFrame)
        assert len(cat_idxs) == 1
        assert len(cat_dims) == 1
        assert input_dim == 2

    def test_preprocess_amazon_tabular_features(self):
        """Test Amazon tabular feature preprocessing."""
        df = pd.DataFrame({
            'categorical': ['A', 'B', 'A', 'C'],
            'numerical': [1.0, 2.0, 3.0, 4.0]
        })
        
        categorical_cols = ['categorical']
        numerical_cols = ['numerical']
        
        result = preprocess_amazon_tabular_features(
            df, categorical_cols, numerical_cols
        )
        
        assert len(result) == 4
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)
        assert isinstance(result[2], list)
        assert isinstance(result[3], int)

    def test_drop_last(self):
        """Test dropping last incomplete batch."""
        df = pd.DataFrame({
            'col1': range(10),
            'col2': range(10)
        })
        
        batch_size = 3
        result = drop_last(df, batch_size)
        
        # Should drop the last incomplete batch
        assert len(result) == 9  # 3 complete batches of 3

    @patch('torchvision.transforms.Compose')
    def test_get_amazon_transforms(self, mock_compose, mock_omegaconf):
        """Test Amazon transforms generation."""
        mock_omegaconf.image_size = 224
        mock_omegaconf.normalize_mean = [0.485, 0.456, 0.406]
        mock_omegaconf.normalize_std = [0.229, 0.224, 0.225]
        
        train_transforms, val_transforms = get_amazon_transforms(mock_omegaconf)
        
        assert train_transforms is not None
        assert val_transforms is not None
        assert mock_compose.call_count == 2  # Called for train and val


class TestMimicUtils:
    """Test cases for MIMIC-IV dataset utilities."""

    def test_load_mimic_data(self, temp_data_dir):
        """Test MIMIC data loading."""
        # Create test CSV file
        test_data = pd.DataFrame({
            'patient_id': ['p1', 'p2', 'p3'],
            'image_path': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'text': ['text1', 'text2', 'text3'],
            'timeseries_path': ['ts1.csv', 'ts2.csv', 'ts3.csv']
        })
        
        csv_file = os.path.join(temp_data_dir, "test_mimic.csv")
        test_data.to_csv(csv_file, index=False)
        
        with patch('lanistr.dataset.mimic_iv.mimic_utils.pd.read_csv') as mock_read:
            mock_read.return_value = test_data
            result = load_mimic_data(csv_file)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

    def test_preprocess_mimic_data(self):
        """Test MIMIC data preprocessing."""
        df = pd.DataFrame({
            'patient_id': ['p1', 'p2', 'p3'],
            'image_path': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
            'text': ['text1', 'text2', 'text3'],
            'timeseries_path': ['ts1.csv', 'ts2.csv', 'ts3.csv']
        })
        
        result = preprocess_mimic_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    @patch('torchvision.transforms.Compose')
    def test_get_mimic_transforms(self, mock_compose, mock_omegaconf):
        """Test MIMIC transforms generation."""
        mock_omegaconf.image_size = 224
        mock_omegaconf.normalize_mean = [0.485, 0.456, 0.406]
        mock_omegaconf.normalize_std = [0.229, 0.224, 0.225]
        
        train_transforms, val_transforms = get_mimic_transforms(mock_omegaconf)
        
        assert train_transforms is not None
        assert val_transforms is not None
        assert mock_compose.call_count == 2  # Called for train and val


class TestTrainingPipelineIntegration:
    """Integration tests for training pipeline."""

    def test_complete_data_loading_flow(self, temp_data_dir):
        """Test complete data loading flow."""
        # Create test data
        test_data = pd.DataFrame({
            'patient_id': ['p1', 'p2'],
            'image_path': ['img1.jpg', 'img2.jpg'],
            'text': ['text1', 'text2'],
            'timeseries_path': ['ts1.csv', 'ts2.csv']
        })
        
        csv_file = os.path.join(temp_data_dir, "test_data.csv")
        test_data.to_csv(csv_file, index=False)
        
        # Test loading
        with patch('lanistr.dataset.mimic_iv.mimic_utils.pd.read_csv') as mock_read:
            mock_read.return_value = test_data
            data = load_mimic_data(csv_file)
            
            # Test preprocessing
            processed_data = preprocess_mimic_data(data)
            
            # Test transforms
            config = Mock()
            config.image_size = 224
            config.normalize_mean = [0.485, 0.456, 0.406]
            config.normalize_std = [0.229, 0.224, 0.225]
            
            train_transforms, val_transforms = get_mimic_transforms(config)
            
            assert len(processed_data) == 2
            assert train_transforms is not None
            assert val_transforms is not None

    def test_model_building_and_checkpointing(self):
        """Test model building and checkpointing flow."""
        # Build model
        config = Mock()
        config.model_type = "lanistr"
        config.hidden_size = 768
        config.num_attention_heads = 12
        
        with patch('lanistr.utils.model_utils.LANISTRModel') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            model = build_model(config)
            
            # Test checkpoint saving
            optimizer = Mock()
            epoch = 5
            loss = 0.1
            
            with patch('torch.save') as mock_save:
                save_checkpoint(model, optimizer, epoch, loss, "test.pth")
                mock_save.assert_called_once()
            
            # Test checkpoint loading
            with patch('torch.load') as mock_load:
                mock_load.return_value = {
                    'model_state_dict': {},
                    'optimizer_state_dict': {},
                    'epoch': 5,
                    'loss': 0.1
                }
                
                loaded_epoch, loaded_loss = load_checkpoint_with_module(
                    model, optimizer, "test.pth"
                )
                
                assert loaded_epoch == 5
                assert loaded_loss == 0.1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(Exception):
            print_df_stats(df, "Empty DataFrame")

    def test_large_dataframe(self):
        """Test handling of large DataFrame."""
        # Create large DataFrame
        large_df = pd.DataFrame({
            'col1': range(10000),
            'col2': [f'value_{i}' for i in range(10000)]
        })
        
        # Should not raise any exceptions
        print_df_stats(large_df, "Large DataFrame")

    def test_none_values_in_data(self):
        """Test handling of None values in data."""
        df = pd.DataFrame({
            'col1': [1, None, 3, None, 5],
            'col2': ['a', 'b', None, 'd', 'e']
        })
        
        # Should handle None values gracefully
        print_df_stats(df, "DataFrame with None values")

    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        df = pd.DataFrame({
            'col1': [1e10, 1e20, 1e30],
            'col2': [1e-10, 1e-20, 1e-30]
        })
        
        # Should handle very large/small numbers
        print_df_stats(df, "DataFrame with large numbers")

    def test_special_characters_in_text(self):
        """Test handling of special characters in text."""
        df = pd.DataFrame({
            'text': [
                'Normal text',
                'Text with "quotes"',
                'Text with \n newlines',
                'Text with \t tabs',
                'Text with unicode: 你好世界'
            ]
        })
        
        # Should handle special characters
        print_df_stats(df, "DataFrame with special characters")


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_file_not_found_error(self):
        """Test handling of file not found errors."""
        with pytest.raises(FileNotFoundError):
            read_gzip("nonexistent_file.json.gz")

    def test_permission_error(self):
        """Test handling of permission errors."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                read_gzip("test_file.json.gz")

    def test_memory_error_large_file(self):
        """Test handling of memory errors with large files."""
        with patch('json.loads', side_effect=MemoryError("Out of memory")):
            with pytest.raises(MemoryError):
                read_gzip("large_file.json.gz")

    def test_invalid_json_format(self):
        """Test handling of invalid JSON format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json.gz', delete=False) as f:
            f.write('{"invalid": json}\n')  # Invalid JSON
            f.flush()
            
            with pytest.raises(Exception):
                read_gzip(f.name)
            
            os.unlink(f.name)

    def test_empty_gzip_file(self):
        """Test handling of empty gzip file."""
        import gzip
        
        with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as f:
            with gzip.open(f.name, 'wt') as gz:
                pass  # Empty file
            
            with pytest.raises(Exception):
                read_gzip(f.name)
            
            os.unlink(f.name)


class TestPerformance:
    """Performance tests."""

    def test_large_dataframe_processing_performance(self):
        """Test performance with large DataFrame."""
        import time
        
        # Create large DataFrame
        large_df = pd.DataFrame({
            'categorical': [f'cat_{i % 100}' for i in range(10000)],
            'numerical': np.random.randn(10000)
        })
        
        categorical_cols = ['categorical']
        numerical_cols = ['numerical']
        
        start_time = time.time()
        result = encode_tabular_features(large_df, categorical_cols, numerical_cols)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 10.0  # 10 seconds
        assert len(result) == 4

    def test_memory_usage_large_data(self):
        """Test memory usage with large data."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large DataFrame
        large_df = pd.DataFrame({
            'col1': range(100000),
            'col2': [f'value_{i}' for i in range(100000)]
        })
        
        # Process data
        print_df_stats(large_df, "Large DataFrame")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB 