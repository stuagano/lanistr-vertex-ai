"""
Unit tests for setup and configuration scripts.
"""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from setup_training_pipeline import (
    print_step,
    print_success,
    print_warning,
    print_error,
    check_python_version,
    check_dependencies,
    validate_dataset,
    create_sample_data,
    setup_directories,
    create_config,
    create_training_script,
    create_monitoring_script,
    setup_training_pipeline
)

from fix_imports import (
    fix_imports_in_file,
    fix_all_imports
)


class TestSetupFunctions:
    """Test cases for setup utility functions."""

    def test_print_step(self, capsys):
        """Test step printing."""
        print_step(1, "Test step")
        captured = capsys.readouterr()
        assert "STEP 1: Test step" in captured.out
        assert "=" in captured.out

    def test_print_success(self, capsys):
        """Test success printing."""
        print_success("Test success")
        captured = capsys.readouterr()
        assert "✅ Test success" in captured.out

    def test_print_warning(self, capsys):
        """Test warning printing."""
        print_warning("Test warning")
        captured = capsys.readouterr()
        assert "⚠️  Test warning" in captured.out

    def test_print_error(self, capsys):
        """Test error printing."""
        print_error("Test error")
        captured = capsys.readouterr()
        assert "❌ Test error" in captured.out

    def test_check_python_version_valid(self):
        """Test Python version check with valid version."""
        with patch('sys.version_info', (3, 8, 0)):
            result = check_python_version()
            assert result == True

    def test_check_python_version_invalid(self):
        """Test Python version check with invalid version."""
        with patch('sys.version_info', (3, 6, 0)):
            result = check_python_version()
            assert result == False

    @patch('subprocess.run')
    def test_check_dependencies_success(self, mock_run):
        """Test dependency check with all packages available."""
        mock_run.return_value.returncode = 0
        
        result = check_dependencies()
        assert result == True

    @patch('subprocess.run')
    def test_check_dependencies_missing(self, mock_run):
        """Test dependency check with missing packages."""
        mock_run.return_value.returncode = 1
        
        result = check_dependencies()
        assert result == False

    def test_validate_dataset_mimic(self, temp_data_dir):
        """Test MIMIC-IV dataset validation."""
        # Create sample MIMIC data
        sample_data = [
            {
                "patient_id": "p1",
                "image_path": "img1.jpg",
                "text": "Sample text 1",
                "timeseries_path": "ts1.csv"
            },
            {
                "patient_id": "p2",
                "image_path": "img2.jpg",
                "text": "Sample text 2",
                "timeseries_path": "ts2.csv"
            }
        ]
        
        jsonl_file = os.path.join(temp_data_dir, "mimic_sample.jsonl")
        with open(jsonl_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        result = validate_dataset("mimic-iv", jsonl_file)
        assert result == True

    def test_validate_dataset_amazon(self, temp_data_dir):
        """Test Amazon dataset validation."""
        # Create sample Amazon data
        sample_data = [
            {
                "asin": "B000123456",
                "title": "Test Product 1",
                "description": "Test description 1",
                "category": "electronics",
                "image_path": "img1.jpg",
                "price": 29.99,
                "rating": 4.5
            }
        ]
        
        jsonl_file = os.path.join(temp_data_dir, "amazon_sample.jsonl")
        with open(jsonl_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        result = validate_dataset("amazon", jsonl_file)
        assert result == True

    def test_validate_dataset_invalid(self, temp_data_dir):
        """Test invalid dataset validation."""
        # Create invalid data
        invalid_data = [{"invalid": "data"}]
        
        jsonl_file = os.path.join(temp_data_dir, "invalid_sample.jsonl")
        with open(jsonl_file, 'w') as f:
            for item in invalid_data:
                f.write(json.dumps(item) + '\n')
        
        result = validate_dataset("mimic-iv", jsonl_file)
        assert result == False

    def test_create_sample_data_mimic(self, temp_data_dir):
        """Test MIMIC-IV sample data creation."""
        output_file = os.path.join(temp_data_dir, "mimic_sample.jsonl")
        
        create_sample_data("mimic-iv", output_file, num_samples=5)
        
        assert os.path.exists(output_file)
        
        # Verify data format
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 5
            
            for line in lines:
                data = json.loads(line.strip())
                assert "patient_id" in data
                assert "image_path" in data
                assert "text" in data
                assert "timeseries_path" in data

    def test_create_sample_data_amazon(self, temp_data_dir):
        """Test Amazon sample data creation."""
        output_file = os.path.join(temp_data_dir, "amazon_sample.jsonl")
        
        create_sample_data("amazon", output_file, num_samples=3)
        
        assert os.path.exists(output_file)
        
        # Verify data format
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3
            
            for line in lines:
                data = json.loads(line.strip())
                assert "asin" in data
                assert "title" in data
                assert "description" in data
                assert "category" in data
                assert "image_path" in data
                assert "price" in data
                assert "rating" in data

    def test_setup_directories(self, temp_data_dir):
        """Test directory setup."""
        directories = [
            os.path.join(temp_data_dir, "data"),
            os.path.join(temp_data_dir, "outputs"),
            os.path.join(temp_data_dir, "logs")
        ]
        
        setup_directories(directories)
        
        for directory in directories:
            assert os.path.exists(directory)
            assert os.path.isdir(directory)

    def test_create_config_mimic(self, temp_data_dir):
        """Test MIMIC-IV configuration creation."""
        config_file = os.path.join(temp_data_dir, "mimic_config.yaml")
        
        create_config("mimic-iv", "pretrain", config_file)
        
        assert os.path.exists(config_file)
        
        # Verify config content
        with open(config_file, 'r') as f:
            content = f.read()
            assert "task: pretrain" in content
            assert "dataset_type: mimic-iv" in content
            assert "batch_size:" in content

    def test_create_config_amazon(self, temp_data_dir):
        """Test Amazon configuration creation."""
        config_file = os.path.join(temp_data_dir, "amazon_config.yaml")
        
        create_config("amazon", "finetune", config_file)
        
        assert os.path.exists(config_file)
        
        # Verify config content
        with open(config_file, 'r') as f:
            content = f.read()
            assert "task: finetune" in content
            assert "dataset_type: amazon" in content
            assert "batch_size:" in content

    def test_create_training_script(self, temp_data_dir):
        """Test training script creation."""
        script_file = os.path.join(temp_data_dir, "train_local.sh")
        
        create_training_script(script_file)
        
        assert os.path.exists(script_file)
        assert os.access(script_file, os.X_OK)  # Should be executable

    def test_create_monitoring_script(self, temp_data_dir):
        """Test monitoring script creation."""
        script_file = os.path.join(temp_data_dir, "monitor_training.sh")
        
        create_monitoring_script(script_file)
        
        assert os.path.exists(script_file)
        assert os.access(script_file, os.X_OK)  # Should be executable


class TestSetupTrainingPipeline:
    """Test cases for main setup function."""

    @patch('setup_training_pipeline.check_python_version')
    @patch('setup_training_pipeline.check_dependencies')
    @patch('setup_training_pipeline.setup_directories')
    @patch('setup_training_pipeline.create_sample_data')
    @patch('setup_training_pipeline.validate_dataset')
    @patch('setup_training_pipeline.create_config')
    @patch('setup_training_pipeline.create_training_script')
    @patch('setup_training_pipeline.create_monitoring_script')
    def test_setup_training_pipeline_success(
        self, mock_monitor, mock_train, mock_config, mock_validate,
        mock_sample, mock_dirs, mock_deps, mock_python
    ):
        """Test successful training pipeline setup."""
        mock_python.return_value = True
        mock_deps.return_value = True
        mock_validate.return_value = True
        
        result = setup_training_pipeline("mimic-iv", "pretrain")
        
        assert result == True
        mock_python.assert_called_once()
        mock_deps.assert_called_once()
        mock_dirs.assert_called()
        mock_sample.assert_called_once()
        mock_validate.assert_called_once()
        mock_config.assert_called_once()
        mock_train.assert_called_once()
        mock_monitor.assert_called_once()

    @patch('setup_training_pipeline.check_python_version')
    def test_setup_training_pipeline_python_version_fail(self, mock_python):
        """Test setup failure due to Python version."""
        mock_python.return_value = False
        
        result = setup_training_pipeline("mimic-iv", "pretrain")
        
        assert result == False

    @patch('setup_training_pipeline.check_python_version')
    @patch('setup_training_pipeline.check_dependencies')
    def test_setup_training_pipeline_dependencies_fail(self, mock_deps, mock_python):
        """Test setup failure due to missing dependencies."""
        mock_python.return_value = True
        mock_deps.return_value = False
        
        result = setup_training_pipeline("mimic-iv", "pretrain")
        
        assert result == False

    @patch('setup_training_pipeline.check_python_version')
    @patch('setup_training_pipeline.check_dependencies')
    @patch('setup_training_pipeline.setup_directories')
    @patch('setup_training_pipeline.create_sample_data')
    @patch('setup_training_pipeline.validate_dataset')
    def test_setup_training_pipeline_validation_fail(
        self, mock_validate, mock_sample, mock_dirs, mock_deps, mock_python
    ):
        """Test setup failure due to validation failure."""
        mock_python.return_value = True
        mock_deps.return_value = True
        mock_validate.return_value = False
        
        result = setup_training_pipeline("mimic-iv", "pretrain")
        
        assert result == False


class TestFixImports:
    """Test cases for import fixing functionality."""

    def test_fix_imports_in_file(self, temp_data_dir):
        """Test fixing imports in a single file."""
        # Create test file with relative imports
        test_file = os.path.join(temp_data_dir, "test_file.py")
        with open(test_file, 'w') as f:
            f.write("""from utils.common_utils import print_df_stats
from dataset.amazon.load_data import load_amazon
from model.lanistr_utils import build_model
from third_party.mvts_transformer import TimeseriesEncoder
""")
        
        fix_imports_in_file(test_file)
        
        # Verify imports were fixed
        with open(test_file, 'r') as f:
            content = f.read()
            assert "from lanistr.utils.common_utils import print_df_stats" in content
            assert "from lanistr.dataset.amazon.load_data import load_amazon" in content
            assert "from lanistr.model.lanistr_utils import build_model" in content
            assert "from lanistr.third_party.mvts_transformer import TimeseriesEncoder" in content

    def test_fix_imports_in_file_no_changes(self, temp_data_dir):
        """Test fixing imports when no changes needed."""
        # Create test file with correct imports
        test_file = os.path.join(temp_data_dir, "test_file.py")
        with open(test_file, 'w') as f:
            f.write("""from lanistr.utils.common_utils import print_df_stats
from lanistr.dataset.amazon.load_data import load_amazon
import os
import sys
""")
        
        original_content = open(test_file, 'r').read()
        fix_imports_in_file(test_file)
        
        # Content should remain the same
        with open(test_file, 'r') as f:
            content = f.read()
            assert content == original_content

    @patch('fix_imports.fix_imports_in_file')
    def test_fix_all_imports(self, mock_fix_file, temp_data_dir):
        """Test fixing imports in all Python files."""
        # Create test Python files
        test_files = [
            os.path.join(temp_data_dir, "file1.py"),
            os.path.join(temp_data_dir, "file2.py"),
            os.path.join(temp_data_dir, "file3.py")
        ]
        
        for file_path in test_files:
            with open(file_path, 'w') as f:
                f.write("from utils.common_utils import print_df_stats\n")
        
        fix_all_imports(temp_data_dir)
        
        # Should be called for each Python file
        assert mock_fix_file.call_count == 3
        for file_path in test_files:
            mock_fix_file.assert_any_call(file_path)

    def test_fix_all_imports_no_python_files(self, temp_data_dir):
        """Test fixing imports when no Python files exist."""
        # Create non-Python files
        test_files = [
            os.path.join(temp_data_dir, "file1.txt"),
            os.path.join(temp_data_dir, "file2.md"),
            os.path.join(temp_data_dir, "file3.json")
        ]
        
        for file_path in test_files:
            with open(file_path, 'w') as f:
                f.write("test content\n")
        
        with patch('fix_imports.fix_imports_in_file') as mock_fix_file:
            fix_all_imports(temp_data_dir)
            mock_fix_file.assert_not_called()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_create_sample_data_zero_samples(self, temp_data_dir):
        """Test creating sample data with zero samples."""
        output_file = os.path.join(temp_data_dir, "empty_sample.jsonl")
        
        create_sample_data("mimic-iv", output_file, num_samples=0)
        
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 0

    def test_create_sample_data_large_number(self, temp_data_dir):
        """Test creating sample data with large number of samples."""
        output_file = os.path.join(temp_data_dir, "large_sample.jsonl")
        
        create_sample_data("mimic-iv", output_file, num_samples=1000)
        
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1000

    def test_setup_directories_existing(self, temp_data_dir):
        """Test setting up directories that already exist."""
        directory = os.path.join(temp_data_dir, "existing_dir")
        os.makedirs(directory, exist_ok=True)
        
        setup_directories([directory])
        
        assert os.path.exists(directory)
        assert os.path.isdir(directory)

    def test_create_config_invalid_dataset(self, temp_data_dir):
        """Test creating config for invalid dataset."""
        config_file = os.path.join(temp_data_dir, "invalid_config.yaml")
        
        with pytest.raises(ValueError):
            create_config("invalid-dataset", "pretrain", config_file)

    def test_create_config_invalid_task(self, temp_data_dir):
        """Test creating config for invalid task."""
        config_file = os.path.join(temp_data_dir, "invalid_config.yaml")
        
        with pytest.raises(ValueError):
            create_config("mimic-iv", "invalid-task", config_file)

    def test_fix_imports_in_file_nonexistent(self, temp_data_dir):
        """Test fixing imports in nonexistent file."""
        nonexistent_file = os.path.join(temp_data_dir, "nonexistent.py")
        
        with pytest.raises(FileNotFoundError):
            fix_imports_in_file(nonexistent_file)

    def test_fix_imports_in_file_permission_error(self, temp_data_dir):
        """Test fixing imports with permission error."""
        test_file = os.path.join(temp_data_dir, "test_file.py")
        with open(test_file, 'w') as f:
            f.write("from utils.common_utils import print_df_stats\n")
        
        # Make file read-only
        os.chmod(test_file, 0o444)
        
        with pytest.raises(PermissionError):
            fix_imports_in_file(test_file)
        
        # Restore permissions
        os.chmod(test_file, 0o666)


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_create_sample_data_io_error(self, temp_data_dir):
        """Test handling of IO error during sample data creation."""
        # Create a directory instead of a file to cause IO error
        output_file = os.path.join(temp_data_dir, "output_dir")
        os.makedirs(output_file, exist_ok=True)
        
        with pytest.raises(Exception):
            create_sample_data("mimic-iv", output_file, num_samples=5)

    def test_setup_directories_permission_error(self, temp_data_dir):
        """Test handling of permission error during directory setup."""
        # Try to create directory in a location without write permissions
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                setup_directories(["/root/test_dir"])

    def test_create_config_io_error(self, temp_data_dir):
        """Test handling of IO error during config creation."""
        # Create a directory instead of a file to cause IO error
        config_file = os.path.join(temp_data_dir, "config_dir")
        os.makedirs(config_file, exist_ok=True)
        
        with pytest.raises(Exception):
            create_config("mimic-iv", "pretrain", config_file)

    def test_fix_imports_memory_error(self, temp_data_dir):
        """Test handling of memory error during import fixing."""
        test_file = os.path.join(temp_data_dir, "test_file.py")
        with open(test_file, 'w') as f:
            f.write("from utils.common_utils import print_df_stats\n")
        
        with patch('builtins.open', side_effect=MemoryError("Out of memory")):
            with pytest.raises(MemoryError):
                fix_imports_in_file(test_file)


class TestPerformance:
    """Performance tests."""

    def test_fix_imports_large_file_performance(self, temp_data_dir):
        """Test performance with large file."""
        import time
        
        # Create large test file
        test_file = os.path.join(temp_data_dir, "large_file.py")
        with open(test_file, 'w') as f:
            for i in range(10000):
                f.write(f"from utils.common_utils_{i} import print_df_stats\n")
        
        start_time = time.time()
        fix_imports_in_file(test_file)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0  # 5 seconds

    def test_create_sample_data_performance(self, temp_data_dir):
        """Test performance with large sample data creation."""
        import time
        
        output_file = os.path.join(temp_data_dir, "performance_test.jsonl")
        
        start_time = time.time()
        create_sample_data("mimic-iv", output_file, num_samples=10000)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 10.0  # 10 seconds
        
        # Verify all samples were created
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 10000

    def test_setup_directories_performance(self, temp_data_dir):
        """Test performance with many directories."""
        import time
        
        directories = [
            os.path.join(temp_data_dir, f"dir_{i}")
            for i in range(1000)
        ]
        
        start_time = time.time()
        setup_directories(directories)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0  # 5 seconds
        
        # Verify all directories were created
        for directory in directories:
            assert os.path.exists(directory)
            assert os.path.isdir(directory) 