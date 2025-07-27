"""
Pytest configuration and fixtures for LANISTR tests.
"""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test data fixtures
@pytest.fixture
def sample_mimic_data():
    """Sample MIMIC-IV data for testing."""
    return [
        {
            "patient_id": "p49746005",
            "image_path": "mimic-cxr-jpg/2.0.0/files/p4/p49746005/s75575983/01246640.jpg",
            "text": "The heart size is normal. The mediastinum is unremarkable.",
            "timeseries_path": "timeseries/patient_p49746005_vitals.csv",
            "split": "train"
        },
        {
            "patient_id": "p68147417", 
            "image_path": "mimic-cxr-jpg/2.0.0/files/p6/p68147417/s66665547/05f3ff3d.jpg",
            "text": "No acute cardiopulmonary abnormality.",
            "timeseries_path": "timeseries/patient_p68147417_vitals.csv",
            "split": "train"
        }
    ]

@pytest.fixture
def sample_amazon_data():
    """Sample Amazon data for testing."""
    return [
        {
            "asin": "B000123456",
            "title": "Test Product 1",
            "description": "This is a test product description.",
            "category": "electronics",
            "image_path": "images/electronics/product_000001.png",
            "price": 29.99,
            "rating": 4.5,
            "split": "train"
        },
        {
            "asin": "B000789012",
            "title": "Test Product 2", 
            "description": "Another test product description.",
            "category": "sports_outdoors",
            "image_path": "images/sports_outdoors/product_000002.jpeg",
            "price": 49.99,
            "rating": 4.2,
            "split": "train"
        }
    ]

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "task": "pretrain",
        "dataset_type": "mimic-iv",
        "data_dir": "./data",
        "batch_size": 4,
        "num_epochs": 2,
        "learning_rate": 0.001,
        "seed": 42
    }

@pytest.fixture
def mock_omegaconf():
    """Mock OmegaConf configuration."""
    config = Mock()
    config.task = "pretrain"
    config.dataset_type = "mimic-iv"
    config.data_dir = "./data"
    config.batch_size = 4
    config.num_epochs = 2
    config.learning_rate = 0.001
    config.seed = 42
    config.category = "mimic-iv"
    config.test_ratio = 0.2
    config.sub_samples = 10
    config.eval_batch_size = 2
    config.test_batch_size = 2
    return config

@pytest.fixture
def sample_jsonl_file(temp_data_dir, sample_mimic_data):
    """Create a sample JSONL file for testing."""
    jsonl_path = os.path.join(temp_data_dir, "test_mimic.jsonl")
    with open(jsonl_path, 'w') as f:
        for item in sample_mimic_data:
            f.write(json.dumps(item) + '\n')
    return jsonl_path

@pytest.fixture
def sample_csv_file(temp_data_dir):
    """Create a sample CSV file for testing."""
    csv_path = os.path.join(temp_data_dir, "test_data.csv")
    data = {
        'patient_id': ['p1', 'p2', 'p3'],
        'age': [25, 30, 35],
        'gender': ['M', 'F', 'M'],
        'diagnosis': ['normal', 'pneumonia', 'normal']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def mock_torch():
    """Mock PyTorch for testing."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.device', return_value='cpu'):
            yield

@pytest.fixture
def mock_gcloud():
    """Mock Google Cloud SDK for testing."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "test-project-id"
        yield mock_run 