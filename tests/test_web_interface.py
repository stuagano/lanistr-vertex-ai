"""
Unit tests for web interface components.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add web_interface to path
web_interface_path = Path(__file__).parent.parent / "web_interface"
sys.path.insert(0, str(web_interface_path))

from api import app


class TestWebInterfaceAPI:
    """Test cases for web interface API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    @patch('api.check_prerequisites')
    def test_status_endpoint(self, mock_check_prereq, client):
        """Test system status endpoint."""
        mock_check_prereq.return_value = {
            "gcloud": True,
            "docker": True,
            "python3": True,
            "gsutil": True
        }
        
        with patch('api.check_authentication', return_value=True):
            with patch('api.get_project_id', return_value="test-project"):
                with patch('api.check_apis_enabled', return_value={
                    "aiplatform.googleapis.com": True,
                    "storage.googleapis.com": True
                }):
                    response = client.get("/status")
                    assert response.status_code == 200
                    data = response.json()
                    assert "prerequisites" in data
                    assert "authenticated" in data
                    assert data["authenticated"] == True

    def test_submit_job_endpoint(self, client):
        """Test job submission endpoint."""
        job_config = {
            "project_id": "test-project",
            "dataset_type": "mimic-iv",
            "environment": "dev",
            "job_name": "test-job"
        }
        
        with patch('api.submit_job_async') as mock_submit:
            mock_submit.return_value = "job-123"
            response = client.post("/jobs/submit", json=job_config)
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["job_id"] == "job-123"

    def test_submit_job_invalid_config(self, client):
        """Test job submission with invalid config."""
        invalid_config = {
            "project_id": "",  # Empty project ID
            "dataset_type": "invalid-type"
        }
        
        response = client.post("/jobs/submit", json=invalid_config)
        assert response.status_code == 422  # Validation error

    def test_list_jobs_endpoint(self, client):
        """Test list jobs endpoint."""
        with patch('api.jobs_db', {
            "job-1": {"id": "job-1", "status": "running"},
            "job-2": {"id": "job-2", "status": "completed"}
        }):
            response = client.get("/jobs")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2

    def test_get_job_endpoint(self, client):
        """Test get specific job endpoint."""
        with patch('api.jobs_db', {
            "job-1": {"id": "job-1", "status": "running"}
        }):
            response = client.get("/jobs/job-1")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "job-1"
            assert data["status"] == "running"

    def test_get_nonexistent_job(self, client):
        """Test get nonexistent job."""
        response = client.get("/jobs/nonexistent-job")
        assert response.status_code == 404

    def test_delete_job_endpoint(self, client):
        """Test delete job endpoint."""
        with patch('api.jobs_db', {"job-1": {"id": "job-1"}}):
            response = client.delete("/jobs/job-1")
            assert response.status_code == 200

    def test_validate_dataset_endpoint(self, client, temp_data_dir):
        """Test dataset validation endpoint."""
        # Create test JSONL file
        test_file = os.path.join(temp_data_dir, "test.jsonl")
        with open(test_file, 'w') as f:
            f.write('{"patient_id": "p1", "image_path": "img.jpg", "text": "test", "timeseries_path": "ts.csv"}\n')
        
        with patch('api.validate_mimic_dataset') as mock_validate:
            mock_validate.return_value = {
                "passed": True,
                "message": "Validation successful",
                "errors": [],
                "warnings": [],
                "stats": {"total_count": 1}
            }
            
            response = client.post("/validate", json={
                "dataset_type": "mimic-iv",
                "data_dir": temp_data_dir,
                "jsonl_file": test_file
            })
            assert response.status_code == 200
            data = response.json()
            assert data["passed"] == True

    def test_upload_data_endpoint(self, client, temp_data_dir):
        """Test file upload endpoint."""
        # Create test file
        test_file = os.path.join(temp_data_dir, "test.jsonl")
        with open(test_file, 'w') as f:
            f.write('{"test": "data"}\n')
        
        with open(test_file, 'rb') as f:
            response = client.post(
                "/upload",
                files={"file": ("test.jsonl", f, "application/json")},
                data={"dataset_type": "mimic-iv"}
            )
            assert response.status_code == 200

    def test_setup_project_endpoint(self, client):
        """Test project setup endpoint."""
        with patch('api.setup_gcs_bucket', return_value=True):
            response = client.post("/setup", json={
                "project_id": "test-project",
                "region": "us-central1"
            })
            assert response.status_code == 200


class TestWebInterfaceValidation:
    """Test cases for web interface validation logic."""

    def test_validate_job_config_valid(self):
        """Test valid job configuration validation."""
        from api import JobConfig
        
        valid_config = {
            "project_id": "test-project",
            "dataset_type": "mimic-iv",
            "environment": "dev",
            "job_name": "test-job"
        }
        
        config = JobConfig(**valid_config)
        assert config.project_id == "test-project"
        assert config.dataset_type == "mimic-iv"
        assert config.environment == "dev"

    def test_validate_job_config_invalid(self):
        """Test invalid job configuration validation."""
        from api import JobConfig
        
        invalid_config = {
            "project_id": "",  # Empty project ID
            "dataset_type": "invalid-type"
        }
        
        with pytest.raises(Exception):  # Should raise validation error
            JobConfig(**invalid_config)

    def test_validate_dataset_type(self):
        """Test dataset type validation."""
        from api import JobConfig
        
        # Valid types
        valid_configs = [
            {"project_id": "test", "dataset_type": "mimic-iv"},
            {"project_id": "test", "dataset_type": "amazon"}
        ]
        
        for config in valid_configs:
            job_config = JobConfig(**config)
            assert job_config.dataset_type in ["mimic-iv", "amazon"]

    def test_validate_environment(self):
        """Test environment validation."""
        from api import JobConfig
        
        # Valid environments
        valid_configs = [
            {"project_id": "test", "dataset_type": "mimic-iv", "environment": "dev"},
            {"project_id": "test", "dataset_type": "mimic-iv", "environment": "prod"}
        ]
        
        for config in valid_configs:
            job_config = JobConfig(**config)
            assert job_config.environment in ["dev", "prod"]


class TestWebInterfaceUtils:
    """Test cases for web interface utility functions."""

    @patch('api.run_command')
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        from api import run_command
        
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "success"
        
        result = run_command("test command")
        assert result.returncode == 0
        assert result.stdout == "success"

    @patch('api.run_command')
    def test_run_command_failure(self, mock_run):
        """Test failed command execution."""
        from api import run_command
        
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "command failed"
        
        with pytest.raises(Exception):
            run_command("test command", check=True)

    @patch('api.run_command')
    def test_get_project_id(self, mock_run):
        """Test getting project ID."""
        from api import get_project_id
        
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "test-project-id"
        
        project_id = get_project_id()
        assert project_id == "test-project-id"

    @patch('api.run_command')
    def test_check_prerequisites(self, mock_run):
        """Test checking prerequisites."""
        from api import check_prerequisites
        
        mock_run.return_value.returncode = 0
        
        status = check_prerequisites()
        assert "gcloud" in status
        assert "docker" in status
        assert "python3" in status
        assert "gsutil" in status

    @patch('api.run_command')
    def test_check_authentication(self, mock_run):
        """Test checking authentication."""
        from api import check_authentication
        
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "user@example.com"
        
        authenticated = check_authentication()
        assert authenticated == True

    @patch('api.run_command')
    def test_check_apis_enabled(self, mock_run):
        """Test checking APIs enabled."""
        from api import check_apis_enabled
        
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "aiplatform.googleapis.com"
        
        apis = check_apis_enabled("test-project")
        assert "aiplatform.googleapis.com" in apis

    @patch('api.run_command')
    def test_setup_gcs_bucket_exists(self, mock_run):
        """Test GCS bucket setup when bucket exists."""
        from api import setup_gcs_bucket
        
        mock_run.return_value.returncode = 0
        
        result = setup_gcs_bucket("test-project", "test-bucket", "us-central1")
        assert result == True

    @patch('api.run_command')
    def test_setup_gcs_bucket_create(self, mock_run):
        """Test GCS bucket setup when bucket needs to be created."""
        from api import setup_gcs_bucket
        
        # First call fails (bucket doesn't exist), second call succeeds (bucket created)
        mock_run.side_effect = [
            Mock(returncode=1),  # Bucket doesn't exist
            Mock(returncode=0)   # Bucket created successfully
        ]
        
        result = setup_gcs_bucket("test-project", "test-bucket", "us-central1")
        assert result == True


class TestWebInterfaceErrorHandling:
    """Test cases for web interface error handling."""

    def test_command_timeout(self, client):
        """Test command timeout handling."""
        with patch('api.run_command', side_effect=Exception("Command timed out")):
            response = client.get("/status")
            assert response.status_code == 500

    def test_authentication_failure(self, client):
        """Test authentication failure handling."""
        with patch('api.check_authentication', return_value=False):
            with patch('api.check_prerequisites', return_value={"gcloud": True}):
                response = client.get("/status")
                assert response.status_code == 200
                data = response.json()
                assert data["authenticated"] == False

    def test_api_enablement_failure(self, client):
        """Test API enablement failure handling."""
        with patch('api.check_apis_enabled', return_value={
            "aiplatform.googleapis.com": False,
            "storage.googleapis.com": False
        }):
            with patch('api.check_authentication', return_value=True):
                response = client.get("/status")
                assert response.status_code == 200
                data = response.json()
                assert data["apis_enabled"]["aiplatform.googleapis.com"] == False

    def test_file_upload_error(self, client):
        """Test file upload error handling."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            response = client.post(
                "/upload",
                files={"file": ("test.jsonl", b"test data", "application/json")},
                data={"dataset_type": "mimic-iv"}
            )
            assert response.status_code == 500


class TestWebInterfaceIntegration:
    """Integration tests for web interface components."""

    def test_complete_job_submission_flow(self, client):
        """Test complete job submission flow."""
        # 1. Check system status
        with patch('api.check_prerequisites', return_value={"gcloud": True, "docker": True}):
            with patch('api.check_authentication', return_value=True):
                with patch('api.get_project_id', return_value="test-project"):
                    status_response = client.get("/status")
                    assert status_response.status_code == 200

        # 2. Submit job
        job_config = {
            "project_id": "test-project",
            "dataset_type": "mimic-iv",
            "environment": "dev",
            "job_name": "integration-test-job"
        }
        
        with patch('api.submit_job_async') as mock_submit:
            mock_submit.return_value = "job-integration-123"
            submit_response = client.post("/jobs/submit", json=job_config)
            assert submit_response.status_code == 200
            job_id = submit_response.json()["job_id"]

        # 3. Check job status
        with patch('api.jobs_db', {job_id: {"id": job_id, "status": "running"}}):
            job_response = client.get(f"/jobs/{job_id}")
            assert job_response.status_code == 200
            assert job_response.json()["status"] == "running"

    def test_dataset_validation_flow(self, client, temp_data_dir):
        """Test complete dataset validation flow."""
        # Create test dataset
        test_file = os.path.join(temp_data_dir, "integration_test.jsonl")
        with open(test_file, 'w') as f:
            f.write('{"patient_id": "p1", "image_path": "img.jpg", "text": "test", "timeseries_path": "ts.csv"}\n')
            f.write('{"patient_id": "p2", "image_path": "img2.jpg", "text": "test2", "timeseries_path": "ts2.csv"}\n')

        # Validate dataset
        with patch('api.validate_mimic_dataset') as mock_validate:
            mock_validate.return_value = {
                "passed": True,
                "message": "Validation successful",
                "errors": [],
                "warnings": [],
                "stats": {"total_count": 2}
            }
            
            response = client.post("/validate", json={
                "dataset_type": "mimic-iv",
                "data_dir": temp_data_dir,
                "jsonl_file": test_file
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["passed"] == True
            assert data["stats"]["total_count"] == 2


class TestWebInterfacePerformance:
    """Performance tests for web interface."""

    def test_large_dataset_validation_performance(self, client, temp_data_dir):
        """Test performance with large dataset."""
        import time
        
        # Create large test dataset
        large_file = os.path.join(temp_data_dir, "large_test.jsonl")
        with open(large_file, 'w') as f:
            for i in range(1000):
                item = {
                    "patient_id": f"p{i}",
                    "image_path": f"img_{i}.jpg",
                    "text": f"Sample text {i}",
                    "timeseries_path": f"ts_{i}.csv"
                }
                f.write(json.dumps(item) + '\n')

        # Measure validation time
        with patch('api.validate_mimic_dataset') as mock_validate:
            mock_validate.return_value = {
                "passed": True,
                "message": "Validation successful",
                "errors": [],
                "warnings": [],
                "stats": {"total_count": 1000}
            }
            
            start_time = time.time()
            response = client.post("/validate", json={
                "dataset_type": "mimic-iv",
                "data_dir": temp_data_dir,
                "jsonl_file": large_file
            })
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 5.0  # Should complete within 5 seconds

    def test_concurrent_job_submissions(self, client):
        """Test concurrent job submissions."""
        import threading
        import time
        
        results = []
        
        def submit_job(job_id):
            job_config = {
                "project_id": "test-project",
                "dataset_type": "mimic-iv",
                "environment": "dev",
                "job_name": f"concurrent-job-{job_id}"
            }
            
            with patch('api.submit_job_async') as mock_submit:
                mock_submit.return_value = f"job-{job_id}"
                response = client.post("/jobs/submit", json=job_config)
                results.append(response.status_code)
        
        # Submit 10 jobs concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=submit_job, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All submissions should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10 