"""
Test suite for refactored web interface classes.

This module tests the functionality of the refactored web interface components
including the command execution, Google Cloud management, and job management classes.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Import the refactored classes
from web_interface.api import (
    CommandExecutor, GoogleCloudManager, DataManager, 
    ContainerManager, JobManager, app, JobConfig
)


class TestCommandExecutor:
    """Test suite for CommandExecutor class."""
    
    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = CommandExecutor.run_command("echo hello")
        
        assert result.returncode == 0
        assert result.stdout == "success output"
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_run_command_failure_with_check(self, mock_run):
        """Test command failure with check=True raises HTTPException."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "command failed"
        mock_run.return_value = mock_result
        
        with pytest.raises(HTTPException) as exc_info:
            CommandExecutor.run_command("false", check=True)
        
        assert exc_info.value.status_code == 500
        assert "Command failed" in str(exc_info.value.detail)
    
    @patch('subprocess.run')
    def test_run_command_failure_without_check(self, mock_run):
        """Test command failure with check=False returns result."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "command failed"
        mock_run.return_value = mock_result
        
        result = CommandExecutor.run_command("false", check=False)
        
        assert result.returncode == 1
        assert result.stderr == "command failed"
    
    @patch('subprocess.run')
    def test_run_command_timeout(self, mock_run):
        """Test command timeout raises HTTPException."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 300)
        
        with pytest.raises(HTTPException) as exc_info:
            CommandExecutor.run_command("sleep 1000", timeout=1)
        
        assert exc_info.value.status_code == 408
        assert "timed out" in str(exc_info.value.detail)
    
    @patch('subprocess.run')
    def test_get_command_output_success(self, mock_run):
        """Test getting command output on success."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "  output with spaces  \n"
        mock_run.return_value = mock_result
        
        output = CommandExecutor.get_command_output("echo hello")
        
        assert output == "output with spaces"
    
    @patch('subprocess.run')
    def test_get_command_output_failure(self, mock_run):
        """Test getting command output on failure returns empty string."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        output = CommandExecutor.get_command_output("false")
        
        assert output == ""
    
    @patch('subprocess.run')
    def test_get_command_output_exception(self, mock_run):
        """Test getting command output when exception occurs."""
        mock_run.side_effect = Exception("Some error")
        
        output = CommandExecutor.get_command_output("bad command")
        
        assert output == ""


class TestGoogleCloudManager:
    """Test suite for GoogleCloudManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.gcp_manager = GoogleCloudManager()
    
    @patch.object(CommandExecutor, 'get_command_output')
    def test_get_project_id(self, mock_get_output):
        """Test getting project ID."""
        mock_get_output.return_value = "my-test-project"
        
        project_id = self.gcp_manager.get_project_id()
        
        assert project_id == "my-test-project"
        mock_get_output.assert_called_once_with("gcloud config get-value project")
    
    @patch.object(CommandExecutor, 'run_command')
    def test_check_prerequisites_all_present(self, mock_run):
        """Test checking prerequisites when all tools are present."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        prerequisites = self.gcp_manager.check_prerequisites()
        
        assert all(prerequisites.values())
        assert "gcloud" in prerequisites
        assert "docker" in prerequisites
        assert "python3" in prerequisites
        assert "gsutil" in prerequisites
    
    @patch.object(CommandExecutor, 'run_command')
    def test_check_prerequisites_some_missing(self, mock_run):
        """Test checking prerequisites when some tools are missing."""
        def side_effect(cmd, check=True):
            if "gcloud" in cmd:
                result = Mock()
                result.returncode = 0
                return result
            else:
                result = Mock()
                result.returncode = 1
                return result
        
        mock_run.side_effect = side_effect
        
        prerequisites = self.gcp_manager.check_prerequisites()
        
        assert prerequisites["gcloud"] == True
        assert prerequisites["docker"] == False
        assert prerequisites["python3"] == False
        assert prerequisites["gsutil"] == False
    
    @patch.object(CommandExecutor, 'run_command')
    def test_check_authentication_authenticated(self, mock_run):
        """Test authentication check when user is authenticated."""
        mock_result = Mock()
        mock_result.stdout = "user@example.com"
        mock_run.return_value = mock_result
        
        is_authenticated = self.gcp_manager.check_authentication()
        
        assert is_authenticated == True
    
    @patch.object(CommandExecutor, 'run_command')
    def test_check_authentication_not_authenticated(self, mock_run):
        """Test authentication check when user is not authenticated."""
        mock_result = Mock()
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        
        is_authenticated = self.gcp_manager.check_authentication()
        
        assert is_authenticated == False
    
    @patch.object(CommandExecutor, 'run_command')
    def test_check_apis_enabled(self, mock_run):
        """Test checking enabled APIs."""
        def side_effect(cmd, check=False):
            if "aiplatform.googleapis.com" in cmd:
                result = Mock()
                result.stdout = "aiplatform.googleapis.com"
                return result
            else:
                result = Mock()
                result.stdout = ""
                return result
        
        mock_run.side_effect = side_effect
        
        apis = self.gcp_manager.check_apis_enabled("test-project")
        
        assert apis["aiplatform.googleapis.com"] == True
        assert apis["storage.googleapis.com"] == False
    
    @patch.object(CommandExecutor, 'run_command')
    def test_setup_gcs_bucket_create_new(self, mock_run):
        """Test creating a new GCS bucket."""
        # First call (check) fails, second call (create) succeeds
        def side_effect(cmd, check=True):
            if "gsutil ls" in cmd:
                if check:
                    raise HTTPException(status_code=500, detail="Not found")
                result = Mock()
                result.returncode = 1
                return result
            else:  # gsutil mb
                result = Mock()
                result.returncode = 0
                return result
        
        mock_run.side_effect = side_effect
        
        success = self.gcp_manager.setup_gcs_bucket("project", "bucket", "region")
        
        assert success == True
    
    @patch.object(CommandExecutor, 'run_command')
    def test_setup_gcs_bucket_already_exists(self, mock_run):
        """Test when GCS bucket already exists."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        success = self.gcp_manager.setup_gcs_bucket("project", "bucket", "region")
        
        assert success == True


class TestDataManager:
    """Test suite for DataManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = DataManager()
    
    @patch.object(CommandExecutor, 'run_command')
    def test_create_sample_data_success(self, mock_run):
        """Test successful sample data creation."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        success = self.data_manager.create_sample_data("mimic-iv", "/data")
        
        assert success == True
        mock_run.assert_called_once()
    
    @patch.object(CommandExecutor, 'run_command')
    def test_create_sample_data_failure(self, mock_run):
        """Test sample data creation failure."""
        mock_run.side_effect = HTTPException(status_code=500, detail="Failed")
        
        success = self.data_manager.create_sample_data("mimic-iv", "/data")
        
        assert success == False
    
    @patch('web_interface.api.validate_amazon_dataset')
    def test_validate_dataset_amazon_success(self, mock_validate):
        """Test successful Amazon dataset validation."""
        mock_validate.return_value = {
            "passed": True,
            "message": "Validation passed",
            "stats": {"total": 100},
            "errors": [],
            "warnings": []
        }
        
        result = self.data_manager.validate_dataset("amazon", "test.jsonl")
        
        assert result.passed == True
        assert result.message == "Validation passed"
        assert result.stats == {"total": 100}
        assert result.errors == []
        assert result.warnings == []
    
    @patch('web_interface.api.validate_mimic_dataset')
    def test_validate_dataset_mimic_success(self, mock_validate):
        """Test successful MIMIC dataset validation."""
        mock_validate.return_value = {
            "passed": True,
            "message": "Validation passed"
        }
        
        result = self.data_manager.validate_dataset("mimic-iv", "test.jsonl")
        
        assert result.passed == True
        mock_validate.assert_called_once_with("test.jsonl", None)
    
    def test_validate_dataset_unknown_type(self):
        """Test validation with unknown dataset type."""
        result = self.data_manager.validate_dataset("unknown", "test.jsonl")
        
        assert result.passed == False
        assert "Unknown dataset type" in result.message
        assert "Unsupported dataset type: unknown" in result.errors
    
    @patch('web_interface.api.validate_amazon_dataset')
    def test_validate_dataset_exception(self, mock_validate):
        """Test validation when exception occurs."""
        mock_validate.side_effect = Exception("Validation error")
        
        result = self.data_manager.validate_dataset("amazon", "test.jsonl")
        
        assert result.passed == False
        assert "Validation failed" in result.message
        assert "Validation error" in result.errors


class TestContainerManager:
    """Test suite for ContainerManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.container_manager = ContainerManager()
    
    @patch.object(CommandExecutor, 'run_command')
    def test_build_and_push_image_success(self, mock_run):
        """Test successful image build and push."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        image_uri = self.container_manager.build_and_push_image("test-project")
        
        assert image_uri == "gcr.io/test-project/lanistr-training:latest"
        assert mock_run.call_count == 2  # build and push
    
    @patch.object(CommandExecutor, 'run_command')
    def test_build_and_push_image_build_failure(self, mock_run):
        """Test image build failure."""
        mock_run.side_effect = HTTPException(status_code=500, detail="Build failed")
        
        with pytest.raises(HTTPException) as exc_info:
            self.container_manager.build_and_push_image("test-project")
        
        assert "Failed to build/push image" in str(exc_info.value.detail)
    
    @patch.object(CommandExecutor, 'run_command')
    def test_build_and_push_image_custom_name(self, mock_run):
        """Test image build with custom name."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        image_uri = self.container_manager.build_and_push_image("test-project", "custom-image")
        
        assert image_uri == "gcr.io/test-project/custom-image:latest"


class TestJobManager:
    """Test suite for JobManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.job_manager = JobManager()
    
    @patch.object(ContainerManager, 'build_and_push_image')
    @patch.object(CommandExecutor, 'run_command')
    def test_submit_job_success(self, mock_run, mock_build):
        """Test successful job submission."""
        mock_build.return_value = "gcr.io/project/image:latest"
        mock_result = Mock()
        mock_result.stdout = "Job submitted successfully"
        mock_run.return_value = mock_result
        
        config = {
            "project_id": "test-project",
            "dataset_type": "mimic-iv",
            "region": "us-central1",
            "machine_type": "n1-standard-4"
        }
        
        result = self.job_manager.submit_job(config)
        
        assert result["status"] == "submitted"
        assert "lanistr-" in result["job_id"]
        assert result["output"] == "Job submitted successfully"
        mock_build.assert_called_once()
        mock_run.assert_called_once()
    
    @patch.object(ContainerManager, 'build_and_push_image')
    def test_submit_job_build_failure(self, mock_build):
        """Test job submission when image build fails."""
        mock_build.side_effect = HTTPException(status_code=500, detail="Build failed")
        
        config = {"project_id": "test-project", "dataset_type": "mimic-iv"}
        
        with pytest.raises(HTTPException) as exc_info:
            self.job_manager.submit_job(config)
        
        assert "Job submission failed" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    @patch.object(JobManager, 'submit_job')
    async def test_run_job_submission_success(self, mock_submit):
        """Test async job submission success."""
        from web_interface.api import jobs_db
        
        mock_submit.return_value = {
            "job_id": "test-job",
            "status": "submitted",
            "output": "Success"
        }
        
        job_config = JobConfig(
            project_id="test-project",
            dataset_type="mimic-iv",
            job_name="test-job"
        )
        
        await self.job_manager.run_job_submission(
            "test-job", job_config, "n1-standard-4", "NVIDIA_TESLA_T4", 1
        )
        
        assert "test-job" in jobs_db
        assert jobs_db["test-job"].status == "running"
        assert jobs_db["test-job"].logs == "Success"
    
    @pytest.mark.asyncio
    @patch.object(JobManager, 'submit_job')
    async def test_run_job_submission_failure(self, mock_submit):
        """Test async job submission failure."""
        from web_interface.api import jobs_db
        
        mock_submit.side_effect = Exception("Submission failed")
        
        job_config = JobConfig(
            project_id="test-project",
            dataset_type="mimic-iv",
            job_name="test-job-fail"
        )
        
        await self.job_manager.run_job_submission(
            "test-job-fail", job_config, "n1-standard-4", "NVIDIA_TESLA_T4", 1
        )
        
        assert "test-job-fail" in jobs_db
        assert jobs_db["test-job-fail"].status == "failed"
        assert "Submission failed" in jobs_db["test-job-fail"].logs


class TestWebInterfaceAPI:
    """Test suite for the web interface API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test the root API endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "LANISTR API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
    
    @patch('web_interface.api.gcp_manager')
    def test_system_status_endpoint(self, mock_gcp):
        """Test the system status endpoint."""
        mock_gcp.get_project_id.return_value = "test-project"
        mock_gcp.check_prerequisites.return_value = {"gcloud": True, "docker": False}
        mock_gcp.check_authentication.return_value = True
        mock_gcp.check_apis_enabled.return_value = {"aiplatform.googleapis.com": True}
        
        response = self.client.get("/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "test-project"
        assert data["authenticated"] == True
        assert data["prerequisites"]["gcloud"] == True
    
    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    @patch('web_interface.api.gcp_manager')
    def test_submit_training_job_no_project(self, mock_gcp):
        """Test job submission with no project configured."""
        mock_gcp.get_project_id.return_value = ""
        
        job_config = {
            "project_id": "test-project",
            "dataset_type": "mimic-iv"
        }
        
        response = self.client.post("/jobs/submit", json=job_config)
        assert response.status_code == 400
        assert "No Google Cloud project configured" in response.json()["detail"]
    
    @patch('web_interface.api.gcp_manager')
    def test_submit_training_job_not_authenticated(self, mock_gcp):
        """Test job submission when not authenticated."""
        mock_gcp.get_project_id.return_value = "test-project"
        mock_gcp.check_authentication.return_value = False
        
        job_config = {
            "project_id": "test-project",
            "dataset_type": "mimic-iv"
        }
        
        response = self.client.post("/jobs/submit", json=job_config)
        assert response.status_code == 400
        assert "Not authenticated" in response.json()["detail"]
    
    def test_list_jobs_empty(self):
        """Test listing jobs when none exist."""
        response = self.client.get("/jobs/")
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"] == []
    
    def test_get_job_not_found(self):
        """Test getting a job that doesn't exist."""
        response = self.client.get("/jobs/nonexistent")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    def test_delete_job_not_found(self):
        """Test deleting a job that doesn't exist."""
        response = self.client.delete("/jobs/nonexistent")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    @patch('web_interface.api.data_manager')
    def test_validate_dataset_endpoint(self, mock_data_manager):
        """Test dataset validation endpoint."""
        mock_result = Mock()
        mock_result.passed = True
        mock_result.message = "Validation passed"
        mock_result.stats = {"total": 100}
        mock_result.errors = []
        mock_result.warnings = []
        mock_data_manager.validate_dataset.return_value = mock_result
        
        response = self.client.post(
            "/validate/dataset",
            params={
                "dataset_type": "mimic-iv",
                "jsonl_file": "test.jsonl"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "validation_id" in data
        assert data["result"]["passed"] == True
    
    def test_get_validation_result_not_found(self):
        """Test getting validation result that doesn't exist."""
        response = self.client.get("/validate/nonexistent")
        assert response.status_code == 404
        assert "Validation not found" in response.json()["detail"]
    
    @patch('web_interface.api.gcp_manager')
    def test_setup_project_endpoint(self, mock_gcp):
        """Test project setup endpoint."""
        mock_gcp.cmd_executor.run_command.return_value = Mock()
        mock_gcp.setup_gcs_bucket.return_value = True
        
        response = self.client.post(
            "/setup/project",
            params={
                "project_id": "test-project",
                "region": "us-central1"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "test-project"
        assert data["region"] == "us-central1"
        assert "bucket_name" in data


class TestIntegrationWorkflows:
    """Integration tests for complete workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @patch('web_interface.api.gcp_manager')
    @patch('web_interface.api.job_manager')
    def test_complete_job_submission_workflow(self, mock_job_manager, mock_gcp):
        """Test complete job submission workflow."""
        # Setup mocks
        mock_gcp.get_project_id.return_value = "test-project"
        mock_gcp.check_authentication.return_value = True
        
        # Mock job submission
        async def mock_run_job_submission(*args):
            from web_interface.api import jobs_db, JobStatus
            from datetime import datetime
            job_id = args[0]
            jobs_db[job_id] = JobStatus(
                job_id=job_id,
                name=job_id,
                status="running",
                created_at=datetime.now(),
                config={"dataset_type": "mimic-iv"}
            )
        
        mock_job_manager.run_job_submission = mock_run_job_submission
        
        # Submit job
        job_config = {
            "project_id": "test-project",
            "dataset_type": "mimic-iv",
            "job_name": "test-integration-job"
        }
        
        response = self.client.post("/jobs/submit", json=job_config)
        assert response.status_code == 200
        data = response.json()
        job_id = data["job_id"]
        
        # List jobs
        response = self.client.get("/jobs/")
        assert response.status_code == 200
        jobs = response.json()["jobs"]
        assert len(jobs) >= 1
        assert any(job["job_id"] == job_id for job in jobs)
        
        # Get specific job
        response = self.client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        job_data = response.json()
        assert job_data["job_id"] == job_id
        
        # Delete job
        response = self.client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200
        
        # Verify job is deleted
        response = self.client.get(f"/jobs/{job_id}")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])