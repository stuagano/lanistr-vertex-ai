#!/usr/bin/env python3
"""
LANISTR Test Suite Runner
This script runs all tests and generates comprehensive reports.
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

def print_header(message):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80)

def print_step(step, message):
    """Print a formatted step message."""
    print(f"\n[{step}] {message}")
    print("-" * 60)

def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"Running: {command}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=check
        )
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ {description} completed in {duration:.2f}s")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return e

def install_test_dependencies():
    """Install test dependencies."""
    print_step(1, "Installing test dependencies")
    
    requirements_file = "tests/requirements-test.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ Test requirements file not found: {requirements_file}")
        return False
    
    result = run_command(
        f"pip install -r {requirements_file}",
        "Installing test dependencies"
    )
    return result.returncode == 0

def run_unit_tests():
    """Run unit tests."""
    print_step(2, "Running unit tests")
    
    # Run tests with coverage
    result = run_command(
        "python -m pytest tests/ -v --cov=lanistr --cov=web_interface --cov=utils --cov-report=html --cov-report=term-missing",
        "Unit tests with coverage"
    )
    return result.returncode == 0

def run_integration_tests():
    """Run integration tests."""
    print_step(3, "Running integration tests")
    
    result = run_command(
        "python -m pytest tests/ -v -m integration",
        "Integration tests"
    )
    return result.returncode == 0

def run_performance_tests():
    """Run performance tests."""
    print_step(4, "Running performance tests")
    
    result = run_command(
        "python -m pytest tests/ -v -m performance",
        "Performance tests"
    )
    return result.returncode == 0

def run_web_interface_tests():
    """Run web interface specific tests."""
    print_step(5, "Running web interface tests")
    
    result = run_command(
        "python -m pytest tests/test_web_interface.py -v",
        "Web interface tests"
    )
    return result.returncode == 0

def run_data_validator_tests():
    """Run data validator specific tests."""
    print_step(6, "Running data validator tests")
    
    result = run_command(
        "python -m pytest tests/test_data_validator.py -v",
        "Data validator tests"
    )
    return result.returncode == 0

def run_training_pipeline_tests():
    """Run training pipeline specific tests."""
    print_step(7, "Running training pipeline tests")
    
    result = run_command(
        "python -m pytest tests/test_training_pipeline.py -v",
        "Training pipeline tests"
    )
    return result.returncode == 0

def run_setup_script_tests():
    """Run setup script specific tests."""
    print_step(8, "Running setup script tests")
    
    result = run_command(
        "python -m pytest tests/test_setup_scripts.py -v",
        "Setup script tests"
    )
    return result.returncode == 0

def generate_test_report():
    """Generate comprehensive test report."""
    print_step(9, "Generating test report")
    
    # Generate HTML report
    result = run_command(
        "python -m pytest tests/ --html=test_reports/report.html --self-contained-html",
        "HTML test report"
    )
    
    # Generate JSON report
    result2 = run_command(
        "python -m pytest tests/ --json-report --json-report-file=test_reports/report.json",
        "JSON test report"
    )
    
    return result.returncode == 0 and result2.returncode == 0

def check_code_quality():
    """Run code quality checks."""
    print_step(10, "Running code quality checks")
    
    # Check for syntax errors
    result = run_command(
        "python -m py_compile lanistr/main.py lanistr/trainer.py web_interface/api.py web_interface/app.py lanistr/utils/data_validator.py lanistr/utils/common_utils.py lanistr/utils/data_utils.py lanistr/utils/model_utils.py lanistr/utils/parallelism_utils.py",
        "Syntax check"
    )
    
    return result.returncode == 0

def create_test_summary():
    """Create a summary of test results."""
    print_step(11, "Creating test summary")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "LANISTR Comprehensive Test Suite",
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "coverage": 0.0,
        "performance_benchmarks": {},
        "recommendations": []
    }
    
    # Try to read coverage report
    coverage_file = "htmlcov/index.html"
    if os.path.exists(coverage_file):
        summary["coverage_report_available"] = True
    
    # Try to read test reports
    if os.path.exists("test_reports/report.json"):
        try:
            with open("test_reports/report.json", 'r') as f:
                report_data = json.load(f)
                summary["total_tests"] = report_data.get("summary", {}).get("total", 0)
                summary["passed"] = report_data.get("summary", {}).get("passed", 0)
                summary["failed"] = report_data.get("summary", {}).get("failed", 0)
        except:
            pass
    
    # Save summary
    with open("test_reports/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Test summary saved to test_reports/summary.json")
    return summary

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="LANISTR Test Suite Runner")
    parser.add_argument("--skip-install", action="store_true", help="Skip installing test dependencies")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--web-only", action="store_true", help="Run only web interface tests")
    parser.add_argument("--validator-only", action="store_true", help="Run only data validator tests")
    parser.add_argument("--pipeline-only", action="store_true", help="Run only training pipeline tests")
    parser.add_argument("--setup-only", action="store_true", help="Run only setup script tests")
    parser.add_argument("--no-report", action="store_true", help="Skip generating test reports")
    parser.add_argument("--no-quality", action="store_true", help="Skip code quality checks")
    
    args = parser.parse_args()
    
    print_header("LANISTR Comprehensive Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create test reports directory
    os.makedirs("test_reports", exist_ok=True)
    
    # Track test results
    test_results = {}
    
    # Install dependencies (unless skipped)
    if not args.skip_install:
        test_results["dependencies"] = install_test_dependencies()
        if not test_results["dependencies"]:
            print("❌ Failed to install test dependencies. Exiting.")
            return 1
    
    # Run specific test categories based on arguments
    if args.unit_only:
        test_results["unit_tests"] = run_unit_tests()
    elif args.integration_only:
        test_results["integration_tests"] = run_integration_tests()
    elif args.performance_only:
        test_results["performance_tests"] = run_performance_tests()
    elif args.web_only:
        test_results["web_interface_tests"] = run_web_interface_tests()
    elif args.validator_only:
        test_results["data_validator_tests"] = run_data_validator_tests()
    elif args.pipeline_only:
        test_results["training_pipeline_tests"] = run_training_pipeline_tests()
    elif args.setup_only:
        test_results["setup_script_tests"] = run_setup_script_tests()
    else:
        # Run all tests
        test_results["unit_tests"] = run_unit_tests()
        test_results["integration_tests"] = run_integration_tests()
        test_results["performance_tests"] = run_performance_tests()
        test_results["web_interface_tests"] = run_web_interface_tests()
        test_results["data_validator_tests"] = run_data_validator_tests()
        test_results["training_pipeline_tests"] = run_training_pipeline_tests()
        test_results["setup_script_tests"] = run_setup_script_tests()
    
    # Generate reports (unless skipped)
    if not args.no_report:
        test_results["reports"] = generate_test_report()
    
    # Code quality checks (unless skipped)
    if not args.no_quality:
        test_results["code_quality"] = check_code_quality()
    
    # Create summary
    summary = create_test_summary()
    
    # Print final results
    print_header("Test Results Summary")
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    print(f"Tests completed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if summary.get("total_tests", 0) > 0:
        print(f"\nTest execution summary:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success rate: {(summary['passed']/summary['total_tests'])*100:.1f}%")
    
    print(f"\nReports generated:")
    print(f"  - HTML coverage report: htmlcov/index.html")
    print(f"  - HTML test report: test_reports/report.html")
    print(f"  - JSON test report: test_reports/report.json")
    print(f"  - Test summary: test_reports/summary.json")
    
    # Return appropriate exit code
    if all(test_results.values()):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the reports for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 