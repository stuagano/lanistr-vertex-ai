# LANISTR Requirements Guide

This document explains the different requirements files available for LANISTR deployment on Google Cloud Vertex AI and how to use them for enterprise-ready deployments.

## Requirements Files Overview

### 1. `requirements_vertex_ai.txt` - Full Enterprise Requirements
**Use for**: Complete enterprise deployment with all features
- **Purpose**: Comprehensive requirements file with all enterprise features, security tools, monitoring, and compliance
- **Includes**: Core ML frameworks, GCP integration, security tools, testing frameworks, monitoring, and enterprise features
- **Size**: ~80+ packages
- **Installation**: `pip install -r requirements_vertex_ai.txt`

### 2. `requirements-prod.txt` - Production Requirements
**Use for**: Minimal production deployment
- **Purpose**: Essential dependencies only for production deployment
- **Includes**: Core ML frameworks, GCP integration, basic monitoring, and security
- **Size**: ~30 packages
- **Installation**: `pip install -r requirements-prod.txt`

### 3. `requirements-dev.txt` - Development Requirements
**Use for**: Development environment setup
- **Purpose**: Includes production requirements plus development tools
- **Includes**: All production dependencies plus testing, linting, debugging, and development tools
- **Size**: ~120+ packages
- **Installation**: `pip install -r requirements-dev.txt`

## Enterprise Features Added

### Security and Compliance
- **Security Scanning**: `bandit`, `safety` for vulnerability detection
- **Encryption**: `cryptography` for secure communication
- **Certificate Management**: `certifi` for SSL/TLS certificates
- **Code Security**: `semgrep`, `trufflehog` for secret detection

### Monitoring and Observability
- **Structured Logging**: `structlog` for production-ready logging
- **Metrics Collection**: `prometheus-client` for monitoring
- **Health Checks**: `psutil`, `GPUtil` for system monitoring
- **Error Reporting**: `google-cloud-error-reporting` for GCP integration
- **Application Performance**: `sentry-sdk` for APM

### Testing and Quality Assurance
- **Testing Framework**: `pytest` with coverage and mocking
- **Code Quality**: `black`, `flake8`, `isort`, `mypy` for code formatting and type checking
- **Property-based Testing**: `hypothesis` for robust testing
- **Performance Testing**: `pytest-benchmark` for performance validation

### Performance and Optimization
- **Memory Profiling**: `memory-profiler`, `line-profiler` for performance analysis
- **Distributed Training**: `horovod` for multi-GPU training
- **Model Serving**: `torchserve`, `mlflow` for model deployment

### Data Validation and Sanitization
- **Data Validation**: `great-expectations`, `pandera` for data quality
- **Configuration Validation**: `pydantic`, `jsonschema` for config validation

### Enterprise Integration
- **API Development**: `fastapi`, `uvicorn` for REST APIs
- **Database Integration**: `sqlalchemy`, `psycopg2-binary` for database connectivity
- **Infrastructure**: Support for Terraform, Ansible, Kubernetes

## Installation Guide

### For Production Deployment

```bash
# Install minimal production requirements
pip install -r requirements-prod.txt

# Or install full enterprise requirements
pip install -r requirements_vertex_ai.txt
```

### For Development Environment

```bash
# Install development requirements (includes production)
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### For Docker Deployment

```dockerfile
# Use production requirements in Dockerfile
COPY requirements-prod.txt /app/
RUN pip install -r requirements-prod.txt
```

## Security Best Practices

### Regular Security Scanning

```bash
# Scan for vulnerabilities
safety check -r requirements-prod.txt

# Scan code for security issues
bandit -r lanistr/ -f json -o security_report.json

# Scan for secrets in code
trufflehog --json . > secrets_report.json
```

### Dependency Management

```bash
# Update dependencies safely
pip install --upgrade --upgrade-strategy only-if-needed -r requirements-prod.txt

# Check for outdated packages
pip list --outdated

# Audit dependencies
pip-audit -r requirements-prod.txt
```

## Monitoring and Logging Setup

### Structured Logging Configuration

```python
import structlog
import logging

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
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
TRAINING_STEPS = Counter('lanistr_training_steps_total', 'Total training steps')
TRAINING_TIME = Histogram('lanistr_training_duration_seconds', 'Training duration')

# Start metrics server
start_http_server(8000)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Security and Quality Checks

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Security scan
        run: |
          safety check -r requirements-prod.txt
          bandit -r lanistr/ -f json -o security_report.json
      - name: Upload security report
        uses: actions/upload-artifact@v3
        with:
          name: security-report
          path: security_report.json
```

## Version Compatibility

### Python Version Requirements
- **Minimum**: Python 3.8
- **Maximum**: Python 3.10 (for torch 1.11.0 compatibility)
- **Recommended**: Python 3.9

### CUDA Requirements
- **CUDA Version**: 11.3
- **cuDNN Version**: 8.x
- **GPU Support**: NVIDIA GPUs (V100, P100, T4, A100)

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Install correct PyTorch version
   pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   python -m memory_profiler your_script.py
   
   # Reduce batch size in config
   ```

3. **Dependency Conflicts**
   ```bash
   # Create clean environment
   python -m venv lanistr_env
   source lanistr_env/bin/activate
   pip install -r requirements-prod.txt
   ```

### Performance Optimization

1. **Enable JIT Compilation**
   ```python
   import torch
   torch.jit.enable_onednn_fusion(True)
   ```

2. **Optimize Data Loading**
   ```python
   # Use multiple workers for data loading
   dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)
   ```

3. **Memory Optimization**
   ```python
   # Use mixed precision training
   from torch.cuda.amp import autocast, GradScaler
   ```

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Update security patches
2. **Monthly**: Review and update dependencies
3. **Quarterly**: Performance benchmarking
4. **Annually**: Major version upgrades

### Monitoring Checklist

- [ ] Application logs are being collected
- [ ] Metrics are being exported to monitoring system
- [ ] Error reporting is configured
- [ ] Security scans are automated
- [ ] Backup and recovery procedures are tested
- [ ] Performance baselines are established

## License

This requirements setup follows the same Apache 2.0 license as the original LANISTR project. 