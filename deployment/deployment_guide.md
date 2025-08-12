# Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Continual Transformer system in production environments. The deployment supports multiple platforms including Docker, Kubernetes, and cloud services.

## Quick Start

### 1. Docker Deployment

```bash
# Build production image
docker build -f Dockerfile.production -t continual-transformer:latest .

# Run with GPU support
docker run -d \
  --name continual-transformer \
  --gpus all \
  -p 8000:8000 \
  -v /data/models:/data/models \
  -v /data/cache:/data/cache \
  -e API_TOKEN=your-secure-token \
  continual-transformer:latest

# Health check
curl http://localhost:8000/health
```

### 2. Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace continual-transformer

# Apply configurations
kubectl apply -f kubernetes/production-manifests.yaml

# Check deployment status
kubectl get pods -n continual-transformer
kubectl get services -n continual-transformer
```

### 3. Development Setup

```bash
# Build development image
docker build -f Dockerfile.production --target development -t continual-transformer:dev .

# Run development server
docker run -it \
  --name continual-transformer-dev \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd):/app \
  -e ENVIRONMENT=development \
  continual-transformer:dev
```

## Environment Configuration

### Required Environment Variables

```bash
# API Configuration
API_TOKEN=your-secure-api-token
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model Configuration
MODEL_NAME=bert-base-uncased
MAX_TASKS=50
DEVICE=cuda

# Storage Paths
MODEL_STORAGE_PATH=/data/models
CACHE_STORAGE_PATH=/data/cache

# Optimization Settings
ENABLE_MIXED_PRECISION=true
MEMORY_ALERT_THRESHOLD=0.85
CACHE_SIZE_GB=10

# Security Settings
ENABLE_INPUT_VALIDATION=true
ENABLE_OUTPUT_SANITIZATION=true
MAX_SEQUENCE_LENGTH=512
RATE_LIMIT_PER_MINUTE=1000

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=info
```

### Optional Environment Variables

```bash
# Database (if using persistent storage)
DATABASE_URL=postgresql://user:password@host:port/db

# Redis (for caching and background tasks)
REDIS_URL=redis://localhost:6379/0

# External APIs
HUGGINGFACE_API_TOKEN=your-hf-token
OPENAI_API_KEY=your-openai-key

# Monitoring and Logging
SENTRY_DSN=your-sentry-dsn
DATADOG_API_KEY=your-datadog-key
PROMETHEUS_GATEWAY=http://prometheus-gateway:9091
```

## Infrastructure Requirements

### Minimum System Requirements

- **CPU**: 4 cores (Intel Xeon or AMD EPYC recommended)
- **Memory**: 16GB RAM (32GB recommended)
- **GPU**: NVIDIA GPU with 8GB VRAM (V100, RTX 3080, or better)
- **Storage**: 100GB SSD for models and cache
- **Network**: 1Gbps bandwidth for API serving

### Recommended Production Setup

- **CPU**: 8+ cores with AVX-512 support
- **Memory**: 64GB RAM
- **GPU**: Multiple NVIDIA A100 or V100 GPUs
- **Storage**: 500GB NVMe SSD + 1TB network storage
- **Network**: 10Gbps with load balancer

### Kubernetes Resource Limits

```yaml
resources:
  requests:
    cpu: "2"
    memory: "8Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "4"
    memory: "16Gi"
    nvidia.com/gpu: "1"
```

## Security Configuration

### Authentication Setup

1. **API Token Authentication**:
   ```bash
   # Generate secure token
   openssl rand -hex 32
   
   # Set as environment variable
   export API_TOKEN=your-generated-token
   ```

2. **TLS/SSL Configuration**:
   ```bash
   # Generate self-signed certificate for testing
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
   
   # Use with uvicorn
   uvicorn app:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
   ```

3. **Network Security**:
   - Configure firewall rules
   - Use VPC/private networks
   - Implement rate limiting
   - Enable CORS appropriately

### Input Validation

The system includes comprehensive input validation:

- Text length limits (max 10,000 characters)
- Task ID validation (alphanumeric + underscore/hyphen)
- Batch size limits (max 100 items)
- Content safety scanning
- SQL injection prevention

### Data Protection

- All sensitive data encrypted at rest
- API tokens securely stored
- Model weights protection
- Audit logging enabled
- GDPR compliance features

## Monitoring and Observability

### Prometheus Metrics

The API exposes metrics at `/metrics`:

- `requests_total`: Total API requests
- `request_duration_seconds`: Request latency
- `active_connections`: Current connections
- `model_memory_mb`: Model memory usage
- `predictions_total`: Total predictions by task
- `errors_total`: Error counts by type

### Health Checks

- **Liveness**: `/health` - Overall system health
- **Readiness**: `/ready` - Service readiness for traffic
- **Deep Health**: Includes memory, GPU, model status

### Logging Configuration

```python
# Structured logging
import structlog

logger = structlog.get_logger()
logger.info("prediction_completed", 
           task_id="sentiment", 
           processing_time_ms=45.2,
           confidence=0.95)
```

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
- name: continual-transformer
  rules:
  - alert: HighErrorRate
    expr: rate(errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: High error rate detected
      
  - alert: HighMemoryUsage
    expr: model_memory_mb > 15000
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: Model memory usage too high
```

## Scaling and Performance

### Horizontal Scaling

1. **Multiple Replicas**:
   ```bash
   # Scale deployment
   kubectl scale deployment continual-transformer-api --replicas=5
   ```

2. **Load Balancing**:
   ```yaml
   # Kubernetes service with session affinity
   apiVersion: v1
   kind: Service
   spec:
     sessionAffinity: ClientIP
   ```

3. **Auto-scaling**:
   ```yaml
   # HPA configuration
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   spec:
     minReplicas: 3
     maxReplicas: 10
     targetCPUUtilizationPercentage: 70
   ```

### Performance Optimization

1. **Model Optimization**:
   - Enable mixed precision training
   - Use gradient checkpointing
   - Implement model quantization
   - Cache frequently used models

2. **API Optimization**:
   - Implement request batching
   - Use async processing
   - Enable response compression
   - Cache prediction results

3. **Infrastructure Optimization**:
   - Use GPU-optimized instances
   - Implement CDN for static assets
   - Use high-performance storage
   - Optimize network configuration

### Performance Tuning

```python
# FastAPI optimization
app = FastAPI(
    debug=False,
    docs_url=None,  # Disable in production
    redoc_url=None
)

# Uvicorn optimization
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    workers=4,
    loop="uvloop",
    http="httptools",
    access_log=False  # Use middleware logging instead
)
```

## Backup and Recovery

### Model Backup Strategy

1. **Automated Backups**:
   ```bash
   # Daily backup cron job
   0 2 * * * /app/scripts/backup_models.sh
   ```

2. **Backup Locations**:
   - Local storage (immediate access)
   - Cloud storage (S3, GCS, Azure)
   - Network storage (NFS, EFS)

3. **Backup Verification**:
   ```bash
   # Verify backup integrity
   python scripts/verify_backup.py --backup-path /data/backups/latest
   ```

### Disaster Recovery

1. **Recovery Procedures**:
   - Document step-by-step recovery process
   - Test recovery regularly
   - Maintain recovery runbooks
   - Train operations team

2. **Data Recovery**:
   ```bash
   # Restore from backup
   python scripts/restore_models.py --backup-path /data/backups/2024-01-15
   ```

3. **Service Recovery**:
   ```bash
   # Health check and restart
   kubectl rollout restart deployment/continual-transformer-api
   kubectl rollout status deployment/continual-transformer-api
   ```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**:
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Clear cache
   docker exec continual-transformer python -c "import torch; torch.cuda.empty_cache()"
   ```

2. **Model Loading Failures**:
   ```bash
   # Check model files
   ls -la /data/models/
   
   # Verify permissions
   docker exec continual-transformer ls -la /data/models/
   ```

3. **API Connectivity Issues**:
   ```bash
   # Check service status
   kubectl get pods -n continual-transformer
   kubectl logs -f deployment/continual-transformer-api
   
   # Test connectivity
   curl -H "Authorization: Bearer your-token" http://api-url/health
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=debug

# Run with debug
docker run -it \
  -e LOG_LEVEL=debug \
  -e ENVIRONMENT=development \
  continual-transformer:latest
```

### Log Analysis

```bash
# View application logs
kubectl logs -f deployment/continual-transformer-api

# Search for errors
kubectl logs deployment/continual-transformer-api | grep ERROR

# Monitor in real-time
kubectl logs -f deployment/continual-transformer-api --tail=100
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Production

on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Build and push Docker image
      run: |
        docker build -f Dockerfile.production -t continual-transformer:${{ github.ref_name }} .
        docker push continual-transformer:${{ github.ref_name }}
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/continual-transformer-api \
          api=continual-transformer:${{ github.ref_name }}
        kubectl rollout status deployment/continual-transformer-api
```

### Deployment Validation

```bash
# Automated deployment validation
python scripts/deployment_validation.py \
  --endpoint https://api.continual-transformer.com \
  --token $API_TOKEN \
  --timeout 300
```

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Review system metrics
   - Check log files for errors
   - Validate backup integrity
   - Update security patches

2. **Monthly**:
   - Performance optimization review
   - Capacity planning assessment
   - Security audit
   - Documentation updates

3. **Quarterly**:
   - Full system upgrade
   - Disaster recovery testing
   - Security penetration testing
   - Performance benchmarking

### Support Contacts

- **Technical Issues**: support@terragon-labs.com
- **Security Issues**: security@terragon-labs.com
- **Emergency**: +1-555-TERRAGON

### Documentation

- [API Documentation](../docs/api/)
- [Architecture Guide](../ARCHITECTURE.md)
- [Security Guidelines](../SECURITY.md)
- [Performance Tuning](../docs/performance/)

---

For additional support or questions, please refer to the project documentation or contact the Terragon Labs support team.