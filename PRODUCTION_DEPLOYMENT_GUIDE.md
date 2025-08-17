# 🚀 Production Deployment Guide

**continual-tiny-transformer v0.1.0**  
**Zero-Parameter Continual Learning Framework**

## 🎯 Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+
- Docker (optional)
- Kubernetes (optional)

### Installation
```bash
# 1. Clone and setup
git clone <repository-url>
cd continual-tiny-transformer

# 2. Install dependencies
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -e .

# 3. Verify installation
python -c "from continual_transformer import ContinualTransformer; print('✅ Ready')"
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t continual-transformer:latest .
```

### Run Container
```bash
# API Service
docker run -d \
  --name continual-transformer \
  -p 8000:8000 \
  -e WORKERS=4 \
  -e LOG_LEVEL=INFO \
  continual-transformer:latest

# Development Mode
docker run -it \
  -v $(pwd):/workspace \
  continual-transformer:latest \
  bash
```

## ☸️ Kubernetes Deployment

### Apply Manifests
```bash
kubectl apply -f kubernetes/production-deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
```

### Monitor Deployment
```bash
kubectl get pods -l app=continual-transformer
kubectl logs -f deployment/continual-transformer
```

## 🔧 Configuration Options

### Environment Variables
```bash
# Core Settings
CONTINUAL_MODEL_NAME=distilbert-base-uncased
CONTINUAL_MAX_TASKS=50
CONTINUAL_DEVICE=auto
CONTINUAL_CACHE_DIR=/tmp/models

# Performance Settings
CONTINUAL_MIXED_PRECISION=true
CONTINUAL_ENABLE_MONITORING=true
CONTINUAL_BATCH_SIZE=32
CONTINUAL_MAX_SEQUENCE_LENGTH=512

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=30

# Monitoring Settings
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
HEALTH_CHECK_INTERVAL=30
```

### Configuration File
```yaml
# config/production.yaml
model:
  name: "distilbert-base-uncased"
  max_tasks: 50
  device: "auto"
  freeze_base_model: true

optimization:
  mixed_precision: true
  gradient_checkpointing: false
  enable_compilation: true

monitoring:
  enabled: true
  prometheus_port: 9090
  health_checks: true
  metrics_interval: 10

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30
```

## 📊 Monitoring & Observability

### Health Check Endpoint
```bash
curl http://localhost:8000/health
```

### Prometheus Metrics
```bash
# Available at http://localhost:9090/metrics
continual_transformer_tasks_total
continual_transformer_inference_duration_seconds
continual_transformer_memory_usage_bytes
continual_transformer_error_rate
```

### Grafana Dashboard
Import dashboard from `monitoring/grafana-dashboard.json`

## 🔒 Security Configuration

### API Security
```bash
# Enable authentication
export API_AUTH_ENABLED=true
export API_SECRET_KEY="your-secret-key"

# Rate limiting
export API_RATE_LIMIT="100/minute"
export API_RATE_LIMIT_BURST=20
```

### Container Security
```dockerfile
# Run as non-root user
USER 1000:1000

# Read-only filesystem
--read-only --tmpfs /tmp

# Security options
--security-opt=no-new-privileges:true
```

## 🚀 API Usage Examples

### Basic Classification
```python
import requests

# Register a task
response = requests.post("http://localhost:8000/tasks", json={
    "task_id": "sentiment_analysis",
    "task_type": "classification",
    "num_labels": 3
})

# Make predictions
response = requests.post("http://localhost:8000/predict", json={
    "task_id": "sentiment_analysis",
    "text": "This product is amazing!"
})

print(response.json())
# {"predictions": [2], "probabilities": [[0.1, 0.2, 0.7]], "task_id": "sentiment_analysis"}
```

### Batch Processing
```python
# Batch predictions
response = requests.post("http://localhost:8000/predict/batch", json={
    "task_id": "sentiment_analysis",
    "texts": [
        "Great product!",
        "Terrible experience.",
        "It's okay."
    ]
})
```

### Model Management
```python
# Get model status
response = requests.get("http://localhost:8000/status")

# List tasks
response = requests.get("http://localhost:8000/tasks")

# Get performance metrics
response = requests.get("http://localhost:8000/metrics")
```

## 📈 Performance Tuning

### CPU Optimization
```yaml
# Increase workers for CPU
workers: 8
threads_per_worker: 2

# Enable torch.compile
torch_compile: true
operator_fusion: true
```

### GPU Optimization
```yaml
# GPU settings
device: "cuda:0"
mixed_precision: true
gradient_checkpointing: true

# Batch optimization
batch_size: 64
max_sequence_length: 256
```

### Memory Optimization
```yaml
# Memory settings
gradient_accumulation_steps: 4
dataloader_num_workers: 4
pin_memory: true
prefetch_factor: 2
```

## 🔄 Scaling Strategies

### Horizontal Scaling
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: continual-transformer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: continual-transformer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Load Balancing
```bash
# NGINX configuration
upstream continual_transformer {
    server continual-transformer-1:8000;
    server continual-transformer-2:8000;
    server continual-transformer-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://continual_transformer;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🚨 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check model cache
ls -la /tmp/models/

# Clear cache and retry
rm -rf /tmp/models/*
```

#### Memory Issues
```bash
# Monitor memory usage
docker stats continual-transformer

# Adjust memory limits
docker run --memory=4g continual-transformer:latest
```

#### Performance Issues
```bash
# Check system resources
top -p $(pgrep -f continual-transformer)

# Enable performance profiling
export CONTINUAL_PROFILING=true
```

### Logs Analysis
```bash
# API logs
docker logs continual-transformer

# Application logs
tail -f logs/continual-transformer.log

# Error tracking
grep "ERROR" logs/*.log | tail -20
```

## 📋 Maintenance

### Backup Strategy
```bash
# Backup model states
tar -czf backup-$(date +%Y%m%d).tar.gz data/models/

# Backup configuration
cp config/production.yaml backups/
```

### Updates
```bash
# Rolling update
kubectl set image deployment/continual-transformer \
  continual-transformer=continual-transformer:v0.2.0

# Health check during update
kubectl rollout status deployment/continual-transformer
```

### Monitoring Alerts
```yaml
# Prometheus alerting rules
groups:
- name: continual-transformer
  rules:
  - alert: HighErrorRate
    expr: continual_transformer_error_rate > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
  
  - alert: HighMemoryUsage
    expr: continual_transformer_memory_usage_bytes > 2e9
    for: 2m
    annotations:
      summary: "Memory usage exceeding 2GB"
```

## 🎯 Best Practices

### Development
- Use virtual environments for isolation
- Pin dependency versions in requirements.txt
- Implement comprehensive testing
- Use configuration files over environment variables

### Production
- Enable monitoring and logging
- Implement health checks
- Use container orchestration
- Set up automated backups
- Configure alerting rules

### Security
- Run containers as non-root
- Use secrets management
- Enable API authentication
- Implement rate limiting
- Regular security updates

## 📞 Support

### Resources
- **Documentation**: `/docs` directory
- **Examples**: `/examples` directory  
- **API Reference**: `/docs/api`
- **Troubleshooting**: `/docs/troubleshooting.md`

### Community
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Contributions**: See CONTRIBUTING.md

---

**🚀 Ready for Production!**

*Your continual learning framework is now deployed and ready to scale.*