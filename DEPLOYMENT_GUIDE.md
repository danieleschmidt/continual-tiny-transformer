# 🚀 Production Deployment Guide

**Continual Tiny Transformer - Zero-Parameter Continual Learning**

This guide provides step-by-step instructions for deploying the continual-tiny-transformer in production environments.

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (recommended: 3.10+)
- **Memory**: Minimum 8GB RAM (recommended: 16GB+)
- **Storage**: 10GB+ available space
- **GPU**: Optional but recommended (CUDA 11.8+)
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+

### Dependencies
- PyTorch 1.12.0+
- Transformers 4.20.0+
- NumPy, SciPy, scikit-learn
- See `requirements.txt` for complete list

---

## 🏗️ Installation Guide

### 1. Clone and Setup Repository
```bash
# Clone repository
git clone https://github.com/your-org/continual-tiny-transformer.git
cd continual-tiny-transformer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### 2. Verify Installation
```bash
# Test CLI installation
continual-transformer --help

# Run basic functionality test
python examples/quick_demo.py

# Run health checks
python -c "
from continual_transformer.monitoring.health import RobustHealthMonitor
monitor = RobustHealthMonitor()
health = monitor.safe_health_check()
print(f'System Status: {health[\"status\"]}')
"
```

### 3. Configuration Setup
```bash
# Create production configuration
cp config/default.yaml config/production.yaml

# Edit production settings
nano config/production.yaml
```

**Key Production Settings:**
```yaml
# Production Configuration
model_name: "distilbert-base-uncased"
device: "auto"  # Will auto-detect CUDA if available
mixed_precision: true
gradient_checkpointing: true

# Security
log_level: "WARNING"
disable_debug: true

# Performance
batch_size: 32
max_sequence_length: 512
num_workers: 4

# Monitoring
health_check_interval: 60
enable_metrics: true
export_metrics: true
```

---

## 🐳 Docker Deployment

### 1. Build Docker Image
```bash
# Build production image
docker build -f Dockerfile -t continual-transformer:latest .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.10 -t continual-transformer:3.10 .
```

### 2. Run Container
```bash
# Basic run
docker run -p 8000:8000 continual-transformer:latest

# Production run with volumes and environment
docker run -d \
  --name continual-transformer \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  -e PYTHONPATH=/app \
  -e LOG_LEVEL=INFO \
  --restart unless-stopped \
  continual-transformer:latest
```

### 3. Docker Compose (Recommended)
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  continual-transformer:
    image: continual-transformer:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
      - CONFIG_PATH=/app/config/production.yaml
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "from continual_transformer.monitoring.health import RobustHealthMonitor; print(RobustHealthMonitor().safe_health_check()['status'])"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  monitoring:
    image: continual-transformer:latest
    command: ["python", "-m", "continual_transformer.monitoring.health", "--server"]
    ports:
      - "8001:8001"
    depends_on:
      - continual-transformer
    restart: unless-stopped
```

Run with:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## ☸️ Kubernetes Deployment

### 1. Create Namespace
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: continual-transformer
```

### 2. ConfigMap for Configuration
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: continual-transformer-config
  namespace: continual-transformer
data:
  production.yaml: |
    model_name: "distilbert-base-uncased"
    device: "auto"
    mixed_precision: true
    batch_size: 32
    log_level: "INFO"
    health_check_interval: 60
```

### 3. Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: continual-transformer
  namespace: continual-transformer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: continual-transformer
  template:
    metadata:
      labels:
        app: continual-transformer
    spec:
      containers:
      - name: continual-transformer
        image: continual-transformer:latest
        ports:
        - containerPort: 8000
        env:
        - name: CONFIG_PATH
          value: "/app/config/production.yaml"
        volumeMounts:
        - name: config
          mountPath: /app/config
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "8Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: continual-transformer-config
```

### 4. Service
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: continual-transformer-service
  namespace: continual-transformer
spec:
  selector:
    app: continual-transformer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 5. Deploy to Kubernetes
```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check deployment status
kubectl get pods -n continual-transformer
kubectl get services -n continual-transformer
```

---

## 🔄 CI/CD Pipeline Setup

### 1. GitHub Actions Workflow
Copy the pre-configured workflows:
```bash
# Copy workflow templates to active directory
mkdir -p .github/workflows
cp docs/workflows/ci-complete.yml .github/workflows/ci.yml
cp docs/workflows/security-complete.yml .github/workflows/security.yml
cp docs/workflows/release-complete.yml .github/workflows/release.yml
```

### 2. Configure GitHub Secrets
Add the following secrets in GitHub repository settings:

**Required Secrets:**
- `PYPI_API_TOKEN`: For publishing to PyPI
- `DOCKER_HUB_USERNAME`: Docker Hub username
- `DOCKER_HUB_TOKEN`: Docker Hub access token

**Optional Secrets:**
- `CODECOV_TOKEN`: For coverage reporting
- `SLACK_WEBHOOK`: For deployment notifications

### 3. Repository Settings
Enable in GitHub repository settings:
- Branch protection rules
- Required status checks
- Security scanning
- Dependabot alerts

---

## 📊 Monitoring & Observability

### 1. Health Monitoring Setup
```python
# monitoring_server.py
from continual_transformer.monitoring.health import RobustHealthMonitor
from flask import Flask, jsonify
import json

app = Flask(__name__)
monitor = RobustHealthMonitor()

@app.route('/health')
def health_check():
    health = monitor.safe_health_check()
    return jsonify(health)

@app.route('/metrics')
def metrics():
    performance = monitor.get_performance_summary()
    return jsonify(performance)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
```

### 2. Logging Configuration
```python
# logging_config.py
import logging
import logging.handlers
import json
from pathlib import Path

def setup_production_logging(log_level="INFO", log_file="app.log"):
    """Setup production logging configuration."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure formatters
    json_formatter = logging.Formatter(
        json.dumps({
            "timestamp": "%(asctime)s",
            "level": "%(levelname)s",
            "logger": "%(name)s",
            "message": "%(message)s",
            "module": "%(module)s",
            "function": "%(funcName)s",
            "line": "%(lineno)d"
        })
    )
    
    # Configure handlers
    handlers = []
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(json_formatter)
    handlers.append(file_handler)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True
    )
```

### 3. Metrics Collection
```python
# Setup Prometheus metrics (optional)
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
task_counter = Counter('continual_tasks_total', 'Total tasks processed')
inference_duration = Histogram('inference_duration_seconds', 'Inference duration')
memory_usage = Gauge('memory_usage_bytes', 'Current memory usage')

# Start metrics server
start_http_server(8002)
```

---

## 🔐 Security Configuration

### 1. Environment Variables
```bash
# .env.production
PYTHONPATH=/app
LOG_LEVEL=WARNING
CONFIG_PATH=/app/config/production.yaml
SECURITY_SCAN_ENABLED=true
HEALTH_CHECK_TOKEN=your-secure-token-here
API_KEY_ENCRYPTION_KEY=your-encryption-key-here
```

### 2. Security Hardening
```bash
# security_setup.sh
#!/bin/bash

# Create non-root user for container
useradd -m -u 1000 appuser

# Set proper file permissions
chmod 755 /app
chmod 644 /app/config/*
chmod 600 /app/config/secrets/*

# Setup firewall rules (if applicable)
ufw allow 8000/tcp
ufw allow 8001/tcp
ufw enable

# Setup log rotation
cat > /etc/logrotate.d/continual-transformer << EOF
/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF
```

### 3. SSL/TLS Configuration
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/private.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://localhost:8001/health;
        access_log off;
    }
}
```

---

## 📈 Performance Optimization

### 1. Production Optimizations
```python
# production_config.py
PRODUCTION_OPTIMIZATIONS = {
    # Model optimizations
    "mixed_precision": True,
    "gradient_checkpointing": True,
    "compile_model": True,
    
    # Memory optimizations
    "adaptive_batch_sizing": True,
    "memory_cleanup_interval": 100,
    "cache_size": 10000,
    
    # Performance optimizations
    "parallel_workers": 4,
    "prefetch_factor": 2,
    "pin_memory": True,
    
    # I/O optimizations
    "async_loading": True,
    "buffer_size": 8192,
}
```

### 2. Resource Allocation
```bash
# Set resource limits
ulimit -n 65536  # Increase file descriptor limit
export OMP_NUM_THREADS=4  # Optimize OpenMP threads
export MKL_NUM_THREADS=4  # Optimize Intel MKL threads

# CUDA optimizations (if applicable)
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
```

### 3. Database Configuration (if applicable)
```python
# database_config.py
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "echo": False,  # Disable SQL logging in production
}
```

---

## 🧪 Testing in Production

### 1. Smoke Tests
```bash
#!/bin/bash
# smoke_test.sh

echo "Running production smoke tests..."

# Test CLI functionality
continual-transformer info --model-path ./models/test_model.pt
if [ $? -ne 0 ]; then
    echo "❌ CLI test failed"
    exit 1
fi

# Test health endpoint
curl -f http://localhost:8001/health || exit 1

# Test basic inference
python -c "
from continual_transformer import ContinualTransformer, ContinualConfig
config = ContinualConfig(device='cpu')
model = ContinualTransformer(config)
print('✅ Basic functionality test passed')
"

echo "✅ All smoke tests passed"
```

### 2. Load Testing
```python
# load_test.py
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def test_inference_load(session, url, data):
    """Test single inference request."""
    async with session.post(url, json=data) as response:
        return await response.json()

async def run_load_test(num_requests=1000, concurrency=50):
    """Run load test with specified parameters."""
    url = "http://localhost:8000/predict"
    data = {
        "text": "This is a test message for load testing",
        "task_id": "sentiment"
    }
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request():
            async with semaphore:
                return await test_inference_load(session, url, data)
        
        tasks = [bounded_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Load test completed:")
    print(f"  Requests: {num_requests}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  RPS: {num_requests/duration:.2f}")
    print(f"  Avg latency: {duration/num_requests*1000:.2f}ms")

if __name__ == "__main__":
    asyncio.run(run_load_test())
```

---

## 🔄 Maintenance & Updates

### 1. Regular Maintenance Tasks
```bash
#!/bin/bash
# maintenance.sh

# Update dependencies
pip list --outdated
pip install --upgrade pip

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# Clear temporary files
find /tmp -name "pytorch_*" -mtime +7 -delete

# Health check
python -c "
from continual_transformer.monitoring.health import RobustHealthMonitor
monitor = RobustHealthMonitor()
report = monitor.get_performance_summary()
print(f'System health: {report}')
"

# Security scan
python src/continual_transformer/security/scanner.py . security_report_$(date +%Y%m%d).json
```

### 2. Backup Strategy
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup models
tar -czf $BACKUP_DIR/models.tar.gz models/

# Backup configurations
cp -r config/ $BACKUP_DIR/config/

# Backup logs (last 7 days)
find logs/ -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/ \;

echo "Backup completed: $BACKUP_DIR"
```

### 3. Update Process
```bash
#!/bin/bash
# update.sh

echo "Starting update process..."

# Create backup
./backup.sh

# Stop services
docker-compose -f docker-compose.prod.yml down

# Pull latest code
git pull origin main

# Update dependencies
pip install --upgrade -e .

# Run tests
python -m pytest tests/ --tb=short

# Security scan
python src/continual_transformer/security/scanner.py . latest_security_scan.json

# Restart services
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
./smoke_test.sh

echo "Update completed successfully"
```

---

## 📞 Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Clear PyTorch cache
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
```

#### 2. GPU Issues
```bash
# Check GPU status
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Fix CUDA out of memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### 3. Performance Issues
```bash
# Profile application
python -m cProfile -o profile.prof your_script.py
python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"

# Check resource utilization
htop
iotop
```

### Support Contacts
- **Technical Issues**: Create issue on GitHub repository
- **Security Concerns**: Email security@your-domain.com
- **Performance Optimization**: Contact DevOps team

---

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Code reviewed and approved
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Configuration validated
- [ ] Backup strategy in place

### Deployment
- [ ] Environment prepared
- [ ] Dependencies installed
- [ ] Configuration deployed
- [ ] Services started
- [ ] Health checks passing
- [ ] Smoke tests completed

### Post-Deployment
- [ ] Monitoring active
- [ ] Logs flowing
- [ ] Performance metrics baseline established
- [ ] Load testing completed
- [ ] Rollback plan tested
- [ ] Team notified

---

## 🏆 Success Metrics

Monitor these metrics for successful production deployment:

### Performance Metrics
- **Response Time**: < 200ms for inference
- **Throughput**: > 100 requests/second
- **Memory Usage**: < 80% of available RAM
- **CPU Usage**: < 70% average
- **Error Rate**: < 0.1%

### Business Metrics
- **Uptime**: > 99.9%
- **Task Success Rate**: > 99%
- **Model Accuracy**: Maintained across tasks
- **User Satisfaction**: Positive feedback
- **Cost Efficiency**: Within budget targets

---

**🎯 Production Deployment Guide Complete**

This guide provides comprehensive instructions for deploying continual-tiny-transformer in production environments. Follow the steps carefully and customize configurations based on your specific requirements.

For additional support, refer to the troubleshooting section or contact the development team.