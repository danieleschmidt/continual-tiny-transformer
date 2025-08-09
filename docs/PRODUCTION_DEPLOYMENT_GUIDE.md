# 🚀 Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Continual Tiny Transformer in production environments with enterprise-grade security, monitoring, and scalability.

## 📋 Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional but recommended)
- **Storage**: 50GB+ SSD storage
- **Network**: Stable internet connection for model downloads

### Software Dependencies
- **Docker**: 20.10+ and Docker Compose 2.0+
- **Kubernetes**: 1.24+ (for Kubernetes deployment)
- **Python**: 3.8+ (for local development)
- **Git**: 2.30+ for version control

## 🐳 Docker Deployment

### Quick Start (Development)

```bash
# Clone repository
git clone https://github.com/your-org/continual-tiny-transformer.git
cd continual-tiny-transformer

# Build and start services
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

### Production Deployment

#### 1. Environment Setup

Create `.env` file:
```bash
# Database Configuration
POSTGRES_PASSWORD=your-secure-password-here
DATABASE_URL=postgresql://postgres:your-password@postgres:5432/continual_transformer

# Redis Configuration
REDIS_PASSWORD=your-redis-password-here
REDIS_URL=redis://:your-redis-password@redis:6379/0

# Security
JWT_SECRET=your-jwt-secret-key-here
API_KEY=your-api-key-here

# Monitoring
GRAFANA_PASSWORD=your-grafana-password-here
SECURITY_WEBHOOK_URL=https://your-webhook-url.com

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
```

#### 2. SSL Certificates

```bash
# Create SSL directory
mkdir -p nginx/ssl

# Generate self-signed certificates (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/server.key \
  -out nginx/ssl/server.crt

# Or copy your production certificates
cp your-ssl-cert.crt nginx/ssl/server.crt
cp your-ssl-key.key nginx/ssl/server.key
```

#### 3. Production Configuration

```bash
# Use production compose file
docker-compose -f docker-compose.production.yml up -d

# Verify all services
docker-compose -f docker-compose.production.yml ps
```

#### 4. Initial Setup

```bash
# Run database migrations
docker-compose exec continual-transformer-app python manage.py migrate

# Create admin user
docker-compose exec continual-transformer-app python manage.py createsuperuser

# Load initial data
docker-compose exec continual-transformer-app python manage.py loaddata initial_data.json
```

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- NVIDIA GPU Operator (for GPU support)
- Ingress controller (nginx)
- Cert-manager (for SSL)

### 1. Cluster Preparation

```bash
# Install NVIDIA GPU Operator (if using GPUs)
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator

# Install cert-manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### 2. Storage Setup

```bash
# Create storage classes
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: none
allowVolumeExpansion: true
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: shared-storage
provisioner: nfs-subdir-external-provisioner
parameters:
  archiveOnDelete: "false"
allowVolumeExpansion: true
EOF
```

### 3. Deploy Application

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/production-deployment.yaml

# Wait for deployment
kubectl rollout status deployment/continual-transformer-app -n continual-transformer

# Check pod status
kubectl get pods -n continual-transformer
```

### 4. Configure Ingress

```bash
# Update ingress hostname
sed -i 's/api.continual-transformer.example.com/your-domain.com/g' \
  kubernetes/production-deployment.yaml

# Apply updated configuration
kubectl apply -f kubernetes/production-deployment.yaml
```

### 5. Verify Deployment

```bash
# Check all resources
kubectl get all -n continual-transformer

# Test API endpoint
curl https://your-domain.com/health

# Check logs
kubectl logs -f deployment/continual-transformer-app -n continual-transformer
```

## 📊 Monitoring Setup

### Prometheus & Grafana

Access monitoring dashboards:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/password from .env)

### Custom Metrics

The application exposes metrics at `/metrics` endpoint:
- Request count and duration
- Memory and CPU usage
- Model inference metrics
- Task performance metrics

### Log Aggregation

Logs are aggregated using ELK stack:
- **Elasticsearch**: http://localhost:9200
- **Kibana**: http://localhost:5601

## 🔐 Security Configuration

### 1. API Security

```python
# Configure authentication in settings
SECURITY_SETTINGS = {
    "enable_api_key_auth": True,
    "enable_jwt_auth": True,
    "rate_limit_requests": 1000,  # per hour
    "max_request_size": "10MB",
    "allowed_origins": ["https://your-domain.com"],
}
```

### 2. Network Security

```bash
# Configure firewall rules
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 5432/tcp  # Block direct database access
ufw deny 6379/tcp  # Block direct Redis access
```

### 3. Input Validation

The application includes comprehensive input validation:
- SQL injection protection
- XSS prevention
- Input sanitization
- Rate limiting

## 🚀 Scaling Configuration

### Horizontal Scaling

#### Docker Swarm
```bash
# Initialize swarm mode
docker swarm init

# Scale services
docker service scale continual-transformer_app=5
```

#### Kubernetes Auto-scaling
```yaml
# HPA configuration (already included in manifests)
spec:
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling

```yaml
# Update resource limits
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
    nvidia.com/gpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4000m"
    nvidia.com/gpu: "2"
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v3
      with:
        push: true
        tags: ${{ secrets.REGISTRY }}/continual-transformer:${{ github.sha }}
    
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          kubernetes/production-deployment.yaml
        images: |
          ${{ secrets.REGISTRY }}/continual-transformer:${{ github.sha }}
```

## 🔧 Performance Optimization

### 1. Model Optimization

```python
# Enable optimization features
api = ContinualLearningAPI(
    model_name="distilbert-base-uncased",
    device="cuda"  # Use GPU
)

# Optimize for deployment
api.optimize_for_deployment("speed")
```

### 2. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_task_performance ON task_metrics(task_id, created_at);
CREATE INDEX idx_user_sessions ON user_sessions(user_id, expires_at);
```

### 3. Caching Strategy

```python
# Redis caching configuration
CACHE_CONFIG = {
    "model_cache_ttl": 3600,  # 1 hour
    "prediction_cache_ttl": 300,  # 5 minutes
    "metadata_cache_ttl": 1800,  # 30 minutes
}
```

## 🆘 Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```bash
# Check memory usage
docker stats

# Increase memory limits
docker-compose up -d --scale app=2
```

#### 2. GPU Not Detected
```bash
# Verify GPU availability
nvidia-smi

# Check Docker GPU support
docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

#### 3. Database Connection Issues
```bash
# Check database logs
docker-compose logs postgres

# Test connection
docker-compose exec app python -c "import psycopg2; print('DB OK')"
```

#### 4. High Latency
```bash
# Check Prometheus metrics
curl http://localhost:9090/metrics | grep response_time

# Enable performance profiling
export ENABLE_PROFILING=true
```

### Log Analysis

```bash
# Application logs
docker-compose logs -f continual-transformer-app

# Database logs
docker-compose logs -f postgres

# Redis logs  
docker-compose logs -f redis

# Nginx access logs
docker-compose logs -f nginx
```

## 📈 Performance Monitoring

### Key Metrics to Monitor

1. **Response Time**: < 200ms for simple requests
2. **Throughput**: > 1000 requests/second
3. **Error Rate**: < 0.1%
4. **Memory Usage**: < 80% of allocated
5. **GPU Utilization**: > 70% when active

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
- name: continual-transformer
  rules:
  - alert: HighResponseTime
    expr: avg(response_time_seconds) > 1.0
    for: 5m
    
  - alert: HighErrorRate
    expr: rate(http_errors_total[5m]) > 0.01
    for: 2m
    
  - alert: HighMemoryUsage
    expr: memory_usage_percent > 90
    for: 5m
```

## 🔒 Backup & Recovery

### Database Backup

```bash
# Automated daily backup
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec postgres pg_dump -U postgres continual_transformer \
  > backups/db_backup_$DATE.sql

# Keep only last 7 days
find backups/ -name "db_backup_*.sql" -mtime +7 -delete
```

### Model Backup

```bash
# Backup trained models
docker-compose exec app python manage.py backup_models \
  --output /backups/models_$(date +%Y%m%d).tar.gz
```

### Disaster Recovery

```bash
# Restore database
docker-compose exec postgres psql -U postgres < backups/db_backup_latest.sql

# Restore models
docker-compose exec app python manage.py restore_models \
  --input /backups/models_latest.tar.gz
```

## 📞 Support

For production support and issues:

1. **Documentation**: Check this guide and API docs
2. **Monitoring**: Review Grafana dashboards and logs
3. **Community**: GitHub Issues and Discussions
4. **Enterprise**: Contact support team for SLA-backed assistance

## 🎯 Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database initialized and migrated
- [ ] Monitoring and alerting setup
- [ ] Backup strategy implemented
- [ ] Security scanning completed
- [ ] Load testing performed
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Incident response plan ready