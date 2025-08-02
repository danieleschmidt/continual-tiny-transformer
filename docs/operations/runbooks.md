# Operational Runbooks

## Overview

This document contains step-by-step procedures for common operational scenarios with the continual-tiny-transformer system.

## Table of Contents

1. [Deployment Procedures](#deployment-procedures)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Incident Response](#incident-response)
4. [Maintenance Tasks](#maintenance-tasks)
5. [Backup and Recovery](#backup-and-recovery)
6. [Performance Tuning](#performance-tuning)

## Deployment Procedures

### 1. Production Deployment

#### Prerequisites
- [ ] Docker and Docker Compose installed
- [ ] Environment variables configured
- [ ] Health checks passing in staging
- [ ] Monitoring stack deployed

#### Steps

1. **Prepare Environment**
   ```bash
   # Set production environment variables
   export ENVIRONMENT=production
   export LOG_LEVEL=INFO
   export ENABLE_METRICS=true
   
   # Verify configuration
   make check-config
   ```

2. **Deploy Application**
   ```bash
   # Pull latest images
   docker-compose pull
   
   # Deploy with zero downtime
   docker-compose up -d
   
   # Verify deployment
   docker-compose ps
   curl http://localhost:8000/health
   ```

3. **Post-Deployment Verification**
   ```bash
   # Check application logs
   docker-compose logs -f app
   
   # Verify metrics endpoint
   curl http://localhost:8000/metrics
   
   # Run smoke tests
   make test-smoke
   ```

#### Rollback Procedure
```bash
# Quick rollback to previous version
docker-compose down
docker-compose -f docker-compose.yml -f docker-compose.rollback.yml up -d

# Verify rollback
curl http://localhost:8000/health
```

### 2. GPU Environment Deployment

#### Prerequisites
- [ ] NVIDIA Docker runtime installed
- [ ] GPU drivers compatible with CUDA 11.8
- [ ] GPU memory available (minimum 8GB)

#### Steps

1. **Deploy GPU-enabled Stack**
   ```bash
   # Deploy GPU service
   docker-compose --profile gpu up -d
   
   # Verify GPU access
   docker exec continual-transformer-gpu nvidia-smi
   
   # Check CUDA availability
   docker exec continual-transformer-gpu python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Monitor GPU Usage**
   ```bash
   # Start GPU monitoring
   docker-compose --profile monitoring up -d prometheus grafana
   
   # Access GPU dashboard
   # http://localhost:3000 (Grafana)
   ```

## Monitoring and Alerting

### 1. Health Check Monitoring

#### Manual Health Checks
```bash
# System health
curl http://localhost:8000/health/system

# Model health
curl http://localhost:8000/health/model

# Comprehensive health check
curl http://localhost:8000/health/all
```

#### Expected Responses
- **Healthy**: HTTP 200, `{"status": "healthy", "checks": {...}}`
- **Degraded**: HTTP 200, `{"status": "degraded", "issues": [...]}`
- **Unhealthy**: HTTP 503, `{"status": "unhealthy", "errors": [...]}`

### 2. Metrics Monitoring

#### Key Metrics to Monitor

| Metric | Threshold | Action |
|--------|-----------|--------|
| `continual_transformer_training_loss` | > 2.0 | Investigate training |
| `continual_transformer_memory_usage_bytes{device="gpu"}` | > 90% | Scale GPU resources |
| `continual_transformer_inference_duration_seconds` | P95 > 1.0s | Performance optimization |
| `continual_transformer_knowledge_retention_score` | < 0.8 | Review model architecture |

#### Prometheus Queries
```promql
# Average training loss over 5 minutes
avg_over_time(continual_transformer_training_loss[5m])

# GPU memory usage percentage
(continual_transformer_memory_usage_bytes{device=~"gpu_.*",type="allocated"} / 
 continual_transformer_memory_usage_bytes{device=~"gpu_.*",type="total"}) * 100

# Inference latency P95
histogram_quantile(0.95, rate(continual_transformer_inference_duration_seconds_bucket[5m]))
```

### 3. Log Analysis

#### Important Log Patterns
```bash
# Training errors
docker-compose logs app | grep "ERROR.*training"

# Memory issues
docker-compose logs app | grep "CUDA out of memory"

# Performance issues
docker-compose logs app | grep "slow.*inference"

# Model convergence issues
docker-compose logs app | grep "loss.*diverged"
```

#### Log Aggregation with ELK Stack
```bash
# Start ELK stack for log aggregation
docker-compose --profile logging up -d elasticsearch logstash kibana

# Access Kibana dashboard
# http://localhost:5601
```

## Incident Response

### 1. High Memory Usage

#### Symptoms
- `continual_transformer_memory_usage_bytes` > 90%
- "CUDA out of memory" errors in logs
- Training failures or crashes

#### Investigation Steps
1. **Check Memory Usage**
   ```bash
   # Current GPU memory
   nvidia-smi
   
   # Application memory metrics
   curl http://localhost:8000/metrics | grep memory_usage
   
   # Container memory usage
   docker stats continual-transformer-app
   ```

2. **Identify Root Cause**
   ```bash
   # Check for memory leaks
   docker-compose logs app | grep -i "memory\|leak\|oom"
   
   # Review recent training jobs
   curl http://localhost:8000/api/jobs/recent
   ```

3. **Immediate Actions**
   ```bash
   # Reduce batch size temporarily
   curl -X POST http://localhost:8000/api/config/batch_size -d '{"value": 16}'
   
   # Clear GPU memory
   curl -X POST http://localhost:8000/api/system/clear_gpu_cache
   
   # Restart service if necessary
   docker-compose restart app
   ```

### 2. Training Convergence Issues

#### Symptoms
- `continual_transformer_training_loss` not decreasing
- Accuracy plateau or degradation
- Knowledge retention score dropping

#### Investigation Steps
1. **Review Training Metrics**
   ```bash
   # Check training history
   curl http://localhost:8000/api/training/history
   
   # Review learning rate schedule
   curl http://localhost:8000/api/training/lr_schedule
   ```

2. **Analyze Data Quality**
   ```bash
   # Check data distribution
   python scripts/analyze_data_quality.py
   
   # Validate training data
   make validate-data
   ```

3. **Adjust Training Parameters**
   ```bash
   # Reduce learning rate
   curl -X POST http://localhost:8000/api/config/learning_rate -d '{"value": 1e-5}'
   
   # Adjust regularization
   curl -X POST http://localhost:8000/api/config/weight_decay -d '{"value": 0.01}'
   ```

### 3. Performance Degradation

#### Symptoms
- Inference latency > 1 second (P95)
- Training time significantly increased
- High CPU/GPU utilization

#### Investigation Steps
1. **Profile Performance**
   ```bash
   # CPU profiling
   python -m cProfile -s cumulative scripts/profile_inference.py
   
   # GPU profiling
   nsys profile --output=profile.nsys python scripts/profile_training.py
   ```

2. **Check Resource Utilization**
   ```bash
   # System resources
   htop
   
   # GPU utilization
   nvidia-smi -l 1
   
   # Application metrics
   curl http://localhost:8000/metrics | grep duration
   ```

3. **Optimization Actions**
   ```bash
   # Enable mixed precision training
   curl -X POST http://localhost:8000/api/config/mixed_precision -d '{"enabled": true}'
   
   # Adjust batch size for optimal throughput
   python scripts/optimize_batch_size.py
   ```

## Maintenance Tasks

### 1. Regular Maintenance Schedule

#### Daily Tasks
- [ ] Check system health status
- [ ] Review error logs
- [ ] Monitor disk space usage
- [ ] Verify backup completion

#### Weekly Tasks
- [ ] Update dependency vulnerabilities
- [ ] Clean old checkpoints and logs
- [ ] Review performance metrics trends
- [ ] Test disaster recovery procedures

#### Monthly Tasks
- [ ] Security audit and updates
- [ ] Capacity planning review
- [ ] Update documentation
- [ ] Performance optimization review

### 2. Maintenance Scripts

#### System Cleanup
```bash
#!/bin/bash
# scripts/maintenance/cleanup.sh

# Clean old Docker images
docker image prune -f

# Clean old logs (keep last 30 days)
find /var/log/continual-transformer -name "*.log" -mtime +30 -delete

# Clean old checkpoints (keep last 10)
python scripts/cleanup_checkpoints.py --keep=10

# Clean GPU cache
python -c "import torch; torch.cuda.empty_cache()"

echo "Cleanup completed at $(date)"
```

#### Health Check Automation
```bash
#!/bin/bash
# scripts/maintenance/health_check.sh

# Run comprehensive health check
curl -f http://localhost:8000/health/all || {
    echo "Health check failed at $(date)"
    # Send alert to monitoring system
    curl -X POST $WEBHOOK_URL -d '{"text": "Health check failed"}'
    exit 1
}

echo "Health check passed at $(date)"
```

### 3. Configuration Management

#### Environment Configuration
```bash
# Backup current configuration
cp .env .env.backup.$(date +%Y%m%d)

# Update configuration
# Edit .env file with new settings

# Validate configuration
make validate-config

# Apply configuration (rolling update)
docker-compose up -d --no-deps app
```

#### Model Configuration Updates
```bash
# Update model configuration
curl -X PUT http://localhost:8000/api/config/model -d @new_model_config.json

# Verify configuration
curl http://localhost:8000/api/config/model

# Restart model if needed
curl -X POST http://localhost:8000/api/model/reload
```

## Backup and Recovery

### 1. Backup Procedures

#### Model Checkpoints
```bash
#!/bin/bash
# scripts/backup/backup_models.sh

BACKUP_DIR="/backup/models/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup model checkpoints
cp -r /app/checkpoints/* $BACKUP_DIR/

# Backup configuration
cp /app/.env $BACKUP_DIR/config.env

# Create manifest
echo "Backup created at $(date)" > $BACKUP_DIR/manifest.txt
echo "Models: $(ls -la $BACKUP_DIR)" >> $BACKUP_DIR/manifest.txt

# Upload to remote storage (if configured)
if [ ! -z "$S3_BUCKET" ]; then
    aws s3 sync $BACKUP_DIR s3://$S3_BUCKET/backups/$(date +%Y%m%d)/
fi
```

#### Database Backup
```bash
#!/bin/bash
# scripts/backup/backup_database.sh

# Backup PostgreSQL (if using)
docker exec continual-transformer-postgres pg_dump -U app continual_transformer > \
    /backup/db/continual_transformer_$(date +%Y%m%d_%H%M%S).sql
```

### 2. Recovery Procedures

#### Model Recovery
```bash
#!/bin/bash
# scripts/recovery/restore_model.sh

BACKUP_DATE=$1
BACKUP_DIR="/backup/models/$BACKUP_DATE"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup directory not found: $BACKUP_DIR"
    exit 1
fi

# Stop application
docker-compose stop app

# Restore checkpoints
rm -rf /app/checkpoints/*
cp -r $BACKUP_DIR/* /app/checkpoints/

# Restore configuration
cp $BACKUP_DIR/config.env /app/.env

# Start application
docker-compose start app

echo "Model restored from backup: $BACKUP_DATE"
```

#### Disaster Recovery
```bash
#!/bin/bash
# scripts/recovery/disaster_recovery.sh

echo "Starting disaster recovery procedure..."

# 1. Restore from latest backup
LATEST_BACKUP=$(ls -1t /backup/models/ | head -1)
scripts/recovery/restore_model.sh $LATEST_BACKUP

# 2. Verify system health
scripts/maintenance/health_check.sh

# 3. Run validation tests
make test-validation

# 4. Notify stakeholders
echo "Disaster recovery completed at $(date)"
```

## Performance Tuning

### 1. Training Performance

#### GPU Optimization
```python
# Optimize GPU memory usage
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Optimize data loading
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

#### Memory Optimization
```python
# Gradient accumulation to simulate larger batches
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    with autocast():
        loss = model(batch) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 2. Inference Performance

#### Model Optimization
```python
# Compile model for faster inference (PyTorch 2.0+)
model = torch.compile(model)

# Use TensorRT for production inference
import torch_tensorrt
trt_model = torch_tensorrt.compile(model, inputs=sample_inputs)
```

#### Batching Strategies
```python
# Dynamic batching for inference
from collections import defaultdict

class DynamicBatcher:
    def __init__(self, max_batch_size=32, timeout_ms=10):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests = []
    
    def add_request(self, request):
        self.pending_requests.append(request)
        
        if len(self.pending_requests) >= self.max_batch_size:
            return self.process_batch()
        
        return None  # Wait for more requests or timeout
    
    def process_batch(self):
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        return batch
```

### 3. System Performance

#### Resource Monitoring
```bash
# Monitor system resources
watch -n 1 'nvidia-smi; echo ""; free -h; echo ""; df -h'

# Profile application
python -m py_spy top --pid $(pgrep -f continual_transformer)
```

#### Optimization Checklist
- [ ] GPU memory utilization > 80%
- [ ] CPU utilization balanced across cores
- [ ] I/O wait time < 10%
- [ ] Network latency < 100ms
- [ ] Memory usage stable (no leaks)
- [ ] Inference latency P95 < 500ms
- [ ] Training throughput > X samples/second

This runbook should be reviewed and updated regularly based on operational experience and system changes.