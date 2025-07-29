# Observability and Monitoring Setup

## Overview

Comprehensive observability strategy for the continual-tiny-transformer project, covering metrics, logging, tracing, and alerting for ML workloads in production.

## Core Observability Stack

### 1. Metrics Collection

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'continual-transformer'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 10s
    
  - job_name: 'gpu-metrics'
    static_configs:
      - targets: ['localhost:9400']
```

#### Application Metrics
```python
# src/continual_transformer/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
import torch

# Training metrics
training_loss = Gauge('training_loss', 'Current training loss')
training_accuracy = Gauge('training_accuracy', 'Current training accuracy')
tasks_learned = Counter('tasks_learned_total', 'Total number of tasks learned')

# Performance metrics
inference_duration = Histogram('inference_duration_seconds', 
                             'Time spent on inference',
                             buckets=[0.001, 0.01, 0.1, 1.0, 10.0])
                             
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes',
                    ['device'])

# Model metrics
model_parameters = Gauge('model_parameters_total', 'Total model parameters')
knowledge_retention = Gauge('knowledge_retention_score', 
                          'Knowledge retention score for task',
                          ['task_id'])

class ModelMetrics:
    def __init__(self):
        self.start_time = time.time()
        
    def record_training_step(self, loss: float, accuracy: float):
        training_loss.set(loss)
        training_accuracy.set(accuracy)
        
    def record_task_completion(self, task_id: str, retention_score: float):
        tasks_learned.inc()
        knowledge_retention.labels(task_id=task_id).set(retention_score)
        
    def record_inference(self, duration: float):
        inference_duration.observe(duration)
        
    def update_memory_usage(self):
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            memory_usage.labels(device='gpu').set(gpu_memory)
```

### 2. Distributed Tracing

#### OpenTelemetry Setup
```python
# src/continual_transformer/monitoring/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.auto_instrumentation import sitecustomize

def setup_tracing():
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return tracer

# Usage in training loop
tracer = setup_tracing()

def train_task(task_id: str, data):
    with tracer.start_as_current_span("train_task") as span:
        span.set_attribute("task.id", task_id)
        span.set_attribute("data.size", len(data))
        
        with tracer.start_as_current_span("data_preprocessing"):
            preprocessed_data = preprocess(data)
            
        with tracer.start_as_current_span("model_training"):
            model_output = train_model(preprocessed_data)
            span.set_attribute("training.loss", model_output.loss)
```

### 3. Structured Logging

#### Logging Configuration
```python
# src/continual_transformer/monitoring/logging.py
import logging
import json
from datetime import datetime
import structlog

def setup_logging():
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

class MLLogger:
    def __init__(self):
        self.logger = structlog.get_logger()
        
    def log_training_start(self, task_id: str, config: dict):
        self.logger.info("Training started",
                        event_type="training_start",
                        task_id=task_id,
                        config=config)
                        
    def log_training_progress(self, task_id: str, epoch: int, 
                            loss: float, accuracy: float):
        self.logger.info("Training progress",
                        event_type="training_progress", 
                        task_id=task_id,
                        epoch=epoch,
                        loss=loss,
                        accuracy=accuracy)
                        
    def log_model_performance(self, task_id: str, metrics: dict):
        self.logger.info("Model performance",
                        event_type="model_performance",
                        task_id=task_id,
                        metrics=metrics)
```

## Production Monitoring Setup

### 1. Grafana Dashboards

#### Training Dashboard
```json
{
  "dashboard": {
    "title": "Continual Learning Training",
    "panels": [
      {
        "title": "Training Loss",
        "type": "graph",
        "targets": [
          {
            "expr": "training_loss",
            "legendFormat": "Training Loss"
          }
        ]
      },
      {
        "title": "Knowledge Retention",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(knowledge_retention_score)",
            "legendFormat": "Avg Retention"
          }
        ]
      },
      {
        "title": "Tasks Learned",
        "type": "counter",
        "targets": [
          {
            "expr": "tasks_learned_total",
            "legendFormat": "Total Tasks"
          }
        ]
      }
    ]
  }
}
```

#### Infrastructure Dashboard
```json
{
  "dashboard": {
    "title": "ML Infrastructure",
    "panels": [
      {
        "title": "GPU Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "memory_usage_bytes{device=\"gpu\"}",
            "legendFormat": "GPU Memory"
          }
        ]
      },
      {
        "title": "Inference Latency",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(inference_duration_seconds_bucket[5m])",
            "legendFormat": "{{le}}"
          }
        ]
      }
    ]
  }
}
```

### 2. Alert Rules

#### Prometheus Alert Rules
```yaml
# alert_rules.yml
groups:
  - name: continual_learning_alerts
    rules:
      - alert: HighTrainingLoss
        expr: training_loss > 2.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Training loss is unusually high"
          description: "Training loss has been above 2.0 for more than 5 minutes"
          
      - alert: LowKnowledgeRetention
        expr: avg(knowledge_retention_score) < 0.8
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Knowledge retention below threshold"
          description: "Average knowledge retention is {{ $value }}, below 0.8 threshold"
          
      - alert: GPUMemoryHigh
        expr: memory_usage_bytes{device="gpu"} > 0.9 * gpu_memory_total
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage is high"
          description: "GPU memory usage is at {{ $value | humanize }}%"
          
      - alert: InferenceLatencyHigh
        expr: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "95th percentile latency is {{ $value }}s"
```

### 3. Alerting Integration

#### Slack Notifications
```yaml
# alertmanager.yml
global:
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    slack_configs:
      - channel: '#ml-alerts'
        title: 'Continual Transformer Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## ML-Specific Monitoring

### 1. Data Drift Detection
```python
# src/continual_transformer/monitoring/drift.py
import numpy as np
from scipy import stats
from prometheus_client import Gauge

data_drift_score = Gauge('data_drift_score', 'Data drift detection score',
                        ['task_id'])

class DriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        
    def detect_drift(self, new_data, task_id: str):
        # KS test for distribution drift
        ks_stat, p_value = stats.ks_2samp(
            self.reference_data.flatten(),
            new_data.flatten()
        )
        
        drift_score = 1 - p_value  # Higher score = more drift
        data_drift_score.labels(task_id=task_id).set(drift_score)
        
        return drift_score > 0.05  # Threshold for significant drift
```

### 2. Model Performance Tracking
```python
# src/continual_transformer/monitoring/performance.py
import mlflow
from typing import Dict, Any

class PerformanceTracker:
    def __init__(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        
    def log_experiment(self, task_id: str, metrics: Dict[str, Any]):
        with mlflow.start_run(run_name=f"task_{task_id}"):
            mlflow.log_params({
                "task_id": task_id,
                "timestamp": datetime.now().isoformat()
            })
            
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
                
    def compare_tasks(self, task_ids: list):
        """Compare performance across multiple tasks"""
        experiments = []
        for task_id in task_ids:
            runs = mlflow.search_runs(
                filter_string=f"params.task_id = '{task_id}'"
            )
            experiments.append(runs)
        return experiments
```

## Container Monitoring

### 1. Docker Health Checks
```dockerfile
# Add to Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"
```

### 2. Kubernetes Monitoring
```yaml
# monitoring/k8s-monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
```

## Best Practices

### 1. Metric Naming
- Use consistent prefixes: `continual_transformer_*`
- Include units in names: `_seconds`, `_bytes`, `_total`
- Use labels for dimensions, not metric names

### 2. Alert Design
- Define clear SLIs/SLOs for ML workloads
- Avoid alert fatigue with proper grouping
- Include runbooks in alert annotations

### 3. Dashboard Organization
- Separate operational and business metrics
- Use consistent time ranges and refresh rates
- Include context and documentation

### 4. Cost Optimization
- Monitor resource usage and costs
- Use sampling for high-cardinality metrics
- Implement retention policies for historical data

## Integration Scripts

### Monitoring Setup Script
```bash
#!/bin/bash
# scripts/setup-monitoring.sh

# Start Prometheus
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Start Grafana
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -v grafana-storage:/var/lib/grafana \
  grafana/grafana

# Start Jaeger
docker run -d \
  --name jaeger \
  -p 16686:16686 \
  -p 14268:14268 \
  jaegertracing/all-in-one:latest

echo "Monitoring stack started:"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000"
echo "- Jaeger: http://localhost:16686"
```

This observability setup provides comprehensive monitoring for ML workloads with focus on training metrics, model performance, and operational health.