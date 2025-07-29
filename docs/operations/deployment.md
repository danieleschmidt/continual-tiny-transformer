# Production Deployment Guide

## Overview

Comprehensive deployment guide for the continual-tiny-transformer in production environments, covering containerization, orchestration, scaling, and operational best practices.

## Container Deployment

### 1. Production Dockerfile

#### Multi-stage Build
```dockerfile
# Production Dockerfile with multi-stage build
FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./
RUN pip install -e .

# Production stage
FROM python:3.9-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /app && chown appuser:appuser /app
USER appuser

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml ./

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Default command
CMD ["python", "-m", "continual_transformer.server"]
```

#### Production Requirements
```txt
# requirements-prod.txt
# Core dependencies from requirements.txt
torch>=1.12.0
transformers>=4.20.0
numpy>=1.21.0

# Production-specific additions
gunicorn>=20.1.0
uvicorn[standard]>=0.18.0
prometheus-client>=0.14.0
structlog>=22.1.0
psutil>=5.9.0

# Monitoring and observability
opentelemetry-api>=1.12.0
opentelemetry-sdk>=1.12.0
opentelemetry-instrumentation>=0.33b0

# Security
cryptography>=3.4.0
```

### 2. Docker Compose for Local Production Testing

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  continual-transformer:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - LOG_LEVEL=info
      - PROMETHEUS_PORT=8000
    volumes:
      - model_data:/app/models
      - logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    networks:
      - app-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - app-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - app-network

volumes:
  model_data:
  logs:
  prometheus_data:
  grafana_data:

networks:
  app-network:
    driver: bridge
```

## Kubernetes Deployment

### 1. Base Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: continual-transformer
  labels:
    app: continual-transformer
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: continual-transformer
  template:
    metadata:
      labels:
        app: continual-transformer
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: continual-transformer
      containers:
      - name: continual-transformer
        image: continual-transformer:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENV
          value: "production"
        - name: LOG_LEVEL
          value: "info"
        - name: KUBERNETES_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      - name: config
        configMap:
          name: continual-transformer-config
      nodeSelector:
        accelerator: nvidia-tesla-v100
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
```

### 2. Service and Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: continual-transformer-service
  labels:
    app: continual-transformer
spec:
  selector:
    app: continual-transformer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: continual-transformer-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.continual-transformer.com
    secretName: continual-transformer-tls
  rules:
  - host: api.continual-transformer.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: continual-transformer-service
            port:
              number: 80
```

### 3. Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: continual-transformer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: continual-transformer
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

## Cloud Deployment

### 1. AWS EKS Deployment

```yaml
# aws/cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: continual-transformer-cluster
  region: us-west-2
  version: "1.24"

managedNodeGroups:
  - name: cpu-nodes
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    
  - name: gpu-nodes
    instanceType: p3.2xlarge
    desiredCapacity: 2
    minSize: 1
    maxSize: 5
    labels:
      accelerator: nvidia-tesla-v100
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule

addons:
  - name: aws-load-balancer-controller
  - name: cluster-autoscaler
  - name: aws-for-fluent-bit
  - name: aws-ebs-csi-driver

cloudWatch:
  clusterLogging:
    enableTypes: ["*"]
```

### 2. GCP GKE Deployment

```yaml
# gcp/cluster.yaml
apiVersion: container.v1
kind: Cluster
metadata:
  name: continual-transformer-cluster
spec:
  location: us-central1
  initialNodeCount: 3
  
  nodePools:
  - name: cpu-pool
    config:
      machineType: n1-standard-4
      diskSizeGb: 100
      preemptible: false
    autoscaling:
      enabled: true
      minNodeCount: 2
      maxNodeCount: 10
      
  - name: gpu-pool
    config:
      machineType: n1-standard-4
      diskSizeGb: 100
      accelerators:
      - acceleratorCount: 1
        acceleratorType: nvidia-tesla-v100
    autoscaling:
      enabled: true
      minNodeCount: 1
      maxNodeCount: 5
      
  addonsConfig:
    horizontalPodAutoscaling:
      disabled: false
    httpLoadBalancing:
      disabled: false
    networkPolicyConfig:
      disabled: false
```

## Scaling Strategies

### 1. Vertical Scaling
```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: continual-transformer-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: continual-transformer
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: continual-transformer
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 500m
        memory: 1Gi
```

### 2. Custom Metrics Autoscaling
```python
# scripts/custom-metrics-adapter.py
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect

class CustomMetricsAdapter:
    def __init__(self):
        config.load_incluster_config()
        self.k8s_client = client.ApiClient()
        self.prometheus = PrometheusConnect(url="http://prometheus:9090")
        
    def get_inference_queue_depth(self):
        query = 'inference_queue_depth'
        result = self.prometheus.custom_query(query)
        return float(result[0]['value'][1])
        
    def get_model_load_average(self):
        query = 'avg(rate(model_load_duration_seconds[5m]))'
        result = self.prometheus.custom_query(query)
        return float(result[0]['value'][1])
```

## Deployment Automation

### 1. Helm Chart
```yaml
# helm/Chart.yaml
apiVersion: v2
name: continual-transformer
description: Helm chart for continual-tiny-transformer
type: application
version: 1.0.0
appVersion: "1.0.0"

dependencies:
- name: prometheus
  version: "15.x.x"
  repository: "https://prometheus-community.github.io/helm-charts"
  condition: prometheus.enabled
  
- name: grafana
  version: "6.x.x" 
  repository: "https://grafana.github.io/helm-charts"
  condition: grafana.enabled
```

```yaml
# helm/values.yaml
replicaCount: 3

image:
  repository: continual-transformer
  pullPolicy: IfNotPresent
  tag: "1.0.0"

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
  - host: api.continual-transformer.com
    paths:
    - path: /
      pathType: Prefix
  tls:
  - secretName: continual-transformer-tls
    hosts:
    - api.continual-transformer.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 1000m
    memory: 2Gi
    nvidia.com/gpu: 1

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

prometheus:
  enabled: true
  
grafana:
  enabled: true
```

### 2. GitOps with ArgoCD
```yaml
# argocd/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: continual-transformer
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/continual-tiny-transformer
    targetRevision: HEAD
    path: helm
    helm:
      valueFiles:
      - values-production.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

## Blue/Green Deployment

### 1. Blue/Green Service
```yaml
# k8s/blue-green.yaml
apiVersion: v1
kind: Service
metadata:
  name: continual-transformer-active
  labels:
    app: continual-transformer
spec:
  selector:
    app: continual-transformer
    version: blue  # Switch between blue/green
  ports:
  - port: 80
    targetPort: 8000

---
apiVersion: v1
kind: Service
metadata:
  name: continual-transformer-preview
  labels:
    app: continual-transformer
spec:
  selector:
    app: continual-transformer
    version: green
  ports:
  - port: 80
    targetPort: 8000
```

### 2. Deployment Script
```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

CURRENT_VERSION=$(kubectl get service continual-transformer-active -o jsonpath='{.spec.selector.version}')
NEW_VERSION="green"
if [[ "$CURRENT_VERSION" == "green" ]]; then
    NEW_VERSION="blue"
fi

echo "Current version: $CURRENT_VERSION"
echo "Deploying version: $NEW_VERSION"

# Deploy new version
kubectl apply -f k8s/deployment-${NEW_VERSION}.yaml

# Wait for deployment to be ready
kubectl rollout status deployment/continual-transformer-${NEW_VERSION}

# Run health checks
./scripts/health-check.sh continual-transformer-preview

# Switch traffic
kubectl patch service continual-transformer-active -p '{"spec":{"selector":{"version":"'${NEW_VERSION}'"}}}'

echo "Traffic switched to $NEW_VERSION"
```

## Operational Best Practices

### 1. Resource Management
- Use resource requests and limits
- Implement node affinity for GPU workloads
- Use PodDisruptionBudgets for availability

### 2. Security
- Run containers as non-root user
- Use network policies for pod-to-pod communication
- Implement RBAC for service accounts
- Scan images for vulnerabilities

### 3. Monitoring
- Implement comprehensive health checks
- Monitor GPU utilization and memory
- Track inference latency and throughput
- Set up alerting for anomalies

### 4. Backup and Recovery
- Regular backup of model weights and checkpoints
- Database backup for configuration and metadata
- Disaster recovery procedures documented

This deployment guide provides production-ready configurations for scaling the continual-tiny-transformer across different cloud providers and orchestration platforms.