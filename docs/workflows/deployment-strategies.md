# Deployment Strategies and Workflows

## Overview

This document outlines comprehensive deployment strategies for the continual-tiny-transformer project, covering different environments, deployment patterns, and automation workflows.

## Deployment Environments

### 1. Environment Hierarchy

```
Development → Testing → Staging → Production
     ↓           ↓        ↓         ↓
   Feature    Integration Preview   Live
   Testing     Testing    Testing   System
```

### 2. Environment Configuration

#### Development Environment
```yaml
# config/environments/development.yml
environment: development
api:
  host: localhost
  port: 8000
  workers: 1
  reload: true

database:
  url: sqlite:///dev.db
  echo: true

logging:
  level: DEBUG
  format: text

monitoring:
  enabled: false

security:
  auth_required: false
  cors_origins: ["*"]

model:
  cache_size: 100MB
  batch_size: 16
  device: auto
```

#### Production Environment
```yaml
# config/environments/production.yml
environment: production
api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: false

database:
  url: ${DATABASE_URL}
  pool_size: 20
  max_overflow: 30

logging:
  level: INFO
  format: json

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30

security:
  auth_required: true
  cors_origins: ${ALLOWED_ORIGINS}
  rate_limiting: true

model:
  cache_size: 1GB
  batch_size: 32
  device: cuda
```

## Deployment Strategies

### 1. Blue-Green Deployment

#### Workflow Template
```yaml
# .github/workflows/blue-green-deployment.yml
name: Blue-Green Deployment

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: continual-tiny-transformer

jobs:
  build:
    name: Build and Test
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=sha,prefix={{branch}}-

      - name: Build and push Docker image
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run security scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ steps.meta.outputs.tags }}
          format: 'sarif'
          output: 'trivy-results.sarif'

  deploy-green:
    name: Deploy to Green Environment
    runs-on: ubuntu-latest
    needs: build
    environment: ${{ github.event.inputs.environment || 'staging' }}
    steps:
      - name: Checkout deployment scripts
        uses: actions/checkout@v4

      - name: Set up kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure kubectl
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBECONFIG }}" | base64 -d > ~/.kube/config

      - name: Deploy to green environment
        run: |
          # Update green deployment with new image
          kubectl set image deployment/continual-transformer-green \
            app=${{ needs.build.outputs.image-tag }} \
            -n ${{ github.event.inputs.environment || 'staging' }}
          
          # Wait for rollout to complete
          kubectl rollout status deployment/continual-transformer-green \
            -n ${{ github.event.inputs.environment || 'staging' }} \
            --timeout=600s

      - name: Run health checks
        run: |
          # Wait for green environment to be ready
          kubectl wait --for=condition=ready pod \
            -l app=continual-transformer,slot=green \
            -n ${{ github.event.inputs.environment || 'staging' }} \
            --timeout=300s
          
          # Get green service endpoint
          GREEN_URL=$(kubectl get service continual-transformer-green \
            -n ${{ github.event.inputs.environment || 'staging' }} \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          
          # Run health checks
          curl -f http://$GREEN_URL:8000/health/all || exit 1

      - name: Run smoke tests
        run: |
          GREEN_URL=$(kubectl get service continual-transformer-green \
            -n ${{ github.event.inputs.environment || 'staging' }} \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          
          # Run comprehensive smoke tests
          python scripts/smoke_tests.py --url http://$GREEN_URL:8000

  switch-traffic:
    name: Switch Traffic to Green
    runs-on: ubuntu-latest
    needs: [build, deploy-green]
    environment: ${{ github.event.inputs.environment || 'staging' }}
    if: success()
    steps:
      - name: Update load balancer
        run: |
          # Switch traffic from blue to green
          kubectl patch service continual-transformer \
            -p '{"spec":{"selector":{"slot":"green"}}}' \
            -n ${{ github.event.inputs.environment || 'staging' }}

      - name: Verify traffic switch
        run: |
          # Verify traffic is going to green
          SERVICE_URL=$(kubectl get service continual-transformer \
            -n ${{ github.event.inputs.environment || 'staging' }} \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          
          # Test the switched service
          curl -f http://$SERVICE_URL:8000/health/all

      - name: Monitor green environment
        run: |
          echo "Monitoring green environment for 5 minutes..."
          sleep 300
          
          # Check error rates and performance
          SERVICE_URL=$(kubectl get service continual-transformer \
            -n ${{ github.event.inputs.environment || 'staging' }} \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          
          # Run extended monitoring
          python scripts/monitor_deployment.py --url http://$SERVICE_URL:8000 --duration 300

  cleanup-blue:
    name: Cleanup Blue Environment
    runs-on: ubuntu-latest
    needs: [build, deploy-green, switch-traffic]
    environment: ${{ github.event.inputs.environment || 'staging' }}
    if: success()
    steps:
      - name: Scale down blue deployment
        run: |
          # Scale down blue environment (keep for quick rollback)
          kubectl scale deployment continual-transformer-blue \
            --replicas=1 \
            -n ${{ github.event.inputs.environment || 'staging' }}

      - name: Update blue with previous version (for rollback)
        run: |
          # Tag current blue as rollback version
          kubectl annotate deployment continual-transformer-blue \
            deployment.kubernetes.io/rollback-image=${{ needs.build.outputs.image-tag }} \
            -n ${{ github.event.inputs.environment || 'staging' }}

  rollback:
    name: Rollback to Blue
    runs-on: ubuntu-latest
    if: failure()
    environment: ${{ github.event.inputs.environment || 'staging' }}
    steps:
      - name: Emergency rollback
        run: |
          echo "Deployment failed, rolling back to blue environment"
          
          # Switch traffic back to blue
          kubectl patch service continual-transformer \
            -p '{"spec":{"selector":{"slot":"blue"}}}' \
            -n ${{ github.event.inputs.environment || 'staging' }}
          
          # Scale up blue if needed
          kubectl scale deployment continual-transformer-blue \
            --replicas=3 \
            -n ${{ github.event.inputs.environment || 'staging' }}

      - name: Notify team
        uses: 8398a7/action-slack@v3
        with:
          status: failure
          text: "🚨 Deployment failed and rolled back to blue environment"
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### 2. Canary Deployment

#### Workflow Template
```yaml
# .github/workflows/canary-deployment.yml
name: Canary Deployment

on:
  push:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      canary_percentage:
        description: 'Canary traffic percentage'
        required: true
        default: '10'
        type: choice
        options:
        - '5'
        - '10'
        - '25'
        - '50'

jobs:
  deploy-canary:
    name: Deploy Canary
    runs-on: ubuntu-latest
    steps:
      - name: Deploy canary version
        run: |
          # Deploy canary with small replica count
          kubectl set image deployment/continual-transformer-canary \
            app=${{ github.sha }} \
            -n production
          
          # Scale canary to appropriate size
          CANARY_PERCENTAGE=${{ github.event.inputs.canary_percentage || '10' }}
          TOTAL_REPLICAS=$(kubectl get deployment continual-transformer-main \
            -n production -o jsonpath='{.spec.replicas}')
          CANARY_REPLICAS=$(( $TOTAL_REPLICAS * $CANARY_PERCENTAGE / 100 ))
          
          kubectl scale deployment continual-transformer-canary \
            --replicas=$CANARY_REPLICAS \
            -n production

      - name: Configure traffic splitting
        run: |
          # Update Istio VirtualService for traffic splitting
          CANARY_PERCENTAGE=${{ github.event.inputs.canary_percentage || '10' }}
          MAIN_PERCENTAGE=$(( 100 - $CANARY_PERCENTAGE ))
          
          kubectl patch virtualservice continual-transformer \
            -n production \
            --type='json' \
            -p="[{
              'op': 'replace',
              'path': '/spec/http/0/match/0/weight',
              'value': $MAIN_PERCENTAGE
            }, {
              'op': 'replace', 
              'path': '/spec/http/0/match/1/weight',
              'value': $CANARY_PERCENTAGE
            }]"

  monitor-canary:
    name: Monitor Canary Deployment
    runs-on: ubuntu-latest
    needs: deploy-canary
    steps:
      - name: Monitor metrics
        run: |
          echo "Monitoring canary deployment for 10 minutes..."
          
          # Monitor key metrics for 10 minutes
          for i in {1..10}; do
            echo "Monitoring cycle $i/10"
            
            # Check error rates
            ERROR_RATE=$(curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\",version=\"canary\"}[5m])" | jq -r '.data.result[0].value[1]')
            
            # Check latency
            LATENCY_P95=$(curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket{version=\"canary\"}[5m]))" | jq -r '.data.result[0].value[1]')
            
            echo "Canary Error Rate: $ERROR_RATE"
            echo "Canary P95 Latency: $LATENCY_P95"
            
            # Check thresholds
            if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
              echo "Error rate too high, triggering rollback"
              exit 1
            fi
            
            if (( $(echo "$LATENCY_P95 > 1.0" | bc -l) )); then
              echo "Latency too high, triggering rollback"
              exit 1
            fi
            
            sleep 60
          done

  promote-canary:
    name: Promote Canary to Main
    runs-on: ubuntu-latest
    needs: [deploy-canary, monitor-canary]
    if: success()
    steps:
      - name: Gradually increase traffic
        run: |
          # Gradually increase canary traffic
          for percentage in 25 50 75 100; do
            echo "Increasing canary traffic to $percentage%"
            
            main_percentage=$(( 100 - $percentage ))
            
            kubectl patch virtualservice continual-transformer \
              -n production \
              --type='json' \
              -p="[{
                'op': 'replace',
                'path': '/spec/http/0/match/0/weight', 
                'value': $main_percentage
              }, {
                'op': 'replace',
                'path': '/spec/http/0/match/1/weight',
                'value': $percentage
              }]"
            
            # Monitor for 2 minutes at each step
            sleep 120
            
            # Check metrics
            ERROR_RATE=$(curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\",version=\"canary\"}[2m])" | jq -r '.data.result[0].value[1]')
            
            if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
              echo "Error rate spike detected, halting promotion"
              exit 1
            fi
          done

      - name: Replace main with canary
        run: |
          # Update main deployment with canary image
          CANARY_IMAGE=$(kubectl get deployment continual-transformer-canary \
            -n production -o jsonpath='{.spec.template.spec.containers[0].image}')
          
          kubectl set image deployment/continual-transformer-main \
            app=$CANARY_IMAGE \
            -n production
          
          # Wait for main deployment rollout
          kubectl rollout status deployment/continual-transformer-main \
            -n production --timeout=600s

      - name: Cleanup canary
        run: |
          # Scale down canary deployment
          kubectl scale deployment continual-transformer-canary \
            --replicas=0 \
            -n production
          
          # Reset traffic to 100% main
          kubectl patch virtualservice continual-transformer \
            -n production \
            --type='json' \
            -p="[{
              'op': 'replace',
              'path': '/spec/http/0/match/0/weight',
              'value': 100
            }, {
              'op': 'replace',
              'path': '/spec/http/0/match/1/weight', 
              'value': 0
            }]"

  rollback-canary:
    name: Rollback Canary
    runs-on: ubuntu-latest
    if: failure()
    steps:
      - name: Emergency canary rollback
        run: |
          echo "Canary deployment failed, rolling back"
          
          # Set traffic to 100% main
          kubectl patch virtualservice continual-transformer \
            -n production \
            --type='json' \
            -p="[{
              'op': 'replace',
              'path': '/spec/http/0/match/0/weight',
              'value': 100
            }, {
              'op': 'replace',
              'path': '/spec/http/0/match/1/weight',
              'value': 0
            }]"
          
          # Scale down canary
          kubectl scale deployment continual-transformer-canary \
            --replicas=0 \
            -n production

      - name: Notify team
        uses: 8398a7/action-slack@v3
        with:
          status: failure
          text: "🚨 Canary deployment failed and was rolled back"
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### 3. Rolling Deployment

#### Workflow Template
```yaml
# .github/workflows/rolling-deployment.yml
name: Rolling Deployment

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  rolling-deploy:
    name: Rolling Deployment
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy with rolling strategy
        run: |
          # Configure rolling update strategy
          kubectl patch deployment continual-transformer \
            -n production \
            -p='{
              "spec": {
                "strategy": {
                  "type": "RollingUpdate",
                  "rollingUpdate": {
                    "maxUnavailable": "25%",
                    "maxSurge": "25%"
                  }
                }
              }
            }'

      - name: Update deployment image
        run: |
          kubectl set image deployment/continual-transformer \
            app=${{ github.sha }} \
            -n production

      - name: Monitor rolling update
        run: |
          # Wait for rollout to complete
          kubectl rollout status deployment/continual-transformer \
            -n production --timeout=900s

      - name: Verify deployment
        run: |
          # Check all pods are ready
          kubectl wait --for=condition=ready pod \
            -l app=continual-transformer \
            -n production \
            --timeout=300s
          
          # Run health checks
          SERVICE_URL=$(kubectl get service continual-transformer \
            -n production \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          
          curl -f http://$SERVICE_URL:8000/health/all

      - name: Run post-deployment tests
        run: |
          SERVICE_URL=$(kubectl get service continual-transformer \
            -n production \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          
          python scripts/post_deployment_tests.py --url http://$SERVICE_URL:8000
```

## Infrastructure as Code

### 1. Terraform Configuration

```hcl
# infrastructure/main.tf
terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

# Kubernetes namespace
resource "kubernetes_namespace" "continual_transformer" {
  metadata {
    name = var.namespace
    labels = {
      environment = var.environment
      application = "continual-tiny-transformer"
    }
  }
}

# ConfigMap for application configuration
resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "continual-transformer-config"
    namespace = kubernetes_namespace.continual_transformer.metadata[0].name
  }

  data = {
    "config.yaml" = templatefile("${path.module}/config/app-config.yaml", {
      environment = var.environment
      db_host     = var.database_host
      redis_host  = var.redis_host
    })
  }
}

# Secret for sensitive configuration
resource "kubernetes_secret" "app_secrets" {
  metadata {
    name      = "continual-transformer-secrets"
    namespace = kubernetes_namespace.continual_transformer.metadata[0].name
  }

  data = {
    database_url = base64encode(var.database_url)
    api_key      = base64encode(var.api_key)
  }

  type = "Opaque"
}

# Main application deployment
resource "kubernetes_deployment" "main" {
  metadata {
    name      = "continual-transformer-main"
    namespace = kubernetes_namespace.continual_transformer.metadata[0].name
    labels = {
      app     = "continual-transformer"
      version = "main"
      slot    = "blue"
    }
  }

  spec {
    replicas = var.replica_count

    selector {
      match_labels = {
        app  = "continual-transformer"
        slot = "blue"
      }
    }

    strategy {
      type = "RollingUpdate"
      rolling_update {
        max_unavailable = "25%"
        max_surge       = "25%"
      }
    }

    template {
      metadata {
        labels = {
          app     = "continual-transformer"
          version = "main"
          slot    = "blue"
        }
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "9090"
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        container {
          name  = "app"
          image = "${var.image_repository}:${var.image_tag}"

          port {
            container_port = 8000
            name          = "http"
          }

          port {
            container_port = 9090
            name          = "metrics"
          }

          env_from {
            config_map_ref {
              name = kubernetes_config_map.app_config.metadata[0].name
            }
          }

          env_from {
            secret_ref {
              name = kubernetes_secret.app_secrets.metadata[0].name
            }
          }

          resources {
            requests = {
              cpu    = var.cpu_request
              memory = var.memory_request
            }
            limits = {
              cpu    = var.cpu_limit
              memory = var.memory_limit
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 30
            period_seconds        = 10
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/health/ready"
              port = 8000
            }
            initial_delay_seconds = 5
            period_seconds        = 5
            failure_threshold     = 3
          }

          volume_mount {
            name       = "config"
            mount_path = "/app/config"
            read_only  = true
          }
        }

        volume {
          name = "config"
          config_map {
            name = kubernetes_config_map.app_config.metadata[0].name
          }
        }

        image_pull_secrets {
          name = "registry-secret"
        }
      }
    }
  }
}

# Green deployment for blue-green strategy
resource "kubernetes_deployment" "green" {
  metadata {
    name      = "continual-transformer-green"
    namespace = kubernetes_namespace.continual_transformer.metadata[0].name
    labels = {
      app     = "continual-transformer"
      version = "green"
      slot    = "green"
    }
  }

  spec {
    replicas = 0  # Start with 0 replicas

    selector {
      match_labels = {
        app  = "continual-transformer"
        slot = "green"
      }
    }

    template {
      metadata {
        labels = {
          app     = "continual-transformer"
          version = "green"
          slot    = "green"
        }
      }

      spec {
        # Same spec as main deployment
        container {
          name  = "app"
          image = "${var.image_repository}:latest"
          
          # ... (same configuration as main)
        }
      }
    }
  }
}

# Service
resource "kubernetes_service" "main" {
  metadata {
    name      = "continual-transformer"
    namespace = kubernetes_namespace.continual_transformer.metadata[0].name
    labels = {
      app = "continual-transformer"
    }
  }

  spec {
    selector = {
      app  = "continual-transformer"
      slot = "blue"  # Default to blue slot
    }

    port {
      name        = "http"
      port        = 80
      target_port = 8000
      protocol    = "TCP"
    }

    type = "LoadBalancer"
  }
}

# HorizontalPodAutoscaler
resource "kubernetes_horizontal_pod_autoscaler_v2" "main" {
  metadata {
    name      = "continual-transformer-hpa"
    namespace = kubernetes_namespace.continual_transformer.metadata[0].name
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment.main.metadata[0].name
    }

    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = 80
        }
      }
    }
  }
}
```

### 2. Helm Chart Structure

```yaml
# charts/continual-transformer/Chart.yaml
apiVersion: v2
name: continual-transformer
description: A Helm chart for Continual Tiny Transformer
type: application
version: 0.1.0
appVersion: "1.0.0"

dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: postgresql.enabled
  - name: redis
    version: "17.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: redis.enabled
  - name: prometheus
    version: "22.x.x"
    repository: "https://prometheus-community.github.io/helm-charts"
    condition: monitoring.prometheus.enabled
```

```yaml
# charts/continual-transformer/values.yaml
# Default values for continual-transformer
replicaCount: 3

image:
  repository: ghcr.io/your-org/continual-tiny-transformer
  pullPolicy: IfNotPresent
  tag: ""

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
  prometheus.io/path: "/metrics"

podSecurityContext:
  fsGroup: 2000

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts: []
  tls: []

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}
tolerations: []
affinity: {}

# Application configuration
config:
  environment: production
  logLevel: INFO
  database:
    host: postgresql
    port: 5432
    name: continual_transformer
  redis:
    host: redis
    port: 6379

# Monitoring configuration
monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true

# Dependencies
postgresql:
  enabled: true
  auth:
    postgresPassword: "changeme"
    database: "continual_transformer"

redis:
  enabled: true
  auth:
    enabled: false
```

This comprehensive deployment documentation provides:

1. **Multiple Deployment Strategies**: Blue-green, canary, and rolling deployments
2. **Environment-Specific Configurations**: Development, staging, and production
3. **Infrastructure as Code**: Terraform and Helm configurations
4. **Automated Workflows**: GitHub Actions for each deployment strategy
5. **Monitoring and Rollback**: Built-in monitoring and automatic rollback capabilities

These deployment strategies ensure safe, reliable, and automated deployments with minimal downtime and quick recovery options.