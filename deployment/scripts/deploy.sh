#!/bin/bash
# Production deployment script for SDLC automation

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

# Default configuration
ENVIRONMENT="production"
AWS_REGION="us-west-2"
CLUSTER_NAME="sdlc-automation-cluster"
NAMESPACE="continual-transformer-sdlc"
IMAGE_TAG="latest"
DRY_RUN=false
SKIP_TESTS=false
SKIP_SECURITY_SCAN=false
FORCE_DEPLOY=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

# Help function
show_help() {
    cat << EOF
SDLC Automation Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
  infrastructure  Deploy infrastructure (Terraform)
  application     Deploy application (Kubernetes)
  all            Deploy both infrastructure and application
  destroy        Destroy deployment (use with caution)

Options:
  -e, --environment ENV     Environment name (default: production)
  -r, --region REGION       AWS region (default: us-west-2)
  -c, --cluster CLUSTER     EKS cluster name (default: sdlc-automation-cluster)
  -n, --namespace NS        Kubernetes namespace (default: continual-transformer-sdlc)
  -t, --tag TAG            Docker image tag (default: latest)
  --dry-run                Show what would be done without executing
  --skip-tests             Skip running tests before deployment
  --skip-security          Skip security scanning
  --force                  Force deployment even if checks fail
  -h, --help               Show this help message

Examples:
  $0 all
  $0 infrastructure --environment staging
  $0 application --tag v1.2.3
  $0 all --dry-run --skip-tests

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -c|--cluster)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-security)
                SKIP_SECURITY_SCAN=true
                shift
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            infrastructure|application|all|destroy)
                COMMAND="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [[ -z "${COMMAND:-}" ]]; then
        log_error "Command required"
        show_help
        exit 1
    fi
}

# Prerequisites check
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_tools=()

    # Check required tools
    if ! command -v aws >/dev/null 2>&1; then
        missing_tools+=("aws")
    fi

    if ! command -v kubectl >/dev/null 2>&1; then
        missing_tools+=("kubectl")
    fi

    if ! command -v terraform >/dev/null 2>&1; then
        missing_tools+=("terraform")
    fi

    if ! command -v docker >/dev/null 2>&1; then
        missing_tools+=("docker")
    fi

    if ! command -v jq >/dev/null 2>&1; then
        missing_tools+=("jq")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again"
        exit 1
    fi

    # Check AWS authentication
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_error "AWS authentication failed. Please configure AWS credentials"
        exit 1
    fi

    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warn "Skipping tests"
        return 0
    fi

    log_info "Running tests..."

    cd "$PROJECT_ROOT"

    # Run unit tests
    if [[ "$DRY_RUN" == "false" ]]; then
        python -m pytest tests/unit/ -v --tb=short || {
            log_error "Unit tests failed"
            return 1
        }

        # Run integration tests
        python -m pytest tests/integration/ -v --tb=short || {
            log_error "Integration tests failed"
            return 1
        }

        # Run SDLC framework tests
        python -m pytest tests/test_sdlc_framework.py -v --tb=short || {
            log_error "SDLC framework tests failed"
            return 1
        }
    else
        log_info "[DRY RUN] Would run pytest tests"
    fi

    log_success "Tests passed"
}

# Run security scan
run_security_scan() {
    if [[ "$SKIP_SECURITY_SCAN" == "true" ]]; then
        log_warn "Skipping security scan"
        return 0
    fi

    log_info "Running security scan..."

    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "false" ]]; then
        # Run security scanner
        python scripts/security_scanner.py . --level strict --output "deployment/reports/security_report.json" || {
            local exit_code=$?
            if [[ "$FORCE_DEPLOY" == "false" && $exit_code -eq 1 ]]; then
                log_error "Security scan found critical issues"
                return 1
            else
                log_warn "Security scan found issues but continuing due to --force flag"
            fi
        }
    else
        log_info "[DRY RUN] Would run security scanner"
    fi

    log_success "Security scan completed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."

    # Get ECR repository URL
    local ecr_repository
    ecr_repository=$(aws ecr describe-repositories --repository-names continual-transformer-sdlc --region "$AWS_REGION" --query 'repositories[0].repositoryUri' --output text 2>/dev/null || echo "")

    if [[ -z "$ecr_repository" ]]; then
        log_warn "ECR repository not found, using local tag"
        ecr_repository="continual-transformer-sdlc"
    fi

    local image_name="${ecr_repository}:${IMAGE_TAG}"

    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "false" ]]; then
        # Build Docker image
        docker build -f deployment/docker/Dockerfile.sdlc -t "$image_name" . || {
            log_error "Docker build failed"
            return 1
        }

        # Push to ECR if repository exists
        if [[ "$ecr_repository" =~ ^[0-9]+\.dkr\.ecr\. ]]; then
            # Login to ECR
            aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ecr_repository" || {
                log_error "ECR login failed"
                return 1
            }

            # Push image
            docker push "$image_name" || {
                log_error "Docker push failed"
                return 1
            }

            log_success "Image pushed to ECR: $image_name"
        else
            log_info "Image built locally: $image_name"
        fi
    else
        log_info "[DRY RUN] Would build and push Docker image: $image_name"
    fi
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."

    cd "$DEPLOYMENT_DIR/terraform"

    # Initialize Terraform
    if [[ "$DRY_RUN" == "false" ]]; then
        terraform init || {
            log_error "Terraform init failed"
            return 1
        }

        # Plan deployment
        terraform plan \
            -var="aws_region=$AWS_REGION" \
            -var="environment=$ENVIRONMENT" \
            -var="cluster_name=$CLUSTER_NAME" \
            -out=tfplan || {
            log_error "Terraform plan failed"
            return 1
        }

        # Apply deployment
        terraform apply tfplan || {
            log_error "Terraform apply failed"
            return 1
        }
    else
        log_info "[DRY RUN] Would run terraform init, plan, and apply"
    fi

    log_success "Infrastructure deployment completed"
}

# Deploy application
deploy_application() {
    log_info "Deploying application to Kubernetes..."

    # Update kubeconfig
    if [[ "$DRY_RUN" == "false" ]]; then
        aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME" || {
            log_error "Failed to update kubeconfig"
            return 1
        }
    else
        log_info "[DRY RUN] Would update kubeconfig"
    fi

    # Apply Kubernetes manifests
    local k8s_manifest="$DEPLOYMENT_DIR/kubernetes/sdlc-deployment.yaml"

    if [[ "$DRY_RUN" == "false" ]]; then
        # Update image tag in deployment
        sed -i.bak "s|continual-transformer-sdlc:latest|continual-transformer-sdlc:${IMAGE_TAG}|g" "$k8s_manifest"

        # Apply manifests
        kubectl apply -f "$k8s_manifest" || {
            log_error "Kubernetes deployment failed"
            return 1
        }

        # Wait for deployment to be ready
        kubectl wait --for=condition=available --timeout=300s deployment/sdlc-automation -n "$NAMESPACE" || {
            log_error "Deployment did not become ready within timeout"
            return 1
        }

        # Restore original manifest
        mv "$k8s_manifest.bak" "$k8s_manifest"
    else
        log_info "[DRY RUN] Would apply Kubernetes manifests"
    fi

    log_success "Application deployment completed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."

    if [[ "$DRY_RUN" == "false" ]]; then
        # Check pod status
        local pod_status
        pod_status=$(kubectl get pods -n "$NAMESPACE" -l app=sdlc-automation -o jsonpath='{.items[0].status.phase}')

        if [[ "$pod_status" != "Running" ]]; then
            log_error "Pod is not in Running state: $pod_status"
            return 1
        fi

        # Run health check
        kubectl exec -n "$NAMESPACE" deployment/sdlc-automation -- /usr/local/bin/healthcheck.sh || {
            log_error "Health check failed"
            return 1
        }

        # Get service information
        kubectl get services -n "$NAMESPACE"
        kubectl get pods -n "$NAMESPACE"
    else
        log_info "[DRY RUN] Would verify deployment health"
    fi

    log_success "Deployment verification completed"
}

# Destroy deployment
destroy_deployment() {
    log_warn "DESTRUCTIVE OPERATION: This will destroy the entire deployment"
    
    if [[ "$FORCE_DEPLOY" == "false" ]]; then
        read -p "Are you sure you want to continue? Type 'yes' to confirm: " confirmation
        if [[ "$confirmation" != "yes" ]]; then
            log_info "Operation cancelled"
            exit 0
        fi
    fi

    log_info "Destroying deployment..."

    # Destroy Kubernetes resources
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl delete -f "$DEPLOYMENT_DIR/kubernetes/sdlc-deployment.yaml" --ignore-not-found=true || true
    else
        log_info "[DRY RUN] Would delete Kubernetes resources"
    fi

    # Destroy Terraform infrastructure
    cd "$DEPLOYMENT_DIR/terraform"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        terraform destroy \
            -var="aws_region=$AWS_REGION" \
            -var="environment=$ENVIRONMENT" \
            -var="cluster_name=$CLUSTER_NAME" \
            -auto-approve || {
            log_error "Terraform destroy failed"
            return 1
        }
    else
        log_info "[DRY RUN] Would run terraform destroy"
    fi

    log_success "Deployment destroyed"
}

# Main deployment function
main() {
    parse_args "$@"

    log_info "Starting SDLC deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Region: $AWS_REGION"
    log_info "Cluster: $CLUSTER_NAME"
    log_info "Namespace: $NAMESPACE"
    log_info "Image Tag: $IMAGE_TAG"
    log_info "Command: $COMMAND"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "DRY RUN MODE - No changes will be made"
    fi

    check_prerequisites

    case $COMMAND in
        infrastructure)
            deploy_infrastructure
            ;;
        application)
            run_tests
            run_security_scan
            build_and_push_image
            deploy_application
            verify_deployment
            ;;
        all)
            run_tests
            run_security_scan
            build_and_push_image
            deploy_infrastructure
            deploy_application
            verify_deployment
            ;;
        destroy)
            destroy_deployment
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac

    log_success "Deployment completed successfully!"
}

# Run main function
main "$@"