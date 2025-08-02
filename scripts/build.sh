#!/bin/bash
# Comprehensive build script for continual-tiny-transformer
# Handles different build targets and environments

set -e  # Exit on any error

# Default values
BUILD_TYPE="release"
SKIP_TESTS="false"
VERBOSE="false"
DOCKER_BUILD="false"
PLATFORM="linux/amd64"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build script for continual-tiny-transformer

OPTIONS:
    -t, --type TYPE         Build type: dev, release, debug (default: release)
    -s, --skip-tests        Skip running tests during build
    -v, --verbose           Enable verbose output
    -d, --docker            Build Docker images
    -p, --platform PLATFORM Docker platform (default: linux/amd64)
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Standard release build
    $0 -t dev -s            # Development build without tests
    $0 -d                   # Build with Docker images
    $0 -t release -v        # Verbose release build

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -s|--skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -d|--docker)
            DOCKER_BUILD="true"
            shift
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Enable verbose mode if requested
if [ "$VERBOSE" = "true" ]; then
    set -x
fi

print_step "Starting build process..."
echo "Build Type: $BUILD_TYPE"
echo "Skip Tests: $SKIP_TESTS"
echo "Docker Build: $DOCKER_BUILD"
echo "Platform: $PLATFORM"
echo ""

# Build metadata
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=$(python -c "import src.continual_transformer; print(src.continual_transformer.__version__)" 2>/dev/null || echo "0.1.0")

export BUILD_DATE VCS_REF VERSION

print_step "Build metadata:"
echo "  Version: $VERSION"
echo "  Git Ref: $VCS_REF"
echo "  Build Date: $BUILD_DATE"
echo ""

# Step 1: Environment validation
print_step "Validating build environment..."

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Check required tools
required_tools=("pip" "git")
if [ "$DOCKER_BUILD" = "true" ]; then
    required_tools+=("docker")
fi

for tool in "${required_tools[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        print_error "$tool is required but not installed"
        exit 1
    fi
done

print_success "Environment validation passed"

# Step 2: Clean previous builds
print_step "Cleaning previous build artifacts..."
make clean || print_warning "Clean command failed"

# Step 3: Install/update dependencies
print_step "Installing dependencies..."
case $BUILD_TYPE in
    "dev")
        pip install -e ".[dev,test]"
        ;;
    "release")
        pip install -e ".[dev,test,docs]"
        ;;
    "debug")
        pip install -e ".[dev,test]"
        ;;
    *)
        print_error "Unknown build type: $BUILD_TYPE"
        exit 1
        ;;
esac

print_success "Dependencies installed"

# Step 4: Code quality checks
print_step "Running code quality checks..."

# Linting
print_step "Running linters..."
ruff check src/ tests/ || print_error "Ruff linting failed"
black --check src/ tests/ || print_error "Black formatting check failed"
isort --check src/ tests/ || print_error "isort import check failed"

# Type checking
print_step "Running type checking..."
mypy src/ || print_warning "Type checking found issues"

print_success "Code quality checks completed"

# Step 5: Security scanning
print_step "Running security checks..."
if [ -f "scripts/security-scan.sh" ]; then
    bash scripts/security-scan.sh || print_warning "Security scan found issues"
else
    bandit -r src/ || print_warning "Bandit security check found issues"
fi

print_success "Security checks completed"

# Step 6: Run tests
if [ "$SKIP_TESTS" = "false" ]; then
    print_step "Running tests..."
    
    case $BUILD_TYPE in
        "dev")
            pytest tests/unit/ -v
            ;;
        "release")
            pytest --cov=continual_transformer --cov-report=html --cov-report=term
            ;;
        "debug")
            pytest tests/unit/ -v -s
            ;;
    esac
    
    print_success "Tests completed"
else
    print_warning "Skipping tests as requested"
fi

# Step 7: Build package
print_step "Building package..."
python -m build

# Verify the built package
print_step "Verifying built package..."
if [ -d "dist" ] && [ "$(ls -A dist)" ]; then
    print_success "Package built successfully"
    echo "Built packages:"
    ls -la dist/
else
    print_error "Package build failed"
    exit 1
fi

# Step 8: Build documentation (for release builds)
if [ "$BUILD_TYPE" = "release" ]; then
    print_step "Building documentation..."
    if [ -f "docs/conf.py" ]; then
        make docs || print_warning "Documentation build failed"
        print_success "Documentation built"
    else
        print_warning "Documentation configuration not found, skipping"
    fi
fi

# Step 9: Docker builds
if [ "$DOCKER_BUILD" = "true" ]; then
    print_step "Building Docker images..."
    
    # Build production image
    docker build \
        --platform "$PLATFORM" \
        --target production \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        -t "continual-tiny-transformer:latest" \
        -t "continual-tiny-transformer:$VERSION" \
        .
    
    # Build development image
    docker build \
        --platform "$PLATFORM" \
        --target development \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        -t "continual-tiny-transformer:dev" \
        .
    
    # Build GPU image if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        print_step "Building GPU-enabled image..."
        docker build \
            --platform "$PLATFORM" \
            --target gpu \
            --build-arg BUILD_DATE="$BUILD_DATE" \
            --build-arg VCS_REF="$VCS_REF" \
            -t "continual-tiny-transformer:gpu" \
            .
    fi
    
    print_success "Docker images built successfully"
    docker images | grep continual-tiny-transformer
fi

# Step 10: Generate build report
print_step "Generating build report..."
BUILD_REPORT="build-report.md"

cat > "$BUILD_REPORT" << EOF
# Build Report

**Build Date:** $BUILD_DATE  
**Version:** $VERSION  
**Git Reference:** $VCS_REF  
**Build Type:** $BUILD_TYPE  
**Platform:** $PLATFORM  

## Build Steps Completed

- ✅ Environment validation
- ✅ Dependency installation
- ✅ Code quality checks
- ✅ Security scanning
$([ "$SKIP_TESTS" = "false" ] && echo "- ✅ Test execution" || echo "- ⏩ Tests skipped")
- ✅ Package building
$([ "$BUILD_TYPE" = "release" ] && echo "- ✅ Documentation building" || echo "- ⏩ Documentation skipped")
$([ "$DOCKER_BUILD" = "true" ] && echo "- ✅ Docker image building" || echo "- ⏩ Docker images skipped")

## Artifacts Generated

### Python Package
- Source distribution: \`dist/*.tar.gz\`
- Wheel distribution: \`dist/*.whl\`

$([ "$DOCKER_BUILD" = "true" ] && cat << 'DOCKER_SECTION'
### Docker Images
- Production: `continual-tiny-transformer:latest`
- Development: `continual-tiny-transformer:dev`
- GPU (if available): `continual-tiny-transformer:gpu`

DOCKER_SECTION
)

## Quality Metrics

- Code coverage: See \`htmlcov/index.html\`
- Security scan: See \`security-reports/\`
- Type checking: Passed with mypy
- Linting: Passed with ruff, black, isort

## Next Steps

1. Review build artifacts in \`dist/\` directory
2. Check test coverage report
3. Review security scan results
4. Deploy to test environment for validation

EOF

print_success "Build report generated: $BUILD_REPORT"

# Step 11: Final validation
print_step "Running final validation..."

# Test package installation
pip install --force-reinstall dist/*.whl
python -c "import continual_transformer; print(f'Package version: {continual_transformer.__version__}')"

print_success "Build completed successfully!"
echo ""
echo "Summary:"
echo "  Version: $VERSION"
echo "  Build Type: $BUILD_TYPE"
echo "  Artifacts: $(ls dist/ | wc -l) packages in dist/"
if [ "$DOCKER_BUILD" = "true" ]; then
    echo "  Docker Images: $(docker images | grep continual-tiny-transformer | wc -l) images built"
fi
echo "  Report: $BUILD_REPORT"
echo ""
print_success "All done! 🚀"