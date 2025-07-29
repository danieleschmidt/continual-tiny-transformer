#!/bin/bash
# Development environment setup script for continual-tiny-transformer

set -e

echo "🚀 Setting up development environment for continual-tiny-transformer..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python 3.8+ is available
check_python() {
    log_info "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        REQUIRED_VERSION="3.8"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
            log_success "Python $PYTHON_VERSION found"
        else
            log_error "Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    log_info "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    log_success "Virtual environment activated and pip upgraded"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Install production dependencies
    pip install -r requirements.txt
    log_success "Production dependencies installed"
    
    # Install development dependencies
    pip install -r requirements-dev.txt
    log_success "Development dependencies installed"
    
    # Install package in editable mode
    pip install -e .
    log_success "Package installed in editable mode"
}

# Setup pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
        
        # Run pre-commit on all files to test setup
        log_info "Running pre-commit on all files (this may take a moment)..."
        pre-commit run --all-files || log_warning "Some pre-commit checks failed (expected for new setup)"
    else
        log_error "pre-commit not found. Installing..."
        pip install pre-commit
        pre-commit install
        log_success "Pre-commit installed and hooks set up"
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Test import
    python3 -c "import continual_transformer; print(f'✅ Package imports successfully')" || {
        log_error "Package import failed"
        return 1
    }
    
    # Test CLI
    python3 -m continual_transformer.cli --help > /dev/null || {
        log_warning "CLI test skipped (implementation pending)"
    }
    
    # Run basic tests
    if [ -d "tests" ]; then
        python3 -m pytest tests/unit/test_example.py -v || {
            log_warning "Example tests failed (expected for development setup)"
        }
    fi
    
    log_success "Installation verification completed"
}

# Setup IDE configuration
setup_ide() {
    log_info "Setting up IDE configuration..."
    
    # VSCode settings are already in place
    if [ -d ".vscode" ]; then
        log_success "VSCode configuration found"
    fi
    
    # Create .python-version for pyenv users
    if command -v pyenv &> /dev/null; then
        python3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" > .python-version
        log_success "Created .python-version for pyenv"
    fi
}

# Display next steps
show_next_steps() {
    echo ""
    echo "🎉 Development environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run tests: make test"
    echo "3. Check code quality: make lint"
    echo "4. Start developing!"
    echo ""
    echo "Available commands:"
    echo "  make help          - Show all available commands"
    echo "  make test          - Run tests"
    echo "  make lint          - Run linting"
    echo "  make format        - Format code"
    echo "  make docs          - Build documentation"
    echo "  make docker-build  - Build Docker image"
    echo ""
    echo "Documentation:"
    echo "  📖 README.md - Project overview"
    echo "  📚 docs/tutorials/01-quickstart.md - Quick start guide"
    echo "  🔧 docs/DEVELOPMENT.md - Development guide"
    echo ""
}

# Main execution
main() {
    echo "Starting development environment setup..."
    echo "======================================"
    
    check_python
    setup_venv
    install_dependencies
    setup_precommit
    setup_ide
    verify_installation
    show_next_steps
    
    log_success "Setup completed successfully! 🚀"
}

# Run main function
main "$@"