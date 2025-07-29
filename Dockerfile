# Multi-stage Docker build for continual-tiny-transformer
# Optimized for both development and production deployments

# Build stage - compile dependencies and prepare application
FROM python:3.10-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DATE
ARG VCS_REF

# Add metadata labels
LABEL maintainer="Daniel Schmidt <daniel@example.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-org/continual-tiny-transformer" \
      org.label-schema.schema-version="1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Production stage - minimal runtime image
FROM python:3.10-slim as production

# Set runtime arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app /app

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import continual_transformer; print('OK')" || exit 1

# Expose port (adjust as needed)
EXPOSE 8000

# Default command - can be overridden
CMD ["python", "-m", "continual_transformer.cli", "--help"]

# Development stage - includes dev dependencies and tools
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipdb \
    jupyter-lab

# Set development environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    DEVELOPMENT=1

# Expose additional ports for development
EXPOSE 8000 8888 8080

# Development command
CMD ["bash"]

# GPU-enabled stage for CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements-dev.txt ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install application
COPY . .
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Create non-root user
RUN groupadd -r gpuuser && useradd -r -g gpuuser gpuuser
RUN chown -R gpuuser:gpuuser /app
USER gpuuser

# Health check with GPU test
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('GPU OK')" || exit 1

# Default GPU command
CMD ["python", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"]