# Development Guide

## Prerequisites

* Python 3.8+ (check `.python-version` if available)
* Git for version control
* Virtual environment tool (venv, conda, etc.)

## Setup

```bash
# Clone and enter repository
git clone <repository-url>
cd continual-tiny-transformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies (if requirements files exist)
pip install -r requirements.txt  # or requirements-dev.txt
```

## Development Workflow

1. Create feature branch: `git checkout -b feature/description`
2. Make changes following existing patterns
3. Test thoroughly before committing
4. Submit PR with clear description

## Project Structure

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed architectural overview.

## Testing

Run tests using project-specific commands (check main README or CI configuration).

## Documentation

Update relevant docs when making changes. Follow existing documentation patterns.