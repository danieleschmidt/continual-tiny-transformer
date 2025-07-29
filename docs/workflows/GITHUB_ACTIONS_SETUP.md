# GitHub Actions Workflows Setup

This document provides comprehensive GitHub Actions workflows for the continual-tiny-transformer project to achieve SDLC maturity.

## Required Workflows

### 1. Main CI/CD Pipeline (`ci.yml`)

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 2 * * 1'  # Weekly dependency check

jobs:
  test:
    name: Test Python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', 'pyproject.toml') }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
        
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
      
    - name: Run tests with coverage
      run: |
        pytest --cov=continual_transformer --cov-report=xml --cov-report=term
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run CodeQL Analysis
      uses: github/codeql-action/init@v2
      with:
        languages: python
        
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      
    - name: Run Bandit Security Scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json
        
    - name: Upload security scan results
      uses: actions/upload-artifact@v3
      with:
        name: security-scan-results
        path: bandit-report.json

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [test, security]
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Build package
      run: |
        pip install build
        python -m build
        
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/

  docs:
    name: Build Documentation
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install -e ".[docs]"
        
    - name: Build docs
      run: |
        cd docs && sphinx-build -b html . _build/html
        
    - name: Upload docs
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/_build/html/
```

### 2. Release Automation (`release.yml`)

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write
      
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install build twine
        
    - name: Build package
      run: python -m build
      
    - name: Create Release Notes
      run: |
        # Extract version from tag
        VERSION=${GITHUB_REF#refs/tags/v}
        echo "VERSION=$VERSION" >> $GITHUB_ENV
        
        # Generate release notes from CHANGELOG.md
        awk "/^## \[$VERSION\]/{flag=1; next} /^## \[/{flag=0} flag" CHANGELOG.md > release_notes.md
        
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ env.VERSION }}
        body_path: release_notes.md
        draft: false
        prerelease: false
        
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### 3. Dependency Updates (`dependency-review.yml`)

Create `.github/workflows/dependency-review.yml`:

```yaml
name: Dependency Review

on:
  pull_request:
    paths:
      - 'requirements*.txt'
      - 'pyproject.toml'
      - '.github/dependabot.yml'

permissions:
  contents: read
  pull-requests: write

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout Repository
      uses: actions/checkout@v4
      
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
      with:
        fail-on-severity: moderate
        allow-licenses: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC
```

## Implementation Instructions

1. **Create Workflows Directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Add Each Workflow**: Copy the YAML content above into respective files in `.github/workflows/`

3. **Configure Secrets**: Add these repository secrets in GitHub Settings:
   - `PYPI_API_TOKEN`: For PyPI publishing
   - `CODECOV_TOKEN`: For coverage reporting (optional)

4. **Enable Dependabot**: See `DEPENDABOT_SETUP.md` for configuration

5. **Branch Protection**: Configure branch protection rules requiring:
   - Status checks to pass
   - Up-to-date branches
   - Dismissal of stale reviews
   - Administrator enforcement

## Expected Benefits

- **Automated Testing**: All PRs tested across Python versions
- **Security Scanning**: CodeQL and Bandit integrated
- **Quality Gates**: Pre-commit hooks enforced in CI  
- **Release Automation**: Tag-based releases with PyPI publishing
- **Documentation**: Automated docs building and validation
- **Dependency Safety**: Automated vulnerability scanning