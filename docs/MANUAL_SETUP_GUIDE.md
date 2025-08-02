# Manual Setup Guide

## Overview

This guide provides step-by-step instructions for completing the SDLC implementation setup that requires manual configuration due to GitHub App permission limitations.

## Prerequisites

- Repository admin access
- GitHub CLI installed (optional but recommended)
- Docker installed (for container testing)
- Python environment with project dependencies

## Required Manual Setup

### 1. GitHub Actions Workflows

#### Step 1: Copy Workflow Files
```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy main CI workflow
cp docs/workflows/ci-complete.yml .github/workflows/ci.yml

# Copy security workflow  
cp docs/workflows/security-complete.yml .github/workflows/security.yml

# Copy release workflow
cp docs/workflows/release-complete.yml .github/workflows/release.yml

# Copy dependency management
cp docs/workflows/dependabot.yml.template .github/dependabot.yml
```

#### Step 2: Configure Workflow Secrets
Navigate to **Settings → Secrets and variables → Actions** and add:

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `PYPI_API_TOKEN` | PyPI publishing token | Release workflow |
| `CODECOV_TOKEN` | Codecov integration token | Coverage reporting |
| `SLACK_WEBHOOK_URL` | Slack notifications webhook | Alert notifications |
| `SNYK_TOKEN` | Snyk security scanning | Security workflow |

#### Step 3: Enable Workflows
```bash
# Commit and push workflow files
git add .github/
git commit -m "feat: add GitHub Actions workflows"
git push origin main

# Workflows will be automatically enabled after push
```

### 2. Repository Settings Configuration

#### Branch Protection Rules
1. Navigate to **Settings → Branches**
2. Click **Add rule** for `main` branch
3. Configure the following settings:

**Required Status Checks:**
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- Select: `CI / quality-gates`, `CI / test-matrix`, `CI / security-scan`

**Restrict Pushes:**
- ✅ Restrict pushes that create files larger than 100 MB
- ✅ Require linear history

**Rules Applied to Administrators:**
- ✅ Include administrators

#### Security Settings
1. Navigate to **Settings → Security**
2. Configure **Code security and analysis**:

**Dependency Scanning:**
- ✅ Enable Dependabot alerts
- ✅ Enable Dependabot security updates
- ✅ Enable Dependabot version updates

**Code Scanning:**
- ✅ Enable CodeQL analysis
- ✅ Enable secret scanning
- ✅ Enable secret scanning push protection

**Private Vulnerability Reporting:**
- ✅ Enable private vulnerability reporting

#### Repository Features
1. Navigate to **Settings → General**
2. Configure **Features**:
   - ✅ Issues
   - ✅ Projects
   - ✅ Wiki (optional)
   - ✅ Discussions (optional)

### 3. GitHub Pages Setup (Optional)

#### For Documentation Hosting
1. Navigate to **Settings → Pages**
2. Configure source:
   - **Source**: Deploy from a branch
   - **Branch**: `main`
   - **Folder**: `/docs`
3. Enable **Enforce HTTPS**

### 4. Issue and PR Templates

#### Step 1: Create Template Files
```bash
# Create GitHub templates directory
mkdir -p .github/ISSUE_TEMPLATE
mkdir -p .github/PULL_REQUEST_TEMPLATE

# Bug report template
cat > .github/ISSUE_TEMPLATE/bug_report.yml << 'EOF'
name: Bug Report
description: Report a bug or issue
title: "[Bug]: "
labels: ["bug"]
body:
  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: A clear and concise description of what the bug is
      placeholder: Describe the bug...
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior
      placeholder: |
        1. Go to '...'
        2. Click on '....'
        3. Scroll down to '....'
        4. See error
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What you expected to happen
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: Your environment details
      placeholder: |
        - OS: [e.g. Ubuntu 20.04]
        - Python version: [e.g. 3.10]
        - Package version: [e.g. 0.1.0]
    validations:
      required: true
EOF

# Feature request template
cat > .github/ISSUE_TEMPLATE/feature_request.yml << 'EOF'
name: Feature Request
description: Suggest a new feature or enhancement
title: "[Feature]: "
labels: ["enhancement"]
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: Is your feature request related to a problem?
      placeholder: A clear description of what the problem is...
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: Describe the solution you'd like
      placeholder: A clear description of what you want to happen...
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Alternative solutions or features you've considered
    validations:
      required: false
EOF

# Security issue template
cat > .github/ISSUE_TEMPLATE/security_report.yml << 'EOF'
name: Security Report
description: Report a security vulnerability (private)
title: "[Security]: "
labels: ["security"]
body:
  - type: markdown
    attributes:
      value: |
        **Please do not report security vulnerabilities publicly.**
        Use GitHub's private vulnerability reporting feature.
  - type: textarea
    id: vulnerability
    attributes:
      label: Vulnerability Description
      description: Describe the security vulnerability
    validations:
      required: true
  - type: textarea
    id: impact
    attributes:
      label: Impact Assessment
      description: What is the potential impact of this vulnerability?
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: How can this vulnerability be reproduced?
    validations:
      required: true
EOF

# Pull request template
cat > .github/PULL_REQUEST_TEMPLATE.md << 'EOF'
## Description
Brief description of changes made in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance testing (if applicable)

## Security
- [ ] Security scan passed
- [ ] No sensitive data exposed
- [ ] Dependencies audited

## Documentation
- [ ] Code comments updated
- [ ] README updated (if needed)
- [ ] API documentation updated (if needed)

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is properly commented
- [ ] Tests added/updated for new functionality
- [ ] All checks are passing
- [ ] No merge conflicts

## Related Issues
Closes #(issue number)

## Additional Notes
Any additional information that would be helpful for reviewers.
EOF
```

#### Step 2: Configure CODEOWNERS
```bash
# Create CODEOWNERS file
cat > .github/CODEOWNERS << 'EOF'
# Global ownership
* @your-username

# Documentation
docs/ @your-username @docs-team
*.md @your-username @docs-team

# CI/CD
.github/ @your-username @devops-team
scripts/ @your-username @devops-team

# Source code
src/ @your-username @dev-team
tests/ @your-username @dev-team

# Security
SECURITY.md @your-username @security-team
.github/workflows/security*.yml @your-username @security-team

# Configuration
pyproject.toml @your-username @dev-team
requirements*.txt @your-username @dev-team
Dockerfile* @your-username @devops-team
docker-compose*.yml @your-username @devops-team
EOF
```

### 5. External Service Integration

#### Codecov Setup (Optional)
1. Visit [codecov.io](https://codecov.io)
2. Sign in with GitHub
3. Enable repository
4. Copy the token to repository secrets

#### Snyk Setup (Optional)
1. Visit [snyk.io](https://snyk.io)
2. Sign in with GitHub
3. Import repository
4. Copy API token to repository secrets

### 6. Local Development Setup

#### Step 1: Install Development Tools
```bash
# Install pre-commit
pip install pre-commit

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Install development dependencies
pip install -e ".[dev,test,docs]"
```

#### Step 2: Setup Git Configuration
```bash
# Configure git aliases
git config alias.ci "commit -m"
git config alias.co "checkout"
git config alias.br "branch"
git config alias.st "status"
git config alias.unstage "reset HEAD --"
git config alias.last "log -1 HEAD"
git config alias.visual "!gitk"

# Set commit template
git config commit.template .gitmessage
```

#### Step 3: Validate Setup
```bash
# Run automation health check
python scripts/automation-health-check.py

# Run metrics collection
python scripts/metrics-collector.py

# Test pre-commit hooks
pre-commit run --all-files

# Run tests
pytest tests/

# Build documentation
make docs
```

### 7. Monitoring Setup

#### Prometheus Configuration (Optional)
```bash
# Create monitoring directory
mkdir -p config/monitoring

# Create Prometheus configuration
cat > config/monitoring/prometheus.yml << 'EOF'
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

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

# Create alert rules
cat > config/monitoring/alert_rules.yml << 'EOF'
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
          
      - alert: LowKnowledgeRetention
        expr: avg(knowledge_retention_score) < 0.8
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Knowledge retention below threshold"
EOF
```

### 8. Production Deployment Setup

#### Docker Production Build
```bash
# Build production image
docker build --target production -t continual-tiny-transformer:latest .

# Test production image
docker run --rm -p 8000:8000 continual-tiny-transformer:latest

# Verify health endpoint
curl http://localhost:8000/health
```

#### Kubernetes Deployment (Optional)
```bash
# Create Kubernetes namespace
kubectl create namespace continual-transformer

# Apply configurations
kubectl apply -f k8s/ -n continual-transformer

# Check deployment status
kubectl get pods -n continual-transformer
```

## Verification Steps

### 1. Workflow Verification
```bash
# Push a test commit to trigger workflows
git commit --allow-empty -m "test: trigger workflows"
git push origin main

# Check workflow status
gh run list --limit 5
```

### 2. Security Verification
```bash
# Run security scan
python scripts/security-scan.sh

# Check for vulnerabilities
safety check
bandit -r src/
```

### 3. Metrics Verification
```bash
# Collect metrics
python scripts/metrics-collector.py

# Check automation health
python scripts/automation-health-check.py

# Review generated reports
cat metrics-report.md
cat automation-health-summary.md
```

### 4. Documentation Verification
```bash
# Build documentation
make docs

# Check links
make docs-linkcheck

# Serve documentation locally
make docs-serve
```

## Troubleshooting

### Common Issues

#### Workflow Permissions
**Issue**: Workflows fail with permission errors
**Solution**: 
1. Check repository settings → Actions → General
2. Ensure "Read and write permissions" is enabled
3. Enable "Allow GitHub Actions to create and approve pull requests"

#### Secret Access
**Issue**: Workflows cannot access secrets
**Solution**:
1. Verify secrets are added to repository settings
2. Check secret names match exactly in workflow files
3. Ensure workflows are running on correct branch

#### Pre-commit Failures
**Issue**: Pre-commit hooks fail
**Solution**:
```bash
# Update pre-commit hooks
pre-commit autoupdate

# Run on all files
pre-commit run --all-files

# Skip hooks if needed (temporarily)
git commit -m "message" --no-verify
```

#### Build Failures
**Issue**: Docker build fails
**Solution**:
1. Check Dockerfile syntax
2. Verify base image availability
3. Clear Docker cache: `docker system prune -f`

### Getting Help

1. **Check Documentation**: Review docs/ directory for detailed guides
2. **Run Health Checks**: Use automation health check script
3. **Review Logs**: Check GitHub Actions workflow logs
4. **Create Issues**: Use GitHub issue templates for bug reports

## Maintenance Schedule

### Daily
- Monitor workflow runs
- Review security alerts
- Check system health

### Weekly
- Run repository maintenance script
- Review metrics reports
- Update dependencies (automated via Dependabot)

### Monthly
- Security audit review
- Performance optimization
- Documentation updates

---

**Note**: This setup guide should be customized based on your specific requirements and organizational policies. Some steps may not be applicable to all environments.