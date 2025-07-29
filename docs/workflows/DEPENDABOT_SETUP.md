# Dependabot Configuration Setup

This document provides Dependabot configuration for automated dependency management.

## Configuration File

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"  
    open-pull-requests-limit: 10
    reviewers:
      - "danieleschmidt"  # Replace with actual maintainer username
    assignees:
      - "danieleschmidt"  # Replace with actual maintainer username
    commit-message:
      prefix: "deps"
      prefix-development: "deps-dev"
      include: "scope"
    labels:
      - "dependencies"
      - "python"
    ignore:
      # Ignore major version updates for torch (breaking changes common)
      - dependency-name: "torch"
        update-types: ["version-update:semver-major"]
      # Ignore pre-release versions
      - dependency-name: "*"
        update-types: ["version-update:semver-major"]
        versions: ["*alpha*", "*beta*", "*rc*"]

  # GitHub Actions dependencies  
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 5
    reviewers:
      - "danieleschmidt"
    assignees:
      - "danieleschmidt"
    commit-message:
      prefix: "ci"
      include: "scope"
    labels:
      - "dependencies"
      - "github-actions"

  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 3
    reviewers:
      - "danieleschmidt"
    assignees:
      - "danieleschmidt"
    commit-message:
      prefix: "docker"
      include: "scope"
    labels:
      - "dependencies"
      - "docker"
```

## Security Configuration

Create `.github/workflows/security-scan.yml`:

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  security:
    name: Security Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install safety pip-audit bandit[toml]

    - name: Run Safety check
      run: safety check --json --output safety-report.json
      continue-on-error: true

    - name: Run pip-audit
      run: pip-audit --format=json --output=pip-audit-report.json
      continue-on-error: true

    - name: Run Bandit security linter
      run: |
        bandit -r src/ -f json -o bandit-report.json
      continue-on-error: true

    - name: Upload security scan results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-scan-results
        path: |
          safety-report.json
          pip-audit-report.json
          bandit-report.json

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
        queries: security-extended,security-and-quality

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      with:
        category: "/language:python"
```

## Vulnerability Management

Create `.github/workflows/vulnerability-check.yml`:

```yaml
name: Vulnerability Check

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
  workflow_dispatch:  # Allow manual trigger

jobs:
  vulnerability-scan:
    name: Check for Vulnerabilities
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        pip install safety pip-audit
        
    - name: Run vulnerability checks
      run: |
        echo "## Safety Check Results" >> vulnerability-report.md
        safety check --json >> vulnerability-report.md || true
        
        echo -e "\n## Pip-Audit Results" >> vulnerability-report.md  
        pip-audit --format=json >> vulnerability-report.md || true
        
    - name: Create Issue on Vulnerabilities
      uses: actions/github-script@v6
      if: failure()
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('vulnerability-report.md', 'utf8');
          
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: `Security Vulnerabilities Detected - ${new Date().toISOString().split('T')[0]}`,
            body: `## Automated Security Scan Results\n\n${report}\n\n**Action Required**: Please review and address these vulnerabilities.\n\n*This issue was automatically created by the vulnerability check workflow.*`,
            labels: ['security', 'vulnerability', 'priority-high']
          });
```

## Automated Dependency Updates Policy

### Update Strategy
- **Weekly Schedule**: Updates run every Monday at 9 AM UTC
- **Staged Rollout**: Development dependencies updated more aggressively than production
- **Version Constraints**: Major version updates require manual review
- **Security Priority**: Security updates are fast-tracked

### Review Process
1. **Automated Testing**: All dependency updates trigger full CI pipeline  
2. **Manual Review**: Major version updates require maintainer approval
3. **Compatibility Check**: Automated tests verify backward compatibility
4. **Security Validation**: Security-related updates are prioritized

### Implementation Instructions

1. **Create Configuration File**:
   ```bash
   mkdir -p .github
   # Copy dependabot.yml content to .github/dependabot.yml
   ```

2. **Configure Repository Settings**:
   - Enable Dependabot alerts in repository settings
   - Configure security advisories notifications
   - Set up branch protection rules to require status checks

3. **Customize Reviewers**: Update reviewer/assignee usernames in configuration

4. **Monitor Results**: Check Dependabot dashboard in repository Insights tab

## Expected Benefits

- **Automated Updates**: Weekly dependency updates with proper testing
- **Security Monitoring**: Continuous vulnerability scanning and alerts  
- **Reduced Maintenance**: Automated PR creation and testing
- **Policy Compliance**: Structured update process with proper review gates
- **Risk Mitigation**: Fast security patch deployment while maintaining stability