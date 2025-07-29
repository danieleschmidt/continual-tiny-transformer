# Monitoring and Observability Setup

This document outlines monitoring, observability, and performance tracking for the continual-tiny-transformer project.

## Repository Health Monitoring

### GitHub Actions Monitoring Workflow

Create `.github/workflows/health-check.yml`:

```yaml
name: Repository Health Check

on:
  schedule:
    - cron: '0 8 * * 1'  # Weekly on Monday at 8 AM UTC
  workflow_dispatch:

jobs:
  health-metrics:
    name: Collect Repository Health Metrics
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for metrics
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install analysis tools
      run: |
        pip install pytest pytest-cov bandit safety pip-audit
        pip install requests python-dateutil  # For metrics collection
        
    - name: Collect Code Quality Metrics
      run: |
        echo "## Repository Health Report" > health-report.md
        echo "Generated: $(date -u)" >> health-report.md
        echo "" >> health-report.md
        
        # Code coverage metrics
        echo "### Code Coverage" >> health-report.md
        pytest --cov=continual_transformer --cov-report=term | grep "TOTAL" >> health-report.md || echo "Coverage data not available" >> health-report.md
        echo "" >> health-report.md
        
        # Security scan summary
        echo "### Security Status" >> health-report.md
        bandit -r src/ -f json > bandit-results.json || true
        python -c "
import json
try:
    with open('bandit-results.json') as f:
        data = json.load(f)
    metrics = data['metrics']
    print(f'Security Issues: {len(data.get(\"results\", []))}')
    print(f'Lines of Code: {metrics[\"_totals\"][\"loc\"]}')
    print(f'Confidence: {metrics[\"_totals\"][\"nosec\"]} skipped')
except:
    print('Security scan data not available')
        " >> health-report.md
        echo "" >> health-report.md
        
        # Dependency health
        echo "### Dependency Status" >> health-report.md
        pip-audit --format=json > audit-results.json || true
        python -c "
import json
try:
    with open('audit-results.json') as f:
        data = json.load(f)
    vuln_count = len([v for v in data if 'vulnerabilities' in v])
    print(f'Vulnerable Dependencies: {vuln_count}')
except:
    print('Dependency audit data not available')
        " >> health-report.md
        
    - name: Collect Git Metrics
      run: |
        echo "### Development Activity" >> health-report.md
        echo "Commits last 30 days: $(git log --since='30 days ago' --oneline | wc -l)" >> health-report.md
        echo "Contributors last 30 days: $(git log --since='30 days ago' --format='%ae' | sort -u | wc -l)" >> health-report.md
        echo "Files changed last 30 days: $(git log --since='30 days ago' --name-only --pretty=format: | sort -u | wc -l)" >> health-report.md
        echo "" >> health-report.md
        
    - name: Create Health Check Issue
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('health-report.md', 'utf8');
          
          // Check if there's already an open health check issue
          const issues = await github.rest.issues.listForRepo({
            owner: context.repo.owner,
            repo: context.repo.repo,
            labels: 'health-check',
            state: 'open'
          });
          
          if (issues.data.length > 0) {
            // Update existing issue
            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: issues.data[0].number,
              body: `## Weekly Health Update\n\n${report}`
            });
          } else {
            // Create new issue
            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Repository Health Check - ${new Date().toISOString().split('T')[0]}`,
              body: `## Weekly Repository Health Report\n\n${report}\n\n*This issue is automatically generated weekly to track repository health metrics.*`,
              labels: ['health-check', 'maintenance']
            });
          }
```

## Performance Monitoring

### Benchmark Tracking Workflow

Create `.github/workflows/benchmark.yml`:

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM UTC

jobs:
  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install -e ".[benchmark]"
        pip install pytest-benchmark
        
    - name: Run benchmarks
      run: |
        pytest tests/benchmarks/ --benchmark-json=benchmark-results.json
        
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark-results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '200%'  # Alert if performance degrades by 200%
        fail-on-alert: false
```

## Code Quality Metrics

### Quality Gate Configuration

Create `scripts/quality-check.py`:

```python
#!/usr/bin/env python3
"""
Quality metrics collection and reporting script.
Run via: python scripts/quality-check.py
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

def run_command(cmd: str) -> tuple[str, int]:
    """Run shell command and return output and exit code."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout, result.returncode

def collect_coverage_metrics() -> Dict[str, Any]:
    """Collect test coverage metrics."""
    output, code = run_command("pytest --cov=continual_transformer --cov-report=json")
    
    if code == 0 and Path("coverage.json").exists():
        with open("coverage.json") as f:
            coverage_data = json.load(f)
        return {
            "line_coverage": coverage_data["totals"]["percent_covered"],
            "branch_coverage": coverage_data["totals"].get("percent_covered_display", "N/A"),
            "missing_lines": coverage_data["totals"]["missing_lines"]
        }
    return {"line_coverage": 0, "branch_coverage": "N/A", "missing_lines": "N/A"}

def collect_security_metrics() -> Dict[str, Any]:
    """Collect security scan metrics."""
    output, code = run_command("bandit -r src/ -f json")
    
    try:
        bandit_data = json.loads(output)
        return {
            "security_issues": len(bandit_data.get("results", [])),
            "high_severity": len([r for r in bandit_data.get("results", []) if r["issue_severity"] == "HIGH"]),
            "medium_severity": len([r for r in bandit_data.get("results", []) if r["issue_severity"] == "MEDIUM"])
        }
    except:
        return {"security_issues": 0, "high_severity": 0, "medium_severity": 0}

def collect_dependency_metrics() -> Dict[str, Any]:
    """Collect dependency health metrics."""
    output, code = run_command("pip-audit --format=json")
    
    try:
        audit_data = json.loads(output)
        vulnerabilities = [item for item in audit_data if "vulnerabilities" in item]
        return {
            "vulnerable_packages": len(vulnerabilities),
            "total_vulnerabilities": sum(len(item["vulnerabilities"]) for item in vulnerabilities)
        }
    except:
        return {"vulnerable_packages": 0, "total_vulnerabilities": 0}

def collect_code_metrics() -> Dict[str, Any]:
    """Collect code complexity and quality metrics."""
    # Line count
    output, _ = run_command("find src/ -name '*.py' -exec wc -l {} + | tail -1")
    total_lines = int(output.split()[0]) if output.strip() else 0
    
    # File count
    output, _ = run_command("find src/ -name '*.py' | wc -l")
    total_files = int(output.strip()) if output.strip() else 0
    
    return {
        "total_lines": total_lines,
        "total_files": total_files,
        "avg_lines_per_file": total_lines / max(total_files, 1)
    }

def generate_quality_report() -> Dict[str, Any]:
    """Generate comprehensive quality report."""
    report = {
        "timestamp": subprocess.check_output(["date", "-u"]).decode().strip(),
        "coverage": collect_coverage_metrics(),
        "security": collect_security_metrics(),
        "dependencies": collect_dependency_metrics(),
        "code_metrics": collect_code_metrics()
    }
    
    return report

def evaluate_quality_gates(report: Dict[str, Any]) -> bool:
    """Evaluate if quality gates pass."""
    gates_passed = True
    issues = []
    
    # Coverage gate (80% minimum)
    if report["coverage"]["line_coverage"] < 80:
        gates_passed = False
        issues.append(f"Coverage below threshold: {report['coverage']['line_coverage']}% < 80%")
    
    # Security gate (no high severity issues)
    if report["security"]["high_severity"] > 0:
        gates_passed = False
        issues.append(f"High severity security issues: {report['security']['high_severity']}")
    
    # Dependency gate (no vulnerable packages)
    if report["dependencies"]["vulnerable_packages"] > 0:
        gates_passed = False
        issues.append(f"Vulnerable dependencies: {report['dependencies']['vulnerable_packages']}")
    
    if not gates_passed:
        print("❌ Quality gates failed:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All quality gates passed")
    
    return gates_passed

def main():
    """Main quality check execution."""
    print("🔍 Collecting quality metrics...")
    
    report = generate_quality_report()
    
    # Output report
    print("\n📊 Quality Report:")
    print(f"Coverage: {report['coverage']['line_coverage']:.1f}%")
    print(f"Security Issues: {report['security']['security_issues']} (High: {report['security']['high_severity']})")
    print(f"Vulnerable Dependencies: {report['dependencies']['vulnerable_packages']}")
    print(f"Code Files: {report['code_metrics']['total_files']} ({report['code_metrics']['total_lines']} lines)")
    
    # Save detailed report
    with open("quality-report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Evaluate gates
    gates_passed = evaluate_quality_gates(report)
    
    # Exit with appropriate code
    sys.exit(0 if gates_passed else 1)

if __name__ == "__main__":
    main()
```

## Alerting and Notifications

### Slack Integration (Optional)

Add to workflow files for Slack notifications:

```yaml
- name: Notify Slack on Failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#dev-alerts'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    fields: repo,message,commit,author,action,eventName,ref,workflow
```

## Dashboard Configuration

### GitHub Insights Configuration

1. **Enable Insights**: Go to repository Settings → Insights
2. **Configure Pulse**: Enable weekly/monthly activity summaries
3. **Traffic Analytics**: Monitor clones, views, and referrers
4. **Dependency Graph**: Enable dependency insights and security alerts

### Third-Party Monitoring Tools

1. **Codecov**: Integrate for advanced coverage analytics
2. **Snyk**: Enhanced security vulnerability monitoring
3. **SonarCloud**: Code quality and technical debt tracking
4. **Dependabot**: Automated dependency updates and alerts

## Implementation Steps

1. **Create Scripts Directory**:
   ```bash
   mkdir -p scripts
   chmod +x scripts/quality-check.py
   ```

2. **Add Monitoring Workflows**: Copy workflow YAML files to `.github/workflows/`

3. **Configure Repository Settings**:
   - Enable vulnerability alerts
   - Configure notification preferences
   - Set up integrations (Slack, email, etc.)

4. **Test Monitoring Setup**:
   ```bash
   # Test quality checks locally
   python scripts/quality-check.py
   
   # Test workflows
   git commit -m "test: trigger monitoring workflows"
   ```

## Metrics Collection

### Key Performance Indicators (KPIs)

- **Code Quality**: Coverage %, complexity, maintainability index
- **Security**: Vulnerability count, security scan results
- **Reliability**: Build success rate, test pass rate
- **Performance**: Benchmark trends, memory usage
- **Community**: Issue response time, PR merge time

### Reporting Schedule

- **Daily**: Security scans, dependency checks
- **Weekly**: Health reports, performance benchmarks  
- **Monthly**: Comprehensive quality review, trend analysis
- **Quarterly**: Third-party security audit, dependency cleanup

This monitoring setup provides comprehensive visibility into repository health, code quality, and project sustainability.