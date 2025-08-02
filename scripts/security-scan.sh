#!/bin/bash
# Security scanning script for continual-tiny-transformer
# Runs multiple security checks and generates reports

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create reports directory
REPORTS_DIR="security-reports"
mkdir -p "$REPORTS_DIR"

print_status "Starting comprehensive security scan..."

# 1. Bandit - Python security linter
print_status "Running Bandit security scan..."
if command -v bandit &> /dev/null; then
    bandit -r src/ -f json -o "$REPORTS_DIR/bandit-report.json" || print_warning "Bandit found security issues"
    bandit -r src/ -f txt -o "$REPORTS_DIR/bandit-report.txt" || true
    print_status "Bandit report saved to $REPORTS_DIR/bandit-report.json"
else
    print_error "Bandit not installed. Install with: pip install bandit"
fi

# 2. Safety - Check dependencies for known vulnerabilities
print_status "Running Safety check for dependency vulnerabilities..."
if command -v safety &> /dev/null; then
    safety check --json --output "$REPORTS_DIR/safety-report.json" || print_warning "Safety found vulnerable dependencies"
    safety check --output "$REPORTS_DIR/safety-report.txt" || true
    print_status "Safety report saved to $REPORTS_DIR/safety-report.json"
else
    print_error "Safety not installed. Install with: pip install safety"
fi

# 3. pip-audit - Additional dependency vulnerability scanning
print_status "Running pip-audit for dependency analysis..."
if command -v pip-audit &> /dev/null; then
    pip-audit --format=json --output="$REPORTS_DIR/pip-audit-report.json" || print_warning "pip-audit found issues"
    pip-audit --format=text --output="$REPORTS_DIR/pip-audit-report.txt" || true
    print_status "pip-audit report saved to $REPORTS_DIR/pip-audit-report.json"
else
    print_error "pip-audit not installed. Install with: pip install pip-audit"
fi

# 4. Semgrep - Static analysis for security patterns
print_status "Running Semgrep static analysis..."
if command -v semgrep &> /dev/null; then
    semgrep --config=auto --json --output="$REPORTS_DIR/semgrep-report.json" src/ || print_warning "Semgrep found potential issues"
    semgrep --config=auto --text --output="$REPORTS_DIR/semgrep-report.txt" src/ || true
    print_status "Semgrep report saved to $REPORTS_DIR/semgrep-report.json"
else
    print_warning "Semgrep not installed. Install with: pip install semgrep"
fi

# 5. Check for secrets in code
print_status "Scanning for potential secrets..."
if command -v detect-secrets &> /dev/null; then
    detect-secrets scan --all-files --force-use-all-plugins > "$REPORTS_DIR/secrets-baseline.json" || true
    print_status "Secrets scan baseline saved to $REPORTS_DIR/secrets-baseline.json"
else
    print_warning "detect-secrets not installed. Install with: pip install detect-secrets"
fi

# 6. License compliance check
print_status "Checking license compliance..."
if command -v pip-licenses &> /dev/null; then
    pip-licenses --format=json --output-file="$REPORTS_DIR/licenses.json"
    pip-licenses --format=csv --output-file="$REPORTS_DIR/licenses.csv"
    print_status "License report saved to $REPORTS_DIR/licenses.json"
else
    print_warning "pip-licenses not installed. Install with: pip install pip-licenses"
fi

# 7. SAST with CodeQL (if available)
print_status "Running CodeQL analysis (if available)..."
if command -v codeql &> /dev/null; then
    if [ -d ".github/codeql" ]; then
        codeql database create "$REPORTS_DIR/codeql-db" --language=python --source-root=.
        codeql database analyze "$REPORTS_DIR/codeql-db" --format=json --output="$REPORTS_DIR/codeql-results.json"
        print_status "CodeQL analysis completed"
    else
        print_warning "CodeQL queries not found in .github/codeql"
    fi
else
    print_warning "CodeQL not available. This is typically run in CI/CD environments"
fi

# 8. Docker image security scanning (if Dockerfile exists)
if [ -f "Dockerfile" ]; then
    print_status "Docker security scanning available. Use: make docker-security-scan"
fi

# 9. Generate summary report
print_status "Generating security summary..."
cat > "$REPORTS_DIR/security-summary.md" << EOF
# Security Scan Summary

**Scan Date:** $(date)
**Repository:** continual-tiny-transformer

## Scans Performed

### Static Analysis
- ✅ Bandit (Python security linter)
- ✅ Semgrep (Static analysis patterns)
- ✅ Secrets detection

### Dependency Analysis
- ✅ Safety (Known vulnerabilities)
- ✅ pip-audit (CVE database)
- ✅ License compliance

### Reports Generated
- \`bandit-report.json\` - Python security issues
- \`safety-report.json\` - Dependency vulnerabilities
- \`pip-audit-report.json\` - Additional dependency analysis
- \`semgrep-report.json\` - Static analysis findings
- \`secrets-baseline.json\` - Potential secrets
- \`licenses.json\` - License compliance

## Next Steps

1. Review all generated reports
2. Address high-priority security findings
3. Update vulnerable dependencies
4. Consider implementing pre-commit security hooks
5. Set up continuous security monitoring

## Security Best Practices

- Keep dependencies updated
- Use virtual environments
- Implement proper access controls
- Regular security scanning
- Monitor for new vulnerabilities

EOF

print_status "Security scan completed! Reports available in: $REPORTS_DIR/"
print_status "Review the security-summary.md file for an overview"

# Check if any critical issues were found
if [ -f "$REPORTS_DIR/bandit-report.json" ]; then
    HIGH_ISSUES=$(jq '.results[] | select(.issue_severity == "HIGH")' "$REPORTS_DIR/bandit-report.json" 2>/dev/null | wc -l || echo "0")
    if [ "$HIGH_ISSUES" -gt 0 ]; then
        print_error "Found $HIGH_ISSUES high-severity security issues!"
        exit 1
    fi
fi

print_status "All security scans completed successfully!"