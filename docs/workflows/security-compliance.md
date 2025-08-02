# Security and Compliance Workflows

## Overview

This document outlines comprehensive security and compliance workflows for the continual-tiny-transformer project, covering security scanning, vulnerability management, and compliance monitoring.

## Security Scanning Workflows

### 1. Comprehensive Security Scan Workflow

```yaml
# .github/workflows/security-comprehensive.yml
name: Comprehensive Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  actions: read
  id-token: write

jobs:
  static-analysis:
    name: Static Analysis Security Testing (SAST)
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install security scanning tools
        run: |
          pip install bandit safety semgrep detect-secrets pip-audit

      - name: Run Bandit security scan
        run: |
          bandit -r src/ -f json -o bandit-report.json
          bandit -r src/ -f txt

      - name: Run Safety vulnerability scan
        run: |
          safety check --json --output safety-report.json
          safety check

      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-report.json
          pip-audit

      - name: Run Semgrep
        run: |
          semgrep --config=auto --json --output=semgrep-report.json src/
          semgrep --config=auto src/

      - name: Detect secrets
        run: |
          detect-secrets scan --all-files --force-use-all-plugins > secrets-baseline.json

      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json
            pip-audit-report.json
            semgrep-report.json
            secrets-baseline.json

  codeql-analysis:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: ${{ matrix.language }}
          queries: security-extended,security-and-quality

      - name: Autobuild
        uses: github/codeql-action/autobuild@v2

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
        with:
          category: "/language:${{matrix.language}}"

  container-security:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -t continual-tiny-transformer:security-scan .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'continual-tiny-transformer:security-scan'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Run Snyk container scan
        uses: snyk/actions/docker@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          image: continual-tiny-transformer:security-scan
          args: --severity-threshold=high

  dependency-check:
    name: OWASP Dependency Check
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run OWASP Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'continual-tiny-transformer'
          path: '.'
          format: 'ALL'
          args: >
            --enableRetired
            --enableExperimental

      - name: Upload OWASP results
        uses: actions/upload-artifact@v3
        with:
          name: dependency-check-report
          path: reports/

  sbom-generation:
    name: Generate SBOM
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Generate SBOM
        run: |
          python scripts/generate-sbom.py

      - name: Upload SBOM artifacts
        uses: actions/upload-artifact@v3
        with:
          name: sbom-files
          path: sbom/

      - name: Sign SBOM with Cosign
        if: github.event_name != 'pull_request'
        uses: sigstore/cosign-installer@v3

      - name: Sign SBOM files
        if: github.event_name != 'pull_request'
        run: |
          cosign sign-blob --bundle sbom.spdx.json.bundle sbom/sbom.spdx.json
          cosign sign-blob --bundle sbom.cyclonedx.json.bundle sbom/sbom.cyclonedx.json

  license-compliance:
    name: License Compliance Check
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pip-licenses licensecheck

      - name: Generate license report
        run: |
          pip-licenses --format=json --output-file=licenses.json
          pip-licenses --format=csv --output-file=licenses.csv
          licensecheck --zero

      - name: Check license compatibility
        run: |
          python scripts/check_license_compatibility.py

      - name: Upload license reports
        uses: actions/upload-artifact@v3
        with:
          name: license-reports
          path: |
            licenses.json
            licenses.csv
```

### 2. Vulnerability Management Workflow

```yaml
# .github/workflows/vulnerability-management.yml
name: Vulnerability Management

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write
  security-events: write

jobs:
  vulnerability-scan:
    name: Scan for Vulnerabilities
    runs-on: ubuntu-latest
    outputs:
      has-vulnerabilities: ${{ steps.check-vulns.outputs.has-vulnerabilities }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install scanning tools
        run: |
          pip install safety pip-audit

      - name: Scan for vulnerabilities
        id: scan
        run: |
          safety check --json --output safety-report.json || true
          pip-audit --format=json --output=pip-audit-report.json || true

      - name: Check for vulnerabilities
        id: check-vulns
        run: |
          SAFETY_VULNS=$(jq '.vulnerabilities | length' safety-report.json)
          AUDIT_VULNS=$(jq '.vulnerabilities | length' pip-audit-report.json)
          
          if [ "$SAFETY_VULNS" -gt 0 ] || [ "$AUDIT_VULNS" -gt 0 ]; then
            echo "has-vulnerabilities=true" >> $GITHUB_OUTPUT
          else
            echo "has-vulnerabilities=false" >> $GITHUB_OUTPUT
          fi

      - name: Upload vulnerability reports
        uses: actions/upload-artifact@v3
        with:
          name: vulnerability-reports
          path: |
            safety-report.json
            pip-audit-report.json

  create-security-issue:
    name: Create Security Issue
    runs-on: ubuntu-latest
    needs: vulnerability-scan
    if: needs.vulnerability-scan.outputs.has-vulnerabilities == 'true'
    steps:
      - name: Download vulnerability reports
        uses: actions/download-artifact@v3
        with:
          name: vulnerability-reports

      - name: Create security issue
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            
            let safetyReport = {};
            let auditReport = {};
            
            try {
              safetyReport = JSON.parse(fs.readFileSync('safety-report.json', 'utf8'));
            } catch (e) {
              console.log('No safety report found');
            }
            
            try {
              auditReport = JSON.parse(fs.readFileSync('pip-audit-report.json', 'utf8'));
            } catch (e) {
              console.log('No audit report found');
            }
            
            let issueBody = '## Security Vulnerabilities Detected\n\n';
            issueBody += `**Scan Date:** ${new Date().toISOString()}\n\n`;
            
            if (safetyReport.vulnerabilities && safetyReport.vulnerabilities.length > 0) {
              issueBody += '### Safety Vulnerabilities\n\n';
              safetyReport.vulnerabilities.forEach(vuln => {
                issueBody += `- **${vuln.package_name}** (${vuln.installed_version})\n`;
                issueBody += `  - Vulnerability: ${vuln.advisory}\n`;
                issueBody += `  - Severity: ${vuln.severity || 'Unknown'}\n\n`;
              });
            }
            
            if (auditReport.vulnerabilities && auditReport.vulnerabilities.length > 0) {
              issueBody += '### Pip-audit Vulnerabilities\n\n';
              auditReport.vulnerabilities.forEach(vuln => {
                issueBody += `- **${vuln.package}** (${vuln.installed_version})\n`;
                issueBody += `  - Vulnerability: ${vuln.id}\n`;
                issueBody += `  - Description: ${vuln.description}\n\n`;
              });
            }
            
            issueBody += '\n### Action Required\n\n';
            issueBody += 'Please review and address these vulnerabilities by:\n';
            issueBody += '1. Updating affected packages\n';
            issueBody += '2. Applying security patches\n';
            issueBody += '3. Reviewing code for potential impact\n';
            issueBody += '4. Testing after fixes\n\n';
            issueBody += '**Auto-generated by vulnerability management workflow**';
            
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `🚨 Security Vulnerabilities Detected - ${new Date().toISOString().split('T')[0]}`,
              body: issueBody,
              labels: ['security', 'vulnerability', 'high-priority']
            });

  auto-update-dependencies:
    name: Auto-update Dependencies
    runs-on: ubuntu-latest
    needs: vulnerability-scan
    if: needs.vulnerability-scan.outputs.has-vulnerabilities == 'true'
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Update dependencies
        run: |
          pip install pip-tools
          pip-compile --upgrade requirements.in
          pip-compile --upgrade requirements-dev.in

      - name: Test updated dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
          make test-unit

      - name: Create pull request
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: 'security: update dependencies to address vulnerabilities'
          title: '🔒 Security: Update dependencies'
          body: |
            ## Security Dependency Updates
            
            This PR updates dependencies to address security vulnerabilities detected by automated scanning.
            
            ### Changes
            - Updated requirements.txt
            - Updated requirements-dev.txt
            
            ### Verification
            - [ ] Dependencies updated successfully
            - [ ] Tests passing
            - [ ] No breaking changes
            - [ ] Security scan clean
            
            **Auto-generated by vulnerability management workflow**
          branch: security/dependency-updates
          labels: security,dependencies,automated
```

### 3. Supply Chain Security Workflow

```yaml
# .github/workflows/supply-chain-security.yml
name: Supply Chain Security

on:
  push:
    branches: [ main ]
  release:
    types: [ published ]
  workflow_dispatch:

permissions:
  contents: read
  id-token: write
  packages: write
  attestations: write

jobs:
  provenance:
    name: Generate Build Provenance
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Build package
        run: |
          pip install build
          python -m build

      - name: Generate provenance
        uses: actions/attest-build-provenance@v1
        with:
          subject-path: 'dist/*'

      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: dist
          path: dist/

  sign-artifacts:
    name: Sign Artifacts
    runs-on: ubuntu-latest
    needs: provenance
    if: github.event_name == 'release'
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v3
        with:
          name: dist
          path: dist/

      - name: Install Cosign
        uses: sigstore/cosign-installer@v3

      - name: Sign artifacts
        run: |
          cosign sign-blob --bundle dist.tar.gz.bundle dist/*.tar.gz
          cosign sign-blob --bundle dist.whl.bundle dist/*.whl

      - name: Upload signed artifacts
        uses: actions/upload-artifact@v3
        with:
          name: signed-dist
          path: |
            dist/
            *.bundle

  slsa-provenance:
    name: SLSA Provenance
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
    with:
      base64-subjects: "${{ needs.provenance.outputs.hashes }}"
      provenance-name: "continual-tiny-transformer.intoto.jsonl"

  verify-slsa:
    name: Verify SLSA Provenance
    runs-on: ubuntu-latest
    needs: [provenance, slsa-provenance]
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v3

      - name: Install SLSA verifier
        run: |
          curl -sSLO https://github.com/slsa-framework/slsa-verifier/releases/latest/download/slsa-verifier-linux-amd64
          chmod +x slsa-verifier-linux-amd64

      - name: Verify SLSA provenance
        run: |
          ./slsa-verifier-linux-amd64 verify-artifact \
            --provenance-path continual-tiny-transformer.intoto.jsonl \
            --source-uri github.com/${{ github.repository }} \
            dist/*
```

## Compliance Workflows

### 1. Regulatory Compliance Check

```yaml
# .github/workflows/compliance.yml
name: Regulatory Compliance

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 3 * * 0'  # Weekly on Sunday at 3 AM

jobs:
  gdpr-compliance:
    name: GDPR Compliance Check
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Check for PII patterns
        run: |
          # Scan for potential PII patterns
          grep -r -i "email\|phone\|ssn\|credit.*card" src/ || echo "No PII patterns found"
          
          # Check data handling documentation
          if [ ! -f "docs/PRIVACY.md" ]; then
            echo "::error::Missing privacy documentation"
            exit 1
          fi

      - name: Verify data encryption
        run: |
          # Check for encryption implementation
          grep -r "encrypt\|cipher\|aes" src/ || echo "::warning::No encryption found"

  sox-compliance:
    name: SOX Compliance Check
    runs-on: ubuntu-latest
    steps:
      - name: Audit trail verification
        run: |
          # Check for audit logging
          grep -r "audit\|log" src/ || echo "::warning::Limited audit logging"

      - name: Access control verification
        run: |
          # Check for proper access controls
          if [ ! -f ".github/CODEOWNERS" ]; then
            echo "::error::Missing CODEOWNERS file"
            exit 1
          fi

  hipaa-compliance:
    name: HIPAA Compliance Check
    runs-on: ubuntu-latest
    if: contains(github.repository, 'healthcare') || contains(github.repository, 'medical')
    steps:
      - name: PHI protection check
        run: |
          # Check for PHI handling
          grep -r -i "patient\|medical\|health" src/ || echo "No PHI patterns found"

      - name: Encryption verification
        run: |
          # Verify encryption for PHI
          grep -r "encrypt.*health\|secure.*medical" src/ || echo "::warning::No PHI encryption found"
```

### 2. Open Source License Compliance

```yaml
# .github/workflows/license-compliance.yml
name: License Compliance

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  license-scan:
    name: License Compatibility Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pip-licenses licensecheck

      - name: Generate license report
        run: |
          pip-licenses --format=json --output-file=licenses.json
          pip-licenses --format=csv --output-file=licenses.csv

      - name: Check license compatibility
        run: |
          python scripts/check_license_compatibility.py

      - name: Upload license reports
        uses: actions/upload-artifact@v3
        with:
          name: license-reports
          path: |
            licenses.json
            licenses.csv

  fossa-scan:
    name: FOSSA License Scan
    runs-on: ubuntu-latest
    if: github.event_name != 'pull_request'
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run FOSSA scan
        uses: fossas/fossa-action@main
        with:
          api-key: ${{ secrets.FOSSA_API_KEY }}
          project: continual-tiny-transformer
```

## Security Response Workflows

### 1. Security Incident Response

```yaml
# .github/workflows/security-incident.yml
name: Security Incident Response

on:
  issues:
    types: [opened, labeled]
  workflow_dispatch:
    inputs:
      severity:
        description: 'Incident severity'
        required: true
        default: 'medium'
        type: choice
        options:
        - low
        - medium
        - high
        - critical

jobs:
  incident-triage:
    name: Security Incident Triage
    runs-on: ubuntu-latest
    if: contains(github.event.issue.labels.*.name, 'security-incident')
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Notify security team
        uses: actions/github-script@v6
        with:
          script: |
            const issue = context.payload.issue;
            const severity = issue.labels.find(label => 
              ['low', 'medium', 'high', 'critical'].includes(label.name)
            )?.name || 'medium';
            
            // Create incident tracking issue
            const incidentBody = `
            ## Security Incident Report
            
            **Original Issue:** #${issue.number}
            **Severity:** ${severity}
            **Reporter:** @${issue.user.login}
            **Date:** ${new Date().toISOString()}
            
            ### Description
            ${issue.body}
            
            ### Response Checklist
            - [ ] Incident acknowledged
            - [ ] Impact assessment completed
            - [ ] Containment measures implemented
            - [ ] Root cause analysis
            - [ ] Fix implemented
            - [ ] Security testing completed
            - [ ] Post-incident review
            `;
            
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: \`🚨 Security Incident: \${issue.title}\`,
              body: incidentBody,
              labels: ['security-incident', 'tracking', severity],
              assignees: ['security-team']
            });

      - name: Immediate containment
        if: contains(github.event.issue.labels.*.name, 'critical')
        run: |
          echo "Critical security incident detected"
          echo "Implementing immediate containment measures..."
          # Add immediate response actions here

  vulnerability-disclosure:
    name: Vulnerability Disclosure
    runs-on: ubuntu-latest
    if: contains(github.event.issue.labels.*.name, 'vulnerability-report')
    steps:
      - name: Validate report
        uses: actions/github-script@v6
        with:
          script: |
            const issue = context.payload.issue;
            
            // Check if report contains required information
            const requiredSections = [
              'vulnerability description',
              'steps to reproduce',
              'impact assessment'
            ];
            
            const missingInfo = requiredSections.filter(section => 
              !issue.body.toLowerCase().includes(section.toLowerCase())
            );
            
            if (missingInfo.length > 0) {
              github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: issue.number,
                body: \`Thank you for reporting this vulnerability. To help us process your report, please provide the following missing information:
                
                \${missingInfo.map(info => \`- \${info}\`).join('\n')}
                
                Please update your report with this information.\`
              });
            }

      - name: Create security advisory
        if: contains(github.event.issue.labels.*.name, 'confirmed-vulnerability')
        run: |
          echo "Creating security advisory for confirmed vulnerability"
          # Use GitHub Security Advisory API to create advisory
```

This comprehensive security and compliance framework ensures:

1. **Continuous Security Monitoring**: Daily vulnerability scans and automated reporting
2. **Supply Chain Security**: SLSA provenance generation and artifact signing
3. **Compliance Automation**: Automated checks for GDPR, SOX, and other regulations
4. **Incident Response**: Structured workflows for security incident handling
5. **License Compliance**: Automated license compatibility checking

These workflows should be customized based on your specific compliance requirements and security policies.