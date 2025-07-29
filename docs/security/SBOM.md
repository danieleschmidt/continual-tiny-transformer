# Software Bill of Materials (SBOM) Configuration

## Overview

This document describes how to generate and manage Software Bill of Materials (SBOM) for the continual-tiny-transformer project. SBOMs provide transparency into software supply chain dependencies and enable automated vulnerability scanning.

## SBOM Generation Tools

### 1. Python-specific Tools

#### pip-audit
```bash
# Install pip-audit
pip install pip-audit

# Generate SBOM in CycloneDX format
pip-audit --format=cyclonedx-json --output=sbom-cyclonedx.json

# Generate SBOM in SPDX format  
pip-audit --format=spdx-json --output=sbom-spdx.json

# Vulnerability scanning with SBOM
pip-audit --format=cyclonedx-json --vulnerability-service=osv
```

#### syft (recommended)
```bash
# Install syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Generate SBOM for Python package
syft packages . -o cyclonedx-json=sbom-cyclonedx.json
syft packages . -o spdx-json=sbom-spdx.json
syft packages . -o table  # Human readable

# Generate SBOM for Docker image
syft packages continual-tiny-transformer:latest -o cyclonedx-json=docker-sbom.json
```

### 2. GitHub Actions Integration

Add to `.github/workflows/sbom.yml`:

```yaml
name: Generate SBOM

on:
  push:
    branches: [main]
  release:
    types: [published]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  sbom:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      security-events: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Install syft
        uses: anchore/sbom-action@v0
        
      - name: Generate Python SBOM
        run: |
          syft packages . -o cyclonedx-json=sbom-python.json
          syft packages . -o spdx-json=sbom-python-spdx.json
          
      - name: Build Docker image
        run: |
          docker build -t ${{ github.repository }}:latest .
          
      - name: Generate Docker SBOM
        run: |
          syft packages ${{ github.repository }}:latest -o cyclonedx-json=sbom-docker.json
          
      - name: Upload SBOMs as artifacts
        uses: actions/upload-artifact@v4
        with:
          name: sbom-files
          path: |
            sbom-*.json
            
      - name: Upload SBOM to release
        if: github.event_name == 'release'
        uses: softprops/action-gh-release@v1
        with:
          files: |
            sbom-*.json
```

## SBOM Validation

### 1. Format Validation
```bash
# Validate CycloneDX SBOM
npm install -g @cyclonedx/cli
cyclonedx validate sbom-cyclonedx.json

# Validate SPDX SBOM
pip install spdx-tools
spdx-tools -i sbom-spdx.json validate
```

### 2. Content Verification
```bash
# Check for required fields
jq '.components[] | select(.name and .version and .purl)' sbom-cyclonedx.json

# Verify package integrity
jq '.components[] | select(.hashes)' sbom-cyclonedx.json
```

## Vulnerability Scanning with SBOM

### 1. Grype Scanner
```bash
# Install grype
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# Scan SBOM for vulnerabilities
grype sbom:sbom-cyclonedx.json -o table
grype sbom:sbom-cyclonedx.json -o sarif > vulnerabilities.sarif
```

### 2. Trivy Scanner
```bash
# Install trivy
sudo apt-get install trivy

# Scan SBOM
trivy sbom sbom-cyclonedx.json
trivy sbom sbom-cyclonedx.json --format sarif > trivy-results.sarif
```

## SBOM Storage and Distribution

### 1. Repository Storage
```
sbom/
├── latest/
│   ├── sbom-python-cyclonedx.json
│   ├── sbom-python-spdx.json
│   └── sbom-docker-cyclonedx.json
├── v1.0.0/
│   └── ...
└── archive/
    └── ...
```

### 2. Attestation and Signing
```bash
# Sign SBOM with cosign
cosign sign-blob --bundle=sbom-signature.bundle sbom-cyclonedx.json

# Verify signature
cosign verify-blob --bundle=sbom-signature.bundle sbom-cyclonedx.json
```

## Compliance Requirements

### SLSA Level 2+ Requirements
- Automated SBOM generation
- Tamper-proof storage
- Version tracking
- Vulnerability scanning integration

### Executive Order 14028 (US Federal)
- CycloneDX or SPDX format
- Complete dependency enumeration
- Regular updates
- Machine-readable format

## Automation Scripts

### Generate SBOM Script
Create `scripts/generate-sbom.sh`:
```bash
#!/bin/bash
set -euo pipefail

VERSION=${1:-"latest"}
OUTPUT_DIR="sbom/${VERSION}"

mkdir -p "${OUTPUT_DIR}"

# Generate Python SBOMs
syft packages . -o cyclonedx-json="${OUTPUT_DIR}/sbom-python-cyclonedx.json"
syft packages . -o spdx-json="${OUTPUT_DIR}/sbom-python-spdx.json"

# Generate Docker SBOM if image exists
if docker images --quiet continual-tiny-transformer:latest; then
    syft packages continual-tiny-transformer:latest -o cyclonedx-json="${OUTPUT_DIR}/sbom-docker-cyclonedx.json"
fi

echo "SBOM files generated in ${OUTPUT_DIR}"
```

### Vulnerability Scan Script
Create `scripts/scan-vulnerabilities.sh`:
```bash
#!/bin/bash
set -euo pipefail

SBOM_FILE=${1:-"sbom/latest/sbom-python-cyclonedx.json"}
OUTPUT_DIR="security-reports"

mkdir -p "${OUTPUT_DIR}"

# Scan with multiple tools
grype "sbom:${SBOM_FILE}" -o json > "${OUTPUT_DIR}/grype-results.json"
trivy sbom "${SBOM_FILE}" --format json > "${OUTPUT_DIR}/trivy-results.json"

# Generate summary report
python scripts/vulnerability-summary.py "${OUTPUT_DIR}"
```

## Integration with CI/CD

### Pre-commit Hook
Add to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: check-sbom-freshness
      name: Check SBOM freshness
      entry: scripts/check-sbom-freshness.sh
      language: script
      pass_filenames: false
```

### Release Process
1. Generate SBOM before release
2. Include SBOM in release artifacts
3. Sign SBOM with release signing key
4. Upload to transparency log (if applicable)

## Best Practices

1. **Regular Updates**: Generate SBOMs on every release and weekly
2. **Multiple Formats**: Support both CycloneDX and SPDX for compatibility
3. **Comprehensive Coverage**: Include all runtime and build dependencies
4. **Vulnerability Integration**: Automate scanning of generated SBOMs
5. **Attestation**: Sign SBOMs for integrity verification
6. **Storage**: Keep historical SBOMs for compliance and analysis

## References

- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [CycloneDX Specification](https://cyclonedx.org/)
- [NTIA SBOM Minimum Elements](https://www.ntia.doc.gov/files/ntia/publications/sbom_minimum_elements_report.pdf)
- [Executive Order 14028](https://www.whitehouse.gov/briefing-room/presidential-actions/2021/05/12/executive-order-on-improving-the-nations-cybersecurity/)