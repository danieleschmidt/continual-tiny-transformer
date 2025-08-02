#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) generator for continual-tiny-transformer.
Generates SPDX and CycloneDX format SBOMs for supply chain security.
"""

import json
import sys
import subprocess
import pkg_resources
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import hashlib
import uuid


class SBOMGenerator:
    """Generate Software Bill of Materials for the project."""
    
    def __init__(self, project_name: str = "continual-tiny-transformer"):
        self.project_name = project_name
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.uuid = str(uuid.uuid4())
        
    def get_installed_packages(self) -> List[Dict[str, Any]]:
        """Get list of installed Python packages with versions."""
        packages = []
        
        for dist in pkg_resources.working_set:
            package_info = {
                "name": dist.project_name,
                "version": dist.version,
                "location": dist.location,
                "requires": [str(req) for req in dist.requires()],
            }
            
            # Try to get additional metadata
            try:
                metadata = dist.get_metadata('METADATA')
                lines = metadata.split('\n')
                for line in lines:
                    if line.startswith('Home-page:'):
                        package_info['homepage'] = line.split(':', 1)[1].strip()
                    elif line.startswith('License:'):
                        package_info['license'] = line.split(':', 1)[1].strip()
                    elif line.startswith('Author:'):
                        package_info['author'] = line.split(':', 1)[1].strip()
            except:
                pass
                
            packages.append(package_info)
        
        return sorted(packages, key=lambda x: x['name'].lower())
    
    def get_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except:
            return ""
    
    def get_project_files(self) -> List[Dict[str, Any]]:
        """Get list of project source files."""
        files = []
        src_path = Path("src")
        
        if src_path.exists():
            for file_path in src_path.rglob("*.py"):
                if not any(part.startswith('.') for part in file_path.parts):
                    files.append({
                        "name": str(file_path),
                        "size": file_path.stat().st_size,
                        "hash": self.get_file_hash(file_path),
                        "type": "source"
                    })
        
        return files
    
    def generate_spdx_sbom(self) -> Dict[str, Any]:
        """Generate SBOM in SPDX format."""
        packages = self.get_installed_packages()
        files = self.get_project_files()
        
        # SPDX document structure
        spdx_doc = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "documentName": f"{self.project_name}-sbom",
            "documentNamespace": f"https://github.com/your-org/{self.project_name}/{self.uuid}",
            "creators": ["Tool: sbom-generator"],
            "created": self.timestamp,
            "packages": [],
            "files": [],
            "relationships": []
        }
        
        # Add main package
        main_package = {
            "SPDXID": "SPDXRef-Package-main",
            "name": self.project_name,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": True,
            "copyrightText": "NOASSERTION",
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT"
        }
        spdx_doc["packages"].append(main_package)
        
        # Add dependency packages
        for i, pkg in enumerate(packages):
            package = {
                "SPDXID": f"SPDXRef-Package-{i}",
                "name": pkg["name"],
                "version": pkg["version"],
                "downloadLocation": pkg.get("homepage", "NOASSERTION"),
                "filesAnalyzed": False,
                "copyrightText": "NOASSERTION",
                "licenseConcluded": pkg.get("license", "NOASSERTION"),
                "licenseDeclared": pkg.get("license", "NOASSERTION")
            }
            spdx_doc["packages"].append(package)
            
            # Add relationship
            spdx_doc["relationships"].append({
                "spdxElementId": "SPDXRef-Package-main",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": f"SPDXRef-Package-{i}"
            })
        
        # Add files
        for i, file_info in enumerate(files):
            file_entry = {
                "SPDXID": f"SPDXRef-File-{i}",
                "fileName": file_info["name"],
                "checksums": [{
                    "algorithm": "SHA256",
                    "checksumValue": file_info["hash"]
                }],
                "copyrightText": "NOASSERTION",
                "licenseConcluded": "MIT"
            }
            spdx_doc["files"].append(file_entry)
            
            # Add file relationship to main package
            spdx_doc["relationships"].append({
                "spdxElementId": "SPDXRef-Package-main",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": f"SPDXRef-File-{i}"
            })
        
        return spdx_doc
    
    def generate_cyclonedx_sbom(self) -> Dict[str, Any]:
        """Generate SBOM in CycloneDX format."""
        packages = self.get_installed_packages()
        
        cyclonedx_doc = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{self.uuid}",
            "version": 1,
            "metadata": {
                "timestamp": self.timestamp,
                "tools": [{
                    "vendor": "continual-tiny-transformer",
                    "name": "sbom-generator",
                    "version": "1.0.0"
                }],
                "component": {
                    "type": "application",
                    "bom-ref": "pkg:github/your-org/continual-tiny-transformer",
                    "name": self.project_name,
                    "version": "0.1.0",
                    "description": "Memory-efficient continual learning for transformers",
                    "licenses": [{"license": {"id": "MIT"}}]
                }
            },
            "components": []
        }
        
        # Add dependency components
        for pkg in packages:
            component = {
                "type": "library",
                "bom-ref": f"pkg:pypi/{pkg['name']}@{pkg['version']}",
                "name": pkg["name"],
                "version": pkg["version"],
                "description": "",
                "scope": "required",
                "purl": f"pkg:pypi/{pkg['name']}@{pkg['version']}"
            }
            
            if "license" in pkg:
                component["licenses"] = [{"license": {"name": pkg["license"]}}]
            
            if "homepage" in pkg:
                component["externalReferences"] = [{
                    "type": "website",
                    "url": pkg["homepage"]
                }]
            
            cyclonedx_doc["components"].append(component)
        
        return cyclonedx_doc
    
    def generate_simple_sbom(self) -> Dict[str, Any]:
        """Generate a simple, human-readable SBOM."""
        packages = self.get_installed_packages()
        files = self.get_project_files()
        
        return {
            "project": {
                "name": self.project_name,
                "version": "0.1.0",
                "generated": self.timestamp,
                "generator": "sbom-generator"
            },
            "dependencies": {
                "total_count": len(packages),
                "packages": packages
            },
            "source_files": {
                "total_count": len(files),
                "files": files
            },
            "summary": {
                "total_dependencies": len(packages),
                "total_source_files": len(files),
                "unique_licenses": list(set(
                    pkg.get("license", "Unknown") for pkg in packages
                ))
            }
        }


def main():
    """Main function to generate SBOM files."""
    generator = SBOMGenerator()
    
    # Create output directory
    output_dir = Path("sbom")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating Software Bill of Materials (SBOM)...")
    
    # Generate SPDX format
    print("  Generating SPDX format...")
    spdx_sbom = generator.generate_spdx_sbom()
    with open(output_dir / "sbom.spdx.json", "w") as f:
        json.dump(spdx_sbom, f, indent=2)
    
    # Generate CycloneDX format
    print("  Generating CycloneDX format...")
    cyclonedx_sbom = generator.generate_cyclonedx_sbom()
    with open(output_dir / "sbom.cyclonedx.json", "w") as f:
        json.dump(cyclonedx_sbom, f, indent=2)
    
    # Generate simple format
    print("  Generating simple format...")
    simple_sbom = generator.generate_simple_sbom()
    with open(output_dir / "sbom.json", "w") as f:
        json.dump(simple_sbom, f, indent=2)
    
    # Generate human-readable report
    print("  Generating human-readable report...")
    with open(output_dir / "sbom-report.md", "w") as f:
        f.write(f"""# Software Bill of Materials

**Project:** {generator.project_name}  
**Generated:** {generator.timestamp}  
**Total Dependencies:** {len(simple_sbom['dependencies']['packages'])}  
**Total Source Files:** {len(simple_sbom['source_files']['files'])}  

## Dependencies

| Package | Version | License |
|---------|---------|---------|
""")
        for pkg in simple_sbom['dependencies']['packages']:
            license_info = pkg.get('license', 'Unknown')
            f.write(f"| {pkg['name']} | {pkg['version']} | {license_info} |\n")
        
        f.write(f"""
## License Summary

""")
        for license_name in simple_sbom['summary']['unique_licenses']:
            count = sum(1 for pkg in simple_sbom['dependencies']['packages'] 
                       if pkg.get('license', 'Unknown') == license_name)
            f.write(f"- **{license_name}**: {count} packages\n")
        
        f.write(f"""
## Files Generated

- `sbom.spdx.json` - SPDX 2.3 format
- `sbom.cyclonedx.json` - CycloneDX 1.4 format  
- `sbom.json` - Simple JSON format
- `sbom-report.md` - This human-readable report

## Usage

These SBOM files can be used for:
- Supply chain security analysis
- License compliance checking
- Vulnerability scanning
- Dependency tracking
- Compliance reporting

""")
    
    print(f"\nSBOM generation complete!")
    print(f"Files generated in: {output_dir}/")
    print(f"  - sbom.spdx.json (SPDX format)")
    print(f"  - sbom.cyclonedx.json (CycloneDX format)")
    print(f"  - sbom.json (Simple format)")
    print(f"  - sbom-report.md (Human-readable report)")


if __name__ == "__main__":
    main()