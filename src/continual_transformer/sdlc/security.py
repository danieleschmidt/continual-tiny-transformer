"""Security validation and scanning for SDLC processes."""

import hashlib
import json
import logging
import os
import re
import subprocess
import time
import threading
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
import uuid

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security scanning levels."""
    BASIC = "basic"
    STANDARD = "standard" 
    STRICT = "strict"
    PARANOID = "paranoid"


class VulnerabilityLevel(Enum):
    """Vulnerability severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability finding."""
    id: str
    title: str
    description: str
    severity: VulnerabilityLevel
    file_path: str
    line_number: Optional[int] = None
    rule_id: Optional[str] = None
    cwe_id: Optional[str] = None
    confidence: str = "medium"
    remediation: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class SecurityScanResult:
    """Results from security scanning."""
    scan_id: str
    scan_type: str
    start_time: float
    end_time: float
    vulnerabilities: List[SecurityVulnerability]
    files_scanned: int
    scan_config: Dict[str, Any]
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def vulnerability_count_by_severity(self) -> Dict[str, int]:
        counts = {level.value: 0 for level in VulnerabilityLevel}
        for vuln in self.vulnerabilities:
            counts[vuln.severity.value] += 1
        return counts
    
    @property
    def has_critical_vulnerabilities(self) -> bool:
        return any(v.severity == VulnerabilityLevel.CRITICAL for v in self.vulnerabilities)
    
    @property
    def risk_score(self) -> float:
        """Calculate risk score based on vulnerabilities."""
        weights = {
            VulnerabilityLevel.LOW: 1,
            VulnerabilityLevel.MEDIUM: 3,
            VulnerabilityLevel.HIGH: 7,
            VulnerabilityLevel.CRITICAL: 10
        }
        
        score = sum(weights[v.severity] for v in self.vulnerabilities)
        return min(100, score)  # Cap at 100


class SecretScanner:
    """Scanner for detecting secrets and sensitive information."""
    
    def __init__(self):
        self.secret_patterns = self._init_secret_patterns()
        self.exclude_patterns = [
            r'test[_\-]?data',
            r'example[s]?',
            r'mock[_\-]?',
            r'\.test\.',
            r'fixtures?',
            r'__pycache__',
            r'\.git'
        ]
        self.logger = logging.getLogger(f"{__name__}.SecretScanner")
    
    def _init_secret_patterns(self) -> Dict[str, str]:
        """Initialize secret detection patterns."""
        return {
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'aws_secret_key': r'[0-9a-zA-Z/+]{40}',
            'api_key': r'(?i)api[_\-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9]{16,}',
            'password': r'(?i)password["\']?\s*[:=]\s*["\'][^"\']{8,}',
            'secret': r'(?i)secret["\']?\s*[:=]\s*["\'][^"\']{8,}',
            'token': r'(?i)token["\']?\s*[:=]\s*["\'][a-zA-Z0-9]{16,}',
            'private_key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
            'jwt_token': r'eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*',
            'github_token': r'ghp_[a-zA-Z0-9]{36}',
            'slack_token': r'xox[baprs]-[a-zA-Z0-9-]+',
            'database_url': r'(?i)(mysql|postgres|mongodb)://[^\s]+',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        }
    
    def scan_file(self, file_path: Path) -> List[SecurityVulnerability]:
        """Scan a single file for secrets."""
        vulnerabilities = []
        
        # Skip excluded paths
        path_str = str(file_path)
        if any(re.search(pattern, path_str, re.IGNORECASE) for pattern in self.exclude_patterns):
            return vulnerabilities
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for pattern_name, pattern in self.secret_patterns.items():
                    matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
                    
                    for match in matches:
                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Check if it's a commented line (likely safe)
                        line_content = lines[line_num - 1].strip() if line_num <= len(lines) else ""
                        is_comment = line_content.startswith('#') or line_content.startswith('//')
                        
                        severity = VulnerabilityLevel.LOW if is_comment else VulnerabilityLevel.HIGH
                        if pattern_name in ['aws_secret_key', 'private_key', 'database_url']:
                            severity = VulnerabilityLevel.CRITICAL if not is_comment else VulnerabilityLevel.MEDIUM
                        
                        vulnerability = SecurityVulnerability(
                            id=f"secret_{pattern_name}_{hashlib.md5(match.group().encode()).hexdigest()[:8]}",
                            title=f"Potential {pattern_name.replace('_', ' ').title()} Found",
                            description=f"Detected potential {pattern_name} in file",
                            severity=severity,
                            file_path=str(file_path),
                            line_number=line_num,
                            rule_id=f"SECRET_{pattern_name.upper()}",
                            confidence="medium" if is_comment else "high",
                            remediation="Remove or encrypt sensitive information"
                        )
                        
                        vulnerabilities.append(vulnerability)
                        
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
        
        return vulnerabilities
    
    def scan_directory(self, directory: Path, extensions: Set[str] = None) -> List[SecurityVulnerability]:
        """Scan directory for secrets."""
        if extensions is None:
            extensions = {'.py', '.js', '.ts', '.json', '.yaml', '.yml', '.env', '.cfg', '.conf', '.ini'}
        
        vulnerabilities = []
        files_scanned = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                file_vulnerabilities = self.scan_file(file_path)
                vulnerabilities.extend(file_vulnerabilities)
                files_scanned += 1
        
        self.logger.info(f"Secret scan completed: {files_scanned} files scanned, {len(vulnerabilities)} issues found")
        return vulnerabilities


class DependencyScanner:
    """Scanner for vulnerable dependencies."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.logger = logging.getLogger(f"{__name__}.DependencyScanner")
    
    def scan_python_dependencies(self) -> List[SecurityVulnerability]:
        """Scan Python dependencies for vulnerabilities using safety."""
        vulnerabilities = []
        
        try:
            # Run safety check
            result = subprocess.run(
                ['safety', 'check', '--json'],
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # No vulnerabilities found
                self.logger.info("No dependency vulnerabilities found")
                return vulnerabilities
            
            # Parse safety output
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    for item in safety_data:
                        vulnerability = SecurityVulnerability(
                            id=f"dep_{item.get('vulnerability_id', 'unknown')}",
                            title=f"Vulnerable dependency: {item.get('package', 'unknown')}",
                            description=item.get('advisory', 'No description available'),
                            severity=self._map_safety_severity(item.get('severity', 'medium')),
                            file_path="requirements/dependencies",
                            rule_id="DEPENDENCY_VULNERABILITY",
                            remediation=f"Update {item.get('package', 'package')} to version {item.get('safe_version', 'latest')}"
                        )
                        vulnerabilities.append(vulnerability)
                        
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse safety output")
            
            # Alternative: use pip-audit if safety fails
            if not vulnerabilities:
                vulnerabilities = self._scan_with_pip_audit()
                
        except subprocess.TimeoutExpired:
            self.logger.error("Dependency scan timed out")
        except FileNotFoundError:
            self.logger.warning("Safety tool not found, trying pip-audit")
            vulnerabilities = self._scan_with_pip_audit()
        except Exception as e:
            self.logger.error(f"Dependency scan failed: {e}")
        
        return vulnerabilities
    
    def _scan_with_pip_audit(self) -> List[SecurityVulnerability]:
        """Scan with pip-audit as fallback."""
        vulnerabilities = []
        
        try:
            result = subprocess.run(
                ['pip-audit', '--format=json'],
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    vulnerabilities_data = audit_data.get('vulnerabilities', [])
                    
                    for vuln_data in vulnerabilities_data:
                        vulnerability = SecurityVulnerability(
                            id=f"audit_{vuln_data.get('id', 'unknown')}",
                            title=f"Vulnerable package: {vuln_data.get('package', 'unknown')}",
                            description=vuln_data.get('description', 'No description available'),
                            severity=self._map_audit_severity(vuln_data.get('fix_available', True)),
                            file_path="pip_dependencies",
                            rule_id="PIP_AUDIT_VULNERABILITY"
                        )
                        vulnerabilities.append(vulnerability)
                        
                except json.JSONDecodeError:
                    self.logger.error("Failed to parse pip-audit output")
                    
        except Exception as e:
            self.logger.warning(f"pip-audit scan failed: {e}")
        
        return vulnerabilities
    
    def _map_safety_severity(self, severity: str) -> VulnerabilityLevel:
        """Map safety severity to our enum."""
        severity_map = {
            'low': VulnerabilityLevel.LOW,
            'medium': VulnerabilityLevel.MEDIUM,
            'high': VulnerabilityLevel.HIGH,
            'critical': VulnerabilityLevel.CRITICAL
        }
        return severity_map.get(severity.lower(), VulnerabilityLevel.MEDIUM)
    
    def _map_audit_severity(self, fix_available: bool) -> VulnerabilityLevel:
        """Map pip-audit data to severity."""
        return VulnerabilityLevel.HIGH if fix_available else VulnerabilityLevel.CRITICAL


class CodeScanner:
    """Scanner for code security issues using Bandit."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.logger = logging.getLogger(f"{__name__}.CodeScanner")
    
    def scan_python_code(self) -> List[SecurityVulnerability]:
        """Scan Python code for security issues."""
        vulnerabilities = []
        
        try:
            # Run bandit scan
            result = subprocess.run(
                ['bandit', '-r', 'src/', '-f', 'json'],
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse bandit output (it returns non-zero even on successful scan with findings)
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    results = bandit_data.get('results', [])
                    
                    for item in results:
                        severity = self._map_bandit_severity(
                            item.get('issue_severity', 'MEDIUM'),
                            item.get('issue_confidence', 'MEDIUM')
                        )
                        
                        vulnerability = SecurityVulnerability(
                            id=f"bandit_{item.get('test_id', 'unknown')}_{hashlib.md5(str(item).encode()).hexdigest()[:8]}",
                            title=item.get('test_name', 'Security Issue'),
                            description=item.get('issue_text', 'No description available'),
                            severity=severity,
                            file_path=item.get('filename', 'unknown'),
                            line_number=item.get('line_number'),
                            rule_id=item.get('test_id'),
                            cwe_id=item.get('cwe', {}).get('id') if item.get('cwe') else None,
                            confidence=item.get('issue_confidence', 'MEDIUM').lower(),
                            remediation=item.get('more_info', 'Review code for security best practices')
                        )
                        
                        vulnerabilities.append(vulnerability)
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse bandit output: {e}")
                    
        except subprocess.TimeoutExpired:
            self.logger.error("Code scan timed out")
        except FileNotFoundError:
            self.logger.warning("Bandit tool not found")
        except Exception as e:
            self.logger.error(f"Code scan failed: {e}")
        
        return vulnerabilities
    
    def _map_bandit_severity(self, severity: str, confidence: str) -> VulnerabilityLevel:
        """Map bandit severity and confidence to our enum."""
        severity_map = {
            'LOW': VulnerabilityLevel.LOW,
            'MEDIUM': VulnerabilityLevel.MEDIUM,
            'HIGH': VulnerabilityLevel.HIGH
        }
        
        base_severity = severity_map.get(severity.upper(), VulnerabilityLevel.MEDIUM)
        
        # Adjust based on confidence
        if confidence.upper() == 'LOW':
            # Downgrade low confidence findings
            if base_severity == VulnerabilityLevel.HIGH:
                return VulnerabilityLevel.MEDIUM
            elif base_severity == VulnerabilityLevel.MEDIUM:
                return VulnerabilityLevel.LOW
        elif confidence.upper() == 'HIGH':
            # Upgrade high confidence findings
            if base_severity == VulnerabilityLevel.MEDIUM:
                return VulnerabilityLevel.HIGH
        
        return base_severity


class SecurityValidator:
    """Main security validation and scanning orchestrator."""
    
    def __init__(self, project_path: str, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.project_path = Path(project_path)
        self.security_level = security_level
        
        # Initialize scanners
        self.secret_scanner = SecretScanner()
        self.dependency_scanner = DependencyScanner(self.project_path)
        self.code_scanner = CodeScanner(self.project_path)
        
        # Security configuration
        self.scan_config = self._init_scan_config()
        
        # Results storage
        self.scan_history = []
        
        self.logger = logging.getLogger(f"{__name__}.SecurityValidator")
    
    def _init_scan_config(self) -> Dict[str, Any]:
        """Initialize scanning configuration based on security level."""
        configs = {
            SecurityLevel.BASIC: {
                'scan_secrets': True,
                'scan_dependencies': False,
                'scan_code': False,
                'fail_on_high': False,
                'fail_on_critical': True
            },
            SecurityLevel.STANDARD: {
                'scan_secrets': True,
                'scan_dependencies': True,
                'scan_code': True,
                'fail_on_high': False,
                'fail_on_critical': True
            },
            SecurityLevel.STRICT: {
                'scan_secrets': True,
                'scan_dependencies': True,
                'scan_code': True,
                'fail_on_high': True,
                'fail_on_critical': True
            },
            SecurityLevel.PARANOID: {
                'scan_secrets': True,
                'scan_dependencies': True,
                'scan_code': True,
                'fail_on_high': True,
                'fail_on_critical': True,
                'additional_checks': True
            }
        }
        return configs[self.security_level]
    
    def run_comprehensive_scan(self) -> SecurityScanResult:
        """Run comprehensive security scan."""
        scan_id = f"security_scan_{int(time.time())}"
        start_time = time.time()
        
        self.logger.info(f"Starting comprehensive security scan (level: {self.security_level.value})")
        
        all_vulnerabilities = []
        files_scanned = 0
        
        try:
            # Secret scanning
            if self.scan_config['scan_secrets']:
                self.logger.info("Running secret scan...")
                secret_vulns = self.secret_scanner.scan_directory(self.project_path)
                all_vulnerabilities.extend(secret_vulns)
                files_scanned += len(list(self.project_path.rglob('*.py')))  # Rough estimate
            
            # Dependency scanning
            if self.scan_config['scan_dependencies']:
                self.logger.info("Running dependency scan...")
                dep_vulns = self.dependency_scanner.scan_python_dependencies()
                all_vulnerabilities.extend(dep_vulns)
            
            # Code scanning
            if self.scan_config['scan_code']:
                self.logger.info("Running code security scan...")
                code_vulns = self.code_scanner.scan_python_code()
                all_vulnerabilities.extend(code_vulns)
            
            # Additional checks for paranoid level
            if self.scan_config.get('additional_checks'):
                additional_vulns = self._run_additional_checks()
                all_vulnerabilities.extend(additional_vulns)
                
        except Exception as e:
            self.logger.error(f"Security scan failed: {e}")
        
        end_time = time.time()
        
        # Create scan result
        result = SecurityScanResult(
            scan_id=scan_id,
            scan_type="comprehensive",
            start_time=start_time,
            end_time=end_time,
            vulnerabilities=all_vulnerabilities,
            files_scanned=files_scanned,
            scan_config=self.scan_config
        )
        
        # Store in history
        self.scan_history.append(result)
        
        # Log results
        counts = result.vulnerability_count_by_severity
        self.logger.info(
            f"Security scan completed in {result.duration:.2f}s: "
            f"Critical={counts['critical']}, High={counts['high']}, "
            f"Medium={counts['medium']}, Low={counts['low']}"
        )
        
        return result
    
    def _run_additional_checks(self) -> List[SecurityVulnerability]:
        """Run additional security checks for paranoid mode."""
        vulnerabilities = []
        
        # Check for overly permissive file permissions
        for file_path in self.project_path.rglob('*'):
            if file_path.is_file():
                mode = file_path.stat().st_mode
                if mode & 0o777 == 0o777:  # World writable
                    vulnerability = SecurityVulnerability(
                        id=f"perm_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}",
                        title="Overly Permissive File Permissions",
                        description=f"File has world-writable permissions: {oct(mode)}",
                        severity=VulnerabilityLevel.MEDIUM,
                        file_path=str(file_path),
                        rule_id="FILE_PERMISSIONS",
                        remediation="Restrict file permissions"
                    )
                    vulnerabilities.append(vulnerability)
        
        # Check for .git directory exposure
        git_dir = self.project_path / '.git'
        if git_dir.exists() and not (self.project_path / '.gitignore').exists():
            vulnerability = SecurityVulnerability(
                id="git_exposure",
                title="Git Directory May Be Exposed",
                description="Git directory exists without proper gitignore",
                severity=VulnerabilityLevel.LOW,
                file_path=str(git_dir),
                rule_id="GIT_EXPOSURE",
                remediation="Ensure .git directory is not exposed in deployments"
            )
            vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    def validate_scan_result(self, result: SecurityScanResult) -> tuple[bool, str]:
        """Validate scan result against security policies."""
        counts = result.vulnerability_count_by_severity
        
        # Check critical vulnerabilities
        if counts['critical'] > 0 and self.scan_config['fail_on_critical']:
            return False, f"Critical vulnerabilities found: {counts['critical']}"
        
        # Check high vulnerabilities
        if counts['high'] > 0 and self.scan_config['fail_on_high']:
            return False, f"High severity vulnerabilities found: {counts['high']}"
        
        # Check risk score threshold
        risk_threshold = {
            SecurityLevel.BASIC: 50,
            SecurityLevel.STANDARD: 30,
            SecurityLevel.STRICT: 15,
            SecurityLevel.PARANOID: 5
        }
        
        if result.risk_score > risk_threshold[self.security_level]:
            return False, f"Risk score too high: {result.risk_score} (max: {risk_threshold[self.security_level]})"
        
        return True, "Security validation passed"
    
    def generate_security_report(self, result: SecurityScanResult, output_path: str):
        """Generate detailed security report."""
        report = {
            'scan_summary': {
                'scan_id': result.scan_id,
                'scan_type': result.scan_type,
                'duration': result.duration,
                'files_scanned': result.files_scanned,
                'security_level': self.security_level.value,
                'risk_score': result.risk_score
            },
            'vulnerability_summary': result.vulnerability_count_by_severity,
            'vulnerabilities': [asdict(vuln) for vuln in result.vulnerabilities],
            'recommendations': self._generate_recommendations(result),
            'scan_config': result.scan_config,
            'generated_at': time.time()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Security report generated: {output_path}")
    
    def _generate_recommendations(self, result: SecurityScanResult) -> List[str]:
        """Generate security recommendations based on scan results."""
        recommendations = []
        counts = result.vulnerability_count_by_severity
        
        if counts['critical'] > 0:
            recommendations.append("URGENT: Fix critical vulnerabilities immediately")
        
        if counts['high'] > 5:
            recommendations.append("Address high severity vulnerabilities promptly")
        
        if counts['medium'] > 10:
            recommendations.append("Review and fix medium severity issues")
        
        # Secret-specific recommendations
        secret_vulns = [v for v in result.vulnerabilities if 'secret' in v.rule_id.lower()]
        if secret_vulns:
            recommendations.append("Remove hardcoded secrets and use environment variables")
        
        # Dependency-specific recommendations
        dep_vulns = [v for v in result.vulnerabilities if 'dependency' in v.rule_id.lower()]
        if dep_vulns:
            recommendations.append("Update vulnerable dependencies to latest versions")
        
        if result.risk_score > 50:
            recommendations.append("Consider increasing security scanning frequency")
        
        return recommendations
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics from scan history."""
        if not self.scan_history:
            return {}
        
        latest_scan = self.scan_history[-1]
        
        # Calculate trends if we have multiple scans
        trend_data = {}
        if len(self.scan_history) > 1:
            prev_scan = self.scan_history[-2]
            trend_data = {
                'risk_score_trend': latest_scan.risk_score - prev_scan.risk_score,
                'vulnerability_trend': len(latest_scan.vulnerabilities) - len(prev_scan.vulnerabilities)
            }
        
        return {
            'latest_scan': {
                'scan_id': latest_scan.scan_id,
                'risk_score': latest_scan.risk_score,
                'vulnerability_counts': latest_scan.vulnerability_count_by_severity,
                'scan_duration': latest_scan.duration
            },
            'historical_data': {
                'total_scans': len(self.scan_history),
                'avg_risk_score': sum(s.risk_score for s in self.scan_history) / len(self.scan_history),
                'trend_data': trend_data
            },
            'security_level': self.security_level.value
        }