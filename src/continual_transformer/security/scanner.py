"""Security scanner and validation for continual learning systems."""

import os
import hashlib
import re
import json
import ast
import logging
import subprocess
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import tempfile
import tarfile
import zipfile

logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Security finding with severity and details."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # vulnerability, misconfiguration, policy_violation, etc.
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    evidence: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None


@dataclass  
class SecurityReport:
    """Comprehensive security assessment report."""
    scan_id: str
    timestamp: datetime
    target: str
    findings: List[SecurityFinding] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    passed_checks: List[str] = field(default_factory=list)
    scan_duration: float = 0.0
    
    def add_finding(self, finding: SecurityFinding):
        """Add finding to report."""
        self.findings.append(finding)
        
        # Update summary
        if finding.severity not in self.summary:
            self.summary[finding.severity] = 0
        self.summary[finding.severity] += 1
    
    def get_risk_score(self) -> float:
        """Calculate overall risk score."""
        severity_weights = {
            'CRITICAL': 10.0,
            'HIGH': 7.0,
            'MEDIUM': 4.0,
            'LOW': 2.0,
            'INFO': 0.5
        }
        
        total_score = 0.0
        for finding in self.findings:
            total_score += severity_weights.get(finding.severity, 0.0)
        
        return min(total_score, 100.0)  # Cap at 100


class SecurityScanner:
    """Comprehensive security scanner for ML models and code."""
    
    def __init__(self, config=None):
        self.config = config
        self.scan_rules = self._load_security_rules()
        self.file_extensions = {'.py', '.yaml', '.yml', '.json', '.txt', '.md'}
        self.binary_extensions = {'.pkl', '.pt', '.pth', '.safetensors', '.onnx'}
        
        # Security patterns
        self.sensitive_patterns = self._compile_security_patterns()
        self.vulnerability_patterns = self._compile_vulnerability_patterns()
        
        logger.info("Security scanner initialized")
    
    def scan_directory(self, directory: str) -> SecurityReport:
        """Perform comprehensive security scan of directory."""
        start_time = datetime.now()
        report = SecurityReport(
            scan_id=hashlib.md5(f"{directory}{start_time}".encode()).hexdigest()[:16],
            timestamp=start_time,
            target=directory
        )
        
        directory_path = Path(directory)
        if not directory_path.exists():
            report.add_finding(SecurityFinding(
                severity="HIGH",
                category="configuration",
                title="Target directory not found",
                description=f"Scan target directory '{directory}' does not exist"
            ))
            return report
        
        try:
            # 1. File system security checks
            self._scan_file_permissions(directory_path, report)
            
            # 2. Source code security analysis
            self._scan_source_code(directory_path, report)
            
            # 3. Configuration security
            self._scan_configurations(directory_path, report)
            
            # 4. Dependency security
            self._scan_dependencies(directory_path, report)
            
            # 5. Model file security
            self._scan_model_files(directory_path, report)
            
            # 6. Data security
            self._scan_data_files(directory_path, report)
            
            # 7. Infrastructure security
            self._scan_infrastructure_configs(directory_path, report)
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            report.add_finding(SecurityFinding(
                severity="HIGH",
                category="scanner_error",
                title="Security scan error",
                description=f"Security scan encountered an error: {e}"
            ))
        
        # Calculate scan duration
        report.scan_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Security scan completed in {report.scan_duration:.2f}s with {len(report.findings)} findings")
        return report
    
    def _scan_file_permissions(self, directory: Path, report: SecurityReport):
        """Scan for insecure file permissions."""
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    
                    # Check for world-writable files
                    if mode.endswith('2') or mode.endswith('6') or mode.endswith('7'):
                        report.add_finding(SecurityFinding(
                            severity="MEDIUM",
                            category="file_permissions",
                            title="World-writable file detected",
                            description=f"File has world-writable permissions: {mode}",
                            file_path=str(file_path),
                            recommendation="Restrict file permissions to prevent unauthorized modifications"
                        ))
                    
                    # Check for executable Python files
                    if file_path.suffix == '.py' and mode.startswith('7'):
                        report.add_finding(SecurityFinding(
                            severity="LOW",
                            category="file_permissions",
                            title="Executable Python file",
                            description=f"Python file has executable permissions: {mode}",
                            file_path=str(file_path),
                            recommendation="Consider if executable permissions are necessary"
                        ))
                        
                except (OSError, PermissionError):
                    continue  # Skip files we can't access
    
    def _scan_source_code(self, directory: Path, report: SecurityReport):
        """Scan source code for security vulnerabilities."""
        python_files = list(directory.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Parse AST for deeper analysis
                try:
                    tree = ast.parse(content)
                    self._analyze_ast_security(tree, py_file, report)
                except SyntaxError:
                    pass  # Skip files with syntax errors
                
                # Pattern-based vulnerability detection
                for line_num, line in enumerate(lines, 1):
                    self._check_line_for_vulnerabilities(line, line_num, py_file, report)
                    
            except (OSError, UnicodeDecodeError):
                continue
    
    def _analyze_ast_security(self, tree: ast.AST, file_path: Path, report: SecurityReport):
        """Analyze Python AST for security issues."""
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, report, file_path):
                self.report = report
                self.file_path = file_path
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    # eval() and exec() usage
                    if func_name in ['eval', 'exec']:
                        self.report.add_finding(SecurityFinding(
                            severity="HIGH",
                            category="code_injection",
                            title=f"Dangerous function: {func_name}",
                            description=f"Use of {func_name}() can lead to code injection vulnerabilities",
                            file_path=str(self.file_path),
                            line_number=node.lineno,
                            cwe_id="CWE-94",
                            recommendation=f"Avoid using {func_name}() or sanitize input thoroughly"
                        ))
                    
                    # pickle usage (deserialization)
                    elif func_name in ['load', 'loads'] and any(
                        isinstance(arg, ast.Attribute) and arg.attr in ['pickle', 'cpickle']
                        for arg in ast.walk(node)
                    ):
                        self.report.add_finding(SecurityFinding(
                            severity="HIGH",
                            category="deserialization",
                            title="Unsafe deserialization with pickle",
                            description="Pickle deserialization can execute arbitrary code",
                            file_path=str(self.file_path),
                            line_number=node.lineno,
                            cwe_id="CWE-502",
                            recommendation="Use safer serialization formats like JSON or validate pickle sources"
                        ))
                
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr
                    
                    # subprocess with shell=True
                    if attr_name in ['call', 'run', 'Popen'] and any(
                        isinstance(kw.value, ast.Constant) and kw.value.value is True and kw.arg == 'shell'
                        for kw in node.keywords
                    ):
                        self.report.add_finding(SecurityFinding(
                            severity="HIGH",
                            category="command_injection",
                            title="Shell command injection risk",
                            description="subprocess call with shell=True can lead to command injection",
                            file_path=str(self.file_path),
                            line_number=node.lineno,
                            cwe_id="CWE-78",
                            recommendation="Use subprocess without shell=True or sanitize input"
                        ))
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for imports of potentially dangerous modules
                dangerous_modules = ['os', 'subprocess', 'pickle', 'marshal', 'shelve']
                for alias in node.names:
                    if alias.name in dangerous_modules:
                        self.report.add_finding(SecurityFinding(
                            severity="INFO",
                            category="import_analysis",
                            title=f"Import of potentially dangerous module: {alias.name}",
                            description=f"Module '{alias.name}' imported - ensure secure usage",
                            file_path=str(self.file_path),
                            line_number=node.lineno,
                            recommendation="Review usage of this module for security implications"
                        ))
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor(report, file_path)
        visitor.visit(tree)
    
    def _check_line_for_vulnerabilities(self, line: str, line_num: int, file_path: Path, report: SecurityReport):
        """Check individual line for vulnerability patterns."""
        line_lower = line.lower().strip()
        
        # Check for hardcoded secrets
        for pattern_name, pattern in self.sensitive_patterns.items():
            matches = pattern.findall(line)
            if matches:
                report.add_finding(SecurityFinding(
                    severity="HIGH",
                    category="secrets",
                    title=f"Potential {pattern_name} exposed",
                    description=f"Possible hardcoded {pattern_name} found in source code",
                    file_path=str(file_path),
                    line_number=line_num,
                    evidence=line.strip(),
                    cwe_id="CWE-798",
                    recommendation="Store sensitive data in environment variables or secure vaults"
                ))
        
        # Check for vulnerability patterns
        for vuln_type, pattern in self.vulnerability_patterns.items():
            if pattern.search(line_lower):
                severity = "HIGH" if vuln_type in ["sql_injection", "xss"] else "MEDIUM"
                report.add_finding(SecurityFinding(
                    severity=severity,
                    category="vulnerability",
                    title=f"Potential {vuln_type.replace('_', ' ').title()}",
                    description=f"Pattern indicating possible {vuln_type} vulnerability",
                    file_path=str(file_path),
                    line_number=line_num,
                    evidence=line.strip(),
                    recommendation="Review and sanitize input handling"
                ))
    
    def _scan_configurations(self, directory: Path, report: SecurityReport):
        """Scan configuration files for security issues."""
        config_patterns = ['*.yaml', '*.yml', '*.json', '*.toml', '*.ini', '*.cfg']
        
        for pattern in config_patterns:
            for config_file in directory.rglob(pattern):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for sensitive data in configs
                    self._check_config_security(content, config_file, report)
                    
                except (OSError, UnicodeDecodeError):
                    continue
    
    def _check_config_security(self, content: str, file_path: Path, report: SecurityReport):
        """Check configuration content for security issues."""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Check for hardcoded credentials
            credential_indicators = [
                'password', 'passwd', 'pwd', 'secret', 'key', 'token', 
                'auth', 'credential', 'api_key', 'private_key'
            ]
            
            for indicator in credential_indicators:
                if indicator in line_lower and '=' in line and not line_lower.strip().startswith('#'):
                    # Extract value
                    value = line.split('=', 1)[1].strip().strip('"\'')
                    
                    # Skip obvious placeholders
                    if value and not any(placeholder in value.lower() for placeholder in 
                                       ['placeholder', 'example', 'your_', 'changeme', '<', '${', 'null']):
                        report.add_finding(SecurityFinding(
                            severity="HIGH",
                            category="configuration",
                            title=f"Hardcoded credential in configuration",
                            description=f"Possible hardcoded {indicator} in configuration file",
                            file_path=str(file_path),
                            line_number=line_num,
                            evidence=line.strip(),
                            recommendation="Use environment variables or secure configuration management"
                        ))
            
            # Check for insecure settings
            insecure_patterns = {
                'debug.*=.*true': 'Debug mode enabled in production configuration',
                'ssl.*=.*false': 'SSL/TLS disabled in configuration',
                'verify.*=.*false': 'Certificate verification disabled',
                'secure.*=.*false': 'Security feature disabled'
            }
            
            for pattern, description in insecure_patterns.items():
                if re.search(pattern, line_lower):
                    report.add_finding(SecurityFinding(
                        severity="MEDIUM",
                        category="configuration",
                        title="Insecure configuration setting",
                        description=description,
                        file_path=str(file_path),
                        line_number=line_num,
                        evidence=line.strip(),
                        recommendation="Review and secure configuration settings"
                    ))
    
    def _scan_dependencies(self, directory: Path, report: SecurityReport):
        """Scan dependencies for known vulnerabilities."""
        # Check requirements.txt
        req_files = list(directory.rglob("requirements*.txt"))
        for req_file in req_files:
            try:
                with open(req_file, 'r') as f:
                    content = f.read()
                self._check_dependency_security(content, req_file, report)
            except OSError:
                continue
        
        # Check pyproject.toml
        pyproject_files = list(directory.rglob("pyproject.toml"))
        for pyproject_file in pyproject_files:
            try:
                with open(pyproject_file, 'r') as f:
                    content = f.read()
                self._check_pyproject_security(content, pyproject_file, report)
            except OSError:
                continue
    
    def _check_dependency_security(self, content: str, file_path: Path, report: SecurityReport):
        """Check dependency requirements for security issues."""
        lines = content.strip().split('\n')
        
        # Known vulnerable packages (simplified - in real implementation, use vulnerability DB)
        vulnerable_packages = {
            'pickle': 'Unsafe serialization library',
            'marshal': 'Unsafe serialization library',
            'yaml': 'Use safe_load instead of load',
            'requests': 'Ensure version >= 2.20.0 for security fixes'
        }
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse package name
                package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('<')[0].split('>')[0].strip()
                
                if package_name.lower() in vulnerable_packages:
                    report.add_finding(SecurityFinding(
                        severity="MEDIUM",
                        category="dependencies",
                        title=f"Potentially vulnerable dependency: {package_name}",
                        description=vulnerable_packages[package_name.lower()],
                        file_path=str(file_path),
                        line_number=line_num,
                        evidence=line,
                        recommendation="Review dependency security and update if necessary"
                    ))
                
                # Check for unpinned versions
                if '==' not in line and '>=' not in line and package_name not in ['pip', 'setuptools']:
                    report.add_finding(SecurityFinding(
                        severity="LOW",
                        category="dependencies",
                        title=f"Unpinned dependency version: {package_name}",
                        description="Dependency version not pinned, could lead to supply chain attacks",
                        file_path=str(file_path),
                        line_number=line_num,
                        evidence=line,
                        recommendation="Pin dependency versions for reproducible and secure builds"
                    ))
    
    def _check_pyproject_security(self, content: str, file_path: Path, report: SecurityReport):
        """Check pyproject.toml for security configuration."""
        # Basic check for now - could be expanded with TOML parsing
        if 'allow-prereleases = true' in content.lower():
            report.add_finding(SecurityFinding(
                severity="LOW",
                category="configuration",
                title="Pre-release dependencies allowed",
                description="Configuration allows pre-release dependencies which may be unstable",
                file_path=str(file_path),
                recommendation="Avoid pre-release dependencies in production"
            ))
    
    def _scan_model_files(self, directory: Path, report: SecurityReport):
        """Scan model files for security issues."""
        for ext in self.binary_extensions:
            for model_file in directory.rglob(f"*{ext}"):
                self._check_model_file_security(model_file, report)
    
    def _check_model_file_security(self, file_path: Path, report: SecurityReport):
        """Check individual model file for security issues."""
        try:
            file_size = file_path.stat().st_size
            
            # Check for suspiciously large models
            if file_size > 10 * 1024 * 1024 * 1024:  # 10GB
                report.add_finding(SecurityFinding(
                    severity="MEDIUM",
                    category="model_security",
                    title="Unusually large model file",
                    description=f"Model file is {file_size / (1024**3):.1f}GB, which may indicate embedded data",
                    file_path=str(file_path),
                    recommendation="Verify model file contents and size requirements"
                ))
            
            # Check pickle files specifically
            if file_path.suffix in ['.pkl', '.pickle']:
                report.add_finding(SecurityFinding(
                    severity="HIGH",
                    category="model_security",
                    title="Pickle model file detected",
                    description="Pickle files can contain arbitrary code and pose security risks",
                    file_path=str(file_path),
                    cwe_id="CWE-502",
                    recommendation="Use safer formats like SafeTensors or validate pickle file sources"
                ))
            
        except OSError:
            pass
    
    def _scan_data_files(self, directory: Path, report: SecurityReport):
        """Scan data files for sensitive information."""
        data_extensions = ['.csv', '.json', '.txt', '.tsv', '.parquet']
        
        for ext in data_extensions:
            for data_file in directory.rglob(f"*{ext}"):
                if data_file.stat().st_size < 100 * 1024 * 1024:  # Only scan files < 100MB
                    self._check_data_file_security(data_file, report)
    
    def _check_data_file_security(self, file_path: Path, report: SecurityReport):
        """Check data file for sensitive information."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first few lines to check for sensitive data
                lines = []
                for _ in range(min(100, 1000)):  # Check first 100 lines max
                    try:
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                    except UnicodeDecodeError:
                        break
            
            # Check for PII patterns
            pii_patterns = {
                'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
                'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
            }
            
            for line_num, line in enumerate(lines, 1):
                for pii_type, pattern in pii_patterns.items():
                    if pattern.search(line):
                        report.add_finding(SecurityFinding(
                            severity="HIGH",
                            category="data_privacy",
                            title=f"Potential {pii_type.upper()} in data file",
                            description=f"Data file may contain {pii_type} information",
                            file_path=str(file_path),
                            line_number=line_num,
                            recommendation="Remove or anonymize sensitive data"
                        ))
                        break  # Only report once per file
            
        except (OSError, UnicodeDecodeError):
            pass
    
    def _scan_infrastructure_configs(self, directory: Path, report: SecurityReport):
        """Scan infrastructure configuration files."""
        # Docker files
        docker_files = list(directory.rglob("Dockerfile*")) + list(directory.rglob("docker-compose*.yml"))
        for docker_file in docker_files:
            self._check_docker_security(docker_file, report)
        
        # GitHub Actions
        gh_actions = list(directory.rglob(".github/workflows/*.yml")) + list(directory.rglob(".github/workflows/*.yaml"))
        for action_file in gh_actions:
            self._check_github_action_security(action_file, report)
    
    def _check_docker_security(self, file_path: Path, report: SecurityReport):
        """Check Docker configuration for security issues."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line_upper = line.strip().upper()
                
                # Check for running as root
                if line_upper.startswith('USER ROOT') or (line_upper.startswith('USER') and '0' in line):
                    report.add_finding(SecurityFinding(
                        severity="MEDIUM",
                        category="docker_security",
                        title="Container running as root",
                        description="Container configured to run as root user",
                        file_path=str(file_path),
                        line_number=line_num,
                        evidence=line.strip(),
                        recommendation="Use non-root user for container execution"
                    ))
                
                # Check for latest tag usage
                if 'FROM' in line_upper and ':LATEST' in line_upper:
                    report.add_finding(SecurityFinding(
                        severity="LOW",
                        category="docker_security",
                        title="Using 'latest' tag in Docker image",
                        description="Using 'latest' tag can lead to unpredictable builds",
                        file_path=str(file_path),
                        line_number=line_num,
                        evidence=line.strip(),
                        recommendation="Use specific version tags for reproducible builds"
                    ))
                
        except OSError:
            pass
    
    def _check_github_action_security(self, file_path: Path, report: SecurityReport):
        """Check GitHub Actions for security issues."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for third-party actions without version pinning
            action_pattern = re.compile(r'uses:\s*([^@\s]+)(?:@([^\s]+))?')
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                match = action_pattern.search(line)
                if match:
                    action_name, version = match.groups()
                    
                    # Skip official GitHub actions
                    if not action_name.startswith('actions/'):
                        if not version:
                            report.add_finding(SecurityFinding(
                                severity="MEDIUM",
                                category="ci_security",
                                title="Unpinned third-party GitHub Action",
                                description=f"Third-party action '{action_name}' not version-pinned",
                                file_path=str(file_path),
                                line_number=line_num,
                                evidence=line.strip(),
                                recommendation="Pin actions to specific commit SHA or version tag"
                            ))
                        elif version in ['main', 'master', 'latest']:
                            report.add_finding(SecurityFinding(
                                severity="LOW",
                                category="ci_security",
                                title="GitHub Action using mutable reference",
                                description=f"Action using mutable reference: {version}",
                                file_path=str(file_path),
                                line_number=line_num,
                                evidence=line.strip(),
                                recommendation="Use immutable commit SHA for maximum security"
                            ))
            
        except OSError:
            pass
    
    def _compile_security_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for detecting sensitive data."""
        patterns = {
            'api_key': re.compile(r'(?i)api[_\-]?key["\']?\s*[:=]\s*["\']?[a-z0-9]{32,}', re.IGNORECASE),
            'secret_key': re.compile(r'(?i)secret[_\-]?key["\']?\s*[:=]\s*["\']?[a-z0-9]{32,}', re.IGNORECASE),
            'password': re.compile(r'(?i)password["\']?\s*[:=]\s*["\'][^"\']{8,}', re.IGNORECASE),
            'private_key': re.compile(r'-----BEGIN [A-Z ]+PRIVATE KEY-----', re.IGNORECASE),
            'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}'),
            'github_token': re.compile(r'ghp_[a-zA-Z0-9]{36}'),
            'jwt_token': re.compile(r'eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+')
        }
        return patterns
    
    def _compile_vulnerability_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for detecting vulnerabilities."""
        patterns = {
            'sql_injection': re.compile(r'(execute|query|cursor)\s*\(\s*["\'].*%s.*["\']', re.IGNORECASE),
            'xss': re.compile(r'innerHTML\s*=.*\+.*\$', re.IGNORECASE),
            'path_traversal': re.compile(r'open\s*\(\s*.*\.\./.*\)', re.IGNORECASE),
            'command_injection': re.compile(r'(system|popen|subprocess)\s*\(.*\+.*\)', re.IGNORECASE)
        }
        return patterns
    
    def _load_security_rules(self) -> Dict[str, Any]:
        """Load security scanning rules."""
        # This would load from a configuration file in a real implementation
        return {
            'max_file_size_mb': 100,
            'scan_binary_files': False,
            'sensitivity_level': 'high',
            'excluded_paths': ['.git', '__pycache__', 'node_modules', '.venv'],
            'excluded_extensions': ['.pyc', '.pyo', '.so', '.dll']
        }
    
    def generate_report(self, report: SecurityReport, format: str = 'json') -> str:
        """Generate security report in specified format."""
        if format.lower() == 'json':
            return self._generate_json_report(report)
        elif format.lower() == 'html':
            return self._generate_html_report(report)
        elif format.lower() == 'text':
            return self._generate_text_report(report)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_json_report(self, report: SecurityReport) -> str:
        """Generate JSON security report."""
        report_dict = {
            'scan_id': report.scan_id,
            'timestamp': report.timestamp.isoformat(),
            'target': report.target,
            'scan_duration': report.scan_duration,
            'risk_score': report.get_risk_score(),
            'summary': report.summary,
            'findings': [
                {
                    'severity': finding.severity,
                    'category': finding.category,
                    'title': finding.title,
                    'description': finding.description,
                    'file_path': finding.file_path,
                    'line_number': finding.line_number,
                    'evidence': finding.evidence,
                    'recommendation': finding.recommendation,
                    'cwe_id': finding.cwe_id,
                    'cvss_score': finding.cvss_score
                }
                for finding in report.findings
            ],
            'passed_checks': report.passed_checks
        }
        return json.dumps(report_dict, indent=2)
    
    def _generate_text_report(self, report: SecurityReport) -> str:
        """Generate text security report."""
        lines = [
            f"Security Scan Report",
            f"=" * 50,
            f"Scan ID: {report.scan_id}",
            f"Target: {report.target}",
            f"Timestamp: {report.timestamp}",
            f"Duration: {report.scan_duration:.2f}s",
            f"Risk Score: {report.get_risk_score():.1f}/100",
            "",
            f"Summary:",
            f"-" * 20
        ]
        
        for severity, count in report.summary.items():
            lines.append(f"{severity}: {count}")
        
        lines.extend(["", "Findings:", "=" * 30])
        
        for i, finding in enumerate(report.findings, 1):
            lines.extend([
                f"{i}. [{finding.severity}] {finding.title}",
                f"   Category: {finding.category}",
                f"   Description: {finding.description}",
                f"   File: {finding.file_path or 'N/A'}",
                f"   Line: {finding.line_number or 'N/A'}",
                f"   Recommendation: {finding.recommendation or 'N/A'}",
                ""
            ])
        
        return "\n".join(lines)
    
    def _generate_html_report(self, report: SecurityReport) -> str:
        """Generate HTML security report."""
        # This would generate a proper HTML report
        # For now, return a simple HTML version
        html_content = f"""
        <html>
        <head><title>Security Scan Report</title></head>
        <body>
        <h1>Security Scan Report</h1>
        <p><strong>Scan ID:</strong> {report.scan_id}</p>
        <p><strong>Target:</strong> {report.target}</p>
        <p><strong>Risk Score:</strong> {report.get_risk_score():.1f}/100</p>
        <h2>Summary</h2>
        <ul>
        """
        
        for severity, count in report.summary.items():
            html_content += f"<li>{severity}: {count}</li>"
        
        html_content += "</ul><h2>Findings</h2><ol>"
        
        for finding in report.findings:
            html_content += f"""
            <li>
                <strong>[{finding.severity}] {finding.title}</strong><br>
                {finding.description}<br>
                <em>File: {finding.file_path or 'N/A'}</em>
            </li>
            """
        
        html_content += "</ol></body></html>"
        return html_content


def run_security_scan(target_directory: str, output_file: Optional[str] = None, format: str = 'json') -> SecurityReport:
    """Run comprehensive security scan and optionally save report."""
    scanner = SecurityScanner()
    report = scanner.scan_directory(target_directory)
    
    if output_file:
        report_content = scanner.generate_report(report, format)
        with open(output_file, 'w') as f:
            f.write(report_content)
        logger.info(f"Security report saved to {output_file}")
    
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python security_scanner.py <directory> [output_file] [format]")
        sys.exit(1)
    
    target = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    fmt = sys.argv[3] if len(sys.argv) > 3 else 'json'
    
    report = run_security_scan(target, output, fmt)
    print(f"Security scan completed. Found {len(report.findings)} issues.")
    print(f"Risk Score: {report.get_risk_score():.1f}/100")