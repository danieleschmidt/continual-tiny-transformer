#!/usr/bin/env python3
"""Comprehensive validation script for SDLC framework implementation."""

import ast
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        self.warnings_list = []
        self.details = {}
    
    def add_success(self, test_name: str, details: str = ""):
        self.passed += 1
        self.details[test_name] = {"status": "PASSED", "details": details}
    
    def add_failure(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        self.details[test_name] = {"status": "FAILED", "error": error}
    
    def add_warning(self, test_name: str, warning: str):
        self.warnings += 1
        self.warnings_list.append(f"{test_name}: {warning}")
        self.details[test_name] = {"status": "WARNING", "warning": warning}
    
    @property
    def total_tests(self) -> int:
        return self.passed + self.failed + self.warnings
    
    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100.0


class SDLCValidator:
    """Main validator for SDLC framework implementation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.result = ValidationResult()
        
        # Key paths
        self.src_path = project_root / "src" / "continual_transformer"
        self.tests_path = project_root / "tests"
        self.scripts_path = project_root / "scripts"
        self.deployment_path = project_root / "deployment"
        self.docs_path = project_root / "docs"
    
    def validate_all(self) -> ValidationResult:
        """Run all validations."""
        logger.info("Starting comprehensive SDLC framework validation...")
        
        validation_methods = [
            ("Project Structure", self.validate_project_structure),
            ("Core Components", self.validate_core_components),
            ("Security Framework", self.validate_security_framework),
            ("Automation System", self.validate_automation_system),
            ("Monitoring System", self.validate_monitoring_system),
            ("Reliability System", self.validate_reliability_system),
            ("Optimization Engine", self.validate_optimization_engine),
            ("Testing Framework", self.validate_testing_framework),
            ("Deployment Configuration", self.validate_deployment_config),
            ("Documentation", self.validate_documentation),
            ("Scripts and Tools", self.validate_scripts),
            ("Code Quality", self.validate_code_quality),
            ("Integration Points", self.validate_integration)
        ]
        
        for test_category, validation_method in validation_methods:
            try:
                logger.info(f"Validating: {test_category}")
                validation_method()
            except Exception as e:
                logger.error(f"Validation error in {test_category}: {e}")
                self.result.add_failure(test_category, str(e))
        
        return self.result
    
    def validate_project_structure(self):
        """Validate project structure and organization."""
        
        # Required directories
        required_dirs = [
            self.src_path / "sdlc",
            self.tests_path,
            self.scripts_path,
            self.deployment_path,
            self.docs_path
        ]
        
        missing_dirs = [d for d in required_dirs if not d.exists()]
        if missing_dirs:
            self.result.add_failure(
                "Required Directories",
                f"Missing directories: {[str(d) for d in missing_dirs]}"
            )
        else:
            self.result.add_success(
                "Required Directories",
                f"All {len(required_dirs)} required directories present"
            )
        
        # Required files
        required_files = [
            self.project_root / "pyproject.toml",
            self.project_root / "Makefile",
            self.project_root / "README.md",
            self.src_path / "__init__.py",
            self.src_path / "sdlc" / "__init__.py"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            self.result.add_failure(
                "Required Files",
                f"Missing files: {[str(f) for f in missing_files]}"
            )
        else:
            self.result.add_success(
                "Required Files",
                f"All {len(required_files)} required files present"
            )
    
    def validate_core_components(self):
        """Validate core SDLC components."""
        
        core_modules = [
            ("core.py", ["SDLCManager", "WorkflowEngine", "WorkflowTask"]),
            ("automation.py", ["AutomatedWorkflow", "TaskRunner", "AutomationConfig"]),
            ("monitoring.py", ["SDLCMonitor", "MetricsCollector"]),
            ("reliability.py", ["ReliabilityManager", "CircuitBreaker"]),
            ("security.py", ["SecurityValidator", "SecretScanner"]),
            ("optimization.py", ["OptimizedWorkflowEngine", "ResourceMonitor"])
        ]
        
        for module_file, expected_classes in core_modules:
            module_path = self.src_path / "sdlc" / module_file
            
            if not module_path.exists():
                self.result.add_failure(
                    f"Core Module {module_file}",
                    f"Module file not found: {module_path}"
                )
                continue
            
            # Parse AST to find classes
            try:
                with open(module_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                found_classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                missing_classes = [cls for cls in expected_classes if cls not in found_classes]
                
                if missing_classes:
                    self.result.add_failure(
                        f"Core Classes in {module_file}",
                        f"Missing classes: {missing_classes}"
                    )
                else:
                    self.result.add_success(
                        f"Core Classes in {module_file}",
                        f"All {len(expected_classes)} required classes found"
                    )
                
            except Exception as e:
                self.result.add_failure(
                    f"Core Module Analysis {module_file}",
                    f"Failed to analyze module: {e}"
                )
    
    def validate_security_framework(self):
        """Validate security framework implementation."""
        
        security_module = self.src_path / "sdlc" / "security.py"
        
        if not security_module.exists():
            self.result.add_failure("Security Framework", "Security module not found")
            return
        
        # Check security components
        try:
            with open(security_module, 'r') as f:
                content = f.read()
            
            required_components = [
                "SecretScanner",
                "DependencyScanner", 
                "CodeScanner",
                "SecurityValidator",
                "VulnerabilityLevel",
                "SecurityLevel"
            ]
            
            missing_components = [comp for comp in required_components if comp not in content]
            
            if missing_components:
                self.result.add_failure(
                    "Security Components",
                    f"Missing components: {missing_components}"
                )
            else:
                self.result.add_success(
                    "Security Components",
                    f"All {len(required_components)} security components found"
                )
            
            # Check for security patterns
            security_patterns = [
                "secret_patterns",
                "vulnerability",
                "scan_file",
                "scan_directory"
            ]
            
            found_patterns = sum(1 for pattern in security_patterns if pattern in content)
            
            if found_patterns < len(security_patterns) * 0.8:  # 80% threshold
                self.result.add_warning(
                    "Security Patterns",
                    f"Only {found_patterns}/{len(security_patterns)} security patterns found"
                )
            else:
                self.result.add_success(
                    "Security Patterns",
                    f"{found_patterns}/{len(security_patterns)} security patterns implemented"
                )
            
        except Exception as e:
            self.result.add_failure("Security Analysis", f"Failed to analyze security module: {e}")
    
    def validate_automation_system(self):
        """Validate automation system implementation."""
        
        automation_module = self.src_path / "sdlc" / "automation.py"
        
        if not automation_module.exists():
            self.result.add_failure("Automation System", "Automation module not found")
            return
        
        try:
            with open(automation_module, 'r') as f:
                content = f.read()
            
            # Check automation components
            automation_components = [
                "TaskRunner",
                "AutomatedWorkflow",
                "TriggerType",
                "AutomationConfig",
                "submit_task",
                "start",
                "stop"
            ]
            
            missing_components = [comp for comp in automation_components if comp not in content]
            
            if missing_components:
                self.result.add_failure(
                    "Automation Components",
                    f"Missing components: {missing_components}"
                )
            else:
                self.result.add_success(
                    "Automation Components",
                    f"All {len(automation_components)} automation components found"
                )
            
            # Check for threading and async patterns
            concurrency_patterns = ["threading", "queue", "executor", "background"]
            found_patterns = sum(1 for pattern in concurrency_patterns if pattern.lower() in content.lower())
            
            if found_patterns >= 3:
                self.result.add_success(
                    "Concurrency Patterns",
                    f"Found {found_patterns} concurrency patterns"
                )
            else:
                self.result.add_warning(
                    "Concurrency Patterns",
                    f"Only {found_patterns} concurrency patterns found"
                )
                
        except Exception as e:
            self.result.add_failure("Automation Analysis", f"Failed to analyze automation module: {e}")
    
    def validate_monitoring_system(self):
        """Validate monitoring system implementation."""
        
        monitoring_module = self.src_path / "sdlc" / "monitoring.py"
        
        if not monitoring_module.exists():
            self.result.add_failure("Monitoring System", "Monitoring module not found")
            return
        
        try:
            with open(monitoring_module, 'r') as f:
                content = f.read()
            
            # Check monitoring components
            monitoring_components = [
                "MetricsCollector",
                "SDLCMonitor", 
                "SDLCMetrics",
                "record_workflow_execution",
                "get_sdlc_metrics",
                "sqlite3"
            ]
            
            missing_components = [comp for comp in monitoring_components if comp not in content]
            
            if missing_components:
                self.result.add_failure(
                    "Monitoring Components",
                    f"Missing components: {missing_components}"
                )
            else:
                self.result.add_success(
                    "Monitoring Components",
                    f"All {len(monitoring_components)} monitoring components found"
                )
            
            # Check for database operations
            db_operations = ["CREATE TABLE", "INSERT INTO", "SELECT", "UPDATE"]
            found_ops = sum(1 for op in db_operations if op in content)
            
            if found_ops >= 3:
                self.result.add_success(
                    "Database Operations",
                    f"Found {found_ops} database operations"
                )
            else:
                self.result.add_warning(
                    "Database Operations",
                    f"Only {found_ops} database operations found"
                )
                
        except Exception as e:
            self.result.add_failure("Monitoring Analysis", f"Failed to analyze monitoring module: {e}")
    
    def validate_reliability_system(self):
        """Validate reliability and resilience system."""
        
        reliability_module = self.src_path / "sdlc" / "reliability.py"
        
        if not reliability_module.exists():
            self.result.add_failure("Reliability System", "Reliability module not found")
            return
        
        try:
            with open(reliability_module, 'r') as f:
                content = f.read()
            
            # Check reliability components
            reliability_components = [
                "CircuitBreaker",
                "ReliabilityManager",
                "FailurePattern",
                "HealthChecker",
                "RecoveryStrategy",
                "analyze_failure",
                "attempt_recovery"
            ]
            
            missing_components = [comp for comp in reliability_components if comp not in content]
            
            if missing_components:
                self.result.add_failure(
                    "Reliability Components",
                    f"Missing components: {missing_components}"
                )
            else:
                self.result.add_success(
                    "Reliability Components",
                    f"All {len(reliability_components)} reliability components found"
                )
            
            # Check for failure patterns
            failure_types = ["timeout", "permission", "network", "dependency"]
            found_types = sum(1 for ftype in failure_types if ftype.lower() in content.lower())
            
            if found_types >= 3:
                self.result.add_success(
                    "Failure Pattern Coverage",
                    f"Found {found_types} failure pattern types"
                )
            else:
                self.result.add_warning(
                    "Failure Pattern Coverage",
                    f"Only {found_types} failure pattern types found"
                )
                
        except Exception as e:
            self.result.add_failure("Reliability Analysis", f"Failed to analyze reliability module: {e}")
    
    def validate_optimization_engine(self):
        """Validate performance optimization engine."""
        
        optimization_module = self.src_path / "sdlc" / "optimization.py"
        
        if not optimization_module.exists():
            self.result.add_failure("Optimization Engine", "Optimization module not found")
            return
        
        try:
            with open(optimization_module, 'r') as f:
                content = f.read()
            
            # Check optimization components
            optimization_components = [
                "OptimizedWorkflowEngine",
                "ResourceMonitor",
                "TaskProfiler",
                "IntelligentScheduler",
                "OptimizationStrategy",
                "execute_optimized_workflow"
            ]
            
            missing_components = [comp for comp in optimization_components if comp not in content]
            
            if missing_components:
                self.result.add_failure(
                    "Optimization Components",
                    f"Missing components: {missing_components}"
                )
            else:
                self.result.add_success(
                    "Optimization Components",
                    f"All {len(optimization_components)} optimization components found"
                )
            
            # Check for performance monitoring
            perf_patterns = ["psutil", "cpu_percent", "memory", "resource", "profil"]
            found_patterns = sum(1 for pattern in perf_patterns if pattern.lower() in content.lower())
            
            if found_patterns >= 4:
                self.result.add_success(
                    "Performance Monitoring",
                    f"Found {found_patterns} performance monitoring patterns"
                )
            else:
                self.result.add_warning(
                    "Performance Monitoring",
                    f"Only {found_patterns} performance monitoring patterns found"
                )
                
        except Exception as e:
            self.result.add_failure("Optimization Analysis", f"Failed to analyze optimization module: {e}")
    
    def validate_testing_framework(self):
        """Validate testing framework completeness."""
        
        test_files = [
            self.tests_path / "test_sdlc_framework.py",
            self.tests_path / "test_sdlc_integration.py"
        ]
        
        missing_test_files = [f for f in test_files if not f.exists()]
        if missing_test_files:
            self.result.add_failure(
                "Test Files",
                f"Missing test files: {[str(f) for f in missing_test_files]}"
            )
            return
        
        try:
            total_test_classes = 0
            total_test_methods = 0
            
            for test_file in test_files:
                with open(test_file, 'r') as f:
                    tree = ast.parse(f.read())
                
                # Count test classes and methods
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                        total_test_classes += 1
                        
                        # Count test methods in class
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                                total_test_methods += 1
            
            if total_test_classes >= 10 and total_test_methods >= 30:
                self.result.add_success(
                    "Test Coverage",
                    f"Found {total_test_classes} test classes with {total_test_methods} test methods"
                )
            elif total_test_classes >= 5:
                self.result.add_warning(
                    "Test Coverage",
                    f"Found {total_test_classes} test classes with {total_test_methods} test methods - consider adding more tests"
                )
            else:
                self.result.add_failure(
                    "Test Coverage",
                    f"Insufficient test coverage: {total_test_classes} classes, {total_test_methods} methods"
                )
                
        except Exception as e:
            self.result.add_failure("Testing Analysis", f"Failed to analyze test files: {e}")
    
    def validate_deployment_config(self):
        """Validate deployment configuration files."""
        
        deployment_files = [
            self.deployment_path / "docker" / "Dockerfile.sdlc",
            self.deployment_path / "kubernetes" / "sdlc-deployment.yaml",
            self.deployment_path / "terraform" / "main.tf",
            self.deployment_path / "scripts" / "deploy.sh"
        ]
        
        for file_path in deployment_files:
            if not file_path.exists():
                self.result.add_failure(
                    f"Deployment File {file_path.name}",
                    f"File not found: {file_path}"
                )
            else:
                # Basic content validation
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    if len(content.strip()) < 100:  # Minimum content check
                        self.result.add_warning(
                            f"Deployment Content {file_path.name}",
                            f"File appears to have minimal content: {len(content)} chars"
                        )
                    else:
                        self.result.add_success(
                            f"Deployment File {file_path.name}",
                            f"File exists with {len(content)} characters"
                        )
                        
                except Exception as e:
                    self.result.add_failure(
                        f"Deployment Content {file_path.name}",
                        f"Failed to read file: {e}"
                    )
        
        # Check for key deployment concepts
        docker_file = self.deployment_path / "docker" / "Dockerfile.sdlc"
        if docker_file.exists():
            with open(docker_file, 'r') as f:
                docker_content = f.read()
            
            docker_concepts = ["FROM", "WORKDIR", "COPY", "RUN", "CMD", "HEALTHCHECK"]
            found_concepts = sum(1 for concept in docker_concepts if concept in docker_content)
            
            if found_concepts >= 5:
                self.result.add_success(
                    "Docker Configuration",
                    f"Found {found_concepts}/{len(docker_concepts)} Docker concepts"
                )
            else:
                self.result.add_warning(
                    "Docker Configuration",
                    f"Only {found_concepts}/{len(docker_concepts)} Docker concepts found"
                )
    
    def validate_documentation(self):
        """Validate documentation completeness."""
        
        doc_files = [
            self.docs_path / "SDLC_FRAMEWORK.md",
            self.project_root / "README.md",
            self.project_root / "ARCHITECTURE.md"
        ]
        
        for doc_file in doc_files:
            if not doc_file.exists():
                self.result.add_failure(
                    f"Documentation {doc_file.name}",
                    f"File not found: {doc_file}"
                )
            else:
                try:
                    with open(doc_file, 'r') as f:
                        content = f.read()
                    
                    word_count = len(content.split())
                    
                    if word_count < 100:
                        self.result.add_warning(
                            f"Documentation Content {doc_file.name}",
                            f"Documentation appears brief: {word_count} words"
                        )
                    elif word_count >= 1000:
                        self.result.add_success(
                            f"Documentation Content {doc_file.name}",
                            f"Comprehensive documentation: {word_count} words"
                        )
                    else:
                        self.result.add_success(
                            f"Documentation Content {doc_file.name}",
                            f"Good documentation: {word_count} words"
                        )
                        
                except Exception as e:
                    self.result.add_failure(
                        f"Documentation Analysis {doc_file.name}",
                        f"Failed to analyze documentation: {e}"
                    )
    
    def validate_scripts(self):
        """Validate automation scripts."""
        
        script_files = [
            self.scripts_path / "security_scanner.py",
            self.scripts_path / "sdlc_automation.py"
        ]
        
        for script_file in script_files:
            if not script_file.exists():
                self.result.add_failure(
                    f"Script {script_file.name}",
                    f"Script not found: {script_file}"
                )
            else:
                # Check if script is executable
                if not os.access(script_file, os.X_OK):
                    self.result.add_warning(
                        f"Script Permissions {script_file.name}",
                        f"Script is not executable: {script_file}"
                    )
                else:
                    self.result.add_success(
                        f"Script Permissions {script_file.name}",
                        "Script has correct permissions"
                    )
                
                # Check for main function or argparse
                try:
                    with open(script_file, 'r') as f:
                        content = f.read()
                    
                    if 'def main(' in content and 'argparse' in content:
                        self.result.add_success(
                            f"Script Structure {script_file.name}",
                            "Script has proper CLI structure"
                        )
                    else:
                        self.result.add_warning(
                            f"Script Structure {script_file.name}",
                            "Script may lack proper CLI structure"
                        )
                        
                except Exception as e:
                    self.result.add_failure(
                        f"Script Analysis {script_file.name}",
                        f"Failed to analyze script: {e}"
                    )
    
    def validate_code_quality(self):
        """Validate code quality metrics."""
        
        # Check for code quality configuration files
        quality_files = [
            self.project_root / "pyproject.toml",
            self.project_root / ".pre-commit-config.yaml"
        ]
        
        found_config = 0
        for config_file in quality_files:
            if config_file.exists():
                found_config += 1
                
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                    
                    # Check for quality tools configuration
                    quality_tools = ["black", "isort", "mypy", "ruff", "pytest"]
                    found_tools = sum(1 for tool in quality_tools if tool in content)
                    
                    if found_tools >= 4:
                        self.result.add_success(
                            f"Quality Configuration {config_file.name}",
                            f"Found {found_tools} quality tools configured"
                        )
                    else:
                        self.result.add_warning(
                            f"Quality Configuration {config_file.name}",
                            f"Only {found_tools} quality tools configured"
                        )
                        
                except Exception as e:
                    self.result.add_failure(
                        f"Quality Config Analysis {config_file.name}",
                        f"Failed to analyze config: {e}"
                    )
        
        if found_config == 0:
            self.result.add_failure(
                "Code Quality Configuration",
                "No quality configuration files found"
            )
        
        # Try to run a basic syntax check on core modules
        try:
            sdlc_init = self.src_path / "sdlc" / "__init__.py"
            if sdlc_init.exists():
                with open(sdlc_init, 'r') as f:
                    ast.parse(f.read())
                
                self.result.add_success(
                    "Core Module Syntax",
                    "SDLC module syntax is valid"
                )
        except SyntaxError as e:
            self.result.add_failure(
                "Core Module Syntax",
                f"Syntax error in SDLC module: {e}"
            )
        except Exception:
            self.result.add_warning(
                "Core Module Syntax",
                "Could not verify module syntax"
            )
    
    def validate_integration(self):
        """Validate integration between components."""
        
        # Check imports in core modules
        core_modules = [
            self.src_path / "sdlc" / "core.py",
            self.src_path / "sdlc" / "automation.py", 
            self.src_path / "sdlc" / "monitoring.py"
        ]
        
        integration_score = 0
        total_modules = len(core_modules)
        
        for module_file in core_modules:
            if not module_file.exists():
                continue
                
            try:
                with open(module_file, 'r') as f:
                    content = f.read()
                
                # Look for cross-module imports
                sdlc_imports = content.count("from continual_transformer.sdlc")
                relative_imports = content.count("from .")
                
                if sdlc_imports > 0 or relative_imports > 0:
                    integration_score += 1
                    
            except Exception:
                continue
        
        integration_ratio = integration_score / total_modules if total_modules > 0 else 0
        
        if integration_ratio >= 0.7:
            self.result.add_success(
                "Component Integration",
                f"Good integration: {integration_score}/{total_modules} modules have cross-imports"
            )
        elif integration_ratio >= 0.5:
            self.result.add_warning(
                "Component Integration",
                f"Moderate integration: {integration_score}/{total_modules} modules have cross-imports"
            )
        else:
            self.result.add_failure(
                "Component Integration",
                f"Poor integration: {integration_score}/{total_modules} modules have cross-imports"
            )


def main():
    """Main validation function."""
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"🔍 Validating SDLC Framework Implementation")
    print(f"📁 Project Root: {project_root}")
    print("=" * 80)
    
    # Run validation
    validator = SDLCValidator(project_root)
    result = validator.validate_all()
    
    # Print results
    print(f"\n📊 VALIDATION RESULTS")
    print("=" * 80)
    print(f"✅ Passed: {result.passed}")
    print(f"❌ Failed: {result.failed}")
    print(f"⚠️  Warnings: {result.warnings}")
    print(f"📈 Success Rate: {result.success_rate:.1f}%")
    print(f"🔢 Total Tests: {result.total_tests}")
    
    # Print errors
    if result.errors:
        print(f"\n❌ FAILURES:")
        for error in result.errors:
            print(f"  • {error}")
    
    # Print warnings
    if result.warnings_list:
        print(f"\n⚠️  WARNINGS:")
        for warning in result.warnings_list:
            print(f"  • {warning}")
    
    # Generate detailed report
    report_file = project_root / "validation_report.json"
    try:
        report_data = {
            "validation_timestamp": time.time(),
            "project_root": str(project_root),
            "summary": {
                "passed": result.passed,
                "failed": result.failed,
                "warnings": result.warnings,
                "success_rate": result.success_rate,
                "total_tests": result.total_tests
            },
            "details": result.details,
            "errors": result.errors,
            "warnings": result.warnings_list
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📋 Detailed report saved to: {report_file}")
        
    except Exception as e:
        print(f"\n⚠️  Could not save detailed report: {e}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT")
    print("=" * 80)
    
    if result.success_rate >= 90:
        print("🎉 EXCELLENT: SDLC framework implementation is comprehensive and production-ready!")
        assessment = "EXCELLENT"
    elif result.success_rate >= 80:
        print("👍 GOOD: SDLC framework implementation is solid with minor improvements needed.")
        assessment = "GOOD"
    elif result.success_rate >= 70:
        print("⚠️  FAIR: SDLC framework implementation needs some improvements.")
        assessment = "FAIR"
    else:
        print("🔴 POOR: SDLC framework implementation requires significant improvements.")
        assessment = "POOR"
    
    # Recommendations
    if result.failed > 0:
        print(f"\n💡 RECOMMENDATIONS:")
        if result.failed >= 5:
            print("  • Focus on addressing critical failures first")
            print("  • Consider reviewing core architecture and design")
        if result.warnings >= 10:
            print("  • Address warnings to improve overall quality")
        print("  • Run tests and validate fixes before deployment")
        print("  • Consider incremental improvements and testing")
    
    print("\n" + "=" * 80)
    
    # Exit code based on results
    if result.failed == 0 and result.success_rate >= 80:
        sys.exit(0)  # Success
    elif result.failed <= 2:
        sys.exit(1)  # Minor issues
    else:
        sys.exit(2)  # Significant issues


if __name__ == "__main__":
    main()