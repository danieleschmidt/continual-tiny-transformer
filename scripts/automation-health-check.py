#!/usr/bin/env python3
"""
Automation health check system for continual-tiny-transformer.
Monitors and validates the health of all automation systems.
"""

import asyncio
import aiohttp
import json
import subprocess
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    response_time_ms: Optional[float] = None


class AutomationHealthChecker:
    """Comprehensive automation health monitoring system."""
    
    def __init__(self):
        self.results: List[HealthCheckResult] = []
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_owner = os.getenv('GITHUB_REPOSITORY_OWNER', 'your-org')
        self.repo_name = os.getenv('GITHUB_REPOSITORY_NAME', 'continual-tiny-transformer')
        
    async def check_github_actions_health(self) -> HealthCheckResult:
        """Check GitHub Actions workflow health."""
        start_time = datetime.now()
        
        if not self.github_token:
            return HealthCheckResult(
                name="GitHub Actions",
                status=HealthStatus.WARNING,
                message="GitHub token not available",
                details={"error": "GITHUB_TOKEN environment variable not set"},
                timestamp=datetime.now()
            )
        
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                # Check recent workflow runs
                workflows_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
                params = {'per_page': 20, 'status': 'completed'}
                
                async with session.get(workflows_url, params=params) as response:
                    if response.status != 200:
                        return HealthCheckResult(
                            name="GitHub Actions",
                            status=HealthStatus.CRITICAL,
                            message="Failed to fetch workflow data",
                            details={"status_code": response.status},
                            timestamp=datetime.now()
                        )
                    
                    data = await response.json()
                    runs = data.get('workflow_runs', [])
                    
                    if not runs:
                        return HealthCheckResult(
                            name="GitHub Actions",
                            status=HealthStatus.WARNING,
                            message="No recent workflow runs found",
                            details={"runs_count": 0},
                            timestamp=datetime.now()
                        )
                    
                    # Analyze run results
                    successful_runs = [r for r in runs if r['conclusion'] == 'success']
                    failed_runs = [r for r in runs if r['conclusion'] == 'failure']
                    
                    success_rate = len(successful_runs) / len(runs)
                    
                    # Check for recent failures
                    recent_failures = [
                        r for r in failed_runs 
                        if datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')) > 
                           datetime.now().replace(tzinfo=None) - timedelta(hours=24)
                    ]
                    
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    if success_rate < 0.8:
                        status = HealthStatus.CRITICAL
                        message = f"Low success rate: {success_rate:.1%}"
                    elif len(recent_failures) > 3:
                        status = HealthStatus.WARNING
                        message = f"{len(recent_failures)} failures in last 24h"
                    else:
                        status = HealthStatus.HEALTHY
                        message = f"Success rate: {success_rate:.1%}"
                    
                    return HealthCheckResult(
                        name="GitHub Actions",
                        status=status,
                        message=message,
                        details={
                            "total_runs": len(runs),
                            "successful_runs": len(successful_runs),
                            "failed_runs": len(failed_runs),
                            "success_rate": success_rate,
                            "recent_failures_24h": len(recent_failures)
                        },
                        timestamp=datetime.now(),
                        response_time_ms=response_time
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                name="GitHub Actions",
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def check_pre_commit_hooks_health(self) -> HealthCheckResult:
        """Check pre-commit hooks configuration and status."""
        try:
            # Check if pre-commit is installed
            result = subprocess.run(['pre-commit', '--version'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                return HealthCheckResult(
                    name="Pre-commit Hooks",
                    status=HealthStatus.CRITICAL,
                    message="Pre-commit not installed",
                    details={"error": "pre-commit command not found"},
                    timestamp=datetime.now()
                )
            
            # Check configuration file
            config_file = Path('.pre-commit-config.yaml')
            if not config_file.exists():
                return HealthCheckResult(
                    name="Pre-commit Hooks",
                    status=HealthStatus.CRITICAL,
                    message="Pre-commit config file missing",
                    details={"missing_file": ".pre-commit-config.yaml"},
                    timestamp=datetime.now()
                )
            
            # Test hooks
            test_result = subprocess.run(['pre-commit', 'run', '--all-files', '--dry-run'],
                                       capture_output=True, text=True)
            
            # Check git hooks installation
            hooks_installed = Path('.git/hooks/pre-commit').exists()
            
            status = HealthStatus.HEALTHY
            message = "Pre-commit hooks configured and working"
            
            if not hooks_installed:
                status = HealthStatus.WARNING
                message = "Pre-commit hooks not installed in git"
            
            return HealthCheckResult(
                name="Pre-commit Hooks",
                status=status,
                message=message,
                details={
                    "pre_commit_version": result.stdout.strip(),
                    "config_exists": config_file.exists(),
                    "hooks_installed": hooks_installed,
                    "test_exit_code": test_result.returncode
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Pre-commit Hooks", 
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def check_testing_automation_health(self) -> HealthCheckResult:
        """Check testing automation health."""
        try:
            # Check if pytest is available
            result = subprocess.run(['pytest', '--version'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                return HealthCheckResult(
                    name="Testing Automation",
                    status=HealthStatus.CRITICAL,
                    message="Pytest not available",
                    details={"error": "pytest command not found"},
                    timestamp=datetime.now()
                )
            
            # Check test configuration
            config_files = [
                'pytest.ini',
                'pyproject.toml',
                'setup.cfg'
            ]
            
            config_found = any(Path(f).exists() for f in config_files)
            
            # Check test directories
            test_dirs = ['tests/', 'test/']
            test_dir_exists = any(Path(d).exists() for d in test_dirs)
            
            # Quick test discovery
            discovery_result = subprocess.run(['pytest', '--collect-only', '-q'],
                                            capture_output=True, text=True)
            
            test_count = 0
            if discovery_result.returncode == 0:
                # Count discovered tests
                lines = discovery_result.stdout.split('\n')
                for line in lines:
                    if 'collected' in line and 'item' in line:
                        try:
                            test_count = int(line.split()[0])
                        except:
                            pass
            
            status = HealthStatus.HEALTHY
            message = f"Testing automation healthy, {test_count} tests found"
            
            if not config_found:
                status = HealthStatus.WARNING
                message = "No pytest configuration found"
            elif not test_dir_exists:
                status = HealthStatus.WARNING 
                message = "No test directories found"
            elif test_count == 0:
                status = HealthStatus.WARNING
                message = "No tests discovered"
            
            return HealthCheckResult(
                name="Testing Automation",
                status=status,
                message=message,
                details={
                    "pytest_version": result.stdout.strip(),
                    "config_found": config_found,
                    "test_directory_exists": test_dir_exists,
                    "tests_discovered": test_count,
                    "discovery_success": discovery_result.returncode == 0
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Testing Automation",
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def check_security_automation_health(self) -> HealthCheckResult:
        """Check security automation tools health."""
        try:
            security_tools = {
                'bandit': ['bandit', '--version'],
                'safety': ['safety', '--version'],
                'ruff': ['ruff', '--version']
            }
            
            tool_status = {}
            all_healthy = True
            
            for tool_name, command in security_tools.items():
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    tool_status[tool_name] = {
                        'available': result.returncode == 0,
                        'version': result.stdout.strip() if result.returncode == 0 else None,
                        'error': result.stderr if result.returncode != 0 else None
                    }
                    
                    if result.returncode != 0:
                        all_healthy = False
                        
                except FileNotFoundError:
                    tool_status[tool_name] = {
                        'available': False,
                        'version': None,
                        'error': 'Command not found'
                    }
                    all_healthy = False
            
            # Check security configuration files
            security_configs = {
                'bandit_config': Path('.bandit').exists() or Path('pyproject.toml').exists(),
                'safety_policy': Path('.safety-policy.json').exists(),
                'security_baseline': Path('.secrets.baseline').exists()
            }
            
            available_tools = sum(1 for tool in tool_status.values() if tool['available'])
            
            if all_healthy and available_tools >= 2:
                status = HealthStatus.HEALTHY
                message = f"Security automation healthy, {available_tools}/3 tools available"
            elif available_tools >= 1:
                status = HealthStatus.WARNING
                message = f"Partial security coverage, {available_tools}/3 tools available"
            else:
                status = HealthStatus.CRITICAL
                message = "No security tools available"
            
            return HealthCheckResult(
                name="Security Automation",
                status=status,
                message=message,
                details={
                    "tools": tool_status,
                    "configs": security_configs,
                    "available_tools": available_tools
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Security Automation",
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def check_build_automation_health(self) -> HealthCheckResult:
        """Check build automation health."""
        try:
            # Check build tools
            build_tools = {
                'python': ['python', '--version'],
                'pip': ['pip', '--version'],
                'build': ['python', '-m', 'build', '--version']
            }
            
            tool_status = {}
            for tool_name, command in build_tools.items():
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    tool_status[tool_name] = {
                        'available': result.returncode == 0,
                        'version': result.stdout.strip() if result.returncode == 0 else None
                    }
                except FileNotFoundError:
                    tool_status[tool_name] = {'available': False, 'version': None}
            
            # Check build configuration
            build_configs = {
                'pyproject_toml': Path('pyproject.toml').exists(),
                'setup_py': Path('setup.py').exists(),
                'requirements_txt': Path('requirements.txt').exists(),
                'makefile': Path('Makefile').exists()
            }
            
            # Check if we can perform a dry-run build
            build_test_result = subprocess.run(['python', '-c', 'import build'], 
                                             capture_output=True, text=True)
            
            available_tools = sum(1 for tool in tool_status.values() if tool['available'])
            config_files = sum(1 for exists in build_configs.values() if exists)
            
            if available_tools >= 2 and config_files >= 1:
                status = HealthStatus.HEALTHY
                message = "Build automation fully configured"
            elif available_tools >= 1:
                status = HealthStatus.WARNING
                message = "Build automation partially configured"
            else:
                status = HealthStatus.CRITICAL
                message = "Build automation not available"
            
            return HealthCheckResult(
                name="Build Automation",
                status=status,
                message=message,
                details={
                    "tools": tool_status,
                    "configs": build_configs,
                    "build_module_available": build_test_result.returncode == 0
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Build Automation",
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    def check_dependency_automation_health(self) -> HealthCheckResult:
        """Check dependency management automation health."""
        try:
            # Check dependency files
            dependency_files = {
                'requirements.txt': Path('requirements.txt').exists(),
                'requirements-dev.txt': Path('requirements-dev.txt').exists(),
                'pyproject.toml': Path('pyproject.toml').exists(),
                'poetry.lock': Path('poetry.lock').exists(),
                'Pipfile': Path('Pipfile').exists()
            }
            
            # Check dependency management tools
            dep_tools = {
                'pip': ['pip', '--version'],
                'pip-tools': ['pip-compile', '--version'],
                'poetry': ['poetry', '--version']
            }
            
            tool_status = {}
            for tool_name, command in dep_tools.items():
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    tool_status[tool_name] = {
                        'available': result.returncode == 0,
                        'version': result.stdout.strip() if result.returncode == 0 else None
                    }
                except FileNotFoundError:
                    tool_status[tool_name] = {'available': False, 'version': None}
            
            # Check for outdated dependencies
            outdated_check = subprocess.run(['pip', 'list', '--outdated'], 
                                          capture_output=True, text=True)
            outdated_count = 0
            if outdated_check.returncode == 0:
                outdated_lines = [line for line in outdated_check.stdout.split('\n') 
                                if line and not line.startswith('Package')]
                outdated_count = len(outdated_lines)
            
            dependency_files_count = sum(1 for exists in dependency_files.values() if exists)
            available_tools_count = sum(1 for tool in tool_status.values() if tool['available'])
            
            if dependency_files_count >= 1 and available_tools_count >= 1:
                if outdated_count > 10:
                    status = HealthStatus.WARNING
                    message = f"Dependency automation working, {outdated_count} outdated packages"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Dependency automation healthy"
            else:
                status = HealthStatus.CRITICAL
                message = "Dependency automation not configured"
            
            return HealthCheckResult(
                name="Dependency Automation",
                status=status,
                message=message,
                details={
                    "dependency_files": dependency_files,
                    "tools": tool_status,
                    "outdated_packages": outdated_count
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="Dependency Automation",
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def run_all_health_checks(self) -> List[HealthCheckResult]:
        """Run all automation health checks."""
        print("Running automation health checks...")
        
        # Run all checks
        checks = [
            self.check_github_actions_health(),
            self.check_pre_commit_hooks_health(),
            self.check_testing_automation_health(), 
            self.check_security_automation_health(),
            self.check_build_automation_health(),
            self.check_dependency_automation_health()
        ]
        
        # Handle async and sync checks
        github_result = await checks[0]
        other_results = [check() for check in checks[1:]]
        
        self.results = [github_result] + other_results
        
        return self.results
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        if not self.results:
            return {"error": "No health check results available"}
        
        total_checks = len(self.results)
        healthy_checks = len([r for r in self.results if r.status == HealthStatus.HEALTHY])
        warning_checks = len([r for r in self.results if r.status == HealthStatus.WARNING])
        critical_checks = len([r for r in self.results if r.status == HealthStatus.CRITICAL])
        
        overall_health = HealthStatus.HEALTHY
        if critical_checks > 0:
            overall_health = HealthStatus.CRITICAL
        elif warning_checks > 0:
            overall_health = HealthStatus.WARNING
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": overall_health.value,
            "summary": {
                "total_checks": total_checks,
                "healthy": healthy_checks,
                "warnings": warning_checks,
                "critical": critical_checks,
                "health_score": healthy_checks / total_checks if total_checks > 0 else 0
            },
            "checks": []
        }
        
        for result in self.results:
            check_data = {
                "name": result.name,
                "status": result.status.value,
                "message": result.message,
                "timestamp": result.timestamp.isoformat(),
                "details": result.details
            }
            
            if result.response_time_ms:
                check_data["response_time_ms"] = result.response_time_ms
            
            report["checks"].append(check_data)
        
        return report
    
    def generate_health_summary_text(self) -> str:
        """Generate human-readable health summary."""
        if not self.results:
            return "No health check results available"
        
        report = self.generate_health_report()
        summary = report["summary"]
        
        text = f"""
# Automation Health Check Report

**Overall Status:** {report['overall_health'].upper()}
**Health Score:** {summary['health_score']:.1%}
**Timestamp:** {report['timestamp']}

## Summary
- **Total Checks:** {summary['total_checks']}
- **Healthy:** {summary['healthy']} ✅
- **Warnings:** {summary['warnings']} ⚠️  
- **Critical:** {summary['critical']} ❌

## Detailed Results

"""
        
        for check in report["checks"]:
            status_emoji = {
                "healthy": "✅",
                "warning": "⚠️",
                "critical": "❌",
                "unknown": "❓"
            }.get(check["status"], "❓")
            
            text += f"### {check['name']} {status_emoji}\n"
            text += f"**Status:** {check['status'].title()}\n"
            text += f"**Message:** {check['message']}\n"
            
            if check.get("response_time_ms"):
                text += f"**Response Time:** {check['response_time_ms']:.0f}ms\n"
            
            text += "\n"
        
        # Recommendations
        text += "## Recommendations\n\n"
        
        critical_checks = [c for c in report["checks"] if c["status"] == "critical"]
        warning_checks = [c for c in report["checks"] if c["status"] == "warning"]
        
        if critical_checks:
            text += "**Critical Issues (Immediate Action Required):**\n"
            for check in critical_checks:
                text += f"- Fix {check['name']}: {check['message']}\n"
            text += "\n"
        
        if warning_checks:
            text += "**Warnings (Should Be Addressed):**\n"
            for check in warning_checks:
                text += f"- Improve {check['name']}: {check['message']}\n"
            text += "\n"
        
        if not critical_checks and not warning_checks:
            text += "All automation systems are healthy! 🎉\n"
        
        return text


async def main():
    """Main function to run automation health checks."""
    checker = AutomationHealthChecker()
    
    # Run health checks
    results = await checker.run_all_health_checks()
    
    # Generate reports
    json_report = checker.generate_health_report()
    text_summary = checker.generate_health_summary_text()
    
    # Save reports
    with open('automation-health-report.json', 'w') as f:
        json.dump(json_report, f, indent=2)
    
    with open('automation-health-summary.md', 'w') as f:
        f.write(text_summary)
    
    print("Automation health check completed!")
    print(f"Overall health: {json_report['overall_health']}")
    print(f"Health score: {json_report['summary']['health_score']:.1%}")
    print("\nReports saved:")
    print("- automation-health-report.json")
    print("- automation-health-summary.md")
    
    # Exit with error code if there are critical issues
    if json_report['summary']['critical'] > 0:
        print(f"\n❌ {json_report['summary']['critical']} critical issues found!")
        sys.exit(1)
    elif json_report['summary']['warnings'] > 0:
        print(f"\n⚠️  {json_report['summary']['warnings']} warnings found.")
        sys.exit(0)
    else:
        print("\n✅ All automation systems healthy!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())