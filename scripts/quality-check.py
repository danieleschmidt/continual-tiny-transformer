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