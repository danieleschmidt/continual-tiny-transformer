#!/usr/bin/env python3
"""
Comprehensive quality check script for the continual learning project.
Performs static analysis, security checks, and generates quality reports.
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast
import tempfile


class QualityChecker:
    """Comprehensive quality checker for the project."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.test_dir = self.project_root / "tests"
        self.quality_report = {
            "timestamp": time.time(),
            "checks": {},
            "overall_score": 0.0,
            "recommendations": []
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all quality checks and return comprehensive report."""
        
        print("🔍 Starting comprehensive quality checks...")
        
        # Core quality checks
        self.check_syntax()
        self.check_imports()
        self.check_code_structure()
        self.check_documentation()
        self.check_security_patterns()
        self.check_performance_patterns()
        self.check_test_coverage()
        
        # Calculate overall score
        self.calculate_overall_score()
        
        # Generate recommendations
        self.generate_recommendations()
        
        print(f"✅ Quality checks completed. Overall score: {self.quality_report['overall_score']:.1f}/100")
        
        return self.quality_report
    
    def check_syntax(self):
        """Check syntax of all Python files."""
        
        print("🔧 Checking Python syntax...")
        
        python_files = list(self.src_dir.rglob("*.py"))
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                syntax_errors.append({
                    "file": str(py_file),
                    "error": str(e),
                    "line": e.lineno
                })
        
        score = 100.0 if len(syntax_errors) == 0 else max(0, 100 - len(syntax_errors) * 10)
        
        self.quality_report["checks"]["syntax"] = {
            "score": score,
            "files_checked": len(python_files),
            "syntax_errors": syntax_errors,
            "status": "pass" if score == 100 else "fail"
        }
    
    def check_imports(self):
        """Check import structure and dependencies."""
        
        print("📦 Checking imports and dependencies...")
        
        python_files = list(self.src_dir.rglob("*.py"))
        import_issues = []
        circular_imports = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                
                # Check for potential issues
                external_deps = [imp for imp in imports if not imp.startswith('.') and not imp.startswith('continual_transformer')]
                internal_deps = [imp for imp in imports if imp.startswith('continual_transformer')]
                
                # Flag files with too many external dependencies
                if len(external_deps) > 15:
                    import_issues.append({
                        "file": str(py_file),
                        "issue": "too_many_external_deps",
                        "count": len(external_deps)
                    })
                
                # Basic circular import detection
                if len(internal_deps) > 10:
                    circular_imports.append({
                        "file": str(py_file),
                        "internal_imports": len(internal_deps)
                    })
                    
            except Exception:
                continue
        
        score = max(0, 100 - len(import_issues) * 5 - len(circular_imports) * 3)
        
        self.quality_report["checks"]["imports"] = {
            "score": score,
            "import_issues": import_issues,
            "potential_circular_imports": circular_imports,
            "status": "pass" if score >= 80 else "warning" if score >= 60 else "fail"
        }
    
    def check_code_structure(self):
        """Check code structure and organization."""
        
        print("🏗️ Checking code structure...")
        
        python_files = list(self.src_dir.rglob("*.py"))
        structure_issues = []
        complexity_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    content = f.read()
                
                # Check file size
                lines = len(content.split('\n'))
                if lines > 1500:
                    structure_issues.append({
                        "file": str(py_file),
                        "issue": "file_too_long",
                        "lines": lines
                    })
                
                # Check function complexity
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        stmt_count = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
                        if stmt_count > 150:  # Adjusted threshold for ML code
                            complexity_issues.append({
                                "file": str(py_file),
                                "function": node.name,
                                "statements": stmt_count
                            })
                
                # Check class structure
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if not node.name[0].isupper():
                            structure_issues.append({
                                "file": str(py_file),
                                "issue": "class_naming",
                                "class": node.name
                            })
                            
            except Exception:
                continue
        
        # More lenient scoring for ML projects
        score = max(0, 100 - len(structure_issues) * 2 - len(complexity_issues) * 1)
        
        self.quality_report["checks"]["code_structure"] = {
            "score": score,
            "structure_issues": structure_issues,
            "complexity_issues": complexity_issues,
            "status": "pass" if score >= 75 else "warning" if score >= 50 else "fail"
        }
    
    def check_documentation(self):
        """Check documentation coverage and quality."""
        
        print("📚 Checking documentation...")
        
        python_files = list(self.src_dir.rglob("*.py"))
        doc_coverage = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                # Check module docstring
                module_doc = ast.get_docstring(tree)
                
                # Count classes and functions
                classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
                functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                
                if classes or functions:
                    has_module_doc = module_doc is not None
                    
                    # Check class docstrings
                    classes_with_docs = 0
                    for cls in classes:
                        if ast.get_docstring(cls):
                            classes_with_docs += 1
                    
                    # Check function docstrings
                    functions_with_docs = 0
                    for func in functions:
                        if ast.get_docstring(func):
                            functions_with_docs += 1
                    
                    total_items = len(classes) + len(functions)
                    documented_items = classes_with_docs + functions_with_docs
                    
                    coverage = documented_items / total_items if total_items > 0 else 1.0
                    
                    doc_coverage.append({
                        "file": str(py_file),
                        "module_doc": has_module_doc,
                        "coverage": coverage,
                        "total_items": total_items,
                        "documented_items": documented_items
                    })
                    
            except Exception:
                continue
        
        # Calculate overall documentation score
        if doc_coverage:
            avg_coverage = sum(item["coverage"] for item in doc_coverage) / len(doc_coverage)
            module_doc_ratio = sum(1 for item in doc_coverage if item["module_doc"]) / len(doc_coverage)
            score = (avg_coverage * 70 + module_doc_ratio * 30)
        else:
            score = 0
        
        self.quality_report["checks"]["documentation"] = {
            "score": score,
            "average_coverage": avg_coverage if doc_coverage else 0,
            "files_with_module_docs": sum(1 for item in doc_coverage if item["module_doc"]),
            "total_files": len(doc_coverage),
            "status": "pass" if score >= 70 else "warning" if score >= 40 else "fail"
        }
    
    def check_security_patterns(self):
        """Check for security best practices."""
        
        print("🛡️ Checking security patterns...")
        
        python_files = list(self.src_dir.rglob("*.py"))
        security_issues = []
        security_features = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for potential security issues
                if "eval(" in content or "exec(" in content:
                    security_issues.append({
                        "file": str(py_file),
                        "issue": "dangerous_eval_exec",
                        "severity": "high"
                    })
                
                if "shell=True" in content:
                    security_issues.append({
                        "file": str(py_file),
                        "issue": "shell_injection_risk",
                        "severity": "medium"
                    })
                
                # Check for security features
                if any(pattern in content for pattern in ["validate_", "sanitize_", "security"]):
                    security_features.append({
                        "file": str(py_file),
                        "feature": "validation_sanitization"
                    })
                
                if "logging" in content and any(level in content for level in ["warning", "error", "critical"]):
                    security_features.append({
                        "file": str(py_file),
                        "feature": "security_logging"
                    })
                    
            except Exception:
                continue
        
        # Calculate security score
        high_issues = sum(1 for issue in security_issues if issue["severity"] == "high")
        medium_issues = sum(1 for issue in security_issues if issue["severity"] == "medium")
        
        score = max(0, 100 - high_issues * 20 - medium_issues * 10)
        score += min(20, len(security_features) * 2)  # Bonus for security features
        score = min(100, score)
        
        self.quality_report["checks"]["security"] = {
            "score": score,
            "security_issues": security_issues,
            "security_features": len(security_features),
            "status": "pass" if score >= 80 else "warning" if score >= 60 else "fail"
        }
    
    def check_performance_patterns(self):
        """Check for performance best practices."""
        
        print("⚡ Checking performance patterns...")
        
        python_files = list(self.src_dir.rglob("*.py"))
        performance_features = []
        performance_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for performance features
                if "torch.no_grad()" in content:
                    performance_features.append({"file": str(py_file), "feature": "no_grad_optimization"})
                
                if "checkpoint" in content.lower():
                    performance_features.append({"file": str(py_file), "feature": "gradient_checkpointing"})
                
                if "cache" in content.lower():
                    performance_features.append({"file": str(py_file), "feature": "caching"})
                
                if "batch" in content.lower():
                    performance_features.append({"file": str(py_file), "feature": "batching"})
                
                # Check for potential performance issues
                if content.count("for ") > 20:  # Many loops might indicate inefficiency
                    performance_issues.append({
                        "file": str(py_file),
                        "issue": "many_loops",
                        "count": content.count("for ")
                    })
                    
            except Exception:
                continue
        
        score = min(100, len(performance_features) * 3 - len(performance_issues) * 5)
        score = max(0, score)
        
        self.quality_report["checks"]["performance"] = {
            "score": score,
            "performance_features": len(performance_features),
            "performance_issues": performance_issues,
            "status": "pass" if score >= 70 else "warning" if score >= 40 else "fail"
        }
    
    def check_test_coverage(self):
        """Check test coverage and structure."""
        
        print("🧪 Checking test coverage...")
        
        src_files = list(self.src_dir.rglob("*.py"))
        test_files = list(self.test_dir.rglob("test_*.py"))
        
        # Basic coverage check - do test files exist for main modules?
        main_modules = [f for f in src_files if f.name != "__init__.py"]
        
        # Simple heuristic: ratio of test files to source files
        coverage_ratio = len(test_files) / max(len(main_modules), 1)
        
        score = min(100, coverage_ratio * 100)
        
        self.quality_report["checks"]["test_coverage"] = {
            "score": score,
            "test_files": len(test_files),
            "source_files": len(main_modules),
            "coverage_ratio": coverage_ratio,
            "status": "pass" if score >= 70 else "warning" if score >= 30 else "fail"
        }
    
    def calculate_overall_score(self):
        """Calculate overall quality score."""
        
        weights = {
            "syntax": 0.25,
            "imports": 0.15,
            "code_structure": 0.20,
            "documentation": 0.15,
            "security": 0.15,
            "performance": 0.05,
            "test_coverage": 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for check_name, check_data in self.quality_report["checks"].items():
            if check_name in weights:
                score = check_data.get("score", 0)
                weight = weights[check_name]
                total_score += score * weight
                total_weight += weight
        
        self.quality_report["overall_score"] = total_score / total_weight if total_weight > 0 else 0
    
    def generate_recommendations(self):
        """Generate improvement recommendations."""
        
        recommendations = []
        
        for check_name, check_data in self.quality_report["checks"].items():
            score = check_data.get("score", 0)
            status = check_data.get("status", "unknown")
            
            if status == "fail":
                if check_name == "syntax":
                    recommendations.append("Fix syntax errors to improve code reliability")
                elif check_name == "imports":
                    recommendations.append("Simplify import structure to reduce complexity")
                elif check_name == "code_structure":
                    recommendations.append("Refactor large functions and improve code organization")
                elif check_name == "documentation":
                    recommendations.append("Add docstrings to improve code maintainability")
                elif check_name == "security":
                    recommendations.append("Address security issues and add input validation")
                elif check_name == "test_coverage":
                    recommendations.append("Add more tests to improve reliability")
            elif status == "warning":
                if check_name == "performance":
                    recommendations.append("Consider performance optimizations for better efficiency")
        
        # Add general recommendations based on overall score
        overall_score = self.quality_report["overall_score"]
        if overall_score < 60:
            recommendations.append("Focus on addressing critical issues first")
        elif overall_score < 80:
            recommendations.append("Good foundation - focus on polish and optimization")
        else:
            recommendations.append("Excellent code quality - maintain current standards")
        
        self.quality_report["recommendations"] = recommendations
    
    def save_report(self, output_file: str):
        """Save quality report to file."""
        
        with open(output_file, 'w') as f:
            json.dump(self.quality_report, f, indent=2)
        
        print(f"📊 Quality report saved to {output_file}")
    
    def print_summary(self):
        """Print quality check summary."""
        
        print("\n" + "="*60)
        print("📊 QUALITY CHECK SUMMARY")
        print("="*60)
        
        overall_score = self.quality_report["overall_score"]
        print(f"Overall Score: {overall_score:.1f}/100")
        
        if overall_score >= 90:
            print("🏆 Excellent - Production ready")
        elif overall_score >= 80:
            print("✅ Good - Minor improvements needed")
        elif overall_score >= 70:
            print("⚠️ Acceptable - Some improvements needed")
        elif overall_score >= 60:
            print("⚠️ Needs work - Multiple issues to address")
        else:
            print("❌ Poor - Major improvements required")
        
        print("\nCheck Results:")
        for check_name, check_data in self.quality_report["checks"].items():
            score = check_data.get("score", 0)
            status = check_data.get("status", "unknown")
            
            status_icon = {
                "pass": "✅",
                "warning": "⚠️",
                "fail": "❌",
                "unknown": "❓"
            }.get(status, "❓")
            
            print(f"  {status_icon} {check_name.replace('_', ' ').title()}: {score:.1f}/100")
        
        print("\nRecommendations:")
        for i, rec in enumerate(self.quality_report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print("="*60)


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = Path(__file__).parent.parent
    
    checker = QualityChecker(project_root)
    
    try:
        report = checker.run_all_checks()
        checker.print_summary()
        
        # Save report
        output_file = Path(project_root) / "quality_report.json"
        checker.save_report(str(output_file))
        
        # Exit with appropriate code
        overall_score = report["overall_score"]
        if overall_score >= 70:
            sys.exit(0)  # Success
        elif overall_score >= 50:
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Failure
            
    except Exception as e:
        print(f"❌ Quality check failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()