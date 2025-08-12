"""
Code quality tests without external dependencies.
Tests code structure, imports, and basic functionality.
"""

import sys
import os
import ast
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCodeStructure(unittest.TestCase):
    """Test code structure and organization."""
    
    def setUp(self):
        """Set up test environment."""
        self.src_dir = Path(__file__).parent.parent / "src" / "continual_transformer"
        self.test_dir = Path(__file__).parent
    
    def test_package_structure(self):
        """Test package structure is correct."""
        
        # Check main package directory exists
        self.assertTrue(self.src_dir.exists())
        self.assertTrue((self.src_dir / "__init__.py").exists())
        
        # Check core modules exist
        expected_modules = [
            "core",
            "adapters", 
            "tasks",
            "utils",
            "metrics",
            "monitoring",
            "optimization",
            "resilience",
            "scaling",
            "security"
        ]
        
        for module in expected_modules:
            module_path = self.src_dir / module
            self.assertTrue(module_path.exists(), f"Module {module} not found")
            self.assertTrue((module_path / "__init__.py").exists(), f"Module {module} missing __init__.py")
    
    def test_python_syntax(self):
        """Test all Python files have valid syntax."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        self.assertGreater(len(python_files), 0, "No Python files found")
        
        syntax_errors = []
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception as e:
                syntax_errors.append(f"{py_file}: {e}")
        
        if syntax_errors:
            self.fail(f"Syntax errors found:\n" + "\n".join(syntax_errors))
    
    def test_import_structure(self):
        """Test import structure is reasonable."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                # Check for potential issues
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                
                # Check for circular imports (basic check)
                relative_imports = [imp for imp in imports if imp and imp.startswith('.')]
                
                # This is a basic check - more sophisticated circular import detection would be needed
                self.assertLess(len(relative_imports), 20, 
                              f"Too many relative imports in {py_file}")
                
            except Exception as e:
                # Skip files that can't be parsed (might have advanced syntax)
                continue
    
    def test_docstring_coverage(self):
        """Test that important modules have docstrings."""
        
        important_files = [
            "core/model.py",
            "core/config.py", 
            "adapters/activation.py",
            "tasks/manager.py"
        ]
        
        missing_docstrings = []
        for file_path in important_files:
            full_path = self.src_dir / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    
                    # Check if module has a docstring
                    if (not ast.get_docstring(tree) and 
                        len([n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]) > 0):
                        missing_docstrings.append(file_path)
                        
                except Exception:
                    continue
        
        # Allow some missing docstrings but flag if too many
        self.assertLess(len(missing_docstrings), 3, 
                       f"Too many files missing docstrings: {missing_docstrings}")
    
    def test_class_structure(self):
        """Test class structure follows conventions."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        
        issues = []
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check class naming convention
                        if not node.name[0].isupper():
                            issues.append(f"{py_file}: Class {node.name} should start with uppercase")
                        
                        # Check for __init__ method in non-trivial classes
                        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        if len(methods) > 1 and "__init__" not in methods:
                            issues.append(f"{py_file}: Class {node.name} might need __init__ method")
                            
            except Exception:
                continue
        
        # Allow some style issues but flag major problems
        self.assertLess(len(issues), 10, f"Class structure issues: {issues[:5]}")
    
    def test_function_complexity(self):
        """Test function complexity is reasonable."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        
        complex_functions = []
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Simple complexity metric: count statements
                        statement_count = len([n for n in ast.walk(node) 
                                             if isinstance(n, (ast.stmt, ast.expr))])
                        
                        if statement_count > 200:  # Very high threshold for complex ML code
                            complex_functions.append(f"{py_file}:{node.name} ({statement_count} statements)")
                            
            except Exception:
                continue
        
        # Allow some complex functions in ML code but flag excessive complexity
        self.assertLess(len(complex_functions), 5, 
                       f"Overly complex functions: {complex_functions}")


class TestCodeQuality(unittest.TestCase):
    """Test code quality metrics."""
    
    def setUp(self):
        """Set up test environment."""
        self.src_dir = Path(__file__).parent.parent / "src" / "continual_transformer"
    
    def test_file_sizes(self):
        """Test file sizes are reasonable."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        large_files = []
        
        for py_file in python_files:
            size_kb = py_file.stat().st_size / 1024
            if size_kb > 200:  # 200KB threshold
                large_files.append(f"{py_file.name}: {size_kb:.1f}KB")
        
        # Allow some large files in ML projects but flag excessive sizes
        self.assertLess(len(large_files), 5, f"Large files found: {large_files}")
    
    def test_line_counts(self):
        """Test line counts are reasonable."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        long_files = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    line_count = len(f.readlines())
                
                if line_count > 2000:  # 2000 lines threshold
                    long_files.append(f"{py_file.name}: {line_count} lines")
                    
            except Exception:
                continue
        
        # Allow some long files in ML projects
        self.assertLess(len(long_files), 3, f"Very long files: {long_files}")
    
    def test_error_handling(self):
        """Test that error handling patterns exist."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        files_with_exceptions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for exception handling patterns
                if any(pattern in content for pattern in ['try:', 'except:', 'raise', 'Exception']):
                    files_with_exceptions += 1
                    
            except Exception:
                continue
        
        # Expect some error handling in a robust system
        self.assertGreater(files_with_exceptions, 5, 
                          "Expected more files with error handling")
    
    def test_logging_usage(self):
        """Test that logging is used appropriately."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        files_with_logging = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for logging patterns
                if any(pattern in content for pattern in ['logger.', 'logging.', 'log.']):
                    files_with_logging += 1
                    
            except Exception:
                continue
        
        # Expect logging in a production system
        self.assertGreater(files_with_logging, 8, 
                          "Expected more files with logging")
    
    def test_type_hints_presence(self):
        """Test that type hints are used in some places."""
        
        python_files = list(self.src_dir.rglob("*.py"))
        files_with_typing = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for type hint patterns
                if any(pattern in content for pattern in ['from typing import', ': str', ': int', ': float', '-> ']):
                    files_with_typing += 1
                    
            except Exception:
                continue
        
        # Expect some type hints in a modern Python project
        self.assertGreater(files_with_typing, 10, 
                          "Expected more files with type hints")


class TestProjectMetadata(unittest.TestCase):
    """Test project metadata and configuration."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent
    
    def test_required_files_exist(self):
        """Test that required project files exist."""
        
        required_files = [
            "README.md",
            "pyproject.toml",
            "LICENSE",
            "ARCHITECTURE.md"
        ]
        
        for filename in required_files:
            file_path = self.project_root / filename
            self.assertTrue(file_path.exists(), f"Required file {filename} not found")
    
    def test_documentation_structure(self):
        """Test documentation structure."""
        
        docs_dir = self.project_root / "docs"
        if docs_dir.exists():
            # Check for common documentation files
            expected_docs = ["README.md", "ROADMAP.md"]
            
            for doc in expected_docs:
                doc_path = docs_dir / doc
                if doc_path.exists():
                    self.assertGreater(doc_path.stat().st_size, 100, 
                                     f"Documentation file {doc} seems too small")
    
    def test_examples_exist(self):
        """Test that examples exist."""
        
        examples_dir = self.project_root / "examples"
        if examples_dir.exists():
            example_files = list(examples_dir.glob("*.py"))
            self.assertGreater(len(example_files), 0, "No example files found")
    
    def test_tests_structure(self):
        """Test tests directory structure."""
        
        tests_dir = self.project_root / "tests"
        self.assertTrue(tests_dir.exists(), "Tests directory not found")
        
        test_files = list(tests_dir.glob("test_*.py"))
        self.assertGreater(len(test_files), 0, "No test files found")


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)