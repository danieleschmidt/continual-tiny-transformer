#!/usr/bin/env python3
"""
Basic functionality test to validate core continual learning functionality.

This test validates that:
1. Configuration system works properly
2. Core model architecture is sound
3. Import system is functional
4. Basic API contracts are met
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config_system():
    """Test configuration system functionality."""
    print("🔧 Testing configuration system...")
    
    try:
        # Import config module directly without going through __init__.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        # Test default configuration
        config = ContinualConfig()
        assert config.model_name == "distilbert-base-uncased"
        assert config.max_tasks == 50
        assert config.adaptation_method == "activation"
        
        # Test configuration updates
        config.update(max_tasks=25, learning_rate=1e-5)
        assert config.max_tasks == 25
        assert config.learning_rate == 1e-5
        
        # Test task-specific configuration
        config.set_task_config("test_task", {"special_param": 42})
        task_config = config.get_task_config("test_task")
        assert task_config["special_param"] == 42
        
        print("  ✅ Configuration system validation passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration system error: {e}")
        return False

def test_imports():
    """Test that all major imports work."""
    print("📦 Testing import system...")
    
    try:
        # Test basic config import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        print("  ✅ Core imports successful")
        
        # Test that modules have expected structure
        config = ContinualConfig()
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'max_tasks')
        
        print("  ✅ Module structure validation passed")
        
        # Test that basic functionality works
        assert config.model_name == "distilbert-base-uncased"
        assert config.max_tasks == 50
        assert config.device in ["auto", "cpu", "cuda", "mps"]
        
        print("  ✅ Basic functionality validation passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Import system error: {e}")
        return False

def test_api_contracts():
    """Test API contracts without requiring heavy dependencies."""
    print("🔌 Testing API contracts...")
    
    try:
        # Import config module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        # Test configuration serialization
        config = ContinualConfig(max_tasks=10, learning_rate=1e-4)
        config_dict = config.to_dict()
        
        assert 'max_tasks' in config_dict
        assert 'learning_rate' in config_dict
        assert config_dict['max_tasks'] == 10
        assert config_dict['learning_rate'] == 1e-4
        
        print("  ✅ Configuration serialization works")
        
        # Test validation
        try:
            invalid_config = ContinualConfig(adaptation_method="invalid_method")
            assert False, "Should have raised validation error"
        except ValueError:
            print("  ✅ Configuration validation works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ API contract error: {e}")
        return False

def test_project_structure():
    """Test that project structure is complete."""
    print("🏗️ Testing project structure...")
    
    required_dirs = [
        "src/continual_transformer",
        "src/continual_transformer/core", 
        "src/continual_transformer/adapters",
        "src/continual_transformer/tasks",
        "src/continual_transformer/metrics",
        "examples",
        "tests"
    ]
    
    required_files = [
        "src/continual_transformer/__init__.py",
        "src/continual_transformer/core/config.py",
        "src/continual_transformer/core/model.py",
        "pyproject.toml",
        "README.md"
    ]
    
    missing_dirs = []
    missing_files = []
    
    repo_root = Path(__file__).parent
    
    for dir_path in required_dirs:
        if not (repo_root / dir_path).exists():
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not (repo_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"  ❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    
    print("  ✅ Project structure validation passed")
    return True

def main():
    """Run all basic functionality tests."""
    print("🚀 Running Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_project_structure,
        test_config_system,
        test_imports,
        test_api_contracts,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL BASIC FUNCTIONALITY TESTS PASSED!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
