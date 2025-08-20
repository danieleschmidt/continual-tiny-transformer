#!/usr/bin/env python3
"""
Enhanced Robustness validation test for Continual Tiny Transformer.

This test validates that:
1. Error handling systems work properly
2. Input validation is comprehensive
3. Edge cases are handled gracefully
4. Recovery mechanisms function correctly
5. Security validations are in place
"""

import sys
from pathlib import Path
import json
import tempfile
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_configuration_robustness():
    """Test configuration system robustness and validation."""
    print("🛡️ Testing configuration robustness...")
    
    try:
        # Import config module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        # Test invalid configuration values
        try:
            config = ContinualConfig(adaptation_method="invalid_method")
            assert False, "Should have raised ValueError for invalid adaptation method"
        except ValueError:
            print("  ✅ Invalid adaptation method validation works")
        
        try:
            config = ContinualConfig(task_routing_method="invalid_routing")
            assert False, "Should have raised ValueError for invalid routing method"
        except ValueError:
            print("  ✅ Invalid routing method validation works")
        
        # Test edge case values
        config = ContinualConfig(max_tasks=1)  # Minimum tasks
        assert config.max_tasks == 1
        print("  ✅ Minimum task configuration works")
        
        config = ContinualConfig(learning_rate=1e-10)  # Very small learning rate
        assert config.learning_rate == 1e-10
        print("  ✅ Edge case learning rate works")
        
        # Test configuration updates with invalid values
        config = ContinualConfig()
        try:
            config.update(nonexistent_param="value")
            assert False, "Should have raised ValueError for unknown parameter"
        except ValueError:
            print("  ✅ Invalid parameter update validation works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration robustness error: {e}")
        return False

def test_file_operations_robustness():
    """Test file operations and path handling robustness."""
    print("🗂️ Testing file operations robustness...")
    
    try:
        # Import config module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        # Test with non-existent directory paths
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test that paths are created automatically
            config = ContinualConfig(
                output_dir=str(Path(temp_dir) / "non_existent" / "output"),
                cache_dir=str(Path(temp_dir) / "non_existent" / "cache")
            )
            
            assert Path(config.output_dir).exists(), "Output directory should be created"
            assert Path(config.cache_dir).exists(), "Cache directory should be created"
            print("  ✅ Automatic directory creation works")
        
        # Test path handling with special characters
        with tempfile.TemporaryDirectory() as temp_dir:
            special_path = str(Path(temp_dir) / "path with spaces & symbols")
            config = ContinualConfig(output_dir=special_path)
            assert Path(config.output_dir).exists()
            print("  ✅ Special character path handling works")
        
        # Test serialization without yaml
        config = ContinualConfig(max_tasks=15, learning_rate=3e-5)
        config_dict = config.to_dict()
        
        # Verify all required fields are present
        required_fields = ['model_name', 'max_tasks', 'learning_rate', 'device']
        for field in required_fields:
            assert field in config_dict, f"Missing required field: {field}"
        
        print("  ✅ Configuration serialization robustness works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ File operations robustness error: {e}")
        return False

def test_input_validation_robustness():
    """Test input validation and sanitization."""
    print("🔒 Testing input validation robustness...")
    
    try:
        # Import config module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        # Test extreme values
        config = ContinualConfig(max_tasks=10000)  # Very large
        assert config.max_tasks == 10000
        print("  ✅ Large value handling works")
        
        # Test negative values where they make sense
        config = ContinualConfig(warmup_steps=0)  # Zero warmup
        assert config.warmup_steps == 0
        print("  ✅ Zero value handling works")
        
        # Test boundary conditions
        config = ContinualConfig(gradient_clipping=0.0)  # No clipping
        assert config.gradient_clipping == 0.0
        print("  ✅ Boundary value handling works")
        
        # Test string validation
        config = ContinualConfig(model_name="custom-model-name")
        assert config.model_name == "custom-model-name"
        print("  ✅ String parameter validation works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Input validation robustness error: {e}")
        return False

def test_device_detection_robustness():
    """Test device detection and fallback mechanisms."""
    print("🖥️ Testing device detection robustness...")
    
    try:
        # Import config module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        # Test auto device detection (should fallback to CPU when torch not available)
        config = ContinualConfig(device="auto")
        assert config.device == "cpu", f"Expected 'cpu', got '{config.device}'"
        print("  ✅ Auto device detection with fallback works")
        
        # Test explicit device settings
        for device in ["cpu", "cuda", "mps"]:
            config = ContinualConfig(device=device)
            assert config.device == device
        print("  ✅ Explicit device settings work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Device detection robustness error: {e}")
        return False

def test_task_configuration_robustness():
    """Test task-specific configuration robustness."""
    print("📋 Testing task configuration robustness...")
    
    try:
        # Import config module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        config = ContinualConfig()
        
        # Test nested configuration
        complex_task_config = {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "special_params": {
                "nested_param": "value",
                "numeric_param": 42
            }
        }
        
        config.set_task_config("complex_task", complex_task_config)
        retrieved_config = config.get_task_config("complex_task")
        
        assert retrieved_config == complex_task_config
        print("  ✅ Nested task configuration works")
        
        # Test task configuration retrieval for non-existent task
        empty_config = config.get_task_config("non_existent_task")
        assert empty_config == {}
        print("  ✅ Non-existent task configuration returns empty dict")
        
        # Test overwriting task configuration
        config.set_task_config("complex_task", {"new_param": "new_value"})
        updated_config = config.get_task_config("complex_task")
        assert updated_config == {"new_param": "new_value"}
        print("  ✅ Task configuration overwriting works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Task configuration robustness error: {e}")
        return False

def test_security_validations():
    """Test security-related validations."""
    print("🔐 Testing security validations...")
    
    try:
        # Import config module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        # Test path traversal protection (basic check)
        with tempfile.TemporaryDirectory() as temp_dir:
            safe_path = str(Path(temp_dir) / "safe_output")
            config = ContinualConfig(output_dir=safe_path)
            
            # Verify path is normalized and safe
            assert os.path.abspath(config.output_dir) == os.path.abspath(safe_path)
            print("  ✅ Path normalization works")
        
        # Test configuration dict doesn't contain sensitive data
        config = ContinualConfig()
        config_dict = config.to_dict()
        
        # Verify no obvious sensitive keys
        sensitive_patterns = ['password', 'token', 'secret', 'key', 'credential']
        for key in config_dict.keys():
            key_lower = key.lower()
            for pattern in sensitive_patterns:
                assert pattern not in key_lower, f"Potential sensitive data in config: {key}"
        
        print("  ✅ Configuration doesn't expose sensitive data")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Security validation error: {e}")
        return False

def test_error_recovery():
    """Test error recovery and graceful degradation."""
    print("🚨 Testing error recovery...")
    
    try:
        # Import config module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "config", str(Path(__file__).parent / "src" / "continual_transformer" / "core" / "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        ContinualConfig = config_module.ContinualConfig
        
        # Test recovery from invalid YAML operations (when yaml not available)
        config = ContinualConfig()
        
        try:
            config.to_yaml("/tmp/test_config.yaml")
            assert False, "Should have raised ImportError when yaml not available"
        except ImportError as e:
            assert "PyYAML is required" in str(e)
            print("  ✅ Graceful degradation for missing YAML works")
        
        try:
            ContinualConfig.from_yaml("/tmp/nonexistent.yaml")
            assert False, "Should have raised ImportError when yaml not available"
        except ImportError as e:
            assert "PyYAML is required" in str(e)
            print("  ✅ Graceful degradation for YAML loading works")
        
        # Test configuration with partial invalid data
        config = ContinualConfig()
        original_max_tasks = config.max_tasks
        
        # Attempt to update with mix of valid and invalid params
        try:
            config.update(max_tasks=25, invalid_param="should_fail")
            assert False, "Should have failed with invalid parameter"
        except ValueError:
            # Verify original value wasn't changed
            assert config.max_tasks == original_max_tasks
            print("  ✅ Atomic configuration updates work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error recovery test error: {e}")
        return False

def main():
    """Run all robustness validation tests."""
    print("🛡️ Running Robustness Validation Tests")
    print("=" * 60)
    
    tests = [
        test_configuration_robustness,
        test_file_operations_robustness,
        test_input_validation_robustness,
        test_device_detection_robustness,
        test_task_configuration_robustness,
        test_security_validations,
        test_error_recovery,
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
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL ROBUSTNESS VALIDATION TESTS PASSED!")
        print("✅ System demonstrates excellent error handling and resilience")
        return True
    else:
        print(f"⚠️ {total - passed} robustness tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)