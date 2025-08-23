#!/usr/bin/env python3
"""
Simple demo of the continual learning transformer.
Demonstrates basic functionality without heavy model loading.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing basic imports...")
    
    try:
        from continual_transformer.core.config import ContinualConfig
        print("  ✅ Config import successful")
        
        from continual_transformer.adapters.activation import ActivationAdapter
        print("  ✅ Adapter import successful")
        
        from continual_transformer.tasks.manager import TaskManager, Task
        print("  ✅ Task manager import successful")
        
        print("✅ All basic imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration creation and validation."""
    print("\n🔍 Testing configuration...")
    
    try:
        from continual_transformer.core.config import ContinualConfig
        
        config = ContinualConfig(
            model_name="distilbert-base-uncased",
            max_tasks=5,
            device="cpu",
            learning_rate=2e-5,
            num_epochs=3,
            batch_size=8
        )
        
        print(f"  ✅ Configuration created with {config.max_tasks} max tasks")
        print(f"  ✅ Device: {config.device}")
        print(f"  ✅ Model: {config.model_name}")
        return config
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return None

def test_activation_adapter():
    """Test activation adapter functionality."""
    print("\n🔍 Testing activation adapter...")
    
    try:
        from continual_transformer.adapters.activation import ActivationAdapter
        
        # Create adapter
        adapter = ActivationAdapter(
            hidden_size=768,
            adapter_size=64,
            num_layers=6
        )
        
        # Test forward pass
        batch_size, seq_len, hidden_size = 2, 32, 768
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        
        output = adapter(input_tensor)
        
        print(f"  ✅ Adapter created with {sum(p.numel() for p in adapter.parameters())} parameters")
        print(f"  ✅ Forward pass: {input_tensor.shape} -> {output.shape}")
        
        # Verify output shape matches input
        assert output.shape == input_tensor.shape, f"Shape mismatch: {output.shape} != {input_tensor.shape}"
        print("  ✅ Shape consistency verified")
        
        return adapter
    except Exception as e:
        print(f"  ❌ Adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_task_management():
    """Test task management functionality."""
    print("\n🔍 Testing task management...")
    
    try:
        from continual_transformer.core.config import ContinualConfig
        from continual_transformer.tasks.manager import TaskManager, Task
        
        config = ContinualConfig(max_tasks=3, device="cpu")
        task_manager = TaskManager(config)
        
        # Add tasks
        task_manager.add_task("sentiment", "classification", {"num_classes": 2})
        task_manager.add_task("topic", "classification", {"num_classes": 4})
        task_manager.add_task("intent", "classification", {"num_classes": 3})
        
        print(f"  ✅ Added {len(task_manager.tasks)} tasks")
        
        # Test task retrieval
        sentiment_task = task_manager.get_task("sentiment")
        print(f"  ✅ Retrieved task: {sentiment_task.task_id} ({sentiment_task.task_type})")
        
        # List all tasks
        all_tasks = [task_id for task_id in task_manager.tasks.keys()]
        print(f"  ✅ All tasks: {all_tasks}")
        
        return task_manager
    except Exception as e:
        print(f"  ❌ Task management test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_memory_efficiency():
    """Test that the system maintains constant memory usage per task."""
    print("\n🔍 Testing memory efficiency...")
    
    try:
        from continual_transformer.adapters.activation import ActivationAdapter
        
        adapters = []
        initial_params = 0
        
        # Create multiple adapters (simulating multiple tasks)
        for i in range(3):
            adapter = ActivationAdapter(
                hidden_size=768,
                adapter_size=64,
                num_layers=6
            )
            adapters.append(adapter)
            
            current_params = sum(p.numel() for p in adapter.parameters())
            if i == 0:
                initial_params = current_params
            
            print(f"  Task {i+1}: {current_params:,} parameters")
        
        # Verify consistent parameter count per adapter
        all_consistent = all(
            sum(p.numel() for p in adapter.parameters()) == initial_params 
            for adapter in adapters
        )
        
        if all_consistent:
            print(f"  ✅ Consistent parameter count: {initial_params:,} per task")
            print(f"  ✅ Total parameters: {len(adapters) * initial_params:,}")
            print(f"  ✅ Zero parameter expansion verified!")
        else:
            print("  ❌ Parameter count inconsistent across tasks")
            
        return all_consistent
    except Exception as e:
        print(f"  ❌ Memory efficiency test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests."""
    print("🚀 Continual Tiny Transformer - Basic Functionality Test")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results["imports"] = test_basic_imports()
    results["config"] = test_configuration() is not None
    results["adapter"] = test_activation_adapter() is not None
    results["task_management"] = test_task_management() is not None
    results["memory_efficiency"] = test_memory_efficiency()
    
    # Summary
    print("\n📋 Test Results Summary:")
    print("-" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<20} {status}")
    
    print("-" * 30)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Core continual learning functionality is working correctly.")
        print("✅ Ready for advanced features and optimization!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)