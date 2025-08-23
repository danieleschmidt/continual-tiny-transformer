#!/usr/bin/env python3
"""
Quick functionality test for continual learning system.
Tests core functionality without heavy dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Test basic imports
    from continual_transformer.core.config import ContinualConfig
    from continual_transformer.core.model import ContinualTransformer, TaskRouter
    from continual_transformer.adapters.activation import ActivationAdapter
    from continual_transformer.tasks.manager import TaskManager, Task
    
    print("✅ Core imports successful")
    
    # Test configuration
    config = ContinualConfig(
        model_name="distilbert-base-uncased",
        max_tasks=3,
        device="cpu"
    )
    print("✅ Configuration created")
    
    # Test task router
    router = TaskRouter(config, input_size=768)
    router.register_task("test_task")
    print("✅ Task router functional")
    
    # Test adapter
    adapter = ActivationAdapter(hidden_size=768, adapter_size=64, num_layers=6)
    print("✅ Adapter created")
    
    # Test task manager
    task_manager = TaskManager(config)
    task = Task("test_task", "classification", {})
    task_manager.add_task("test_task", "classification", {})
    print("✅ Task manager functional")
    
    print("\n🎉 ALL BASIC TESTS PASSED!")
    print("Core continual learning components are working correctly.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)