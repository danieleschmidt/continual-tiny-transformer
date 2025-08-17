#!/usr/bin/env python3
"""
Basic functionality test for continual transformer framework.
Tests core imports and basic model instantiation.
"""

import sys
import os
sys.path.insert(0, 'src')

try:
    from continual_transformer import ContinualTransformer
    from continual_transformer.config import ContinualConfig
    print("✅ Core imports successful")
    
    # Test basic configuration
    config = ContinualConfig(
        model_name="distilbert-base-uncased",
        max_tasks=5,
        device="cpu",
        freeze_base_model=True
    )
    print("✅ Configuration created successfully")
    
    # Test model instantiation
    model = ContinualTransformer(config)
    print("✅ Model instantiated successfully")
    
    # Test task registration
    model.register_task("test_task", num_labels=2, task_type="classification")
    print("✅ Task registration successful")
    
    print("\n🎯 Generation 1 (MAKE IT WORK) - COMPLETED SUCCESSFULLY")
    print("   - Core imports working")
    print("   - Configuration system functional") 
    print("   - Model instantiation working")
    print("   - Task management working")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)