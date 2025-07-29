"""
Continual Tiny Transformer: Zero-Parameter Continual Learning

A memory-efficient continual learning framework for transformers that adds
ZERO new parameters per task, based on Amazon Research's breakthrough methodology.

This package provides:
- ContinualTransformer: Core model for zero-parameter continual learning
- Task management utilities for multi-task scenarios
- Knowledge distillation components to prevent catastrophic forgetting
- Evaluation metrics and benchmarking tools
"""

from continual_transformer.core import ContinualTransformer
from continual_transformer.config import ContinualConfig
from continual_transformer.tasks import TaskManager, Task
from continual_transformer.adapters import ActivationAdapter
from continual_transformer.metrics import ContinualMetrics

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

__all__ = [
    "ContinualTransformer",
    "ContinualConfig", 
    "TaskManager",
    "Task",
    "ActivationAdapter",
    "ContinualMetrics",
]

# Package metadata
__title__ = "continual-tiny-transformer"
__description__ = "Memory-efficient continual learning for transformers with zero parameter expansion"
__url__ = "https://github.com/your-org/continual-tiny-transformer"
__license__ = "MIT"
__copyright__ = "Copyright 2025 Daniel Schmidt"