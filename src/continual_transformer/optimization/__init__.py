"""Advanced optimization modules for continual learning."""

from .performance_optimizer import (
    PerformanceOptimizer,
    MemoryOptimizer,
    ComputeOptimizer,
    AdaptiveOptimizer
)
from .knowledge_transfer import (
    KnowledgeTransferOptimizer,
    CrossTaskTransfer,
    MetaLearningOptimizer
)
from .neural_architecture_search import (
    NASOptimizer,
    AdapterArchitectureSearch,
    TaskSpecificNAS
)

__all__ = [
    "PerformanceOptimizer",
    "MemoryOptimizer", 
    "ComputeOptimizer",
    "AdaptiveOptimizer",
    "KnowledgeTransferOptimizer",
    "CrossTaskTransfer",
    "MetaLearningOptimizer",
    "NASOptimizer",
    "AdapterArchitectureSearch",
    "TaskSpecificNAS",
]