"""Data handling and processing for continual learning."""

from .loaders import (
    ContinualDataLoader,
    TaskDataset,
    MemoryReplayDataLoader
)
from .processors import (
    TextProcessor,
    TaskBatchProcessor,
    ContinualBatchSampler
)
from .storage import (
    TaskDataStorage,
    ModelCheckpointManager,
    MetricsStorage
)

__all__ = [
    # Data loading
    "ContinualDataLoader",
    "TaskDataset", 
    "MemoryReplayDataLoader",
    
    # Processing
    "TextProcessor",
    "TaskBatchProcessor",
    "ContinualBatchSampler",
    
    # Storage
    "TaskDataStorage",
    "ModelCheckpointManager",
    "MetricsStorage"
]