"""Utility functions for continual learning."""

from .knowledge_distillation import (
    KnowledgeDistillation,
    ProgressiveKnowledgeDistillation,
    MemoryReplay
)

__all__ = [
    "KnowledgeDistillation",
    "ProgressiveKnowledgeDistillation", 
    "MemoryReplay"
]