"""Adapter modules for task-specific adaptations."""

from .activation import (
    ActivationAdapter,
    MultiLayerActivationAdapter,
    AttentionAdapter,
    HyperAdapter,
    create_adapter
)

__all__ = [
    "ActivationAdapter",
    "MultiLayerActivationAdapter", 
    "AttentionAdapter",
    "HyperAdapter",
    "create_adapter"
]