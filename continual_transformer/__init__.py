"""Continual Tiny Transformer — small transformer with EWC continual learning."""

from .model import TinyTransformer
from .ewc import EWC
from .tasks import SyntheticTask, generate_task

__all__ = ["TinyTransformer", "EWC", "SyntheticTask", "generate_task"]
