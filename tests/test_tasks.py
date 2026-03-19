"""Tests for synthetic task generators."""

import pytest
import torch
from continual_transformer.tasks import generate_task, SyntheticTask


def test_generate_low_range():
    task = generate_task("low_range", n_train=200, n_val=50, seq_len=16, vocab_size=16, batch_size=32)
    assert isinstance(task, SyntheticTask)
    assert task.name == "low_range"


def test_generate_high_range():
    task = generate_task("high_range", n_train=200, n_val=50, seq_len=16, vocab_size=16, batch_size=32)
    assert isinstance(task, SyntheticTask)
    assert task.name == "high_range"


def test_generate_majority():
    task = generate_task("majority", n_train=200, n_val=50, seq_len=16, vocab_size=16, batch_size=32)
    assert isinstance(task, SyntheticTask)


def test_generate_freq():
    task = generate_task("freq", n_train=200, n_val=50, seq_len=16, vocab_size=16, batch_size=32)
    assert isinstance(task, SyntheticTask)


def test_generate_parity():
    task = generate_task("parity", n_train=200, n_val=50, seq_len=16, vocab_size=16, batch_size=32)
    assert isinstance(task, SyntheticTask)


def test_unknown_task():
    with pytest.raises(ValueError, match="Unknown task"):
        generate_task("nonexistent")


def test_loader_shapes():
    task = generate_task("low_range", n_train=64, n_val=32, seq_len=8, vocab_size=16, batch_size=16)
    x, y = next(iter(task.train_loader))
    assert x.shape == (16, 8), f"Expected (16, 8), got {x.shape}"
    assert y.shape == (16,)
    assert y.dtype == torch.long


def test_labels_are_binary():
    task = generate_task("low_range", n_train=200, n_val=50)
    for x, y in task.train_loader:
        assert y.min() >= 0
        assert y.max() <= 1


def test_domain_shift():
    """low_range and high_range use different token ID ranges."""
    ta = generate_task("low_range",  n_train=200, n_val=50, vocab_size=32, batch_size=200, seed=5)
    tb = generate_task("high_range", n_train=200, n_val=50, vocab_size=32, batch_size=200, seed=5)
    xa, _ = next(iter(ta.train_loader))
    xb, _ = next(iter(tb.train_loader))
    # Task A tokens should all be ≤ vocab//2
    assert xa.max().item() <= 16
    # Task B tokens should all be > vocab//2
    assert xb.min().item() > 16
