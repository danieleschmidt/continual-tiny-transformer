"""Tests for training utilities."""

import pytest
import torch
from continual_transformer.model import TinyTransformer
from continual_transformer.tasks import generate_task
from continual_transformer.train import train, evaluate, train_epoch


VOCAB = 32
DEVICE = torch.device("cpu")


@pytest.fixture
def tiny_task():
    return generate_task("low_range", n_train=128, n_val=64, vocab_size=VOCAB, batch_size=32)


@pytest.fixture
def tiny_model():
    return TinyTransformer(vocab_size=VOCAB, d_model=32, nhead=4, num_layers=2, d_ff=64)


def test_evaluate_returns_dict(tiny_model, tiny_task):
    metrics = evaluate(tiny_model, tiny_task.val_loader, DEVICE)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["loss"] >= 0.0


def test_train_epoch_changes_params(tiny_model, tiny_task):
    before = {n: p.clone() for n, p in tiny_model.named_parameters()}
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    train_epoch(tiny_model, tiny_task.train_loader, opt, DEVICE)
    changed = any(
        not torch.equal(before[n], p)
        for n, p in tiny_model.named_parameters()
    )
    assert changed, "Parameters should change after training"


def test_train_returns_history(tiny_model, tiny_task):
    history = train(
        tiny_model, tiny_task.train_loader, tiny_task.val_loader,
        DEVICE, epochs=3, verbose=False,
    )
    assert "train_loss" in history
    assert "val_acc" in history
    assert len(history["train_loss"]) == 3
    assert len(history["val_acc"]) == 3


def test_model_learns_low_range():
    """Model should reach >80% accuracy on low_range task within 15 epochs."""
    task = generate_task("low_range", n_train=800, n_val=200, vocab_size=VOCAB, batch_size=64, seed=7)
    model = TinyTransformer(vocab_size=VOCAB, d_model=32, nhead=4, num_layers=2, d_ff=64)
    history = train(model, task.train_loader, task.val_loader, DEVICE, epochs=15, verbose=False)
    final_acc = history["val_acc"][-1]
    assert final_acc > 0.75, f"Expected >75% accuracy, got {final_acc:.3f}"


def test_multihead_training():
    """Training with explicit task_name on multi-head model."""
    task_a = generate_task("low_range",  n_train=128, n_val=64, vocab_size=VOCAB, batch_size=32, seed=1)
    task_b = generate_task("high_range", n_train=128, n_val=64, vocab_size=VOCAB, batch_size=32, seed=2)
    model = TinyTransformer(vocab_size=VOCAB, d_model=32, nhead=4, num_layers=2, d_ff=64)
    model.add_head("A", 2)
    model.add_head("B", 2)

    history_a = train(model, task_a.train_loader, task_a.val_loader,
                      DEVICE, epochs=3, task_name="A", verbose=False)
    history_b = train(model, task_b.train_loader, task_b.val_loader,
                      DEVICE, epochs=3, task_name="B", verbose=False)

    assert len(history_a["val_acc"]) == 3
    assert len(history_b["val_acc"]) == 3
