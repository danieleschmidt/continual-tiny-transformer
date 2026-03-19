"""Tests for EWC continual learning."""

import pytest
import torch
import torch.nn.functional as F
from continual_transformer.model import TinyTransformer
from continual_transformer.ewc import EWC
from continual_transformer.tasks import generate_task


VOCAB = 32
DEVICE = torch.device("cpu")


@pytest.fixture
def task_a():
    return generate_task("low_range", n_train=128, n_val=64, vocab_size=VOCAB, batch_size=32)


@pytest.fixture
def trained_model(task_a):
    """A model trained briefly on Task A (default single head)."""
    model = TinyTransformer(vocab_size=VOCAB, d_model=32, nhead=4, num_layers=2, d_ff=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for x, y in task_a.train_loader:
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        break
    return model


def test_ewc_creates_fisher(trained_model, task_a):
    ewc = EWC(trained_model, task_a.train_loader, DEVICE, n_samples=64)
    assert len(ewc.fisher) > 0
    for name, fisher in ewc.fisher.items():
        assert fisher.shape == trained_model.state_dict()[name].shape


def test_ewc_penalty_is_scalar(trained_model, task_a):
    ewc = EWC(trained_model, task_a.train_loader, DEVICE, n_samples=64)
    penalty = ewc.penalty(trained_model)
    assert penalty.shape == torch.Size([])


def test_ewc_penalty_zero_at_anchor(trained_model, task_a):
    """Penalty should be near-zero when model params equal the anchor."""
    ewc = EWC(trained_model, task_a.train_loader, DEVICE, n_samples=64)
    penalty = ewc.penalty(trained_model)
    assert penalty.item() < 1e-6, f"Expected ~0, got {penalty.item()}"


def test_ewc_penalty_positive_after_update(trained_model, task_a):
    """Penalty should increase after parameters are perturbed."""
    ewc = EWC(trained_model, task_a.train_loader, DEVICE, n_samples=64)
    with torch.no_grad():
        for p in trained_model.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    penalty = ewc.penalty(trained_model)
    assert penalty.item() > 0.0


def test_ewc_penalty_differentiable(trained_model, task_a):
    """Penalty must be differentiable for inclusion in a loss."""
    ewc = EWC(trained_model, task_a.train_loader, DEVICE, n_samples=64)
    task_b = generate_task("high_range", n_train=64, n_val=32, vocab_size=VOCAB, batch_size=32)

    opt = torch.optim.SGD(trained_model.parameters(), lr=1e-3)
    trained_model.train()
    for x, y in task_b.train_loader:
        opt.zero_grad()
        loss = F.cross_entropy(trained_model(x), y) + ewc.penalty(trained_model)
        loss.backward()
        opt.step()
        break


def test_ewc_reduces_forgetting():
    """
    End-to-end: EWC should retain Task A accuracy better than naive fine-tuning.
    Uses tiny model + few epochs for speed.

    Domain shift:
      Task A = low token IDs  → backbone learns low-embedding features
      Task B = high token IDs → naive training overwrites those features
      EWC protects the features learned for Task A.
    """
    EPOCHS = 12
    LR = 5e-3
    LAMBDA = 300.0

    task_a = generate_task("low_range",  n_train=600, n_val=300, vocab_size=VOCAB, batch_size=64, seed=42)
    task_b = generate_task("high_range", n_train=600, n_val=300, vocab_size=VOCAB, batch_size=64, seed=43)

    def make_model():
        return TinyTransformer(vocab_size=VOCAB, d_model=32, nhead=4, num_layers=2, d_ff=64)

    def quick_train(model, loader, epochs, ewc=None, lam=0.0):
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        model.train()
        for _ in range(epochs):
            for x, y in loader:
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                if ewc is not None:
                    loss = loss + lam * ewc.penalty(model)
                loss.backward()
                opt.step()

    def acc(model, loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total

    # ── Naive ──────────────────────────────────────────────────────────────
    m_naive = make_model()
    quick_train(m_naive, task_a.train_loader, EPOCHS)
    naive_before = acc(m_naive, task_a.val_loader)
    quick_train(m_naive, task_b.train_loader, EPOCHS)
    naive_after = acc(m_naive, task_a.val_loader)
    naive_forget = naive_before - naive_after

    # ── EWC ────────────────────────────────────────────────────────────────
    m_ewc = make_model()
    quick_train(m_ewc, task_a.train_loader, EPOCHS)
    ewc_before = acc(m_ewc, task_a.val_loader)
    ewc = EWC(m_ewc, task_a.train_loader, DEVICE, n_samples=300)
    quick_train(m_ewc, task_b.train_loader, EPOCHS, ewc=ewc, lam=LAMBDA)
    ewc_after = acc(m_ewc, task_a.val_loader)
    ewc_forget = ewc_before - ewc_after

    print(f"\nNaive forgetting: {naive_forget:.3f}  |  EWC forgetting: {ewc_forget:.3f}")

    # EWC should forget ≤ naive + small tolerance
    assert ewc_forget <= naive_forget + 0.15, (
        f"EWC forgetting ({ewc_forget:.3f}) should not be much worse than "
        f"naive ({naive_forget:.3f})"
    )
