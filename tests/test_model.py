"""Tests for TinyTransformer model."""

import pytest
import torch
from continual_transformer.model import TinyTransformer


VOCAB = 32
D = 64
BATCH = 4
SEQ = 16


@pytest.fixture
def model():
    return TinyTransformer(vocab_size=VOCAB, d_model=D, nhead=4, num_layers=2, d_ff=128)


def test_forward_default_head(model):
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    out = model(x)
    assert out.shape == (BATCH, 2), f"Expected ({BATCH}, 2), got {out.shape}"


def test_forward_named_head(model):
    model.add_head("task_a", 3)
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    out = model(x, task_name="task_a")
    assert out.shape == (BATCH, 3)


def test_forward_with_padding_mask(model):
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
    mask[:, -4:] = True
    out = model(x, src_key_padding_mask=mask)
    assert out.shape == (BATCH, 2)


def test_unknown_head_raises(model):
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    with pytest.raises(KeyError, match="Unknown task"):
        model(x, task_name="nonexistent")


def test_param_count(model):
    n = model.param_count()
    assert n > 0
    assert n < 500_000, f"Model is too large: {n:,} params"


def test_backbone_param_count(model):
    backbone = model.backbone_param_count()
    total = model.param_count()
    assert backbone < total  # heads add parameters


def test_4_layer_model():
    m = TinyTransformer(vocab_size=VOCAB, d_model=64, nhead=4, num_layers=4, d_ff=128)
    x = torch.randint(1, VOCAB, (2, 8))
    out = m(x)
    assert out.shape == (2, 2)


def test_output_changes_with_input(model):
    x1 = torch.randint(1, VOCAB, (1, SEQ), generator=torch.Generator().manual_seed(0))
    x2 = torch.randint(1, VOCAB, (1, SEQ), generator=torch.Generator().manual_seed(1))
    o1 = model(x1)
    o2 = model(x2)
    assert not torch.allclose(o1, o2)


def test_multiple_heads_independent(model):
    model.add_head("A", 2)
    model.add_head("B", 4)
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    out_a = model(x, task_name="A")
    out_b = model(x, task_name="B")
    assert out_a.shape == (BATCH, 2)
    assert out_b.shape == (BATCH, 4)


def test_add_head_idempotent(model):
    model.add_head("A", 2)
    model.add_head("A", 2)  # should not raise or add duplicate
    assert "A" in model.heads
