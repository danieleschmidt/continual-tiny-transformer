"""
Synthetic sequence classification tasks.

All tasks share the same output vocabulary (binary labels 0/1) and can be
used with a single shared head or per-task heads.

For the cleanest catastrophic-forgetting demonstration, Tasks A and B use
different token ID ranges ("domains") but the same label rule.  When trained
naively, the backbone is forced to overwrite the embedding representation it
learned for Task A's tokens in order to handle Task B's tokens.  EWC
prevents this by penalising changes to parameters that were important for A.

Built-in tasks
--------------
"low_range"   – tokens uniformly from [1 .. vocab//2],
                label = 1 if mean token value > vocab // 4
"high_range"  – tokens uniformly from [vocab//2+1 .. vocab-1],
                label = 1 if mean token value > vocab * 3 // 4

For a multi-head variant, or for per-experiment flexibility, "majority",
"freq", "maxdiff", and "parity" are also available.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor):
        assert sequences.shape[0] == labels.shape[0]
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Task
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SyntheticTask:
    name: str
    train_loader: DataLoader
    val_loader: DataLoader
    vocab_size: int
    num_classes: int = 2
    description: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Label functions  (all receive a single sequence tensor + vocab_size)
# ──────────────────────────────────────────────────────────────────────────────

def _majority_label(seq: torch.Tensor, vocab_size: int) -> int:
    """Label = 1 if majority of tokens are in lower half of vocab (IDs ≤ vocab//2)."""
    threshold = vocab_size // 2
    active = (seq > 0)
    if active.sum() == 0:
        return 0
    lower = ((seq <= threshold) & active).float().sum()
    return int(lower / active.float().sum() > 0.5)


def _upper_majority_label(seq: torch.Tensor, vocab_size: int) -> int:
    """Label = 1 if majority of tokens are in upper half of vocab."""
    threshold = vocab_size // 2
    active = (seq > 0)
    if active.sum() == 0:
        return 0
    upper = ((seq > threshold) & active).float().sum()
    return int(upper / active.float().sum() > 0.5)


def _high_token_label(seq: torch.Tensor, vocab_size: int) -> int:
    """Label = 1 if majority of tokens are in upper half of vocab (alias)."""
    return _upper_majority_label(seq, vocab_size)


def _freq_label(seq: torch.Tensor, vocab_size: int) -> int:
    """Label = 1 if the most-frequent token ID is odd."""
    counts = torch.bincount(seq, minlength=vocab_size + 1)
    counts[0] = 0
    most_common = int(counts.argmax().item())
    return int(most_common % 2 == 1)


def _maxdiff_label(seq: torch.Tensor, vocab_size: int) -> int:
    """Label = 1 if mean adjacent difference > vocab_size // 3."""
    diffs = (seq[1:].float() - seq[:-1].float()).abs()
    return int(diffs.mean().item() > vocab_size // 3)


def _parity_label(seq: torch.Tensor, _vocab_size: int) -> int:
    """Label = 1 if count of odd-valued (non-pad) tokens is even."""
    odd_count = int(((seq % 2 == 1) & (seq > 0)).sum().item())
    return int(odd_count % 2 == 0)


# ──────────────────────────────────────────────────────────────────────────────
# Specialised split generators for the domain-shift tasks
# ──────────────────────────────────────────────────────────────────────────────

def _gen_split_domain(
    n: int,
    seq_len: int,
    tok_lo: int,
    tok_hi: int,
    label_fn: Callable,
    vocab_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sequences with token IDs in [tok_lo, tok_hi)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    seqs = torch.randint(tok_lo, tok_hi, (n, seq_len), generator=rng)
    labels = torch.tensor(
        [label_fn(seqs[i], vocab_size) for i in range(n)], dtype=torch.long
    )
    return seqs, labels


# ──────────────────────────────────────────────────────────────────────────────
# Generic generator
# ──────────────────────────────────────────────────────────────────────────────

_LABEL_FNS: Dict[str, Callable] = {
    "majority":       _majority_label,
    "upper_majority": _upper_majority_label,
    "high_token":     _high_token_label,
    "freq":           _freq_label,
    "maxdiff":        _maxdiff_label,
    "parity":         _parity_label,
}


def _generate_split(
    n: int, seq_len: int, vocab_size: int, label_fn: Callable, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    seqs = torch.randint(1, vocab_size, (n, seq_len), generator=rng)
    labels = torch.tensor(
        [label_fn(seqs[i], vocab_size) for i in range(n)], dtype=torch.long
    )
    return seqs, labels


def generate_task(
    name: str,
    n_train: int = 2000,
    n_val: int = 500,
    seq_len: int = 32,
    vocab_size: int = 32,
    batch_size: int = 64,
    seed: int = 42,
) -> SyntheticTask:
    """
    Generate a synthetic continual-learning task by name.

    Domain-shift tasks (recommended for EWC demo):
        ``"low_range"``  – tokens from lower half of vocab, mean-threshold label
        ``"high_range"`` – tokens from upper half of vocab, mean-threshold label

    Miscellaneous tasks:
        ``"majority"``, ``"upper_majority"``, ``"high_token"``,
        ``"freq"``, ``"maxdiff"``, ``"parity"``
    """
    # ── Domain-shift tasks ────────────────────────────────────────────────────
    if name == "low_range":
        lo, hi = 1, vocab_size // 2 + 1
        threshold = (lo + hi) // 2
        label_fn = lambda s, v: int(s.float().mean().item() > threshold)
        train_seqs, train_labels = _gen_split_domain(n_train, seq_len, lo, hi, label_fn, vocab_size, seed)
        val_seqs,   val_labels   = _gen_split_domain(n_val,   seq_len, lo, hi, label_fn, vocab_size, seed + 1)

    elif name == "high_range":
        lo, hi = vocab_size // 2 + 1, vocab_size
        threshold = (lo + hi) // 2
        label_fn = lambda s, v: int(s.float().mean().item() > threshold)
        train_seqs, train_labels = _gen_split_domain(n_train, seq_len, lo, hi, label_fn, vocab_size, seed)
        val_seqs,   val_labels   = _gen_split_domain(n_val,   seq_len, lo, hi, label_fn, vocab_size, seed + 1)

    # ── General tasks ─────────────────────────────────────────────────────────
    elif name in _LABEL_FNS:
        fn = _LABEL_FNS[name]
        train_seqs, train_labels = _generate_split(n_train, seq_len, vocab_size, fn, seed)
        val_seqs,   val_labels   = _generate_split(n_val,   seq_len, vocab_size, fn, seed + 1)
    else:
        raise ValueError(
            f"Unknown task '{name}'. "
            f"Choose from {['low_range', 'high_range'] + list(_LABEL_FNS)}"
        )

    train_ds = SyntheticDataset(train_seqs, train_labels)
    val_ds   = SyntheticDataset(val_seqs,   val_labels)

    pos = train_labels.float().mean().item()
    import warnings
    if pos < 0.1 or pos > 0.9:
        warnings.warn(f"Task '{name}' is imbalanced: {pos:.1%} positive labels")

    return SyntheticTask(
        name=name,
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        vocab_size=vocab_size,
        description=(
            f"'{name}' task: {n_train} train / {n_val} val, seq_len={seq_len}"
        ),
    )
