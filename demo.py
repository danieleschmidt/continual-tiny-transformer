#!/usr/bin/env python3
"""
Catastrophic Forgetting Demo
==============================
Trains a TinyTransformer on two sequential tasks and compares:
  - Naive fine-tuning  → catastrophic forgetting on Task A
  - EWC               → retains Task A while learning Task B

Setup: domain-incremental continual learning with a shared single head.

  Task A ("low_range"):  sequences from the lower half of the vocabulary,
                          label = 1 if mean token value > mid-low threshold.
  Task B ("high_range"): sequences from the upper half of the vocabulary,
                          same structural rule but completely different tokens.

The backbone must re-learn token embeddings to handle Task B, directly
overwriting what it learned for Task A.  EWC penalises changes to the
important parameters, preserving Task A performance.

Usage:
    python demo.py [--epochs 30] [--ewc-lambda 400] [--device cpu]
"""

import argparse

import torch
import torch.nn.functional as F

from continual_transformer import TinyTransformer, EWC, generate_task


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_task(model, train_loader, device, epochs, lr=3e-4,
               ewc=None, ewc_lambda=0.0, verbose=True):
    """Train with optional EWC penalty. Single-head model (task_name omitted)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if ewc is not None:
                loss = loss + ewc_lambda * ewc.penalty(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.size(0)
        sched.step()
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | train_loss={total_loss / len(train_loader.dataset):.4f}")


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(epochs: int, ewc_lambda: float, device_str: str):
    device = torch.device(device_str)
    print(f"\nDevice: {device} | epochs={epochs} | ewc_lambda={ewc_lambda}\n")

    VOCAB = 64  # larger vocab → clearer domain shift
    task_a = generate_task("low_range",  vocab_size=VOCAB, seed=1)
    task_b = generate_task("high_range", vocab_size=VOCAB, seed=2)
    print(f"Task A: {task_a.description}")
    print(f"Task B: {task_b.description}")
    print(f"  Token range A: [1, {VOCAB//2}]")
    print(f"  Token range B: [{VOCAB//2+1}, {VOCAB-1}]")

    def make_model():
        # Single shared head — forgetting is most visible here
        return TinyTransformer(
            vocab_size=VOCAB, d_model=64, nhead=4, num_layers=2, d_ff=128,
            num_classes=2,
        ).to(device)

    print(f"\nModel parameters: {make_model().param_count():,}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 1 — Naive sequential fine-tuning
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXPERIMENT 1 — Naive fine-tuning (expects forgetting)")
    print("=" * 60)

    m_naive = make_model()
    print("\n[Naive] Training on Task A …")
    train_task(m_naive, task_a.train_loader, device, epochs)
    acc_a0 = accuracy(m_naive, task_a.val_loader, device)
    print(f"\n[Naive] After Task A — A_acc={acc_a0:.3f}")

    print("\n[Naive] Fine-tuning on Task B (no EWC) …")
    train_task(m_naive, task_b.train_loader, device, epochs)
    naive_a = accuracy(m_naive, task_a.val_loader, device)
    naive_b = accuracy(m_naive, task_b.val_loader, device)
    naive_forget = acc_a0 - naive_a
    print(f"\n[Naive] After Task B — A_acc={naive_a:.3f}  B_acc={naive_b:.3f}")
    print(f"[Naive] Forgetting on A: {naive_forget:+.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 2 — EWC
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — EWC (expects low forgetting)")
    print("=" * 60)

    m_ewc = make_model()
    print("\n[EWC] Training on Task A …")
    train_task(m_ewc, task_a.train_loader, device, epochs)
    acc_a0_ewc = accuracy(m_ewc, task_a.val_loader, device)
    print(f"\n[EWC] After Task A — A_acc={acc_a0_ewc:.3f}")

    print("\n[EWC] Computing Fisher information from Task A …")
    ewc = EWC(m_ewc, task_a.train_loader, device, n_samples=500)

    print(f"\n[EWC] Fine-tuning on Task B with EWC (λ={ewc_lambda}) …")
    train_task(m_ewc, task_b.train_loader, device, epochs,
               ewc=ewc, ewc_lambda=ewc_lambda)
    ewc_a = accuracy(m_ewc, task_a.val_loader, device)
    ewc_b = accuracy(m_ewc, task_b.val_loader, device)
    ewc_forget = acc_a0_ewc - ewc_a
    print(f"\n[EWC] After Task B — A_acc={ewc_a:.3f}  B_acc={ewc_b:.3f}")
    print(f"[EWC] Forgetting on A: {ewc_forget:+.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<12} {'A acc after A':>13} {'A acc after B':>13} {'Forgetting':>10} {'B acc':>7}")
    print("-" * 60)
    print(f"{'Naive':<12} {acc_a0:>13.3f} {naive_a:>13.3f} {naive_forget:>+10.3f} {naive_b:>7.3f}")
    print(f"{'EWC':<12} {acc_a0_ewc:>13.3f} {ewc_a:>13.3f} {ewc_forget:>+10.3f} {ewc_b:>7.3f}")
    print("=" * 60)

    improvement = naive_forget - ewc_forget
    print(f"\nEWC reduced forgetting by {improvement:.3f} accuracy points.")
    if improvement > 0.05:
        print("✓ EWC successfully mitigated catastrophic forgetting.")
    else:
        print("△ Difference small — try increasing --epochs or --ewc-lambda.")

    return {
        "naive_forgetting": naive_forget,
        "ewc_forgetting": ewc_forget,
        "naive_b_acc": naive_b,
        "ewc_b_acc": ewc_b,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--ewc-lambda", type=float, default=400.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run(epochs=args.epochs, ewc_lambda=args.ewc_lambda, device_str=args.device)
