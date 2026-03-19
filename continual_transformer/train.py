"""
Training utilities.

train_epoch  – one epoch of SGD
evaluate     – accuracy + loss on a DataLoader
train        – full training loop, returns history dict

All functions accept an optional ``task_name`` parameter for multi-head models.
If omitted they default to ``"default"`` (single-head usage).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .ewc import EWC


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_name: str = "default",
    ewc: Optional[EWC] = None,
    ewc_lambda: float = 400.0,
) -> float:
    """Train for one epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if hasattr(model, "heads"):
            # Multi-head model
            logits = model(x, task_name=task_name)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits, y)
        if ewc is not None:
            loss = loss + ewc_lambda * ewc.penalty(model)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_name: str = "default",
) -> Dict[str, float]:
    """Evaluate model. Returns dict with 'loss' and 'accuracy'."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if hasattr(model, "heads"):
            logits = model(x, task_name=task_name)
        else:
            logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    task_name: str = "default",
    ewc: Optional[EWC] = None,
    ewc_lambda: float = 400.0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Full training loop.

    Returns history dict with keys:
        'train_loss', 'val_loss', 'val_acc'
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device, task_name, ewc, ewc_lambda)
        val_metrics = evaluate(model, val_loader, device, task_name)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={tr_loss:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['accuracy']:.3f}"
            )

    return history
