"""
Elastic Weight Consolidation (EWC)
===================================
Kirkpatrick et al., 2017 — "Overcoming catastrophic forgetting in neural networks"
https://arxiv.org/abs/1612.00796

Usage (task-incremental, multi-head)
-------------------------------------
After training task A with head "A":

    ewc = EWC(model, task_a_loader, device, task_name="A")

While training task B, add the penalty to your loss:

    loss = ce_loss + ewc_lambda * ewc.penalty(model)

The penalty is:

    λ/2 · Σ_i  F_i · (θ_i − θ*_i)²

where F_i is the diagonal Fisher information for parameter i and θ*_i are
the parameters after task A.

Usage (single-head / wrapped model)
--------------------------------------
If the model's forward() already encodes the task (e.g. a wrapper):

    ewc = EWC(wrapped_model, loader, device)   # no task_name needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional


class EWC:
    """
    EWC penalty computed over a single previous task.

    Parameters
    ----------
    model       : nn.Module  – the trained model (after task A).
                               ``forward(x)`` must return logits.
    dataloader  : DataLoader – task-A data for Fisher estimation.
    device      : str/device
    task_name   : str | None – if set, calls ``model(x, task_name=task_name)``
                               instead of ``model(x)``.
    n_samples   : int        – max samples to use for Fisher estimation.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        task_name: Optional[str] = None,
        n_samples: int = 200,
    ):
        self.device = device
        self.task_name = task_name

        # Snapshot θ*
        self.params: Dict[str, torch.Tensor] = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        # Estimate diagonal Fisher
        self.fisher: Dict[str, torch.Tensor] = self._compute_fisher(
            model, dataloader, n_samples
        )

    def _forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.task_name is not None:
            return model(x, task_name=self.task_name)
        return model(x)

    def _compute_fisher(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        n_samples: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate the diagonal of the Fisher information matrix using the
        empirical Fisher (squared gradients of the log-likelihood).
        """
        model.eval()
        fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        seen = 0

        for x, y in dataloader:
            if seen >= n_samples:
                break
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            model.zero_grad()
            logits = self._forward(model, x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2) * batch_size

            seen += batch_size

        # Normalise
        for n in fisher:
            fisher[n] /= max(seen, 1)

        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty:  Σ_i  F_i · (θ_i − θ*_i)²

        Pass the model whose current parameters should be compared to the
        post-Task-A anchor.
        """
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss
