"""
TinyTransformer: a small transformer classifier built from scratch.

Multi-head variant: a shared encoder backbone with one linear head per task.
This is the standard setup for task-incremental continual learning with EWC.

At test time the task identity is provided (the model is told which head to use).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TinyTransformer(nn.Module):
    """
    Small transformer encoder for task-incremental continual learning.

    Architecture:
        Embedding → Positional Encoding → N × TransformerEncoderLayer
        → mean-pool → LayerNorm → task-specific Linear head

    Task heads are registered via ``add_head(task_name, num_classes)``.
    During forward, specify the active head with ``task_name``.

    Parameters
    ----------
    vocab_size   : int   – number of input token IDs
    d_model      : int   – embedding / hidden dimension   (default 64)
    nhead        : int   – number of attention heads      (default 4)
    num_layers   : int   – number of encoder layers       (default 2)
    d_ff         : int   – feed-forward hidden size        (default 128)
    num_classes  : int   – default head classes (added as "default" head)
    dropout      : float – dropout rate                    (default 0.1)
    max_len      : int   – max sequence length             (default 64)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model

        # ── Backbone ──────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN: more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # ── Task heads ────────────────────────────────────────────────────────
        self.heads: Dict[str, nn.Linear] = nn.ModuleDict()
        # Register a default head so single-task use still works
        self.add_head("default", num_classes)

        self._init_weights()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.01)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_head(self, task_name: str, num_classes: int) -> None:
        """Register a new task head. Idempotent if head already exists."""
        if task_name not in self.heads:
            head = nn.Linear(self.d_model, num_classes)
            nn.init.zeros_(head.bias)
            nn.init.xavier_uniform_(head.weight)
            self.heads[task_name] = head

    def backbone_parameters(self):
        """Iterator over backbone parameters (all except task heads)."""
        head_ids = {id(p) for h in self.heads.values() for p in h.parameters()}
        return [p for p in self.parameters() if id(p) not in head_ids]

    def forward(
        self,
        x: torch.Tensor,
        task_name: str = "default",
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                    : (batch, seq) int token IDs
        task_name            : which head to use
        src_key_padding_mask : (batch, seq) bool mask, True where padded

        Returns
        -------
        logits : (batch, num_classes_for_task)
        """
        if task_name not in self.heads:
            raise KeyError(
                f"Unknown task '{task_name}'. Call add_head() first. "
                f"Known: {list(self.heads)}"
            )

        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)                                               # (B, S, D)
        out = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)    # (B, S, D)

        # Mean pool (ignoring padding if mask provided)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = out.mean(1)

        pooled = self.norm(pooled)
        return self.heads[task_name](pooled)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def backbone_param_count(self) -> int:
        return sum(p.numel() for p in self.backbone_parameters())
