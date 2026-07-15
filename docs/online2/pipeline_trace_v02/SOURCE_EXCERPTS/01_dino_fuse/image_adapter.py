"""Shared frozen-DINO → IMAGE_SLOT adapter (finetune v0.2).

Stage A injects per-slot residuals; IMAGE event is later mean+max pooled to one
event vector. This module is fold-trainable; DINO itself stays frozen.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.online2.v2.finetune_v02.config import ImageAdapterConfig


class SharedImageAdapter(nn.Module):
    """u = A(v / ||v||); residual = g * u  (added to IMAGE_SLOT embedding)."""

    def __init__(self, cfg: Optional[ImageAdapterConfig] = None):
        super().__init__()
        self.cfg = cfg or ImageAdapterConfig()
        self.proj = nn.Sequential(
            nn.Linear(self.cfg.dino_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Linear(self.cfg.model_dim, self.cfg.model_dim),
        )
        self.gate = nn.Parameter(torch.tensor(float(self.cfg.gate_init)))

    def forward(self, dino_vecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dino_vecs: [..., dino_dim] frozen external embeddings
        Returns:
            residual: [..., model_dim] gated adapter output
        """
        x = dino_vecs
        if self.cfg.normalize_l2:
            x = F.normalize(x, p=2, dim=-1)
        return self.gate * self.proj(x)


def inject_slot_residuals(
    slot_embeddings: torch.Tensor,
    residuals: torch.Tensor,
) -> torch.Tensor:
    """x'_slot = x_slot + residual (broadcast-safe)."""
    return slot_embeddings + residuals
