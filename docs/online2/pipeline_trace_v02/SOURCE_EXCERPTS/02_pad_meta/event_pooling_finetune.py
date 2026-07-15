"""Event-pooling diagnosis finetune: PAD + classification head.

Stage A produces offline event summaries (mean/max). This module implements:
  Stage B — Pooled Attention Event-Sequence Decoder
  Stage C — Diagnosis Classification Head (10-way)

See: outputs/online2/v2_finetune/EVENT_POOLING_FINETUNE_ARCH.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


VIEW_TO_ID = {
    "ENVIRONMENT": 0,
    "ROOTZONE": 1,
    "ACTUATOR": 2,
    "GROWTH": 3,
    "IMAGE": 4,
    "UNKNOWN": 5,
}
NUM_VIEWS = len(VIEW_TO_ID)
NUM_ZONES = 8  # zone ids 1–4 (+ slack)
NUM_HOURS = 24


@dataclass
class EventPoolingConfig:
    input_dim: int = 384
    hidden_dim: int = 192
    num_heads: int = 4
    ff_dim: int = 384
    dropout: float = 0.2
    num_classes: int = 10
    max_case_age_hours: float = 336.0
    use_mean_max: bool = True  # concat mean||max → 2*input_dim before P_e


class EventMetadataEncoder(nn.Module):
    """Map Stage A mean/max (+ metadata) → z_i in hidden_dim."""

    def __init__(self, cfg: EventPoolingConfig):
        super().__init__()
        self.cfg = cfg
        feat_dim = cfg.input_dim * 2 if cfg.use_mean_max else cfg.input_dim
        self.event_proj = nn.Linear(feat_dim, cfg.hidden_dim)
        self.case_age_proj = nn.Linear(1, cfg.hidden_dim)
        self.view_emb = nn.Embedding(NUM_VIEWS, cfg.hidden_dim)
        self.zone_emb = nn.Embedding(NUM_ZONES, cfg.hidden_dim)
        self.hour_emb = nn.Embedding(NUM_HOURS, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        event_mean: torch.Tensor,
        event_max: torch.Tensor,
        case_age_hours: torch.Tensor,
        view_id: torch.Tensor,
        zone_id: torch.Tensor,
        local_hour: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            event_mean/max: [B, T, H]
            case_age_hours: [B, T]
            view_id/zone_id/local_hour: [B, T] long
            padding_mask: [B, T] True = valid
        Returns:
            z: [B, T, hidden_dim]
        """
        if self.cfg.use_mean_max:
            feats = torch.cat([event_mean, event_max], dim=-1)
        else:
            feats = event_mean
        age = (case_age_hours / self.cfg.max_case_age_hours).clamp(0.0, 2.0).unsqueeze(-1)
        z = (
            self.event_proj(feats)
            + self.case_age_proj(age)
            + self.view_emb(view_id.clamp(0, NUM_VIEWS - 1))
            + self.zone_emb(zone_id.clamp(0, NUM_ZONES - 1))
            + self.hour_emb(local_hour.clamp(0, NUM_HOURS - 1))
        )
        z = self.dropout(self.norm(z))
        z = z * padding_mask.unsqueeze(-1).to(z.dtype)
        return z


class PooledAttentionEventDecoder(nn.Module):
    """Learned query cross-attends over event sequence → h_case."""

    def __init__(self, cfg: EventPoolingConfig):
        super().__init__()
        self.cfg = cfg
        self.query = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(
        self, z: torch.Tensor, padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, T, D]
            padding_mask: [B, T] True = valid
        Returns:
            h_case: [B, D]
            attn: [B, 1, T]
        """
        bsz = z.size(0)
        q = self.query.expand(bsz, -1, -1)
        # nn.MultiheadAttention: True means ignore (pad)
        key_padding_mask = ~padding_mask.bool()
        pooled, attn = self.mha(
            q, z, z, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True
        )
        # attn: [B, 1, T] when average_attn_weights=True with one query
        self.last_attn = attn
        h = self.norm1(pooled + q)
        h = self.norm2(h + self.ff(h))
        return h.squeeze(1), attn


class DiagnosisHead(nn.Module):
    def __init__(self, cfg: EventPoolingConfig):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.linear = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def forward(self, h_case: torch.Tensor) -> torch.Tensor:
        return self.linear(self.drop(self.ln(h_case)))


class EventPoolingDiagnosisModel(nn.Module):
    """Full Stage B+C classifier over cached Stage A event summaries."""

    def __init__(self, cfg: Optional[EventPoolingConfig] = None):
        super().__init__()
        self.cfg = cfg or EventPoolingConfig()
        self.meta = EventMetadataEncoder(self.cfg)
        self.pad = PooledAttentionEventDecoder(self.cfg)
        self.head = DiagnosisHead(self.cfg)

    def forward(
        self,
        event_mean: torch.Tensor,
        event_max: torch.Tensor,
        case_age_hours: torch.Tensor,
        view_id: torch.Tensor,
        zone_id: torch.Tensor,
        local_hour: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z = self.meta(
            event_mean,
            event_max,
            case_age_hours,
            view_id,
            zone_id,
            local_hour,
            padding_mask,
        )
        h_case, attn = self.pad(z, padding_mask)
        logits = self.head(h_case)
        return {"logits": logits, "h_case": h_case, "attn": attn, "z": z}

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels.long())
