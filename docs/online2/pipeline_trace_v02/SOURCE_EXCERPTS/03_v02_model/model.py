"""Stage B+C model for finetune v0.2 — PAD + semantic heads + IMAGE DINO fusion."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.online2.v2.event_pooling_finetune import (
    DiagnosisHead,
    EventMetadataEncoder,
    PooledAttentionEventDecoder,
)
from src.online2.v2.finetune_v02.config import FinetuneV02Config, ImageAdapterConfig
from src.online2.v2.finetune_v02.image_adapter import SharedImageAdapter


class SemanticProjectionHead(nn.Module):
    """h_case → L2-normalized semantic vector (state or cause)."""

    def __init__(self, hidden_dim: int, semantic_dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, semantic_dim)

    def forward(self, h_case: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.drop(self.ln(h_case)))
        return F.normalize(z, p=2, dim=-1)


class EventPoolingDiagnosisModelV02(nn.Module):
    """PAD + diagnosis + state/cause heads + optional IMAGE-slot DINO residual.

    Image path (fold-trainable):
      for IMAGE events with dino_vec:
        event_mean ← event_mean + SharedImageAdapter(dino)
      (mirrors IMAGE_SLOT residual after Stage A pooling)

    Semantic heads stay separate from classification logits (no fusion).
    """

    def __init__(self, cfg: Optional[FinetuneV02Config] = None, *, use_image_adapter: bool = True):
        super().__init__()
        self.cfg = cfg or FinetuneV02Config()
        pad_cfg = self.cfg.pad_config()
        self.meta = EventMetadataEncoder(pad_cfg)
        self.pad = PooledAttentionEventDecoder(pad_cfg)
        self.head = DiagnosisHead(pad_cfg)
        self.state_head = SemanticProjectionHead(
            self.cfg.hidden_dim, self.cfg.semantic_dim, self.cfg.dropout
        )
        self.cause_head = SemanticProjectionHead(
            self.cfg.hidden_dim, self.cfg.semantic_dim, self.cfg.dropout
        )
        self.use_image_adapter = bool(use_image_adapter)
        if self.use_image_adapter:
            img_cfg = ImageAdapterConfig(
                dino_dim=self.cfg.image.dino_dim,
                model_dim=self.cfg.input_dim,  # residual on event_mean (Stage A H)
                gate_init=self.cfg.image.gate_init,
                normalize_l2=self.cfg.image.normalize_l2,
            )
            self.image_adapter = SharedImageAdapter(img_cfg)
        else:
            self.image_adapter = None

    def fuse_image_into_events(
        self,
        event_mean: torch.Tensor,
        event_max: torch.Tensor,
        *,
        dino_vec: Optional[torch.Tensor],
        dino_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add gated DINO residual onto IMAGE event summaries (token-slot surrogate)."""
        if (
            self.image_adapter is None
            or dino_vec is None
            or dino_mask is None
            or not bool(dino_mask.any())
        ):
            return event_mean, event_max
        residual = self.image_adapter(dino_vec)  # [B,T,H]
        m = dino_mask.unsqueeze(-1).to(dtype=event_mean.dtype)
        event_mean = event_mean + residual * m
        event_max = event_max + residual * m
        return event_mean, event_max

    def encode_events(
        self,
        event_mean: torch.Tensor,
        event_max: torch.Tensor,
        case_age_hours: torch.Tensor,
        view_id: torch.Tensor,
        zone_id: torch.Tensor,
        local_hour: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        dino_vec: Optional[torch.Tensor] = None,
        dino_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        event_mean, event_max = self.fuse_image_into_events(
            event_mean, event_max, dino_vec=dino_vec, dino_mask=dino_mask
        )
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
        return {"z": z, "h_case": h_case, "attn": attn, "event_mean": event_mean}

    def forward(
        self,
        event_mean: torch.Tensor,
        event_max: torch.Tensor,
        case_age_hours: torch.Tensor,
        view_id: torch.Tensor,
        zone_id: torch.Tensor,
        local_hour: torch.Tensor,
        padding_mask: torch.Tensor,
        dino_vec: Optional[torch.Tensor] = None,
        dino_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode_events(
            event_mean,
            event_max,
            case_age_hours,
            view_id,
            zone_id,
            local_hour,
            padding_mask,
            dino_vec=dino_vec,
            dino_mask=dino_mask,
        )
        logits = self.head(enc["h_case"])
        z_state = self.state_head(enc["h_case"])
        z_cause = self.cause_head(enc["h_case"])
        return {
            "logits": logits,
            "h_case": enc["h_case"],
            "attn": enc["attn"],
            "z": enc["z"],
            "z_state": z_state,
            "z_cause": z_cause,
        }
