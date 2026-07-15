"""Stage B+C multi-task model v0.3 (task queries + optional attention residual).

PLAN: docs/online2/DIAGNOSIS_FINETUNE_V03_PLAN.md
- Image adapter always on by default (not a sweep knob).
- Task-specific learned queries share MHA key/value.
- Event-level attention residual over mean/max (token cache optional later).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.online2.v2.event_pooling_finetune import (
    DiagnosisHead,
    EventMetadataEncoder,
    EventPoolingConfig,
    PooledAttentionEventDecoder,
)
from src.online2.v2.finetune_v02.config import ImageAdapterConfig
from src.online2.v2.finetune_v02.image_adapter import SharedImageAdapter
from src.online2.v2.finetune_v03.config import FinetuneV03Config


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.drop(self.ln(h))), p=2, dim=-1)


class BinaryHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(self.drop(self.ln(h))).squeeze(-1)


class EventAttentionResidual(nn.Module):
    """Attention over event sequence as residual on mean/max fused features.

    Without token-level cache this operates on event representations (stage-B z).
    When token_hidden is provided later, the same module can pool tokens first.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.ln = nn.LayerNorm(hidden_dim)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, z: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = z.size(0)
        q = self.query.expand(bsz, -1, -1)
        key_padding_mask = ~padding_mask.bool()
        pooled, attn = self.mha(
            q, z, z, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True
        )
        self.last_attn = attn
        g = torch.sigmoid(self.gate)
        # residual broadcast to each event: optional skip — return pooled residual signal
        h = self.ln(pooled.squeeze(1))
        return h * g, attn


class SharedKVTaskQueries(nn.Module):
    """Shared MHA KV with task-specific learned queries → h_fine / h_binary / h_proj."""

    def __init__(self, cfg: EventPoolingConfig):
        super().__init__()
        self.cfg = cfg
        self.q_fine = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.q_binary = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.q_proj = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.last_attn: Dict[str, torch.Tensor] = {}

    def _pool(self, q: torch.Tensor, z: torch.Tensor, padding_mask: torch.Tensor, name: str):
        bsz = z.size(0)
        qq = q.expand(bsz, -1, -1)
        key_padding_mask = ~padding_mask.bool()
        pooled, attn = self.mha(
            qq, z, z, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True
        )
        self.last_attn[name] = attn
        h = self.norm(pooled + qq)
        h = self.norm2(h + self.ff(h))
        return h.squeeze(1), attn

    def forward(self, z: torch.Tensor, padding_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h_fine, a_f = self._pool(self.q_fine, z, padding_mask, "fine")
        h_bin, a_b = self._pool(self.q_binary, z, padding_mask, "binary")
        h_proj, a_p = self._pool(self.q_proj, z, padding_mask, "proj")
        return {
            "h_fine": h_fine,
            "h_binary": h_bin,
            "h_proj": h_proj,
            "attn_fine": a_f,
            "attn_binary": a_b,
            "attn_proj": a_p,
        }


class EventPoolingDiagnosisModelV03(nn.Module):
    """PAD / task-query multitask + DINO IMAGE residual (adapter always default ON)."""

    def __init__(self, cfg: Optional[FinetuneV03Config] = None, *, use_image_adapter: Optional[bool] = None):
        super().__init__()
        self.cfg = cfg or FinetuneV03Config()
        pad_cfg = self.cfg.pad_config()
        # Image: plan fixes use_image_adapter=True; allow explicit override only for diagnostics
        if use_image_adapter is None:
            use_image_adapter = bool(self.cfg.arch.use_image_adapter)
        self.use_image_adapter = bool(use_image_adapter)

        self.meta = EventMetadataEncoder(pad_cfg)
        self.task_specific_query = bool(self.cfg.arch.task_specific_query)
        if self.task_specific_query:
            self.task_pad = SharedKVTaskQueries(pad_cfg)
            self.pad = None
        else:
            self.pad = PooledAttentionEventDecoder(pad_cfg)
            self.task_pad = None

        self.attention_residual = bool(self.cfg.arch.attention_residual)
        if self.attention_residual:
            self.event_attn_res = EventAttentionResidual(
                pad_cfg.hidden_dim, pad_cfg.num_heads, pad_cfg.dropout
            )
        else:
            self.event_attn_res = None

        self.fine_head = DiagnosisHead(pad_cfg)
        self.binary_head = BinaryHead(self.cfg.hidden_dim, self.cfg.dropout)
        self.proj_head = ProjectionHead(self.cfg.hidden_dim, self.cfg.proj_dim, self.cfg.dropout)

        if self.use_image_adapter:
            img_cfg = ImageAdapterConfig(
                dino_dim=self.cfg.image.dino_dim,
                model_dim=self.cfg.input_dim,
                gate_init=self.cfg.image.gate_init,
                normalize_l2=self.cfg.image.normalize_l2,
            )
            self.image_adapter = SharedImageAdapter(img_cfg)
        else:
            self.image_adapter = None

        self._last_image_stats: Dict[str, float] = {}

    def fuse_image_into_events(
        self,
        event_mean: torch.Tensor,
        event_max: torch.Tensor,
        *,
        dino_vec: Optional[torch.Tensor],
        dino_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.image_adapter is None
            or dino_vec is None
            or dino_mask is None
            or not bool(dino_mask.any())
        ):
            self._last_image_stats = {
                "dino_active_frac": 0.0,
                "adapter_gate_mean": float("nan"),
                "dino_residual_norm": 0.0,
            }
            return event_mean, event_max
        residual = self.image_adapter(dino_vec)
        m = dino_mask.unsqueeze(-1).to(dtype=event_mean.dtype)
        with torch.no_grad():
            gate = getattr(self.image_adapter, "gate", None)
            gmean = float(gate.detach().mean().item()) if gate is not None else float("nan")
            self._last_image_stats = {
                "dino_active_frac": float(dino_mask.float().mean().item()),
                "adapter_gate_mean": gmean,
                "dino_residual_norm": float((residual * m).norm(dim=-1).mean().item()),
            }
        return event_mean + residual * m, event_max + residual * m

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
        token_hidden: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # token_attention_pool: if token cache provided, average-pool tokens → residual on events (stub path)
        if (
            bool(self.cfg.arch.token_attention_pool)
            and token_hidden is not None
            and token_mask is not None
            and self.event_attn_res is not None
        ):
            # expect token_hidden [B,T,L,D] → mean over L as event token hint (placeholder until full attn)
            tm = token_mask.unsqueeze(-1).to(token_hidden.dtype)
            tok = (token_hidden * tm).sum(dim=2) / tm.sum(dim=2).clamp_min(1.0)
            event_mean = event_mean + 0.1 * tok[..., : event_mean.size(-1)]

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

        attn_res = None
        if self.event_attn_res is not None:
            h_res, attn_res = self.event_attn_res(z, padding_mask)
            # add residual to each valid event embedding via broadcast from case pool
            z = z + h_res.unsqueeze(1) * padding_mask.unsqueeze(-1).to(z.dtype)

        out: Dict[str, torch.Tensor] = {
            "z": z,
            "event_mean": event_mean,
            "event_max": event_max,
        }
        if attn_res is not None:
            out["attn_event_res"] = attn_res

        if self.task_pad is not None:
            pools = self.task_pad(z, padding_mask)
            out.update(pools)
            out["h_case"] = pools["h_fine"]  # compat for saliency helpers
            out["attn"] = pools["attn_fine"]
        else:
            assert self.pad is not None
            h_case, attn = self.pad(z, padding_mask)
            out["h_case"] = h_case
            out["h_fine"] = h_case
            out["h_binary"] = h_case
            out["h_proj"] = h_case
            out["attn"] = attn
        return out

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
        token_hidden: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
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
            token_hidden=token_hidden,
            token_mask=token_mask,
        )
        logits = self.fine_head(enc["h_fine"])
        abnormal_logit = self.binary_head(enc["h_binary"])
        z_proj = self.proj_head(enc["h_proj"])
        return {
            "logits": logits,
            "abnormal_logit": abnormal_logit,
            "z_proj": z_proj,
            "h_case": enc["h_case"],
            "h_fine": enc["h_fine"],
            "h_binary": enc["h_binary"],
            "h_proj": enc["h_proj"],
            "attn": enc.get("attn"),
            "attn_fine": enc.get("attn_fine", enc.get("attn")),
            "attn_binary": enc.get("attn_binary"),
            "z": enc["z"],
            "image_stats": self._last_image_stats,
        }
