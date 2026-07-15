"""Classification + quality-weighted semantic alignment losses (finetune v0.2)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.online2.v2.finetune_v02.config import FinetuneV02Config, QualityWeights


def _quality_weight_tensor(
    quality: QualityWeights, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    # order: high, medium, low
    return torch.tensor(
        [quality.high, quality.medium, quality.low], device=device, dtype=dtype
    )


def weighted_cosine_alignment(
    z: torch.Tensor,
    targets: torch.Tensor,
    *,
    quality: QualityWeights,
    present_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quality-weighted mean of (1 - cos) over High/Medium/Low targets.

    Args:
        z: [B, D] L2-normalized predictions
        targets: [B, 3, D] L2-normalized (H, M, L) text embeddings
        present_mask: optional [B, 3] bool — False skips that quality tier
    """
    cos = (z.unsqueeze(1) * targets).sum(dim=-1)
    loss_q = 1.0 - cos
    w = _quality_weight_tensor(quality, z.device, z.dtype).view(1, 3)
    if present_mask is not None:
        w = w * present_mask.to(dtype=z.dtype)
    denom = w.sum(dim=-1).clamp_min(1e-6)
    return (loss_q * w).sum(dim=-1) / denom


def quality_ranking_loss(
    z: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 0.05,
    present_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encourage sim_H > sim_M > sim_L when all three exist."""
    cos = (z.unsqueeze(1) * targets).sum(dim=-1)  # [B, 3]
    pairs = [(0, 1), (1, 2), (0, 2)]
    losses = []
    for i, j in pairs:
        gap = cos[:, i] - cos[:, j]
        hinge = F.relu(margin - gap)
        if present_mask is not None:
            ok = present_mask[:, i] & present_mask[:, j]
            hinge = hinge * ok.to(hinge.dtype)
        losses.append(hinge)
    return torch.stack(losses, dim=-1).mean(dim=-1)


def total_finetune_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    cfg: FinetuneV02Config,
    state_targets: Optional[torch.Tensor] = None,
    cause_targets: Optional[torch.Tensor] = None,
    state_present: Optional[torch.Tensor] = None,
    cause_present: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Return dict with total and component losses (means over batch)."""
    w = cfg.loss
    ce = F.cross_entropy(outputs["logits"], labels.long())
    parts: Dict[str, torch.Tensor] = {"classification_ce": ce}

    state_l = ce.new_zeros(())
    cause_l = ce.new_zeros(())
    rank_l = ce.new_zeros(())

    if state_targets is not None and w.state_alignment > 0:
        state_l = weighted_cosine_alignment(
            outputs["z_state"],
            state_targets,
            quality=cfg.quality,
            present_mask=state_present,
        ).mean()
        parts["state_alignment"] = state_l

    if cause_targets is not None and w.cause_alignment > 0:
        cause_l = weighted_cosine_alignment(
            outputs["z_cause"],
            cause_targets,
            quality=cfg.quality,
            present_mask=cause_present,
        ).mean()
        parts["cause_alignment"] = cause_l

    if w.quality_ranking > 0 and state_targets is not None and cause_targets is not None:
        rank_l = (
            quality_ranking_loss(
                outputs["z_state"], state_targets, present_mask=state_present
            ).mean()
            + quality_ranking_loss(
                outputs["z_cause"], cause_targets, present_mask=cause_present
            ).mean()
        ) * 0.5
        parts["quality_ranking"] = rank_l

    total = (
        w.classification_ce * ce
        + w.state_alignment * state_l
        + w.cause_alignment * cause_l
        + w.quality_ranking * rank_l
    )
    parts["total"] = total
    return parts
