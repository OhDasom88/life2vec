"""Multi-task losses for finetune v0.3 (PLAN: consistency + prototype + SupCon)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.online2.v2.finetune_v03.config import FinetuneV03Config


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SupCon on L2-normalized z [B, D] with Fine diagnosis labels [B].

    Returns loss and diagnostic stats (positive-pair health).
    """
    stats: Dict[str, float] = {
        "valid_supcon_anchor_count": 0.0,
        "valid_supcon_anchor_ratio": 0.0,
        "positive_pair_count": 0.0,
        "mean_positive_pairs_per_anchor": 0.0,
        "batch_without_positive": 1.0,
        "supcon_loss_zero": 1.0,
    }
    if z.size(0) < 2:
        return z.new_zeros(()), stats
    sim = z @ z.T / temperature
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask
    pos_per_anchor = mask.sum(dim=1)
    valid = pos_per_anchor > 0
    stats["valid_supcon_anchor_count"] = float(valid.sum().item())
    stats["valid_supcon_anchor_ratio"] = float(valid.float().mean().item())
    stats["positive_pair_count"] = float(mask.sum().item())
    stats["mean_positive_pairs_per_anchor"] = float(pos_per_anchor.mean().item())
    stats["batch_without_positive"] = float(1.0 if mask.sum() < 1 else 0.0)
    if mask.sum() < 1:
        return z.new_zeros(()), stats
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    mean_log_prob = (mask * log_prob).sum(dim=1) / pos_per_anchor.clamp_min(1.0)
    loss = -mean_log_prob[valid].mean() if bool(valid.any()) else z.new_zeros(())
    stats["supcon_loss_zero"] = float(1.0 if float(loss.item()) == 0.0 else 0.0)
    return loss, stats


def prototype_center_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_classes: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L2 distance to class mean prototypes (Fine labels). No binary collapse."""
    if z.size(0) < 1:
        return z.new_zeros(()), {"n_prototypes": 0.0}
    centers = []
    for c in range(n_classes):
        m = labels == c
        if bool(m.any()):
            centers.append(z[m].mean(dim=0))
        else:
            centers.append(z.new_zeros(z.size(-1)))
    mu = torch.stack(centers, dim=0)  # [C, D]
    target = mu[labels.long()]
    loss = ((z - target) ** 2).sum(dim=-1).mean()
    # diagnostic distances
    with torch.no_grad():
        norms = mu.norm(dim=-1)
        n_alive = int((norms > 1e-6).sum().item())
    return loss, {"n_prototypes": float(n_alive), "prototype_norm_mean": float(norms.mean().item())}


def fine_abnormal_prob(logits: torch.Tensor, normal_class_id: int) -> torch.Tensor:
    """p_abnormal^fine = 1 - p(normal)."""
    probs = F.softmax(logits, dim=-1)
    return 1.0 - probs[:, int(normal_class_id)]


def consistency_loss(
    logits: torch.Tensor,
    abnormal_logit: torch.Tensor,
    *,
    normal_class_id: int,
    mode: str = "stopgrad_bce",
) -> torch.Tensor:
    p_fine = fine_abnormal_prob(logits, normal_class_id)
    p_bin = torch.sigmoid(abnormal_logit)
    if mode == "js":
        m = 0.5 * (p_fine + p_bin)
        eps = 1e-8
        kl_fm = p_fine * (torch.log(p_fine.clamp_min(eps)) - torch.log(m.clamp_min(eps)))
        kl_bm = p_bin * (torch.log(p_bin.clamp_min(eps)) - torch.log(m.clamp_min(eps)))
        # also for 1-p
        qf, qb = 1.0 - p_fine, 1.0 - p_bin
        mm = 0.5 * (qf + qb)
        kl_fm = kl_fm + qf * (torch.log(qf.clamp_min(eps)) - torch.log(mm.clamp_min(eps)))
        kl_bm = kl_bm + qb * (torch.log(qb.clamp_min(eps)) - torch.log(mm.clamp_min(eps)))
        return (0.5 * (kl_fm + kl_bm)).mean()
    # stop-gradient BCE: binary → fine target
    return F.binary_cross_entropy(p_bin.clamp(1e-6, 1 - 1e-6), p_fine.detach())


def total_finetune_loss_v03(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    cfg: FinetuneV03Config,
    binary_pos_weight: Optional[torch.Tensor] = None,
    normal_class_id: int = 6,
) -> Dict[str, torch.Tensor]:
    """labels: fine class ids. outputs must include y_abnormal [B]."""
    w = cfg.loss
    fine = F.cross_entropy(outputs["logits"], labels.long())
    y_bin = outputs.get("y_abnormal")
    if y_bin is None:
        raise KeyError("outputs must include y_abnormal [B] float {0,1}")
    bce = F.binary_cross_entropy_with_logits(
        outputs["abnormal_logit"],
        y_bin.float(),
        pos_weight=binary_pos_weight,
    )
    parts: Dict[str, torch.Tensor] = {
        "fine_ce": fine,
        "binary_bce": bce,
    }
    extras: Dict[str, float] = {}

    cons = fine.new_zeros(())
    if w.consistency > 0:
        cons = consistency_loss(
            outputs["logits"],
            outputs["abnormal_logit"],
            normal_class_id=normal_class_id,
            mode=cfg.consistency_mode,
        )
        parts["consistency"] = cons

    supcon = fine.new_zeros(())
    if w.proj_supcon > 0:
        supcon, sc_stats = supervised_contrastive_loss(
            outputs["z_proj"], labels, temperature=float(cfg.supcon_temperature)
        )
        parts["proj_supcon"] = supcon
        extras.update({f"supcon/{k}": v for k, v in sc_stats.items()})

    proto = fine.new_zeros(())
    if w.prototype > 0:
        proto, pr_stats = prototype_center_loss(
            outputs["z_proj"], labels, n_classes=int(cfg.num_classes)
        )
        parts["prototype"] = proto
        extras.update({f"proto/{k}": v for k, v in pr_stats.items()})

    total = (
        w.fine_ce * fine
        + w.binary_bce * bce
        + w.consistency * cons
        + w.proj_supcon * supcon
        + w.prototype * proto
    )
    parts["total"] = total
    for k, v in extras.items():
        parts[f"_stat_{k}"] = fine.new_tensor(v)
    return parts
