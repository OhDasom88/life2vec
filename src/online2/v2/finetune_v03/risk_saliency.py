"""Fuse-aware risk + event IxG for v0.3 (does not mutate v0.2 saliency.py)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def abnormal_class_ids(num_classes: int, normal_class_id: int) -> List[int]:
    return [i for i in range(num_classes) if i != int(normal_class_id)]


def risk_margin_from_logits(
    logits: torch.Tensor,
    *,
    normal_class_id: int,
) -> torch.Tensor:
    """R_margin = logsumexp(z_abn) - z_normal  [B]."""
    abn = abnormal_class_ids(logits.size(-1), normal_class_id)
    return torch.logsumexp(logits[:, abn], dim=-1) - logits[:, int(normal_class_id)]


def risk_from_binary_logit(abnormal_logit: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(abnormal_logit)


@torch.enable_grad()
def event_ixg_abnormal_margin(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    normal_class_id: int,
    use_binary: bool = False,
) -> torch.Tensor:
    """IxG on z_i w.r.t. abnormal-vs-normal objective; requires encode_events path.

    Returns scores [T] for batch size 1 (signed; >0 increases abnormal risk).
    """
    if batch["event_mean"].size(0) != 1:
        raise ValueError("event_ixg_abnormal_margin expects batch_size=1")

    model.eval()
    event_mean = batch["event_mean"].detach()
    event_max = batch["event_max"].detach()
    dino_vec = batch.get("dino_vec")
    dino_mask = batch.get("dino_mask")
    if dino_vec is not None:
        dino_vec = dino_vec.detach()
    if dino_mask is not None:
        dino_mask = dino_mask.detach()

    # Go through fuse + meta (matches diagnosis forward)
    if hasattr(model, "fuse_image_into_events"):
        event_mean, event_max = model.fuse_image_into_events(
            event_mean, event_max, dino_vec=dino_vec, dino_mask=dino_mask
        )
    z = model.meta(
        event_mean,
        event_max,
        batch["case_age_hours"],
        batch["view_id"],
        batch["zone_id"],
        batch["local_hour"],
        batch["padding_mask"],
    )
    z = z.detach().requires_grad_(True)
    if getattr(model, "task_pad", None) is not None:
        pools = model.task_pad(z, batch["padding_mask"])
        h_case = pools["h_binary"] if use_binary else pools["h_fine"]
    else:
        h_case, _attn = model.pad(z, batch["padding_mask"])

    if use_binary and hasattr(model, "binary_head"):
        objective = model.binary_head(h_case)[0]
    else:
        logits = model.fine_head(h_case) if hasattr(model, "fine_head") else model.head(h_case)
        objective = risk_margin_from_logits(logits, normal_class_id=normal_class_id)[0]

    model.zero_grad(set_to_none=True)
    if z.grad is not None:
        z.grad = None
    objective.backward()
    assert z.grad is not None
    score = (z.grad * z).sum(dim=-1)[0]
    mask = batch["padding_mask"][0].to(score.dtype)
    return (score * mask).detach()


def ensemble_risk_stats(fold_risks: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(fold_risks, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
