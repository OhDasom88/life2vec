"""Embedding-cache perturbations standing in for Stage-A re-encode on frozen critic path.

Diagnosis finetune inference consumes Stage-A *cached* event_mean/max. Full token→encoder
re-encode is optional; M1 critic evaluates fuse-aware risk on modified caches, which is the
same forward path as diagnosis after Stage A.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch


def noop_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def zero_event_indices(
    batch: Dict[str, torch.Tensor],
    event_indices: Sequence[int],
) -> Dict[str, torch.Tensor]:
    """Path B B0 model-space edit: zero selected event embeddings (OFF / removed)."""
    out = noop_batch(batch)
    for idx in event_indices:
        i = int(idx)
        if 0 <= i < out["event_mean"].shape[1]:
            out["event_mean"][0, i].zero_()
            out["event_max"][0, i].zero_()
            if "dino_mask" in out and out["dino_mask"] is not None:
                out["dino_mask"][0, i] = False
    return out


def interpolate_event_indices(
    batch: Dict[str, torch.Tensor],
    event_indices: Sequence[int],
    *,
    alpha: float = 0.35,
    toward: str = "zero",
) -> Dict[str, torch.Tensor]:
    """Path A model-space edit on cached Stage-A event embeddings."""
    out = noop_batch(batch)
    a = float(max(0.0, min(1.0, abs(alpha))))
    for idx in event_indices:
        i = int(idx)
        if not (0 <= i < out["event_mean"].shape[1]):
            continue
        if toward == "zero":
            out["event_mean"][0, i] = (1.0 - a) * out["event_mean"][0, i]
            out["event_max"][0, i] = (1.0 - a) * out["event_max"][0, i]
        elif toward == "amplify":
            out["event_mean"][0, i] = (1.0 + a) * out["event_mean"][0, i]
            out["event_max"][0, i] = (1.0 + a) * out["event_max"][0, i]
    return out
