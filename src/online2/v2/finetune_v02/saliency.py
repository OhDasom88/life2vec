"""Hierarchical saliency scaffolding (finetune v0.2).

Computes Input×Gradient on contextual event vectors z_i by default.
Occlusion / token / image paths are staged to control cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.online2.v2.finetune_v02.config import SaliencyConfig


@dataclass
class EventSaliencyResult:
    event_ids: List[str]
    scores: np.ndarray  # [T]
    method: str
    target: str  # classification | state | cause


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)


def event_input_x_gradient(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    target: str = "classification",
    class_id: Optional[int] = None,
    state_proto: Optional[torch.Tensor] = None,
    cause_proto: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gradient×input on z_i for v0.1 or v0.2 models (duck-typed meta/pad/head).

    Returns saliency [T] for batch size 1.
    """
    if batch["event_mean"].size(0) != 1:
        raise ValueError("event_input_x_gradient expects batch_size=1")

    model.eval()
    event_mean = batch["event_mean"].detach()
    event_max = batch["event_max"].detach()
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
    h_case, _attn = model.pad(z, batch["padding_mask"])

    if target == "classification":
        logits = model.head(h_case)
        cid = int(class_id) if class_id is not None else int(logits.argmax(dim=-1).item())
        objective = logits[0, cid]
    elif target == "state":
        if state_proto is None:
            raise ValueError("state_proto required for state saliency")
        if not hasattr(model, "state_head"):
            raise AttributeError("model has no state_head (need finetune v0.2)")
        z_state = model.state_head(h_case)
        objective = (z_state * F.normalize(state_proto, dim=-1)).sum()
    elif target == "cause":
        if cause_proto is None:
            raise ValueError("cause_proto required for cause saliency")
        if not hasattr(model, "cause_head"):
            raise AttributeError("model has no cause_head (need finetune v0.2)")
        z_cause = model.cause_head(h_case)
        objective = (z_cause * F.normalize(cause_proto, dim=-1)).sum()
    else:
        raise ValueError(f"unknown saliency target: {target}")

    model.zero_grad(set_to_none=True)
    if z.grad is not None:
        z.grad = None
    objective.backward()
    assert z.grad is not None
    score = (z.grad * z).sum(dim=-1)[0]  # [T]
    mask = batch["padding_mask"][0].to(score.dtype)
    return (score * mask).detach()


def select_top_event_indices(
    scores: np.ndarray,
    *,
    top_frac: float,
    min_k: int = 1,
) -> List[int]:
    t = scores.shape[0]
    k = max(min_k, int(np.ceil(t * top_frac)))
    order = np.argsort(-scores)
    return order[:k].tolist()


def leave_one_image_logit_delta(
    logit_full: float,
    logit_without: float,
) -> float:
    """S_img = logit_y(X) - logit_y(X \\ image_j)."""
    return float(logit_full - logit_without)


def hierarchical_plan(cfg: SaliencyConfig, n_events: int) -> Dict[str, int]:
    """Cost-control plan sizes for one case."""
    per_model = max(1, int(np.ceil(n_events * cfg.top_frac_per_model)))
    return {
        "n_events": n_events,
        "per_model_top": per_model,
        "occlusion_cap": cfg.occlusion_candidate_cap,
        "consensus_min": cfg.consensus_events_min,
        "consensus_max": cfg.consensus_events_max,
    }
