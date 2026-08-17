"""Event-level activation + gradient extraction for online2's diagnosis model.

Adapted from `risk_saliency.py::event_ixg_abnormal_margin`, which computes the
same forward/backward pass but returns only the product `z.grad * z`
(input x gradient saliency). TCAV needs the activation `z` and the gradient
`z.grad` as SEPARATE tensors — the CAV is fit on activations, the score is
computed against gradients — so this module re-runs that pass and returns
both, rather than importing the fused saliency value.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


@torch.enable_grad()
def event_activation_and_gradient(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    normal_class_id: int,
    use_binary: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (activations, gradients), each (T_valid, D), for batch_size=1.

    `activations` is `z` (per-event representation before task-query pooling,
    same position `risk_saliency.py` uses). `gradients` is d(objective)/dz
    where objective is the binary abnormal logit (`use_binary=True`) or the
    fine-head abnormal-risk margin (`use_binary=False`), matching
    `risk_saliency.py::event_ixg_abnormal_margin`'s objective choice.
    """
    if batch["event_mean"].size(0) != 1:
        raise ValueError("event_activation_and_gradient expects batch_size=1")

    from src.online2.v2.finetune_v03.risk_saliency import risk_margin_from_logits

    model.eval()
    event_mean = batch["event_mean"].detach()
    event_max = batch["event_max"].detach()
    dino_vec = batch.get("dino_vec")
    dino_mask = batch.get("dino_mask")
    if dino_vec is not None:
        dino_vec = dino_vec.detach()
    if dino_mask is not None:
        dino_mask = dino_mask.detach()

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

    mask = batch["padding_mask"][0].bool()
    activations = z[0][mask].detach().cpu().numpy()
    gradients = z.grad[0][mask].detach().cpu().numpy()
    return activations, gradients
