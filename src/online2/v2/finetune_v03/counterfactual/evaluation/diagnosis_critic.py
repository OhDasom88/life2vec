"""Frozen diagnosis critic: risk_before / risk_after via binary head."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from src.online2.v2.finetune_v03.checkpoint import load_fold_model_v03
from src.online2.v2.finetune_v03.risk_saliency import risk_from_binary_logit, risk_margin_from_logits


@torch.no_grad()
def risk_from_batch(
    model,
    batch: Dict[str, torch.Tensor],
    *,
    normal_class_id: int,
    use_binary: bool = True,
) -> float:
    # Prefer encode_events so task_specific_query models (pad=None, task_pad set) work.
    if hasattr(model, "encode_events"):
        enc = model.encode_events(
            batch["event_mean"],
            batch["event_max"],
            batch["case_age_hours"],
            batch["view_id"],
            batch["zone_id"],
            batch["local_hour"],
            batch["padding_mask"],
            dino_vec=batch.get("dino_vec"),
            dino_mask=batch.get("dino_mask"),
        )
        h_bin = enc.get("h_binary", enc["h_case"])
        h_fine = enc.get("h_fine", enc["h_case"])
        if use_binary and hasattr(model, "binary_head"):
            return float(risk_from_binary_logit(model.binary_head(h_bin))[0].item())
        logits = model.fine_head(h_fine) if hasattr(model, "fine_head") else model.head(h_fine)
        return float(
            torch.sigmoid(risk_margin_from_logits(logits, normal_class_id=normal_class_id))[0].item()
        )

    event_mean = batch["event_mean"]
    event_max = batch["event_max"]
    if hasattr(model, "fuse_image_into_events"):
        event_mean, event_max = model.fuse_image_into_events(
            event_mean,
            event_max,
            dino_vec=batch.get("dino_vec"),
            dino_mask=batch.get("dino_mask"),
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
    if getattr(model, "task_pad", None) is not None:
        pools = model.task_pad(z, batch["padding_mask"])
        h_case = pools["h_binary"] if use_binary else pools["h_fine"]
    else:
        h_case, _ = model.pad(z, batch["padding_mask"])
    if use_binary and hasattr(model, "binary_head"):
        return float(risk_from_binary_logit(model.binary_head(h_case))[0].item())
    logits = model.fine_head(h_case) if hasattr(model, "fine_head") else model.head(h_case)
    return float(torch.sigmoid(risk_margin_from_logits(logits, normal_class_id=normal_class_id))[0].item())


def score_folds(
    ckpt_paths: Sequence[Path],
    batch_before: Mapping[str, Any],
    batch_after: Mapping[str, Any],
    *,
    normal_class_id: int,
    device: str = "cuda",
    gpu_fraction: float = 0.4,
    use_binary: bool = True,
) -> Dict[str, Any]:
    """Score ΔR on the given checkpoint list (caller controls search vs holdout)."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(gpu_fraction), device=0)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    before = []
    after = []
    for ckpt in ckpt_paths:
        model, meta = load_fold_model_v03(Path(ckpt), device=dev)
        nid = int(meta.get("normal_class_id") if meta.get("normal_class_id") is not None else normal_class_id)
        bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch_before.items()}
        ba = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch_after.items()}
        rb = risk_from_batch(model, bb, normal_class_id=nid, use_binary=use_binary)
        ra = risk_from_batch(model, ba, normal_class_id=nid, use_binary=use_binary)
        before.append(rb)
        after.append(ra)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    deltas = [a - b for a, b in zip(after, before)]
    folds_improved = sum(1 for d in deltas if d < 0)
    mean_dr = float(sum(deltas) / max(len(deltas), 1))
    return {
        "risk_before": before,
        "risk_after": after,
        "delta_r_folds": deltas,
        "delta_r": mean_dr,
        "folds_improved": int(folds_improved),
        "model_space_delta_r": mean_dr,
    }


def ensemble_delta_r(
    ckpt_paths: Sequence[Path],
    batch_before: Dict[str, torch.Tensor],
    batch_after: Dict[str, torch.Tensor],
    *,
    normal_class_id: int,
    device: str = "cuda",
    gpu_fraction: float = 0.4,
    use_binary: bool = True,
) -> Dict[str, Any]:
    return score_folds(
        ckpt_paths,
        batch_before,
        batch_after,
        normal_class_id=normal_class_id,
        device=device,
        gpu_fraction=gpu_fraction,
        use_binary=use_binary,
    )
