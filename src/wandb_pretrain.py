"""Utilities for linking W&B pretrain sweeps to downstream finetune runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)

PRETRAIN_SWEEP_ID = "ugf1f2ld"
PRETRAIN_WEIGHT_DIR = Path.home() / "weights" / "agri" / "mlm" / "pre_training"
PRETRAIN_WEIGHT_TEMPLATE = "/weights/agri/mlm/pre_training/sweep_{run_id}.pth"

HACKATHON_PRETRAIN_SWEEP_ID = "qj4w2ktc"
HACKATHON_IMPLEMENTATION = "hackathon"


def pretrain_weight_dir(implementation: str = "agri") -> Path:
    return Path.home() / "weights" / implementation / "mlm" / "pre_training"


def pretrain_weight_path(run_id: str, *, implementation: str = "agri") -> Path:
    """Local path to transformer weights exported by AgriSaveWeights."""
    return pretrain_weight_dir(implementation) / f"sweep_{run_id}.pth"


def pretrain_weight_hparam(run_id: str, *, implementation: str = "agri") -> str:
    """Hydra hparam path (prepended with HOME_PATH in cls_model)."""
    return f"/weights/{implementation}/mlm/pre_training/sweep_{run_id}.pth"


def list_pretrain_sweep_runs(
    entity: str = "dasom-oh",
    project: str = "Berry2Vec",
    sweep_id: str = PRETRAIN_SWEEP_ID,
    *,
    implementation: str = "agri",
    top_k: int | None = None,
    finished_only: bool = True,
) -> list[dict[str, Any]]:
    """Return pretrain sweep runs sorted by val/pretrain_score (desc)."""
    import wandb

    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = list(sweep.runs)
    if finished_only:
        runs = [r for r in runs if r.state == "finished"]

    def score(run) -> float:
        return float(run.summary.get("val/pretrain_score", 0) or 0)

    runs = sorted(runs, key=score, reverse=True)
    if top_k is not None:
        runs = runs[:top_k]

    out: list[dict[str, Any]] = []
    for run in runs:
        weight = pretrain_weight_path(run.id, implementation=implementation)
        out.append(
            {
                "run_id": run.id,
                "score": score(run),
                "weight_path": str(weight),
                "weight_exists": weight.is_file(),
                "config": {
                    k: run.config.get(k)
                    for k in (
                        "learning_rate",
                        "mlm_weight",
                        "dropout",
                        "weight_decay",
                        "batch_size",
                    )
                    if k in run.config
                },
            }
        )
    return out


def summarize_weight_transfer(
    model_state: dict[str, torch.Tensor],
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare checkpoint keys/shapes against the live model state dict."""
    loaded: list[str] = []
    shape_mismatch: list[str] = []
    missing_in_ckpt: list[str] = []
    unexpected_in_ckpt: list[str] = []

    for key, value in checkpoint_state.items():
        if key not in model_state:
            unexpected_in_ckpt.append(key)
            continue
        if model_state[key].shape != value.shape:
            shape_mismatch.append(key)
            continue
        loaded.append(key)

    for key in model_state:
        if key not in checkpoint_state:
            missing_in_ckpt.append(key)

    total_model = len(model_state)
    transfer_ratio = len(loaded) / total_model if total_model else 0.0
    return {
        "loaded_keys": len(loaded),
        "total_model_keys": total_model,
        "transfer_ratio": transfer_ratio,
        "shape_mismatch_keys": shape_mismatch,
        "missing_in_checkpoint": missing_in_ckpt,
        "unexpected_in_checkpoint": unexpected_in_ckpt,
    }


def load_transformer_weights(
    transformer: torch.nn.Module,
    weight_path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load pretrained encoder weights and return transfer diagnostics."""
    path = Path(weight_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {path}")

    checkpoint_state = torch.load(path, map_location=map_location)
    summary = summarize_weight_transfer(transformer.state_dict(), checkpoint_state)
    missing, unexpected = transformer.load_state_dict(checkpoint_state, strict=False)
    summary["strict_missing_keys"] = list(missing)
    summary["strict_unexpected_keys"] = list(unexpected)

    log.info(
        "Pretrain transfer: %d/%d keys (%.1f%%), shape_mismatch=%d",
        summary["loaded_keys"],
        summary["total_model_keys"],
        100 * summary["transfer_ratio"],
        len(summary["shape_mismatch_keys"]),
    )
    if summary["shape_mismatch_keys"]:
        log.warning(
            "Shape mismatch (check vocab/arch): %s",
            summary["shape_mismatch_keys"][:5],
        )
    return summary
