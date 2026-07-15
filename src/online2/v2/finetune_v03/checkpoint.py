"""Checkpoint helpers for v0.3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.online2.v2.finetune_v02.config import ImageAdapterConfig, SaliencyConfig
from src.online2.v2.finetune_v03.config import (
    ArchitectureV03,
    CvPlanV03,
    FinetuneV03Config,
    LossWeightsV03,
    OpenSetInferCfg,
    OpenSetTrainCfg,
    RoutingThresholds,
)
from src.online2.v2.finetune_v03.model import EventPoolingDiagnosisModelV03
from src.online2.v2.finetune_v03.version import FINETUNE_VERSION


def fold_ckpt_path(run_dir: Path, fold: int, *, repeat: int | None = None) -> Path:
    if repeat is None:
        return Path(run_dir) / f"fold{fold}_best.pt"
    return Path(run_dir) / f"r{repeat}_fold{fold}_best.pt"


def _sub(cls, raw: Dict[str, Any], key: str):
    val = raw.get(key)
    if isinstance(val, dict):
        fields = getattr(cls, "__dataclass_fields__", {})
        return cls(**{k: v for k, v in val.items() if k in fields})
    return cls()


def _cfg_from_dict(raw: Dict[str, Any]) -> FinetuneV03Config:
    fields = FinetuneV03Config.__dataclass_fields__
    nested = {
        "loss",
        "image",
        "saliency",
        "arch",
        "open_set_train",
        "open_set_infer",
        "routing",
        "cv",
    }
    flat = {k: raw[k] for k in fields if k in raw and k not in nested}
    # migrate old loss key proj_supcon-only configs
    loss_raw = raw.get("loss", {})
    if isinstance(loss_raw, dict) and "binary_bce" not in loss_raw and "fine_ce" in loss_raw:
        pass
    return FinetuneV03Config(
        **flat,
        loss=_sub(LossWeightsV03, raw, "loss"),
        image=_sub(ImageAdapterConfig, raw, "image"),
        saliency=_sub(SaliencyConfig, raw, "saliency"),
        arch=_sub(ArchitectureV03, raw, "arch"),
        open_set_train=_sub(OpenSetTrainCfg, raw, "open_set_train"),
        open_set_infer=_sub(OpenSetInferCfg, raw, "open_set_infer"),
        routing=_sub(RoutingThresholds, raw, "routing"),
        cv=_sub(CvPlanV03, raw, "cv"),
    )


def build_v03_ckpt_payload(
    *,
    model: EventPoolingDiagnosisModelV03,
    cfg: FinetuneV03Config,
    epoch: int,
    train_cases: List[str],
    val_cases: List[str],
    metrics: Dict[str, Any],
    normal_class_id: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        "model": model.state_dict(),
        "cfg": cfg.to_dict(),
        "epoch": int(epoch),
        "train_cases": list(train_cases),
        "val_cases": list(val_cases),
        "metrics": metrics,
        "finetune_version": FINETUNE_VERSION,
        "model_class": "EventPoolingDiagnosisModelV03",
        "normal_class_id": int(normal_class_id),
    }
    if extra:
        payload.update(extra)
    return payload


def load_fold_model_v03(
    ckpt_path: Path, device: torch.device
) -> Tuple[EventPoolingDiagnosisModelV03, Dict[str, Any]]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _cfg_from_dict(dict(blob["cfg"]))
    model = EventPoolingDiagnosisModelV03(cfg)
    model.load_state_dict(blob["model"], strict=False)
    model.to(device)
    model.eval()
    meta = {
        "path": str(ckpt_path),
        "epoch": blob.get("epoch"),
        "train_cases": [str(c) for c in blob.get("train_cases", [])],
        "val_cases": [str(c) for c in blob.get("val_cases", [])],
        "metrics": blob.get("metrics"),
        "finetune_version": blob.get("finetune_version"),
        "normal_class_id": blob.get("normal_class_id"),
    }
    return model, meta
