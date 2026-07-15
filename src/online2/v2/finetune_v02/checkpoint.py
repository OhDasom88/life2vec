"""Checkpoint load/save helpers shared by v0.2 train + eval.

v0.1-compatible naming:
  <run_dir>/fold{i}_best.pt

Payload keys (superset of v0.1):
  model, cfg, epoch, train_cases, val_cases, metrics,
  finetune_version, model_class
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.online2.v2.event_pooling_finetune import (
    EventPoolingConfig,
    EventPoolingDiagnosisModel,
)
from src.online2.v2.finetune_v02.config import (
    FinetuneV02Config,
    ImageAdapterConfig,
    LossWeights,
    QualityWeights,
    SaliencyConfig,
)
from src.online2.v2.finetune_v02.model import EventPoolingDiagnosisModelV02
from src.online2.v2.finetune_v02.version import FINETUNE_VERSION

ModelVersion = Literal["v01", "v02"]


def fold_ckpt_path(run_dir: Path, fold: int) -> Path:
    """Canonical path matching v0.1 train/eval: fold{i}_best.pt."""
    return Path(run_dir) / f"fold{fold}_best.pt"


def detect_model_version(blob: Dict[str, Any]) -> ModelVersion:
    ver = str(blob.get("finetune_version") or blob.get("model_class") or "")
    if ver.startswith("0.2") or "V02" in ver or blob.get("model_class") == "EventPoolingDiagnosisModelV02":
        return "v02"
    keys = set(blob.get("model", {}).keys()) if isinstance(blob.get("model"), dict) else set()
    if any(k.startswith("state_head.") or k.startswith("cause_head.") for k in keys):
        return "v02"
    return "v01"


def _cfg_v02_from_dict(raw: Dict[str, Any]) -> FinetuneV02Config:
    def _sub(cls, key: str, default_factory):
        val = raw.get(key)
        if isinstance(val, dict):
            return cls(**{k: v for k, v in val.items() if k in cls.__dataclass_fields__})
        return default_factory()

    fields = FinetuneV02Config.__dataclass_fields__
    flat = {k: raw[k] for k in fields if k in raw and k not in {"loss", "quality", "image", "saliency"}}
    return FinetuneV02Config(
        **flat,
        loss=_sub(LossWeights, "loss", LossWeights),
        quality=_sub(QualityWeights, "quality", QualityWeights),
        image=_sub(ImageAdapterConfig, "image", ImageAdapterConfig),
        saliency=_sub(SaliencyConfig, "saliency", SaliencyConfig),
    )


def load_fold_model(
    ckpt_path: Path,
    device: torch.device,
    *,
    force_version: Optional[ModelVersion] = None,
) -> Tuple[nn.Module, Dict[str, Any], ModelVersion]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    version = force_version or detect_model_version(blob)
    if version == "v01":
        cfg = EventPoolingConfig(**blob["cfg"])
        model: nn.Module = EventPoolingDiagnosisModel(cfg)
    else:
        raw_cfg = blob["cfg"]
        if not isinstance(raw_cfg, dict):
            raw_cfg = dict(raw_cfg)
        cfg_v02 = _cfg_v02_from_dict(raw_cfg)
        model = EventPoolingDiagnosisModelV02(cfg_v02)
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    meta = {
        "path": str(ckpt_path),
        "version": version,
        "epoch": blob.get("epoch"),
        "train_cases": [str(c) for c in blob.get("train_cases", [])],
        "val_cases": [str(c) for c in blob.get("val_cases", [])],
        "metrics": blob.get("metrics"),
        "finetune_version": blob.get("finetune_version"),
    }
    return model, meta, version


def load_five_fold_models(
    run_dir: Path,
    device: torch.device,
    *,
    force_version: Optional[ModelVersion] = None,
) -> Tuple[List[nn.Module], List[Dict[str, Any]], ModelVersion]:
    models: List[nn.Module] = []
    metas: List[Dict[str, Any]] = []
    versions: List[ModelVersion] = []
    for i in range(5):
        path = fold_ckpt_path(run_dir, i)
        if not path.exists():
            raise FileNotFoundError(path)
        model, meta, ver = load_fold_model(path, device, force_version=force_version)
        meta["fold"] = i
        models.append(model)
        metas.append(meta)
        versions.append(ver)
    if len(set(versions)) != 1:
        raise RuntimeError(f"mixed fold versions: {versions}")
    return models, metas, versions[0]


def build_v02_ckpt_payload(
    *,
    model: EventPoolingDiagnosisModelV02,
    cfg: FinetuneV02Config,
    epoch: int,
    train_cases: List[str],
    val_cases: List[str],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "cfg": cfg.to_dict(),
        "epoch": int(epoch),
        "train_cases": list(train_cases),
        "val_cases": list(val_cases),
        "metrics": metrics,
        "finetune_version": FINETUNE_VERSION,
        "model_class": "EventPoolingDiagnosisModelV02",
    }
