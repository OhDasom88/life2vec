"""P0-A diagnosis artifact freeze / preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import file_sha256, utc_now, write_json


REQUIRED = [
    "diagnosis_model_artifact",
    "tokenizer_artifact",
    "binning_registry_artifact",
    "risk_definition",
    "stage_a_encoder_version",
]


def discover_fold_ckpts(run_dir: Path, *, repeat: int = 0) -> List[Path]:
    run_dir = Path(run_dir)
    paths = sorted(run_dir.glob(f"r{repeat}_fold*_best.pt"))
    if not paths:
        paths = sorted(run_dir.glob("fold*_best.pt"))
    return paths


def build_diagnosis_manifest(
    *,
    run_dir: Path,
    tokenizer_path: Path,
    binning_registry_path: Path,
    cells_path: Path,
    embeddings_dir: Path,
    out_path: Path,
    risk_definition: str = "calibrated_binary_abnormal_probability",
    stage_a_encoder_version: str = "finetune_v03.encode_events+fuse_image_into_events",
    routing_config: Optional[str] = None,
    repeat: int = 0,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ckpts = discover_fold_ckpts(run_dir, repeat=repeat)
    missing = []
    for label, p in [
        ("model checkpoint", ckpts[0] if ckpts else None),
        ("tokenizer", tokenizer_path if Path(tokenizer_path).exists() else None),
        ("binning registry", binning_registry_path if Path(binning_registry_path).exists() else None),
        ("raw occurrence mapping", cells_path if Path(cells_path).exists() else None),
        ("stage a embeddings", embeddings_dir if Path(embeddings_dir).exists() else None),
    ]:
        if p is None or not Path(p).exists():
            missing.append(label)
    if missing:
        raise FileNotFoundError(f"M1 preflight missing: {missing}")

    fold_ids = []
    ckpt_hashes = []
    for p in ckpts:
        # r0_fold2_best.pt or fold2_best.pt
        stem = p.stem
        if "_fold" in stem:
            fold_ids.append(int(stem.split("_fold")[-1].split("_")[0]))
        else:
            fold_ids.append(int(stem.replace("fold", "").replace("_best", "")))
        ckpt_hashes.append(file_sha256(p))

    payload = {
        "diagnosis_model_artifact": str(run_dir),
        "checkpoint_paths": [str(p) for p in ckpts],
        "checkpoint_hashes": ckpt_hashes,
        "tokenizer_artifact": str(tokenizer_path),
        "tokenizer_hash": file_sha256(tokenizer_path) if Path(tokenizer_path).is_file() else None,
        "binning_registry_artifact": str(binning_registry_path),
        "binning_registry_hash": file_sha256(binning_registry_path),
        "cells_path": str(cells_path),
        "cells_hash": file_sha256(cells_path),
        "embeddings_dir": str(embeddings_dir),
        "routing_config": routing_config,
        "risk_definition": risk_definition,
        "stage_a_encoder_version": stage_a_encoder_version,
        "fold_ids": fold_ids,
        "repeat": repeat,
        "created_at": utc_now(),
        "frozen": True,
    }
    write_json(out_path, payload)
    return payload


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
