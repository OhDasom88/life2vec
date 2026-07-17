"""CF M2 prereq preflight + manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import file_sha256, utc_now, write_json
from .manifest import discover_fold_ckpts, load_manifest


ARTIFACT_SCHEMA = "cf_m2_prereq_v1"


def build_m2_prereq_manifest(
    cfg: Dict[str, Any],
    *,
    out_path: Path,
) -> Dict[str, Any]:
    required_files = {
        "run_dir": Path(cfg["run_dir"]),
        "tokenizer_path": Path(cfg["tokenizer_path"]),
        "binning_registry_path": Path(cfg["binning_registry_path"]),
        "cells_path": Path(cfg["cells_path"]),
        "embeddings_dir": Path(cfg["embeddings_dir"]),
        "events_path": Path(cfg["events_path"]),
        "stage_a_ckpt": Path(cfg["stage_a_ckpt"]),
        "case_manifest_path": Path(cfg.get("case_manifest_path") or cfg.get("labels_path")),
    }
    missing = [k for k, p in required_files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"M2 preflight missing: {missing}")

    ckpts = discover_fold_ckpts(Path(cfg["run_dir"]), repeat=int(cfg.get("repeat", 0)))
    if not ckpts:
        raise FileNotFoundError("no fold checkpoints")

    fold_ids = []
    for p in ckpts:
        stem = p.stem
        if "_fold" in stem:
            fold_ids.append(int(stem.split("_fold")[-1].split("_")[0]))
        else:
            fold_ids.append(int(stem.replace("fold", "").replace("_best", "")))

    critic = cfg.get("critic") or {}
    m2 = cfg.get("cf_m2_prereq") or {}
    feature_schema = Path(
        cfg.get("feature_schema_path") or "outputs/online2/v2_build/feature_schema_v2.yaml"
    )
    if not feature_schema.is_absolute():
        feature_schema = Path("/home/dasom/life2vec") / feature_schema
    vocab_path = Path(cfg.get("vocabulary_path") or cfg["tokenizer_path"])
    tokenizer_code = Path(__file__).resolve().parents[2] / "tokenizer.py"
    # parents: counterfactual -> finetune_v03 -> v2
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA,
        "created_at": utc_now(),
        "fallback_policy": m2.get("fallback_policy", "error"),
        "case_id": cfg.get("case_id"),
        "checkpoint_paths": [str(p) for p in ckpts],
        "checkpoint_hashes": [file_sha256(p) for p in ckpts],
        "fold_ids": fold_ids,
        "search_fold_ids": list(critic.get("search_fold_ids") or fold_ids[:-1]),
        "holdout_fold_ids": list(critic.get("holdout_fold_ids") or [fold_ids[-1]]),
        "stage_a_ckpt": str(required_files["stage_a_ckpt"]),
        "stage_a_ckpt_hash": file_sha256(required_files["stage_a_ckpt"]),
        "events_path": str(required_files["events_path"]),
        "events_hash": file_sha256(required_files["events_path"]),
        "embeddings_dir": str(required_files["embeddings_dir"]),
        "tokenizer_path": str(required_files["tokenizer_path"]),
        "tokenizer_hash": file_sha256(required_files["tokenizer_path"]),
        "tokenization_registry_hash": file_sha256(required_files["tokenizer_path"]),
        "binning_registry_path": str(required_files["binning_registry_path"]),
        "binning_registry_hash": file_sha256(required_files["binning_registry_path"]),
        "feature_schema_path": str(feature_schema) if feature_schema.exists() else None,
        "feature_schema_hash": file_sha256(feature_schema) if feature_schema.exists() else None,
        "vocab_path": str(vocab_path),
        "vocab_hash": file_sha256(vocab_path) if vocab_path.exists() else None,
        "tokenizer_code_path": str(tokenizer_code) if tokenizer_code.exists() else None,
        "tokenizer_code_hash": file_sha256(tokenizer_code) if tokenizer_code.exists() else None,
        "cells_path": str(required_files["cells_path"]),
        "cells_hash": file_sha256(required_files["cells_path"]),
        "frozen": True,
    }
    # isolation check
    if set(payload["search_fold_ids"]) & set(payload["holdout_fold_ids"]):
        raise ValueError("search/holdout fold overlap in M2 manifest")
    write_json(out_path, payload)
    return payload


def assert_strict_no_fallback(cfg: Dict[str, Any], *, attribution_mode: str) -> None:
    m2 = cfg.get("cf_m2_prereq") or {}
    if not m2.get("enabled"):
        return
    policy = str(m2.get("fallback_policy") or "error")
    if policy == "error" and attribution_mode in {"synthetic_fallback", "cache_perturbation"}:
        raise RuntimeError(
            f"strict M2 smoke forbids fallback mode={attribution_mode}; "
            "set fallback_policy=m1_synthetic only for compatibility runs"
        )
