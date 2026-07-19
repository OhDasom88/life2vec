"""Production preflight checks for CF-1S (paths, folds, case-scoped cutoff)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd


REQUIRED_PRODUCTION_PATH_KEYS = (
    "run_dir",
    "embeddings_dir",
    "labels_path",
    "label_map_path",
    "stage_a_ckpt",
    "abspos_reference_path",
    "tokenizer_path",
    "vocabulary_path",
    "feature_schema_path",
    "binning_registry_path",
    "events_path",
    "cells_path",
)


class ProductionPreflightError(RuntimeError):
    pass


def resolve_prediction_cutoff_time(case_id: str) -> Dict[str, Any]:
    """Timezone-aware period_end from case_id; parse failure → NOT_EVALUABLE."""
    try:
        parts = str(case_id).split("_", 2)
        if len(parts) != 3:
            raise ValueError("case_id must be farm_start_end")
        period_end = pd.Timestamp(parts[2])
        if period_end.tzinfo is None:
            period_end = period_end.tz_localize("UTC")
        else:
            period_end = period_end.tz_convert("UTC")
        return {
            "ok": True,
            "prediction_cutoff_time": period_end.isoformat(),
            "prediction_cutoff_source": "PREDICTION_CUTOFF_TIME_PERIOD_END",
            "recency_status": "EVALUABLE",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "prediction_cutoff_time": None,
            "prediction_cutoff_source": "PREDICTION_CUTOFF_TIME_PERIOD_END",
            "recency_status": "NOT_EVALUABLE",
            "error": str(exc),
        }


def validate_case_scoped_cutoff(
    *,
    case_id: str,
    case_scoped_event_times: Sequence[Any],
    prediction_cutoff_time: Optional[str],
) -> Dict[str, Any]:
    """Only case-scoped model-input events may fail cutoff checks."""
    if not prediction_cutoff_time:
        return {
            "ok": False,
            "prediction_cutoff_time": None,
            "case_scoped_event_count": len(list(case_scoped_event_times)),
            "future_case_input_event_count": 0,
            "future_case_input_event_policy": "CASE_NOT_EVALUABLE",
            "case_evaluable": False,
            "reason": "CUTOFF_PARSE_FAILED",
        }
    cutoff = pd.Timestamp(prediction_cutoff_time)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    future = 0
    scoped = list(case_scoped_event_times)
    for ts in scoped:
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")
            if t > cutoff:
                future += 1
        except Exception:
            future += 1
    ok = future == 0
    return {
        "ok": ok,
        "prediction_cutoff_time": cutoff.isoformat(),
        "case_scoped_event_count": len(scoped),
        "future_case_input_event_count": future,
        "future_case_input_event_policy": "CASE_NOT_EVALUABLE",
        "case_evaluable": ok,
        "case_id": case_id,
    }


def check_fold_partition(
    *,
    search_fold_ids: Sequence[int],
    holdout_fold_ids: Sequence[int],
    saliency_search_fold_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    search = [int(x) for x in search_fold_ids]
    holdout = [int(x) for x in holdout_fold_ids]
    overlap = sorted(set(search) & set(holdout))
    complete = bool(search) and bool(holdout)
    saliency_match = True
    if saliency_search_fold_ids is not None:
        saliency_match = sorted(int(x) for x in saliency_search_fold_ids) == sorted(search)
    return {
        "fold_partition_disjoint": len(overlap) == 0,
        "fold_partition_complete": complete,
        "search_fold_ids": search,
        "holdout_fold_ids": holdout,
        "overlap_fold_ids": overlap,
        "config_matches_saliency_search_folds": saliency_match,
        "ok": len(overlap) == 0 and complete and saliency_match,
    }


def production_preflight(
    cfg: Mapping[str, Any],
    *,
    saliency_policy: Optional[Mapping[str, Any]] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Hard checks before any production candidate generation."""
    errors: List[str] = []
    missing_paths: List[str] = []
    path_status = {}
    for key in REQUIRED_PRODUCTION_PATH_KEYS:
        raw = cfg.get(key)
        if not raw:
            missing_paths.append(key)
            path_status[key] = {"present": False, "path": None}
            continue
        p = Path(str(raw))
        if root is not None and not p.is_absolute():
            p = root / p
        exists = p.exists()
        path_status[key] = {"present": exists, "path": str(p)}
        if not exists:
            missing_paths.append(key)

    if missing_paths:
        errors.append(f"missing_or_absent_paths:{','.join(missing_paths)}")

    if cfg.get("repeat") is None:
        errors.append("repeat_missing")

    critic = cfg.get("critic") or {}
    search_ids = list(critic.get("search_fold_ids") or cfg.get("search_fold_ids") or [])
    holdout_ids = list(
        critic.get("holdout_fold_ids") or cfg.get("holdout_fold_ids") or []
    )
    sal_search = None
    if saliency_policy is not None:
        sal_search = list(
            ((saliency_policy.get("fold_partition") or {}).get("search_fold_ids"))
            or ((saliency_policy.get("folds") or {}).get("search_fold_ids"))
            or ((saliency_policy.get("saliency_selector") or {}).get("search_fold_ids"))
            or (saliency_policy.get("search_fold_ids") or [])
        ) or None
    folds = check_fold_partition(
        search_fold_ids=search_ids,
        holdout_fold_ids=holdout_ids,
        saliency_search_fold_ids=sal_search,
    )
    if not folds["ok"]:
        errors.append("fold_partition_invalid")

    ok = not errors
    return {
        "ok": ok,
        "production_preflight_pass": ok,
        "errors": errors,
        "path_status": path_status,
        "fold_partition": folds,
        "required_keys": list(REQUIRED_PRODUCTION_PATH_KEYS),
        "fixture_default_callback_use_count_allowed": 0,
        "simulated_production_result_count_allowed": 0,
    }


def assert_production_preflight(cfg: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
    report = production_preflight(cfg, **kwargs)
    if not report["ok"]:
        raise ProductionPreflightError(
            f"CF-1S production preflight failed: {report['errors']}"
        )
    return report
