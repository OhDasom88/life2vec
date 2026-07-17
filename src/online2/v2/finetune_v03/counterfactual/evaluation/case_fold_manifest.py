"""Case fold / cohort manifest for CF M2 (example OOF vs problem crossfit)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..io_utils import write_json
from .fold_execution_context import EVAL_CROSSFIT, EVAL_OOF_REF


def load_split_manifest(path: Path, *, repeat: int = 0) -> List[Dict[str, Any]]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"split_manifest must be a list: {path}")
    return [row for row in raw if int(row.get("repeat", 0)) == int(repeat)]


def oof_fold_for_case(case_id: str, splits: Sequence[Mapping[str, Any]]) -> Optional[int]:
    cid = str(case_id)
    for row in splits:
        val = set(str(x) for x in (row.get("val") or []))
        if cid in val:
            return int(row["fold"])
    return None


def case_seen_by_fold(
    case_id: str,
    splits: Sequence[Mapping[str, Any]],
    *,
    fold_ids: Sequence[int] = (0, 1, 2),
) -> Dict[int, bool]:
    """True if case was in that fold's train set (seen during training)."""
    cid = str(case_id)
    out = {int(f): False for f in fold_ids}
    for row in splits:
        fid = int(row["fold"])
        train = set(str(x) for x in (row.get("train") or []))
        if cid in train:
            out[fid] = True
    return out


def case_present_in_split(
    case_id: str,
    splits: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Locate case in any fold train/val (for problem cohort integrity)."""
    cid = str(case_id)
    hits = []
    for row in splits:
        fid = int(row["fold"])
        train = set(str(x) for x in (row.get("train") or []))
        val = set(str(x) for x in (row.get("val") or []))
        if cid in train:
            hits.append({"fold": fid, "split": "train"})
        if cid in val:
            hits.append({"fold": fid, "split": "val"})
    return {"present": bool(hits), "hits": hits}


def resolve_cohort(
    case_id: str,
    *,
    labels_path: Optional[Path] = None,
    cohort_override: Optional[str] = None,
    example_case_ids: Optional[Sequence[str]] = None,
    problem_case_ids: Optional[Sequence[str]] = None,
) -> str:
    if cohort_override:
        c = str(cohort_override)
        if c not in {"example_set", "problem_set"}:
            raise ValueError(f"invalid cohort_override={c}")
        return c
    cid = str(case_id)
    if example_case_ids is not None and cid in set(str(x) for x in example_case_ids):
        return "example_set"
    if problem_case_ids is not None and cid in set(str(x) for x in problem_case_ids):
        return "problem_set"
    if labels_path and Path(labels_path).exists():
        import pandas as pd

        df = pd.read_csv(labels_path)
        if "case_id" in df.columns and "set" in df.columns:
            hit = df[df["case_id"].astype(str) == cid]
            if len(hit):
                return str(hit.iloc[0]["set"])
    # default: if present in split val/train → example; else problem
    return "example_set"


def build_case_fold_record(
    *,
    case_id: str,
    cohort: str,
    splits: Sequence[Mapping[str, Any]],
    fold_ids: Sequence[int] = (0, 1, 2),
    search_fold_ids: Optional[Sequence[int]] = None,
    holdout_fold_ids: Optional[Sequence[int]] = None,
    holdout_rotation: Optional[int] = None,
) -> Dict[str, Any]:
    """Build per-case evaluation record.

    Problem set: 3-fold crossfit (search/holdout rotation).
    Example set: oof_reference_only (single OOF critic; cf_valid=null).
    """
    seen = case_seen_by_fold(case_id, splits, fold_ids=fold_ids)
    oof = oof_fold_for_case(case_id, splits)
    fids = [int(x) for x in fold_ids]

    if cohort == "problem_set":
        presence = case_present_in_split(case_id, splits)
        if presence["present"]:
            raise ValueError(
                f"cohort mismatch: problem_set case {case_id} appears in split "
                f"train/val: {presence['hits']} (config override cannot force unseen)"
            )
        if holdout_rotation is not None:
            holdout = [int(holdout_rotation)]
            search = [f for f in fids if f not in set(holdout)]
        elif holdout_fold_ids is not None:
            holdout = [int(x) for x in holdout_fold_ids]
            search = (
                [int(x) for x in search_fold_ids]
                if search_fold_ids is not None
                else [f for f in fids if f not in set(holdout)]
            )
        else:
            holdout = [fids[-1]]
            search = [f for f in fids if f not in set(holdout)]
        # Verified absent from all train/val → all folds unseen
        seen = {int(f): False for f in fids}
        return {
            "case_id": str(case_id),
            "cohort": "problem_set",
            "case_seen_by_fold": {str(k): v for k, v in seen.items()},
            "oof_fold_id": None,
            "evaluation_mode": EVAL_CROSSFIT,
            "search_fold_ids": search,
            "holdout_fold_ids": holdout,
            "split_presence_verified": True,
            "split_presence_hits": [],
        }

    # example_set
    if oof is None:
        raise KeyError(f"example case {case_id} not found in split_manifest val sets")
    return {
        "case_id": str(case_id),
        "cohort": "example_set",
        "case_seen_by_fold": {str(k): v for k, v in seen.items()},
        "oof_fold_id": int(oof),
        "evaluation_mode": EVAL_OOF_REF,
        # OOF-only: both search and holdout reference the single OOF fold for scoring;
        # selection/eval separation is intentionally unavailable.
        "search_fold_ids": [int(oof)],
        "holdout_fold_ids": [int(oof)],
    }


def crossfit_rotations(fold_ids: Sequence[int] = (0, 1, 2)) -> List[Tuple[List[int], List[int]]]:
    """Return (search, holdout) for each leave-one-out rotation."""
    fids = [int(x) for x in fold_ids]
    out = []
    for h in fids:
        holdout = [h]
        search = [f for f in fids if f != h]
        out.append((search, holdout))
    return out


def write_case_fold_manifest(
    records: Sequence[Mapping[str, Any]],
    out_path: Path,
) -> Dict[str, Any]:
    payload = {
        "n_cases": len(records),
        "n_example": sum(1 for r in records if r.get("cohort") == "example_set"),
        "n_problem": sum(1 for r in records if r.get("cohort") == "problem_set"),
        "cases": list(records),
    }
    write_json(out_path, payload)
    return payload
