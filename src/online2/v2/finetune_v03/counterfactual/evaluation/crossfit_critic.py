"""Search/holdout fold isolation for CF M2 prereq.

Holdout folds must not be used for attribution, locus selection, or ranking.
They evaluate only already-selected candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .diagnosis_critic import ensemble_delta_r, score_folds


@dataclass
class FoldSplit:
    search_fold_ids: List[int]
    holdout_fold_ids: List[int]
    checkpoint_paths: List[Path]
    fold_ids: List[int]

    def search_ckpts(self) -> List[Path]:
        return [self.checkpoint_paths[i] for i in self._indices(self.search_fold_ids)]

    def holdout_ckpts(self) -> List[Path]:
        return [self.checkpoint_paths[i] for i in self._indices(self.holdout_fold_ids)]

    def _indices(self, wanted: Sequence[int]) -> List[int]:
        id_to_i = {int(f): i for i, f in enumerate(self.fold_ids)}
        out = []
        for fid in wanted:
            if int(fid) not in id_to_i:
                raise KeyError(f"fold_id {fid} not in manifest fold_ids={self.fold_ids}")
            out.append(id_to_i[int(fid)])
        return out


def resolve_fold_split(
    *,
    checkpoint_paths: Sequence[Path],
    fold_ids: Sequence[int],
    search_fold_ids: Optional[Sequence[int]] = None,
    holdout_fold_ids: Optional[Sequence[int]] = None,
    holdout_mode: str = "fixed",
) -> FoldSplit:
    paths = [Path(p) for p in checkpoint_paths]
    fids = [int(x) for x in fold_ids]
    if len(paths) != len(fids):
        raise ValueError(f"checkpoint/fold length mismatch: {len(paths)} vs {len(fids)}")

    if holdout_mode == "fixed":
        if holdout_fold_ids is None:
            holdout = [fids[-1]] if fids else []
        else:
            holdout = [int(x) for x in holdout_fold_ids]
        if search_fold_ids is None:
            search = [f for f in fids if f not in set(holdout)]
        else:
            search = [int(x) for x in search_fold_ids]
    elif holdout_mode == "leave_one_out":
        # default: last fold holdout (caller rotates for batch)
        holdout = [int(x) for x in (holdout_fold_ids or [fids[-1]])]
        search = [f for f in fids if f not in set(holdout)]
        if search_fold_ids is not None:
            search = [int(x) for x in search_fold_ids]
    else:
        raise ValueError(f"unknown holdout_mode={holdout_mode}")

    if set(search) & set(holdout):
        # OOF-reference mode intentionally uses the same fold for search+holdout scoring;
        # callers must still set cf_valid=null via FoldExecutionContext.
        if not (
            len(search) == 1
            and len(holdout) == 1
            and int(search[0]) == int(holdout[0])
            and holdout_mode in {"fixed", "leave_one_out", "oof_reference"}
        ):
            raise ValueError(f"search/holdout overlap: search={search} holdout={holdout}")
    if not search:
        raise ValueError("search_fold_ids empty")
    return FoldSplit(
        search_fold_ids=list(search),
        holdout_fold_ids=list(holdout),
        checkpoint_paths=paths,
        fold_ids=fids,
    )


def score_candidate_folds(
    ckpt_paths: Sequence[Path],
    batch_before: Mapping[str, Any],
    batch_after: Mapping[str, Any],
    *,
    normal_class_id: int,
    fold_ids: Sequence[int],
    device: str = "cuda",
    gpu_fraction: float = 0.4,
    use_binary: bool = True,
) -> Dict[str, Any]:
    """Score ΔR on an explicit fold/ckpt subset."""
    stats = score_folds(
        list(ckpt_paths),
        batch_before,
        batch_after,
        normal_class_id=normal_class_id,
        device=device,
        gpu_fraction=gpu_fraction,
        use_binary=use_binary,
    )
    stats["fold_ids"] = [int(x) for x in fold_ids]
    return stats


def _search_delta_variance(c: Mapping[str, Any]) -> float:
    folds = c.get("delta_r_folds_search") or c.get("delta_r_folds")
    if folds is None:
        return 0.0
    try:
        import numpy as np

        arr = np.asarray(list(folds), dtype=np.float64)
        if arr.size <= 1:
            return 0.0
        return float(arr.var())
    except Exception:
        return 0.0


def rank_candidates_on_search_folds(
    candidates: Sequence[Mapping[str, Any]],
    *,
    delta_key: str = "delta_r_search",
    folds_improved_key: str = "folds_improved_search",
    material_delta_r_abs: float = 0.01,
    min_search_folds_improved: int = 2,
    require_all_search_folds_negative: bool = True,
) -> List[Dict[str, Any]]:
    """Stable rank (Holdout unused):

    1. delta_r_search (lower better)
    2. folds_improved_search (higher)
    3. search fold ΔR variance (lower)
    4. abs_raw_delta (lower)
    5. token_edit_count / token_hamming (lower)
    6. stable candidate_id
    """

    def sort_key(c: Mapping[str, Any]) -> Tuple:
        dr = float(c.get(delta_key) if c.get(delta_key) is not None else 0.0)
        fi = int(c.get(folds_improved_key) or 0)
        var = _search_delta_variance(c)
        raw_dist = float(c.get("raw_edit_distance") or c.get("abs_raw_delta") or 0.0)
        edits = float(
            c.get("token_edit_count")
            if c.get("token_edit_count") is not None
            else c.get("token_hamming")
            or 0.0
        )
        cid = str(c.get("candidate_id") or c.get("source") or "")
        is_noop = 0 if str(c.get("source") or "").startswith("noop") or c.get("is_noop") else 1
        return (dr, -fi, var, raw_dist, edits, cid, is_noop)

    ranked = sorted((dict(c) for c in candidates), key=sort_key)
    for i, row in enumerate(ranked):
        row["search_rank"] = i
        dr = row.get(delta_key)
        fi = int(row.get(folds_improved_key) or 0)
        folds = row.get("delta_r_folds_search") or row.get("delta_r_folds")
        all_neg = True
        if require_all_search_folds_negative and folds is not None:
            try:
                all_neg = all(float(x) < 0.0 for x in list(folds))
            except Exception:
                all_neg = False
        structural_ok = bool(row.get("structurally_valid", True))
        row["search_material"] = (
            dr is not None
            and float(dr) <= -abs(material_delta_r_abs)
            and fi >= int(min_search_folds_improved)
            and all_neg
            and structural_ok
            and not bool(row.get("is_noop"))
            and not str(row.get("source") or "").startswith("noop")
        )
        row["candidate_selected_without_holdout"] = True
    return ranked


def select_best_on_search(
    ranked: Sequence[Mapping[str, Any]],
    *,
    require_material: bool = True,
) -> Optional[Dict[str, Any]]:
    """Select material candidate on search folds only; else canonical NO_OP.

    When require_material=True (default), never fall back to a non-material edit.
    """
    for row in ranked:
        if require_material and not row.get("search_material"):
            continue
        return dict(row)
    # Prefer canonical NO_OP over any non-material edit
    for row in ranked:
        if row.get("is_noop") or str(row.get("source") or "").startswith("noop"):
            out = dict(row)
            out["selection_reason"] = "no_material_candidate_canonical_noop"
            return out
    return None


def evaluate_selected_on_holdout(
    selected: Mapping[str, Any],
    holdout_ckpts: Sequence[Path],
    batch_before: Mapping[str, Any],
    batch_after: Mapping[str, Any],
    *,
    normal_class_id: int,
    holdout_fold_ids: Sequence[int],
    device: str = "cuda",
    gpu_fraction: float = 0.4,
    use_binary: bool = True,
    material_delta_r_abs: float = 0.01,
) -> Dict[str, Any]:
    stats = score_candidate_folds(
        holdout_ckpts,
        batch_before,
        batch_after,
        normal_class_id=normal_class_id,
        fold_ids=holdout_fold_ids,
        device=device,
        gpu_fraction=gpu_fraction,
        use_binary=use_binary,
    )
    dr = float(stats["delta_r"])
    out = dict(selected)
    out["delta_r_holdout"] = dr
    out["risk_before_holdout"] = stats["risk_before"]
    out["risk_after_holdout"] = stats["risk_after"]
    out["delta_r_folds_holdout"] = stats["delta_r_folds"]
    out["folds_improved_holdout"] = int(stats["folds_improved"])
    out["holdout_fold_ids"] = [int(x) for x in holdout_fold_ids]
    out["material_improvement_holdout"] = dr <= -abs(material_delta_r_abs)
    out["candidate_selected_without_holdout"] = True
    return out


# Re-export for callers that still use all-fold ensemble
__all__ = [
    "FoldSplit",
    "resolve_fold_split",
    "score_candidate_folds",
    "rank_candidates_on_search_folds",
    "select_best_on_search",
    "evaluate_selected_on_holdout",
    "ensemble_delta_r",
]
