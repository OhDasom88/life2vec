"""Runtime fold-usage traces for leakage_free acceptance (no hardcoded True)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Set


SELECTION_STAGES = (
    "event_preselection",
    "token_ixg",
    "locus_selection",
    "candidate_ranking",
)


def build_fold_usage_trace(
    *,
    event_preselection_folds: Sequence[int],
    token_ixg_folds: Sequence[int],
    locus_selection_folds: Sequence[int],
    candidate_ranking_folds: Sequence[int],
    holdout_evaluation_folds: Sequence[int],
    evaluation_mode: str,
    cohort: str,
    case_id: str,
) -> Dict[str, Any]:
    return {
        "case_id": str(case_id),
        "cohort": str(cohort),
        "evaluation_mode": str(evaluation_mode),
        "event_preselection_folds": [int(x) for x in event_preselection_folds],
        "token_ixg_folds": [int(x) for x in token_ixg_folds],
        "locus_selection_folds": [int(x) for x in locus_selection_folds],
        "candidate_ranking_folds": [int(x) for x in candidate_ranking_folds],
        "holdout_evaluation_folds": [int(x) for x in holdout_evaluation_folds],
    }


def selection_folds_used(trace: Mapping[str, Any]) -> Set[int]:
    used: Set[int] = set()
    for key in (
        "event_preselection_folds",
        "token_ixg_folds",
        "locus_selection_folds",
        "candidate_ranking_folds",
    ):
        used |= {int(x) for x in (trace.get(key) or [])}
    return used


def compute_leakage_free(
    trace: Mapping[str, Any],
    *,
    evaluation_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Holdout folds must not appear in any selection-stage usage.

    OOF reference mode is not independently evaluable; leakage_free is True only
    when selection folds == holdout folds == {oof} (intentional shared fold).
    """
    mode = str(evaluation_mode or trace.get("evaluation_mode") or "")
    holdout = {int(x) for x in (trace.get("holdout_evaluation_folds") or [])}
    selection = selection_folds_used(trace)
    if mode == "oof_reference_only":
        # Same-fold OOF is by design; not a crossfit holdout leak.
        ok = bool(selection) and selection == holdout
        return {
            "leakage_free": ok,
            "selection_folds": sorted(selection),
            "holdout_folds": sorted(holdout),
            "leaked_folds": [],
            "mode": mode,
            "reason": None if ok else "oof_fold_mismatch",
        }
    leaked = sorted(selection & holdout)
    ok = len(leaked) == 0 and bool(selection) and bool(holdout)
    return {
        "leakage_free": ok,
        "selection_folds": sorted(selection),
        "holdout_folds": sorted(holdout),
        "leaked_folds": leaked,
        "mode": mode,
        "reason": None if ok else ("holdout_in_selection" if leaked else "empty_fold_usage"),
    }
