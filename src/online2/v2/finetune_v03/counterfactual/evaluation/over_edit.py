"""Normal-case over-edit audit metrics (GT vs route normal separated)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def _safe_rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def summarize_over_edit(
    case_results: Sequence[Mapping[str, Any]],
    *,
    normal_key: str = "is_normal",
) -> Dict[str, Any]:
    """Aggregate over-edit rates for a set of normal-case pipeline results.

    A non-NO_OP final selection on a normal case counts as over-edit by default,
    regardless of small risk decreases.
    """
    n_cases = len(case_results)
    n_with_candidates = 0
    n_edit_selected = 0
    n_harmful = 0
    n_immaterial = 0
    token_edits = []

    for row in case_results:
        if not bool(row.get(normal_key, True)):
            continue
        cands = row.get("candidates") or []
        if cands:
            n_with_candidates += 1
        selected = row.get("selected") or row.get("best_candidate")
        if selected is None:
            continue
        is_noop = bool(
            selected.get("is_noop")
            or str(selected.get("source") or "").startswith("noop")
            or str(selected.get("operation") or "") == "NO_OP"
        )
        if is_noop:
            continue
        n_edit_selected += 1
        token_edits.append(int(selected.get("token_edit_count") or selected.get("n_token_edits") or 0))
        dr = selected.get("delta_r_holdout")
        if dr is None:
            dr = selected.get("delta_r_search")
        if dr is not None and float(dr) > 0:
            n_harmful += 1
        elif dr is not None and abs(float(dr)) < 0.01:
            n_immaterial += 1

    return {
        "n_normal_cases": n_cases,
        "normal_candidate_rate": _safe_rate(n_with_candidates, n_cases),
        "normal_edit_selection_rate": _safe_rate(n_edit_selected, n_cases),
        "normal_harmful_given_edit_rate": _safe_rate(n_harmful, n_edit_selected),
        "normal_immaterial_given_edit_rate": _safe_rate(n_immaterial, n_edit_selected),
        "normal_over_edit_rate": _safe_rate(n_edit_selected, n_cases),
        "mean_token_edit_count_normal": (
            float(sum(token_edits) / len(token_edits)) if token_edits else None
        ),
        "n_edit_selected": n_edit_selected,
    }


def build_over_edit_report(
    *,
    gt_normal_results: Sequence[Mapping[str, Any]],
    route_normal_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "over_edit_gt_normal": summarize_over_edit(gt_normal_results),
        "over_edit_route_normal": summarize_over_edit(route_normal_results),
    }
