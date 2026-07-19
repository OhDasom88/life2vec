"""Search-first eligibility, material/control selection, Search-derived direction."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError
from .core_scientific import material


SEARCH_MATERIAL_STABLE = "SEARCH_MATERIAL_STABLE"
SEARCH_CONTROL_NO_MATERIAL = "SEARCH_CONTROL_NO_MATERIAL"
SEARCH_UNSTABLE_OR_MIXED = "SEARCH_UNSTABLE_OR_MIXED"

SELECTED_MATERIAL = "SELECTED_MATERIAL"
SELECTED_CONTROL_NO_MATERIAL = "SELECTED_CONTROL_NO_MATERIAL"
EXCLUDED_NO_MATERIAL = "EXCLUDED_NO_MATERIAL"
EXCLUDED_UNSTABLE_SEARCH = "EXCLUDED_UNSTABLE_SEARCH"
EXCLUDED_DIRECTION_DISAGREEMENT = "EXCLUDED_DIRECTION_DISAGREEMENT"
EXCLUDED_PARTIAL_MATERIAL = "EXCLUDED_PARTIAL_MATERIAL"
EXCLUDED_GATE_FAILURE = "EXCLUDED_GATE_FAILURE"
EXCLUDED_PARENT_GATE_FAILURE = "EXCLUDED_PARENT_GATE_FAILURE"
EXCLUDED_DEPENDENCY_BUDGET = "EXCLUDED_DEPENDENCY_BUDGET"


def classify_search_materiality(
    fold_deltas: Mapping[str, float],
    *,
    required_fold_ids: Sequence[str] = ("0", "1"),
    locked_threshold: float,
) -> Dict[str, Any]:
    """Fold 2/2 materiality. Aggregate is diagnostic only — not used for materiality."""
    vals: List[float] = []
    for fid in required_fold_ids:
        if fid not in fold_deltas:
            return {
                "class": SEARCH_UNSTABLE_OR_MIXED,
                "reason": "MISSING_FOLD",
                "fold_deltas": dict(fold_deltas),
                "aggregate_delta": None,
                "expected_effect_direction": None,
            }
        vals.append(float(fold_deltas[fid]))
    mats = [material(v, locked_threshold=locked_threshold) for v in vals]
    signs = [0 if abs(v) < float(locked_threshold) else (1 if v > 0 else -1) for v in vals]
    aggregate = float(sum(vals) / len(vals))

    if all(not m for m in mats):
        return {
            "class": SEARCH_CONTROL_NO_MATERIAL,
            "reason": "BOTH_FOLDS_BELOW_THRESHOLD",
            "fold_deltas": {fid: float(fold_deltas[fid]) for fid in required_fold_ids},
            "aggregate_delta": aggregate,
            "expected_effect_direction": None,  # control: no direction
        }
    if all(mats) and len(set(signs)) == 1 and signs[0] != 0:
        direction = "RISK_INCREASE" if signs[0] > 0 else "RISK_DECREASE"
        return {
            "class": SEARCH_MATERIAL_STABLE,
            "reason": "BOTH_FOLDS_MATERIAL_SAME_SIGN",
            "fold_deltas": {fid: float(fold_deltas[fid]) for fid in required_fold_ids},
            "aggregate_delta": aggregate,
            "expected_effect_direction": direction,
        }
    if any(mats) and not all(mats):
        return {
            "class": SEARCH_UNSTABLE_OR_MIXED,
            "reason": "PARTIAL_MATERIAL",
            "fold_deltas": {fid: float(fold_deltas[fid]) for fid in required_fold_ids},
            "aggregate_delta": aggregate,
            "expected_effect_direction": None,
        }
    if all(mats) and len(set(signs)) > 1:
        return {
            "class": SEARCH_UNSTABLE_OR_MIXED,
            "reason": "DIRECTION_DISAGREEMENT",
            "fold_deltas": {fid: float(fold_deltas[fid]) for fid in required_fold_ids},
            "aggregate_delta": aggregate,
            "expected_effect_direction": None,
        }
    return {
        "class": SEARCH_UNSTABLE_OR_MIXED,
        "reason": "UNSTABLE_OR_MIXED",
        "fold_deltas": {fid: float(fold_deltas[fid]) for fid in required_fold_ids},
        "aggregate_delta": aggregate,
        "expected_effect_direction": None,
    }


def disposition_from_search_class(search_class: str) -> str:
    if search_class == SEARCH_MATERIAL_STABLE:
        return SELECTED_MATERIAL  # provisional until budget
    if search_class == SEARCH_CONTROL_NO_MATERIAL:
        return SELECTED_CONTROL_NO_MATERIAL  # provisional until budget
    if search_class == SEARCH_UNSTABLE_OR_MIXED:
        return EXCLUDED_UNSTABLE_SEARCH
    raise CoreContractError(f"unknown search class: {search_class}")


def refine_exclusion_reason(search_class: str, reason: Optional[str]) -> str:
    if reason == "PARTIAL_MATERIAL":
        return EXCLUDED_PARTIAL_MATERIAL
    if reason == "DIRECTION_DISAGREEMENT":
        return EXCLUDED_DIRECTION_DISAGREEMENT
    if search_class == SEARCH_UNSTABLE_OR_MIXED:
        return EXCLUDED_UNSTABLE_SEARCH
    return EXCLUDED_UNSTABLE_SEARCH


def select_after_search(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    locked_threshold: float,
    max_pair_budget: int = 10,
    max_control: int = 1,
) -> Dict[str, Any]:
    """
    Search all pairs first → eligibility → select material then at most one control.
    Each pair_row must include: candidate_id, event_ids, fold_deltas (search folds).
    Input order must not affect manifest SHA (canonical sort).
    """
    evaluated: List[Dict[str, Any]] = []
    for row in pair_rows:
        cls = classify_search_materiality(
            row["fold_deltas"],
            locked_threshold=locked_threshold,
        )
        entry = {
            "candidate_id": str(row["candidate_id"]),
            "event_ids": list(row["event_ids"]),
            "kind": row.get("kind") or "PAIR",
            "search_class": cls["class"],
            "search_reason": cls["reason"],
            "fold_deltas": cls["fold_deltas"],
            "aggregate_delta": cls["aggregate_delta"],
            "expected_effect_direction": cls["expected_effect_direction"],
            "transaction_sha": row.get("transaction_sha"),
        }
        evaluated.append(entry)

    # Canonical order for deterministic ranking
    evaluated.sort(
        key=lambda e: (
            0 if e["search_class"] == SEARCH_MATERIAL_STABLE else 1,
            -abs(float(e["aggregate_delta"] or 0.0)),
            str(e["candidate_id"]),
        )
    )

    selected: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    control_slots = int(max_control)

    for e in evaluated:
        if e["search_class"] == SEARCH_MATERIAL_STABLE:
            if len([s for s in selected if s["disposition"] == SELECTED_MATERIAL]) >= max_pair_budget:
                excluded.append({**e, "disposition": EXCLUDED_DEPENDENCY_BUDGET, "exclusion_reason": EXCLUDED_DEPENDENCY_BUDGET})
                continue
            selected.append({**e, "disposition": SELECTED_MATERIAL, "exclusion_reason": None})
        elif e["search_class"] == SEARCH_CONTROL_NO_MATERIAL:
            if control_slots > 0 and len(selected) < max_pair_budget:
                selected.append(
                    {
                        **e,
                        "disposition": SELECTED_CONTROL_NO_MATERIAL,
                        "exclusion_reason": None,
                    }
                )
                control_slots -= 1
            else:
                excluded.append(
                    {
                        **e,
                        "disposition": EXCLUDED_NO_MATERIAL,
                        "exclusion_reason": EXCLUDED_NO_MATERIAL,
                    }
                )
        else:
            disp = refine_exclusion_reason(e["search_class"], e.get("search_reason"))
            excluded.append({**e, "disposition": disp, "exclusion_reason": disp})

    # Re-sort selected for stable SHA (material first, then control, then id)
    selected.sort(
        key=lambda e: (
            0 if e["disposition"] == SELECTED_MATERIAL else 1,
            str(e["candidate_id"]),
        )
    )
    excluded.sort(key=lambda e: str(e["candidate_id"]))

    manifest = {
        "version": "CF1S_SELECTION_MANIFEST_V1",
        "locked_threshold": float(locked_threshold),
        "max_pair_budget": int(max_pair_budget),
        "max_control": int(max_control),
        "evaluated": sorted(evaluated, key=lambda e: str(e["candidate_id"])),
        "selected": selected,
        "excluded": excluded,
        "selected_candidate_ids": [s["candidate_id"] for s in selected],
        "execution_scope": "TWO_EVENT_ONLY",
        "three_event_execution_status": "OUT_OF_SCOPE",
    }
    return {
        **manifest,
        "selection_manifest_sha256": canonical_json_sha256(manifest),
    }


def classify_control_reeval(
    *,
    reeval_deltas: Mapping[str, float],
    locked_threshold: float,
    required_fold_ids: Sequence[str] = ("2",),
) -> Dict[str, Any]:
    """Control branch: no directed/incremental claims."""
    vals = []
    for fid in required_fold_ids:
        if fid not in reeval_deltas:
            return {
                "control_status": "CONTROL_NOT_EVALUABLE",
                "development_scientific_result": "NOT_EVALUABLE",
                "reason": "CONTROL_NOT_EVALUABLE",
                "highest_claim_level": "NONE",
            }
        vals.append(float(reeval_deltas[fid]))
    if any(material(v, locked_threshold=locked_threshold) for v in vals):
        return {
            "control_status": "CONTROL_BECAME_MATERIAL_IN_REEVALUATION",
            "development_scientific_result": "INCONCLUSIVE",
            "reason": "CONTROL_BECAME_MATERIAL_IN_REEVALUATION",
            "highest_claim_level": "NONE",
        }
    return {
        "control_status": "CONTROL_NO_MATERIAL_REPLICATED",
        "development_scientific_result": "NOT_SUPPORTED",
        "reason": "NO_MATERIAL_EFFECT_DETECTED_ABOVE_LOCKED_THRESHOLD",
        "highest_claim_level": "NONE",
    }


def case_coverage_eligible(
    *,
    baseline_identity_valid: bool,
    selected_bundles: Sequence[Mapping[str, Any]],
    parent_evidence_complete: bool,
    search_folds_complete: bool,
    reeval_folds_complete: bool,
    candidate_ledger_complete: bool,
    trace_and_lock_valid: bool,
) -> bool:
    if not baseline_identity_valid:
        return False
    ok_disp = {SELECTED_MATERIAL, SELECTED_CONTROL_NO_MATERIAL}
    has_bundle = any(
        b.get("disposition") in ok_disp and len(b.get("event_ids") or []) == 2
        for b in selected_bundles
    )
    return bool(
        has_bundle
        and parent_evidence_complete
        and search_folds_complete
        and reeval_folds_complete
        and candidate_ledger_complete
        and trace_and_lock_valid
    )
