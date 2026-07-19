"""CF-1S acceptance policy evaluator (candidate → case → cohort)."""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .status import (
    LIMITED,
    NOT_EVALUABLE,
    NOT_SUPPORTED,
    SUPPORTED,
)


def _policy_epsilon(acceptance_policy: Mapping[str, Any]) -> float:
    effect = acceptance_policy.get("effect_fold_aggregation") or {}
    return float(effect.get("effect_pass_epsilon") or 0.01)


def _min_incremental(acceptance_policy: Mapping[str, Any]) -> float:
    inc = acceptance_policy.get("incremental_gain") or {}
    if "min_incremental_abs_gain" in inc:
        return float(inc["min_incremental_abs_gain"])
    return float(_policy_epsilon(acceptance_policy))


def evaluate_candidate_acceptance(
    candidate: Mapping[str, Any],
    *,
    acceptance_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map one candidate's finalized metrics to feasibility contribution."""
    stable = bool(candidate.get("stable_model_sensitivity_valid"))
    inc_pass = bool(candidate.get("stable_incremental_multi_event_pass"))
    me_only = bool(candidate.get("stable_multi_event_only_effect_pass"))
    n_events = int(candidate.get("n_events") or 0)
    parent_ok = bool(candidate.get("parent_family_complete"))
    finalized = bool(candidate.get("finalized_effect_values", True))

    if not finalized or not parent_ok:
        status = NOT_EVALUABLE
    elif inc_pass or me_only:
        status = SUPPORTED
    elif stable and n_events >= 2:
        status = LIMITED
    elif stable and n_events == 1:
        # Single-only stable cannot support primary.
        status = NOT_SUPPORTED
    elif bool(candidate.get("model_effects_evaluated")):
        status = NOT_SUPPORTED
    else:
        status = NOT_EVALUABLE

    return {
        "candidate_id": candidate.get("multi_event_candidate_id")
        or candidate.get("candidate_id"),
        "candidate_acceptance_status": status,
        "stable_incremental_multi_event_pass": inc_pass,
        "stable_multi_event_only_effect_pass": me_only,
        "stable_model_sensitivity_valid": stable,
        "single_only_stable_can_support_primary": False,
        "min_incremental_abs_gain": _min_incremental(acceptance_policy),
    }


def evaluate_selector_acceptance(
    bundles: Sequence[Mapping[str, Any]],
    *,
    acceptance_policy: Mapping[str, Any],
    model_effects_evaluated: bool,
    identity_and_determinism_ok: bool,
    parity_ok: bool,
) -> Dict[str, Any]:
    if not identity_and_determinism_ok or not parity_ok:
        return {
            "feasibility_status": NOT_EVALUABLE,
            "n_stable_single_event": 0,
            "n_stable_multi_event": 0,
            "n_stable_multi_event_only": 0,
            "n_stable_incremental_multi_event": 0,
            "n_supported_candidates": 0,
            "n_limited_candidates": 0,
            "candidate_acceptances": [],
            "reason": "IDENTITY_OR_PARITY_GATE",
        }

    if not model_effects_evaluated:
        return {
            "feasibility_status": NOT_EVALUABLE,
            "n_stable_single_event": 0,
            "n_stable_multi_event": 0,
            "n_stable_multi_event_only": 0,
            "n_stable_incremental_multi_event": 0,
            "n_supported_candidates": 0,
            "n_limited_candidates": 0,
            "candidate_acceptances": [],
            "reason": "MODEL_EFFECTS_NOT_EVALUATED",
        }

    accs = [
        evaluate_candidate_acceptance(b, acceptance_policy=acceptance_policy)
        for b in bundles
    ]
    n_single = sum(
        1
        for b in bundles
        if int(b.get("n_events") or 0) == 1 and b.get("stable_model_sensitivity_valid")
    )
    n_multi = sum(
        1
        for b in bundles
        if int(b.get("n_events") or 0) >= 2 and b.get("stable_model_sensitivity_valid")
    )
    n_me_only = sum(1 for a in accs if a["stable_multi_event_only_effect_pass"])
    n_inc = sum(1 for a in accs if a["stable_incremental_multi_event_pass"])
    n_supported = sum(1 for a in accs if a["candidate_acceptance_status"] == SUPPORTED)
    n_limited = sum(1 for a in accs if a["candidate_acceptance_status"] == LIMITED)

    if n_supported > 0:
        status = SUPPORTED
    elif n_limited > 0:
        status = LIMITED
    elif len(bundles) > 0:
        status = NOT_SUPPORTED
    else:
        status = NOT_EVALUABLE

    return {
        "feasibility_status": status,
        "n_stable_single_event": n_single,
        "n_stable_multi_event": n_multi,
        "n_stable_multi_event_only": n_me_only,
        "n_stable_incremental_multi_event": n_inc,
        "n_supported_candidates": n_supported,
        "n_limited_candidates": n_limited,
        "candidate_acceptances": accs,
        "single_only_stable_can_support_primary": False,
        "acceptance_policy_evaluator_active": True,
    }


def aggregate_random_matched_subset(
    seed_realizations: Sequence[Mapping[str, Any]],
    *,
    matched_budget: int,
) -> Dict[str, Any]:
    """MEDIAN_OF_SEED_MAXIMA / MEAN_OF_SEED_CASE_PASS over matched truncate."""
    maxima: List[float] = []
    case_pass: List[float] = []
    for real in seed_realizations:
        bundles = list(real.get("bundles") or [])
        # Matched truncate: keep top-|search delta| within budget.
        ranked = sorted(
            bundles,
            key=lambda b: -abs(float(b.get("delta_risk_search_aggregate") or 0.0)),
        )[: max(int(matched_budget), 0)]
        stables = [
            b
            for b in ranked
            if b.get("stable_incremental_multi_event_pass")
            or b.get("stable_multi_event_only_effect_pass")
        ]
        maxima.append(float(len(stables)))
        case_pass.append(1.0 if stables else 0.0)
    med_max = float(median(maxima)) if maxima else 0.0
    mean_pass = float(sum(case_pass) / len(case_pass)) if case_pass else 0.0
    return {
        "random_matched_subset_statistics_active": True,
        "matched_budget": int(matched_budget),
        "MEDIAN_OF_SEED_MAXIMA": med_max,
        "MEAN_OF_SEED_CASE_PASS": mean_pass,
        "seed_maxima": maxima,
        "seed_case_pass": case_pass,
    }


def evaluate_cohort_acceptance(
    case_results: Sequence[Mapping[str, Any]],
    *,
    acceptance_policy: Mapping[str, Any],
    cohort: str,
) -> Dict[str, Any]:
    mins = acceptance_policy.get("minimum_evaluable_cases") or {}
    key = {
        "development_3": "development_3",
        "primary_internal_validation_32": "primary_internal_validation_32",
        "problem20_blind": "problem20_blind",
    }.get(cohort, cohort)
    min_eval = int(mins.get(key) or 0)
    evaluable = [
        c
        for c in case_results
        if str(
            c.get("primary_multi_event_edit_feasibility_status")
            or c.get("feasibility_status")
            or NOT_EVALUABLE
        )
        != NOT_EVALUABLE
    ]
    supported = sum(
        1
        for c in evaluable
        if str(
            c.get("primary_multi_event_edit_feasibility_status")
            or c.get("feasibility_status")
        )
        == SUPPORTED
    )
    limited = sum(
        1
        for c in evaluable
        if str(
            c.get("primary_multi_event_edit_feasibility_status")
            or c.get("feasibility_status")
        )
        == LIMITED
    )
    if len(evaluable) < min_eval:
        status = NOT_EVALUABLE
    elif supported > 0:
        status = SUPPORTED
    elif limited > 0:
        status = LIMITED
    elif evaluable:
        status = NOT_SUPPORTED
    else:
        status = NOT_EVALUABLE
    return {
        "cohort": cohort,
        "cohort_acceptance_status": status,
        "n_cases": len(case_results),
        "n_evaluable_cases": len(evaluable),
        "minimum_evaluable_cases": min_eval,
        "n_supported_cases": supported,
        "n_limited_cases": limited,
        "acceptance_policy_evaluator_active": True,
    }
