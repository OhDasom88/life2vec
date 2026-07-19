"""Machine-readable CF-1S Core scientific acceptance engine."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .core_contract import fold_median, required_strict_majority_count, sign_with_threshold


def _as_floats(xs: Sequence[float]) -> List[float]:
    return [float(x) for x in xs]


def evaluate_stable_effect_scope(
    deltas: Sequence[float],
    *,
    effect_threshold: float,
) -> Dict[str, Any]:
    vals = _as_floats(deltas)
    n = len(vals)
    required = required_strict_majority_count(n)
    if n == 0:
        return {
            "stable_effect_scope": False,
            "aggregate_delta": 0.0,
            "aggregate_sign": 0,
            "directional_support_count": 0,
            "required_count": required,
            "requested_fold_count": 0,
        }
    aggregate = fold_median(vals)
    agg_sign = sign_with_threshold(aggregate, threshold=effect_threshold)
    directional = 0
    for v in vals:
        if abs(v) >= effect_threshold and sign_with_threshold(v, threshold=effect_threshold) == agg_sign:
            directional += 1
    ok = agg_sign != 0 and directional >= required
    return {
        "stable_effect_scope": ok,
        "aggregate_delta": aggregate,
        "aggregate_sign": agg_sign,
        "directional_support_count": directional,
        "required_count": required,
        "requested_fold_count": n,
    }


def evaluate_stable_bundle(
    search_deltas: Sequence[float],
    holdout_deltas: Sequence[float],
    *,
    effect_threshold: float,
) -> Dict[str, Any]:
    search = evaluate_stable_effect_scope(search_deltas, effect_threshold=effect_threshold)
    holdout = evaluate_stable_effect_scope(holdout_deltas, effect_threshold=effect_threshold)
    sign_agree = (
        search["aggregate_sign"] != 0
        and holdout["aggregate_sign"] != 0
        and search["aggregate_sign"] == holdout["aggregate_sign"]
    )
    stable = bool(search["stable_effect_scope"] and holdout["stable_effect_scope"] and sign_agree)
    return {
        "stable_bundle": stable,
        "search": search,
        "holdout": holdout,
        "aggregate_sign_agreement": sign_agree,
    }


def select_best_parents(
    parent_abs_deltas: Mapping[str, float],
) -> Dict[str, Any]:
    if not parent_abs_deltas:
        return {
            "best_parent_ids": [],
            "representative_best_parent_id": None,
            "best_parent_tie": False,
            "max_parent_abs_delta": 0.0,
        }
    max_abs = max(float(v) for v in parent_abs_deltas.values())
    tied = sorted(
        [pid for pid, v in parent_abs_deltas.items() if float(v) == float(max_abs)],
        key=lambda x: str(x),
    )
    return {
        "best_parent_ids": tied,
        "representative_best_parent_id": tied[0] if tied else None,
        "best_parent_tie": len(tied) > 1,
        "max_parent_abs_delta": float(max_abs),
    }


def incremental_gain_fold(
    bundle_delta: float,
    parent_deltas: Mapping[str, float],
) -> Dict[str, Any]:
    abs_parents = {str(k): abs(float(v)) for k, v in parent_deltas.items()}
    best = select_best_parents(abs_parents)
    gain = abs(float(bundle_delta)) - float(best["max_parent_abs_delta"])
    bundle_sign = 1 if bundle_delta > 0 else (-1 if bundle_delta < 0 else 0)
    flips = []
    for pid in best["best_parent_ids"]:
        pd = float(parent_deltas[pid])
        psign = 1 if pd > 0 else (-1 if pd < 0 else 0)
        flips.append(bool(bundle_sign != 0 and psign != 0 and bundle_sign != psign))
    return {
        "incremental_gain": float(gain),
        **best,
        "parent_bundle_direction_flip_any": any(flips) if flips else False,
        "parent_bundle_direction_flip_all": all(flips) if flips else False,
    }


def evaluate_stable_incremental_scope(
    bundle_deltas: Sequence[float],
    parent_deltas_by_fold: Sequence[Mapping[str, float]],
    *,
    effect_threshold: float,
    incremental_threshold: float,
) -> Dict[str, Any]:
    vals = _as_floats(bundle_deltas)
    n = len(vals)
    required = required_strict_majority_count(n)
    if n == 0 or len(parent_deltas_by_fold) != n:
        return {
            "stable_incremental_scope": False,
            "joint_support_count": 0,
            "required_count": required,
            "median_gain": 0.0,
            "requested_fold_count": n,
        }
    effect = evaluate_stable_effect_scope(vals, effect_threshold=effect_threshold)
    agg_sign = int(effect["aggregate_sign"])
    joint = 0
    gains: List[float] = []
    provenance = []
    for i, b in enumerate(vals):
        ginfo = incremental_gain_fold(b, parent_deltas_by_fold[i])
        gains.append(float(ginfo["incremental_gain"]))
        directional = abs(b) >= effect_threshold and sign_with_threshold(
            b, threshold=effect_threshold
        ) == agg_sign
        joint_ok = directional and float(ginfo["incremental_gain"]) >= incremental_threshold
        if joint_ok:
            joint += 1
        provenance.append({"fold_index": i, **ginfo, "joint_incremental_supported": joint_ok})
    median_gain = fold_median(gains) if gains else 0.0
    ok = joint >= required and median_gain >= incremental_threshold and agg_sign != 0
    return {
        "stable_incremental_scope": ok,
        "joint_support_count": joint,
        "required_count": required,
        "median_gain": median_gain,
        "requested_fold_count": n,
        "fold_provenance": provenance,
        "effect": effect,
    }


def evaluate_stable_incremental(
    search_bundle: Sequence[float],
    search_parents: Sequence[Mapping[str, float]],
    holdout_bundle: Sequence[float],
    holdout_parents: Sequence[Mapping[str, float]],
    *,
    effect_threshold: float,
    incremental_threshold: float,
) -> Dict[str, Any]:
    search = evaluate_stable_incremental_scope(
        search_bundle,
        search_parents,
        effect_threshold=effect_threshold,
        incremental_threshold=incremental_threshold,
    )
    holdout = evaluate_stable_incremental_scope(
        holdout_bundle,
        holdout_parents,
        effect_threshold=effect_threshold,
        incremental_threshold=incremental_threshold,
    )
    return {
        "stable_incremental": bool(
            search["stable_incremental_scope"] and holdout["stable_incremental_scope"]
        ),
        "search": search,
        "holdout": holdout,
    }


def evaluate_multi_event_only_scope(
    bundle_deltas: Sequence[float],
    single_parent_deltas_by_fold: Sequence[Mapping[str, float]],
    *,
    effect_threshold: float,
) -> Dict[str, Any]:
    vals = _as_floats(bundle_deltas)
    n = len(vals)
    required = required_strict_majority_count(n)
    if n == 0 or len(single_parent_deltas_by_fold) != n:
        return {
            "stable_multi_event_only_scope": False,
            "joint_support_count": 0,
            "required_count": required,
            "requested_fold_count": n,
        }
    effect = evaluate_stable_effect_scope(vals, effect_threshold=effect_threshold)
    agg_sign = int(effect["aggregate_sign"])
    joint = 0
    for i, b in enumerate(vals):
        directional = abs(b) >= effect_threshold and sign_with_threshold(
            b, threshold=effect_threshold
        ) == agg_sign
        parents_ok = all(
            abs(float(v)) < effect_threshold for v in single_parent_deltas_by_fold[i].values()
        )
        if directional and parents_ok:
            joint += 1
    # aggregate condition
    parent_ids = sorted({k for m in single_parent_deltas_by_fold for k in m})
    parent_aggs = {}
    for pid in parent_ids:
        series = [float(single_parent_deltas_by_fold[i].get(pid, 0.0)) for i in range(n)]
        parent_aggs[pid] = fold_median(series)
    aggregate_ok = abs(float(effect["aggregate_delta"])) >= effect_threshold and all(
        abs(v) < effect_threshold for v in parent_aggs.values()
    )
    ok = joint >= required and aggregate_ok and agg_sign != 0
    return {
        "stable_multi_event_only_scope": ok,
        "joint_support_count": joint,
        "required_count": required,
        "requested_fold_count": n,
        "multi_event_only_aggregate_scope": aggregate_ok,
        "parent_aggregate_deltas": parent_aggs,
        "effect": effect,
    }


def evaluate_stable_multi_event_only(
    search_bundle: Sequence[float],
    search_singles: Sequence[Mapping[str, float]],
    holdout_bundle: Sequence[float],
    holdout_singles: Sequence[Mapping[str, float]],
    *,
    effect_threshold: float,
) -> Dict[str, Any]:
    search = evaluate_multi_event_only_scope(
        search_bundle, search_singles, effect_threshold=effect_threshold
    )
    holdout = evaluate_multi_event_only_scope(
        holdout_bundle, holdout_singles, effect_threshold=effect_threshold
    )
    return {
        "stable_multi_event_only": bool(
            search["stable_multi_event_only_scope"] and holdout["stable_multi_event_only_scope"]
        ),
        "search": search,
        "holdout": holdout,
    }


def evaluate_candidate_status(
    *,
    execution_complete: bool,
    parent_complete: bool,
    fold_execution_failed: bool,
    search_bundle_deltas: Sequence[float],
    holdout_bundle_deltas: Sequence[float],
    search_incremental_parents: Sequence[Mapping[str, float]],
    holdout_incremental_parents: Sequence[Mapping[str, float]],
    search_single_parents: Sequence[Mapping[str, float]],
    holdout_single_parents: Sequence[Mapping[str, float]],
    effect_threshold: float,
    incremental_threshold: float,
    n_events: int,
) -> Dict[str, Any]:
    if (not execution_complete) or (not parent_complete) or fold_execution_failed:
        return {
            "candidate_scientific_status": "NOT_EVALUABLE",
            "stable_bundle": False,
            "stable_incremental": False,
            "stable_multi_event_only": False,
            "reason": "EXECUTION_OR_PARENT_INCOMPLETE",
        }
    if int(n_events) < 2:
        return {
            "candidate_scientific_status": "NOT_SUPPORTED",
            "stable_bundle": False,
            "stable_incremental": False,
            "stable_multi_event_only": False,
            "reason": "SINGLE_ONLY_NOT_PROMOTED",
        }
    bundle = evaluate_stable_bundle(
        search_bundle_deltas,
        holdout_bundle_deltas,
        effect_threshold=effect_threshold,
    )
    incremental = evaluate_stable_incremental(
        search_bundle_deltas,
        search_incremental_parents,
        holdout_bundle_deltas,
        holdout_incremental_parents,
        effect_threshold=effect_threshold,
        incremental_threshold=incremental_threshold,
    )
    meo = evaluate_stable_multi_event_only(
        search_bundle_deltas,
        search_single_parents,
        holdout_bundle_deltas,
        holdout_single_parents,
        effect_threshold=effect_threshold,
    )
    if bundle["stable_bundle"] and (
        incremental["stable_incremental"] or meo["stable_multi_event_only"]
    ):
        status = "SUPPORTED"
    elif bundle["stable_bundle"]:
        status = "LIMITED"
    else:
        status = "NOT_SUPPORTED"
    return {
        "candidate_scientific_status": status,
        "stable_bundle": bundle["stable_bundle"],
        "stable_incremental": incremental["stable_incremental"],
        "stable_multi_event_only": meo["stable_multi_event_only"],
        "bundle": bundle,
        "incremental": incremental,
        "multi_event_only": meo,
    }


def evaluate_case_status(candidate_statuses: Sequence[str]) -> str:
    statuses = [str(s) for s in candidate_statuses]
    evaluable = [s for s in statuses if s != "NOT_EVALUABLE"]
    if not evaluable:
        return "NOT_EVALUABLE"
    if any(s == "SUPPORTED" for s in evaluable):
        return "SUPPORTED"
    if any(s == "LIMITED" for s in evaluable):
        return "LIMITED"
    return "NOT_SUPPORTED"


def evaluate_case_multi_event_status(
    *,
    two_event_candidate_statuses: Sequence[str],
    three_event_candidate_statuses: Sequence[str],
) -> Dict[str, Any]:
    two = evaluate_case_status(two_event_candidate_statuses)
    three = (
        evaluate_case_status(three_event_candidate_statuses)
        if three_event_candidate_statuses
        else "NOT_EVALUABLE"
    )
    # overall from 2-event; 3-event absence does not force NOT_EVALUABLE
    overall = two
    return {
        "case_two_event_status": two,
        "case_three_event_status": three,
        "case_multi_event_evaluable": two != "NOT_EVALUABLE",
        "case_overall_scientific_status": overall,
        "two_event_family_evaluable": two != "NOT_EVALUABLE",
        "three_event_family_evaluable": three != "NOT_EVALUABLE",
        "two_event_supported": two == "SUPPORTED",
        "three_event_supported": three == "SUPPORTED",
    }


def evaluate_cohort_status(
    case_statuses: Sequence[str],
    *,
    locked_cohort_case_count: int,
    minimum_evaluable_case_count: int,
    minimum_supported_case_count: int,
    minimum_supported_fraction: float,
) -> Dict[str, Any]:
    locked = int(locked_cohort_case_count)
    statuses = [str(s) for s in case_statuses]
    evaluable = [s for s in statuses if s != "NOT_EVALUABLE"]
    supported = [s for s in statuses if s == "SUPPORTED"]
    limited = [s for s in statuses if s == "LIMITED"]
    evaluable_count = len(evaluable)
    supported_count = len(supported)
    evaluable_fraction = evaluable_count / float(locked) if locked else 0.0
    supported_fraction = supported_count / float(locked) if locked else 0.0
    supported_among_evaluable = (
        supported_count / float(evaluable_count) if evaluable_count else 0.0
    )
    if evaluable_count < int(minimum_evaluable_case_count):
        status = "NOT_EVALUABLE"
    elif supported_count >= int(minimum_supported_case_count) and supported_fraction >= float(
        minimum_supported_fraction
    ):
        status = "SUPPORTED"
    elif evaluable_count >= int(minimum_evaluable_case_count) and (
        supported_count > 0 or len(limited) > 0
    ):
        status = "LIMITED"
    else:
        status = "NOT_SUPPORTED"
    return {
        "cohort_scientific_status": status,
        "locked_cohort_case_count": locked,
        "evaluable_case_count": evaluable_count,
        "supported_case_count": supported_count,
        "limited_case_count": len(limited),
        "evaluable_fraction": evaluable_fraction,
        "supported_fraction": supported_fraction,
        "supported_among_evaluable_fraction": supported_among_evaluable,
        "fraction_denominator": "LOCKED_COHORT_CASE_COUNT",
    }


def development_readiness_status(
    *,
    execution_status: str,
    case_statuses: Sequence[str],
    noise_ceiling_pass: bool,
) -> Dict[str, Any]:
    scientific = evaluate_case_status(case_statuses) if case_statuses else "NOT_EVALUABLE"
    # cohort-level scientific for development is separate; readiness uses evaluability
    evaluable = [s for s in case_statuses if s != "NOT_EVALUABLE"]
    ready = (
        execution_status == "PASS"
        and len(evaluable) == 3
        and len(case_statuses) == 3
        and bool(noise_ceiling_pass)
    )
    return {
        "development_execution_status": execution_status,
        "development_scientific_status": scientific,
        "development_readiness_status": "READY" if ready else "BLOCKED",
        "evaluable_case_count": len(evaluable),
        "noise_ceiling_pass": bool(noise_ceiling_pass),
    }
