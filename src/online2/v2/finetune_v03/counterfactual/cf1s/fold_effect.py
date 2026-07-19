"""Fold-level effect aggregation and strict-majority consensus for CF-1S."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Dict, List, Optional, Sequence


def required_strict_majority_count(n_evaluable: int) -> int:
    n = int(n_evaluable)
    if n <= 0:
        return 1
    return (n // 2) + 1


def sign_of(value: float, *, zero_tol: float = 0.001) -> int:
    v = float(value)
    if abs(v) <= float(zero_tol):
        return 0
    return 1 if v > 0 else -1


def aggregate_fold_deltas(
    deltas: Sequence[float],
    *,
    statistic: str = "MEDIAN",
) -> float:
    vals = [float(x) for x in deltas]
    if not vals:
        return 0.0
    if statistic.upper() == "MEDIAN":
        return float(median(vals))
    return float(sum(vals) / len(vals))


def effect_pass_fraction(
    deltas: Sequence[float],
    *,
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    vals = [float(x) for x in deltas]
    if not vals:
        return {
            "evaluable_fold_count": 0,
            "effect_pass_count": 0,
            "effect_pass_fraction": 0.0,
            "required_agreement_count": 1,
            "strict_majority_pass": False,
        }
    passes = sum(1 for v in vals if abs(v) >= float(epsilon))
    n = len(vals)
    required = required_strict_majority_count(n)
    return {
        "evaluable_fold_count": n,
        "effect_pass_count": passes,
        "effect_pass_fraction": passes / float(n),
        "required_agreement_count": required,
        "strict_majority_pass": passes >= required,
        "ties_pass": False,
    }


def sign_consensus_fraction(
    deltas: Sequence[float],
    *,
    zero_sign_tolerance: float = 0.001,
) -> Dict[str, Any]:
    vals = [float(x) for x in deltas]
    signed = [sign_of(v, zero_tol=zero_sign_tolerance) for v in vals]
    evaluable = [s for s in signed if s != 0]
    if not evaluable:
        return {
            "evaluable_fold_count": 0,
            "sign_consensus_fraction": 0.0,
            "required_agreement_count": 1,
            "strict_majority_pass": False,
            "dominant_sign": 0,
            "ties_pass": False,
        }
    pos = sum(1 for s in evaluable if s > 0)
    neg = sum(1 for s in evaluable if s < 0)
    top = max(pos, neg)
    n = len(evaluable)
    required = required_strict_majority_count(n)
    dominant = 1 if pos > neg else (-1 if neg > pos else 0)
    # Exact 0.5 / ties never pass.
    return {
        "evaluable_fold_count": n,
        "sign_consensus_fraction": top / float(n),
        "required_agreement_count": required,
        "strict_majority_pass": top >= required and dominant != 0,
        "dominant_sign": dominant,
        "ties_pass": False,
    }


def stable_model_sensitivity_valid(
    *,
    delta_risk_search_by_fold: Sequence[float],
    delta_risk_holdout_by_fold: Sequence[float],
    effect_pass_epsilon: float = 0.01,
    zero_sign_tolerance: float = 0.001,
    magnitude_retention_min: float = 0.5,
    sign_evaluable_min_abs_delta: float = 0.001,
) -> Dict[str, Any]:
    search_agg = aggregate_fold_deltas(delta_risk_search_by_fold)
    holdout_agg = aggregate_fold_deltas(delta_risk_holdout_by_fold)
    effect = effect_pass_fraction(
        delta_risk_holdout_by_fold, epsilon=effect_pass_epsilon
    )
    signs = sign_consensus_fraction(
        delta_risk_holdout_by_fold, zero_sign_tolerance=zero_sign_tolerance
    )

    if (
        abs(search_agg) < float(sign_evaluable_min_abs_delta)
        or abs(holdout_agg) < float(sign_evaluable_min_abs_delta)
    ):
        sign_agreement_status = "NOT_EVALUABLE"
        sign_agree = False
    else:
        sign_agree = sign_of(search_agg, zero_tol=0.0) == sign_of(
            holdout_agg, zero_tol=0.0
        )
        sign_agreement_status = "AGREE" if sign_agree else "DISAGREE"

    denom = max(abs(search_agg), 0.001)
    retention = abs(holdout_agg) / denom
    stable = (
        abs(holdout_agg) >= float(effect_pass_epsilon)
        and bool(effect["strict_majority_pass"])
        and bool(signs["strict_majority_pass"])
        and sign_agree
        and retention >= float(magnitude_retention_min)
    )
    return {
        "delta_risk_search_aggregate": search_agg,
        "delta_risk_holdout_aggregate": holdout_agg,
        "search_model_sensitivity_valid": abs(search_agg) >= float(effect_pass_epsilon),
        "holdout_model_sensitivity_valid": abs(holdout_agg) >= float(effect_pass_epsilon),
        "stable_model_sensitivity_valid": bool(stable),
        "holdout_effect_pass_fraction": effect["effect_pass_fraction"],
        "holdout_sign_consensus_fraction": signs["sign_consensus_fraction"],
        "holdout_effect_strict_majority": bool(effect["strict_majority_pass"]),
        "holdout_sign_strict_majority": bool(signs["strict_majority_pass"]),
        "magnitude_retention": retention,
        "sign_agreement_status": sign_agreement_status,
        "required_effect_agreement_count": effect["required_agreement_count"],
        "required_sign_agreement_count": signs["required_agreement_count"],
        "fold_ties_pass": False,
    }


def multi_event_only_effect_pass(
    *,
    bundle_abs_delta: float,
    atomic_parent_abs_deltas: Sequence[float],
    epsilon: float = 0.01,
) -> bool:
    if abs(float(bundle_abs_delta)) < float(epsilon):
        return False
    return all(abs(float(a)) < float(epsilon) for a in atomic_parent_abs_deltas)


def incremental_abs_gain_by_fold(
    *,
    bundle_deltas_by_fold: Sequence[float],
    atomic_parent_deltas_by_fold: Sequence[Sequence[float]],
) -> List[float]:
    """Per-fold abs(bundle) - max(abs(atomic parents))."""
    gains: List[float] = []
    for i, b in enumerate(bundle_deltas_by_fold):
        parents = []
        for parent_series in atomic_parent_deltas_by_fold:
            if i < len(parent_series):
                parents.append(abs(float(parent_series[i])))
        max_parent = max(parents) if parents else 0.0
        gains.append(abs(float(b)) - float(max_parent))
    return gains


def incremental_gain_pass(
    *,
    incremental_abs_gain_by_fold: Sequence[float],
    min_incremental_abs_gain: float,
    require_strict_majority: bool,
) -> Dict[str, Any]:
    vals = [float(x) for x in incremental_abs_gain_by_fold]
    if not vals:
        return {
            "incremental_abs_gain_aggregate": 0.0,
            "pass_count": 0,
            "required_count": 1,
            "strict_majority": False,
            "median_pass": False,
            "incremental_gain_pass": False,
        }
    agg = float(median(vals))
    passes = sum(1 for v in vals if v >= float(min_incremental_abs_gain))
    required = required_strict_majority_count(len(vals))
    median_ok = agg >= float(min_incremental_abs_gain)
    majority_ok = passes >= required if require_strict_majority else True
    return {
        "incremental_abs_gain_aggregate": agg,
        "pass_count": passes,
        "required_count": required,
        "strict_majority": bool(majority_ok),
        "median_pass": bool(median_ok),
        "incremental_gain_pass": bool(median_ok and majority_ok),
    }


def multi_event_only_fold_pass(
    *,
    bundle_deltas_by_fold: Sequence[float],
    atomic_parent_deltas_by_fold: Sequence[Sequence[float]],
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """Fold-level multi-event-only with strict majority + aggregate median check."""
    vals = [float(x) for x in bundle_deltas_by_fold]
    if not vals:
        return {
            "pass_count": 0,
            "required_count": 1,
            "strict_majority": False,
            "aggregate_pass": False,
            "multi_event_only_pass": False,
            "fold_pass_flags": [],
        }
    flags = []
    for i, b in enumerate(vals):
        parents = []
        for parent_series in atomic_parent_deltas_by_fold:
            if i < len(parent_series):
                parents.append(abs(float(parent_series[i])))
        flags.append(
            multi_event_only_effect_pass(
                bundle_abs_delta=abs(b),
                atomic_parent_abs_deltas=parents or [0.0],
                epsilon=epsilon,
            )
        )
    passes = sum(1 for f in flags if f)
    required = required_strict_majority_count(len(vals))
    bundle_med = abs(aggregate_fold_deltas(vals))
    parent_meds = []
    for parent_series in atomic_parent_deltas_by_fold:
        parent_meds.append(abs(aggregate_fold_deltas(parent_series)))
    aggregate_pass = multi_event_only_effect_pass(
        bundle_abs_delta=bundle_med,
        atomic_parent_abs_deltas=parent_meds or [0.0],
        epsilon=epsilon,
    )
    majority = passes >= required
    return {
        "pass_count": passes,
        "required_count": required,
        "strict_majority": bool(majority),
        "aggregate_pass": bool(aggregate_pass),
        "multi_event_only_pass": bool(majority and aggregate_pass),
        "fold_pass_flags": flags,
    }


def stable_incremental_multi_event_pass(
    *,
    stable_model_sensitivity_valid_flag: bool,
    parent_completion_search_complete: bool,
    parent_completion_holdout_complete: bool,
    search_incremental_gain_pass: bool,
    holdout_incremental_gain_pass: bool,
) -> bool:
    return bool(
        stable_model_sensitivity_valid_flag
        and parent_completion_search_complete
        and parent_completion_holdout_complete
        and search_incremental_gain_pass
        and holdout_incremental_gain_pass
    )


def stable_multi_event_only_effect_pass_fold(
    *,
    stable_model_sensitivity_valid_flag: bool,
    parent_completion_search_complete: bool,
    parent_completion_holdout_complete: bool,
    search_multi_event_only_pass: bool,
    holdout_multi_event_only_pass: bool,
) -> bool:
    return bool(
        stable_model_sensitivity_valid_flag
        and parent_completion_search_complete
        and parent_completion_holdout_complete
        and search_multi_event_only_pass
        and holdout_multi_event_only_pass
    )
