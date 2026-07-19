"""Cohort and selector feasibility status mapping for CF-1S."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


SUPPORTED = "SUPPORTED"
LIMITED = "LIMITED"
NOT_SUPPORTED = "NOT_SUPPORTED"
NOT_EVALUABLE = "NOT_EVALUABLE"

PRIMARY_SELECTOR = "SALIENCY_TOP_K"


def derive_selector_feasibility_status(
    *,
    model_effects_evaluated: bool,
    n_evaluable_composites: int,
    n_stable_model_sensitivity: int,
    identity_and_determinism_ok: bool,
    saliency_required_and_missing: bool = False,
    n_stable_incremental_multi_event: int = 0,
    n_stable_multi_event_only: int = 0,
    n_stable_multi_event: int = 0,
    n_stable_single_event: int = 0,
) -> str:
    """Map evaluation evidence to feasibility status for one selector.

    Primary SUPPORTED requires stable incremental or stable multi-event-only.
    Stable multi-event without incremental evidence → LIMITED.
    Single-only stable never rescues primary (NOT_SUPPORTED if that is all).
    """
    if saliency_required_and_missing:
        return NOT_EVALUABLE
    if not identity_and_determinism_ok:
        return NOT_EVALUABLE
    if not model_effects_evaluated:
        # Fixture / no real critic scores → not a sensitivity result.
        return NOT_EVALUABLE
    if int(n_evaluable_composites) <= 0:
        return NOT_EVALUABLE
    if int(n_stable_incremental_multi_event) > 0 or int(n_stable_multi_event_only) > 0:
        return SUPPORTED
    if int(n_stable_multi_event) > 0:
        return LIMITED
    if int(n_stable_model_sensitivity) > 0 and int(n_stable_single_event) > 0:
        # Single-only stable cannot support primary.
        if int(n_stable_multi_event) == 0:
            return NOT_SUPPORTED
    if int(n_stable_model_sensitivity) > 0:
        return LIMITED
    # Real evaluation completed with zero stable effect.
    return NOT_SUPPORTED


def primary_feasibility_from_selectors(
    feasibility_status_by_selector: Mapping[str, str],
    *,
    saliency_evaluable: bool = True,
    exploratory_from_baselines: Optional[str] = None,
) -> Dict[str, Any]:
    """Primary feasibility is SALIENCY_TOP_K only; baselines cannot rescue."""
    by_sel = {str(k).upper(): str(v) for k, v in feasibility_status_by_selector.items()}
    if not saliency_evaluable:
        primary = NOT_EVALUABLE
    else:
        primary = by_sel.get(PRIMARY_SELECTOR, NOT_EVALUABLE)

    exploratory = exploratory_from_baselines
    if exploratory is None:
        for sel in ("RANDOM_EDITABLE_LOCUS", "RECENCY_TOP_K"):
            st = by_sel.get(sel)
            if st in (SUPPORTED, LIMITED, NOT_SUPPORTED):
                exploratory = st
                break
        if exploratory is None:
            exploratory = by_sel.get("RANDOM_EDITABLE_LOCUS") or by_sel.get(
                "RECENCY_TOP_K"
            )

    return {
        "primary_feasibility_selector": PRIMARY_SELECTOR,
        "primary_multi_event_edit_feasibility_status": primary,
        "feasibility_status_by_selector": by_sel,
        "selector_agnostic_exploratory_feasibility_status": exploratory,
        "baseline_success_can_rescue_primary_feasibility": False,
        "random_role": "BASELINE_ONLY",
        "recency_role": "BASELINE_ONLY",
    }


def control_like_authority(
    *,
    primary_control_like_status: str,
    observational_sensitivity_status: str,
) -> Dict[str, Any]:
    return {
        "primary_control_like_feasibility_status": primary_control_like_status,
        "observational_sensitivity_status": observational_sensitivity_status,
        "overall_control_like_status_rescued_by_observational": False,
    }


def overall_cf1s_status(
    *,
    problem20_blind_status: str,
    primary_internal_validation_32_status: str,
    development_3_status: str,
    problem20_min_evaluable_met: bool = True,
) -> Dict[str, Any]:
    """Overall authority: Problem20 blind first; development never drives overall."""
    if not problem20_min_evaluable_met:
        overall = NOT_EVALUABLE
    else:
        p20 = str(problem20_blind_status)
        if p20 == SUPPORTED:
            overall = SUPPORTED
        elif p20 == LIMITED:
            overall = LIMITED
        elif p20 == NOT_SUPPORTED:
            overall = NOT_SUPPORTED
        else:
            overall = NOT_EVALUABLE

    generalization = None
    if (
        str(primary_internal_validation_32_status) == SUPPORTED
        and str(problem20_blind_status) == NOT_SUPPORTED
    ):
        generalization = "INTERNAL_TO_BLIND_NOT_REPLICATED"
    elif (
        str(primary_internal_validation_32_status) != str(problem20_blind_status)
        and str(problem20_blind_status) not in (NOT_EVALUABLE,)
    ):
        generalization = "INTERNAL_BLIND_STATUS_CONFLICT"

    return {
        "development_3_status": development_3_status,
        "primary_internal_validation_32_status": primary_internal_validation_32_status,
        "problem20_blind_status": problem20_blind_status,
        "overall_cf1s_status": overall,
        "generalization_status": generalization,
        "development_excluded_from_overall": True,
        "primary_authority": "PROBLEM20_BLIND_x_CONTROL_LIKE_x_SALIENCY_TOP_K",
    }


def saliency_efficiency_status(
    *,
    saliency_evaluable: bool,
    saliency_metric: float,
    random_metric: float,
    recency_metric: float,
    min_gain: float = 0.0,
) -> str:
    if not saliency_evaluable:
        return NOT_EVALUABLE
    if float(saliency_metric) > max(float(random_metric), float(recency_metric)) + float(
        min_gain
    ):
        return "GAIN"
    return "NO_EFFICIENCY_GAIN"
