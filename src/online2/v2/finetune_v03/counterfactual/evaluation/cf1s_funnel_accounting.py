"""CF-1S funnel accounting with grounding/inversion/Gate0/Gate4 and budget checks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Tuple


def empty_cf1s_funnel_counts() -> Dict[str, int]:
    return {
        "n_all_events": 0,
        "n_edit_class_eligible": 0,
        "n_forbidden_outcome_excluded": 0,
        "n_raw_mg_reconstructable": 0,
        "n_after_distinct_mg_dedup": 0,
        "n_feature_edit_profile_valid": 0,
        "n_atomic_schema_edit_possible": 0,
        "n_common_eligible_loci": 0,
        "n_selector_ranked": 0,
        "n_round_robin_selected": 0,
        "n_atomic_candidates": 0,
        "n_grounding_valid": 0,
        "n_inversion_valid": 0,
        "n_gate0_pass": 0,
        "n_gate4_pass": 0,
        "n_single_screened": 0,
        "n_multi_event_composites": 0,
        "n_two_event_composites": 0,
        "n_three_event_composites": 0,
        "n_total_selected": 0,
        "n_length_position_valid": 0,
        "n_parity_audited": 0,
        "n_search_effect_evaluated": 0,
        "n_holdout_effect_evaluated": 0,
        "n_stable_model_sensitivity": 0,
        "grounding_failure_count": 0,
        "inversion_failure_count": 0,
        "gate0_failure_count": 0,
        "gate4_failure_count": 0,
        "locus_level_skip_count": 0,
        "atomic_rejection_count": 0,
        "composite_rejection_count": 0,
        "case_execution_error_count": 0,
        "raw_mg_duplicate_locus_count": 0,
        "raw_mg_duplicate_detected_count": 0,
        "raw_mg_duplicate_remaining_count": 0,
        "cross_profile_bundle_count": 0,
        "forbidden_outcome_edit_count": 0,
        "holdout_reranking_count": 0,
        "parent_completion_used_for_selection_count": 0,
        "budget_two_event_exceeded": 0,
        "budget_three_event_exceeded": 0,
        "budget_total_exceeded": 0,
    }


def merge_universe_counts(
    funnel: MutableMapping[str, int],
    universe_counts: Mapping[str, Any],
) -> None:
    for key in (
        "n_all_events",
        "n_edit_class_eligible",
        "n_forbidden_outcome_excluded",
        "n_raw_mg_reconstructable",
        "n_after_distinct_mg_dedup",
        "n_feature_edit_profile_valid",
        "n_atomic_schema_edit_possible",
        "n_common_eligible_loci",
        "raw_mg_duplicate_locus_count",
        "raw_mg_duplicate_detected_count",
        "raw_mg_duplicate_remaining_count",
        "forbidden_outcome_edit_count",
        "cross_profile_bundle_count",
    ):
        if key in universe_counts:
            funnel[key] = int(universe_counts[key] or 0)


def validate_cf1s_funnel_monotonicity(
    funnel: Mapping[str, Any],
    *,
    two_event_max: int = 32,
    three_event_max: int = 24,
    total_max: int = 100,
    require_single_screening: bool = True,
) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    chain = (
        "n_common_eligible_loci",
        "n_selector_ranked",
        "n_round_robin_selected",
        "n_atomic_candidates",
        "n_grounding_valid",
        "n_inversion_valid",
        "n_gate0_pass",
        "n_gate4_pass",
    )
    vals = [int(funnel.get(k) or 0) for k in chain]
    for a, b, ka, kb in zip(vals, vals[1:], chain, chain[1:]):
        if ka == "n_common_eligible_loci" and kb == "n_selector_ranked":
            if b > a:
                errs.append(f"{kb}({b}) > {ka}({a})")
            continue
        if a < b:
            errs.append(f"{kb}({b}) > {ka}({a})")

    if int(funnel.get("raw_mg_duplicate_remaining_count") or 0) != 0:
        errs.append("raw_mg_duplicate_remaining_count != 0")
    if int(funnel.get("holdout_reranking_count") or 0) != 0:
        errs.append("holdout_reranking_count != 0")
    if int(funnel.get("parent_completion_used_for_selection_count") or 0) != 0:
        errs.append("parent_completion_used_for_selection_count != 0")
    if int(funnel.get("forbidden_outcome_edit_count") or 0) != 0:
        errs.append("forbidden_outcome_edit_count != 0")
    if int(funnel.get("cross_profile_bundle_count") or 0) != 0:
        errs.append("cross_profile_bundle_count != 0")

    n_single = int(funnel.get("n_single_screened") or 0)
    n_multi = int(funnel.get("n_multi_event_composites") or 0)
    n_two = int(funnel.get("n_two_event_composites") or 0)
    n_three = int(funnel.get("n_three_event_composites") or 0)
    n_total = int(funnel.get("n_total_selected") or 0)
    if require_single_screening and n_multi > 0 and n_single <= 0:
        errs.append("n_single_screened == 0 with multi-event composites")
    if n_two > int(two_event_max):
        errs.append(f"n_two_event_composites({n_two}) > two_event_max({two_event_max})")
    if n_three > int(three_event_max):
        errs.append(
            f"n_three_event_composites({n_three}) > three_event_max({three_event_max})"
        )
    if n_total > int(total_max):
        errs.append(f"n_total_selected({n_total}) > total_max({total_max})")

    n_search = int(funnel.get("n_search_effect_evaluated") or 0)
    if n_search > n_total and n_total > 0:
        errs.append("n_search_effect_evaluated > n_total_selected")

    return len(errs) == 0, errs


def funnel_accounting_status(
    funnel: Mapping[str, Any],
    *,
    two_event_max: int = 32,
    three_event_max: int = 24,
    total_max: int = 100,
    require_single_screening: bool = True,
) -> str:
    ok, _ = validate_cf1s_funnel_monotonicity(
        funnel,
        two_event_max=two_event_max,
        three_event_max=three_event_max,
        total_max=total_max,
        require_single_screening=require_single_screening,
    )
    return "PASS" if ok else "FAIL"
