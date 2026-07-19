from __future__ import annotations

from src.online2.v2.finetune_v03.counterfactual.cf1s.fold_effect import (
    required_strict_majority_count,
    sign_consensus_fraction,
    stable_model_sensitivity_valid,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.interaction import (
    ADDITIVE,
    AMPLIFICATION,
    CANCELLATION,
    EMERGENT,
    INDETERMINATE,
    classify_interaction,
    stable_interaction_fields,
    three_event_residuals,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.status import (
    NOT_EVALUABLE,
    NOT_SUPPORTED,
    SUPPORTED,
    overall_cf1s_status,
    primary_feasibility_from_selectors,
)


def test_interaction_classes_mutually_exclusive_boundaries():
    # Exact additive boundary is ADDITIVE, not amplification.
    row = classify_interaction(bundle_delta=0.021, atomic_deltas=[0.01, 0.01])
    assert abs(row["interaction_delta"] - 0.001) < 1e-12
    assert row["interaction_class"] == ADDITIVE

    amp = classify_interaction(bundle_delta=0.025, atomic_deltas=[0.01, 0.01])
    assert amp["interaction_class"] == AMPLIFICATION

    cancel = classify_interaction(bundle_delta=0.005, atomic_deltas=[0.02, 0.02])
    assert cancel["interaction_class"] == CANCELLATION

    emergent = classify_interaction(bundle_delta=0.02, atomic_deltas=[0.0001, -0.0001])
    assert emergent["interaction_class"] == EMERGENT

    # No double classification at additive boundary.
    assert row["interaction_class"] != AMPLIFICATION


def test_strict_majority_rejects_half_tie():
    assert required_strict_majority_count(2) == 2
    assert required_strict_majority_count(3) == 2
    assert required_strict_majority_count(4) == 3
    signs = sign_consensus_fraction([0.015, -0.014], zero_sign_tolerance=0.001)
    assert signs["sign_consensus_fraction"] == 0.5
    assert signs["strict_majority_pass"] is False
    assert signs["ties_pass"] is False


def test_stable_validity_requires_strict_majority():
    # Two opposing holdout folds → not stable even if median mag large.
    out = stable_model_sensitivity_valid(
        delta_risk_search_by_fold=[0.02, 0.03],
        delta_risk_holdout_by_fold=[0.015, -0.014],
    )
    assert out["holdout_sign_strict_majority"] is False
    assert out["stable_model_sensitivity_valid"] is False


def test_stable_interaction_requires_fold_consensus():
    fields = stable_interaction_fields(
        search_class=AMPLIFICATION,
        holdout_class=AMPLIFICATION,
        holdout_classes_by_fold=[AMPLIFICATION, CANCELLATION],
        holdout_aligned_directions_by_fold=[1, -1],
        holdout_interaction_magnitude=0.01,
    )
    assert fields["interaction_class_stable"] is False

    ok = stable_interaction_fields(
        search_class=ADDITIVE,
        holdout_class=ADDITIVE,
        holdout_classes_by_fold=[ADDITIVE, ADDITIVE, ADDITIVE],
        holdout_aligned_directions_by_fold=[1, 1, 1],
        holdout_interaction_magnitude=0.002,
    )
    assert ok["interaction_class_stable"] is True
    assert ok["stable_interaction_class"] == ADDITIVE


def test_three_event_residuals_separated():
    res = three_event_residuals(
        delta_a=0.01,
        delta_b=0.01,
        delta_c=0.01,
        delta_ab=0.03,
        delta_ac=0.02,
        delta_bc=0.02,
        delta_abc=0.08,
    )
    assert abs(res["total_nonadditivity_abc"] - 0.05) < 1e-12
    assert "third_order_residual_abc" in res


def test_primary_selector_authority_no_baseline_rescue():
    out = primary_feasibility_from_selectors(
        {
            "SALIENCY_TOP_K": NOT_SUPPORTED,
            "RANDOM_EDITABLE_LOCUS": SUPPORTED,
            "RECENCY_TOP_K": SUPPORTED,
        },
        saliency_evaluable=True,
    )
    assert out["primary_multi_event_edit_feasibility_status"] == NOT_SUPPORTED
    assert out["baseline_success_can_rescue_primary_feasibility"] is False
    assert out["selector_agnostic_exploratory_feasibility_status"] == SUPPORTED

    uneval = primary_feasibility_from_selectors(
        {
            "SALIENCY_TOP_K": NOT_EVALUABLE,
            "RANDOM_EDITABLE_LOCUS": SUPPORTED,
            "RECENCY_TOP_K": LIMITED if False else "LIMITED",
        },
        saliency_evaluable=False,
    )
    assert uneval["primary_multi_event_edit_feasibility_status"] == NOT_EVALUABLE


def test_overall_status_problem20_first():
    out = overall_cf1s_status(
        problem20_blind_status=NOT_SUPPORTED,
        primary_internal_validation_32_status=SUPPORTED,
        development_3_status=SUPPORTED,
        problem20_min_evaluable_met=True,
    )
    assert out["overall_cf1s_status"] == NOT_SUPPORTED
    assert out["generalization_status"] == "INTERNAL_TO_BLIND_NOT_REPLICATED"
    assert out["development_excluded_from_overall"] is True

    missing = overall_cf1s_status(
        problem20_blind_status=SUPPORTED,
        primary_internal_validation_32_status=SUPPORTED,
        development_3_status=SUPPORTED,
        problem20_min_evaluable_met=False,
    )
    assert missing["overall_cf1s_status"] == NOT_EVALUABLE
