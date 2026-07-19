"""Boundary fixtures for CF-1S Core acceptance engine."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_acceptance import (
    development_readiness_status,
    evaluate_candidate_status,
    evaluate_cohort_status,
    evaluate_stable_bundle,
    evaluate_stable_effect_scope,
    evaluate_stable_incremental,
    evaluate_stable_multi_event_only,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
    derive_thresholds_from_noise,
    fold_median,
    required_strict_majority_count,
)


def test_even_fold_median_is_arithmetic_mean_of_two_central():
    assert fold_median([0.01, 0.03, 0.02, 0.04]) == pytest.approx(0.025)
    assert fold_median([0.02, 0.01, 0.03]) == pytest.approx(0.02)


def test_strict_majority_even_tie_fails():
    assert required_strict_majority_count(4) == 3
    assert required_strict_majority_count(2) == 2
    assert required_strict_majority_count(5) == 3


def test_directional_fold_majority_rejects_zero_aggregate_and_half_ties():
    # aggregate median exactly below threshold -> sign 0 -> fail
    scope = evaluate_stable_effect_scope(
        [0.04, 0.031, -0.04, -0.039, 0.0], effect_threshold=0.03
    )
    assert scope["stable_effect_scope"] is False
    assert scope["aggregate_sign"] == 0
    # exact half same-direction on even folds (2/4) fails strict majority
    scope2 = evaluate_stable_effect_scope(
        [0.05, 0.05, -0.05, -0.05], effect_threshold=0.01
    )
    # median 0 -> sign 0
    assert scope2["aggregate_sign"] == 0
    assert scope2["stable_effect_scope"] is False
    # nonzero aggregate but only 2/5 directional supports
    scope3 = evaluate_stable_effect_scope(
        [0.05, 0.05, -0.02, -0.02, -0.02], effect_threshold=0.01
    )
    # sorted [-0.02,-0.02,-0.02,0.05,0.05], median=-0.02 -> -
    # directional -: three -0.02 = 3 >= 3 passes. Use thr so negatives fail magnitude:
    scope4 = evaluate_stable_effect_scope(
        [0.05, 0.05, -0.005, -0.005, -0.005], effect_threshold=0.01
    )
    # median -0.005 -> sign 0
    assert scope4["stable_effect_scope"] is False


def test_joint_incremental_requires_same_fold_intersection():
    # effect majority folds 0,1,2 ; gain-only majority would be 2,3,4 if independent
    search_bundle = [0.05, 0.05, 0.05, 0.0, 0.0]
    # parents large on early folds so gain fails there; small on late folds
    search_parents = [
        {"A": 0.04, "B": 0.04},  # gain 0.01
        {"A": 0.04, "B": 0.04},
        {"A": 0.04, "B": 0.04},
        {"A": 0.0, "B": 0.0},  # gain 0 but bundle 0 -> not directional
        {"A": 0.0, "B": 0.0},
    ]
    # Make late folds have gain but no directional effect; early have effect but tiny gain
    search_bundle = [0.05, 0.05, 0.05, 0.0001, 0.0001]
    search_parents = [
        {"A": 0.049, "B": 0.049},  # gain 0.001
        {"A": 0.049, "B": 0.049},
        {"A": 0.049, "B": 0.049},
        {"A": 0.0, "B": 0.0},  # gain 0.0001 < thr if thr=0.001
        {"A": 0.0, "B": 0.0},
    ]
    # Better fixture from plan: effect folds 1,2,3 and incremental 3,4,5
    search_bundle = [0.05, 0.05, 0.05, 0.0, 0.0]
    search_parents = [
        {"A": 0.05, "B": 0.05},  # gain 0
        {"A": 0.05, "B": 0.05},  # gain 0
        {"A": 0.0, "B": 0.0},  # gain 0.05 joint
        {"A": 0.0, "B": 0.0},  # bundle 0 not directional
        {"A": 0.0, "B": 0.0},
    ]
    # Need incremental-looking majority if computed independently on gain>=thr:
    # folds 2,3,4 with artificial gains — but 3,4 have bundle 0.
    # Independent gain majority would need gains on 3 folds; only fold2 has gain.
    # Construct: effect on 0,1,2; gain on 2,3,4 where 3,4 have tiny effect below thr
    search_bundle = [0.05, 0.05, 0.05, 0.0005, 0.0005]
    search_parents = [
        {"A": 0.0495, "B": 0.0},  # gain 0.0005 < 0.001
        {"A": 0.0495, "B": 0.0},
        {"A": 0.0, "B": 0.0},  # gain 0.05 joint yes
        {"A": 0.0, "B": 0.0},  # gain 0.0005, not directional
        {"A": 0.0, "B": 0.0},
    ]
    holdout_bundle = [0.05]
    holdout_parents = [{"A": 0.0, "B": 0.0}]
    out = evaluate_stable_incremental(
        search_bundle,
        search_parents,
        holdout_bundle,
        holdout_parents,
        effect_threshold=0.001,
        incremental_threshold=0.001,
    )
    # joint support only fold index 2 => 1 < 3 required
    assert out["search"]["joint_support_count"] == 1
    assert out["search"]["stable_incremental_scope"] is False
    assert out["stable_incremental"] is False


def test_joint_multi_event_only_intersection():
    # directional on 0,1,2; parent-noneffect only on 2,3,4
    search_bundle = [0.05, 0.05, 0.05, 0.05, 0.05]
    search_singles = [
        {"A": 0.02, "B": 0.0},  # parent effect -> not MEO
        {"A": 0.02, "B": 0.0},
        {"A": 0.0, "B": 0.0},  # joint
        {"A": 0.0, "B": 0.0},
        {"A": 0.0, "B": 0.0},
    ]
    # Wait folds 2,3,4 all joint if parents noneffect and directional — that's 3.
    # Make 3,4 non-directional:
    search_bundle = [0.05, 0.05, 0.05, 0.0, 0.0]
    search_singles = [
        {"A": 0.02, "B": 0.0},
        {"A": 0.02, "B": 0.0},
        {"A": 0.0, "B": 0.0},  # only joint
        {"A": 0.0, "B": 0.0},
        {"A": 0.0, "B": 0.0},
    ]
    holdout_bundle = [0.05]
    holdout_singles = [{"A": 0.0, "B": 0.0}]
    out = evaluate_stable_multi_event_only(
        search_bundle,
        search_singles,
        holdout_bundle,
        holdout_singles,
        effect_threshold=0.001,
    )
    assert out["search"]["joint_support_count"] == 1
    assert out["stable_multi_event_only"] is False


def test_candidate_supported_requires_stable_bundle_and_incremental_or_meo():
    search = [0.05, 0.05]
    holdout = [0.05]
    parents_s = [{"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0}]
    parents_h = [{"A": 0.0, "B": 0.0}]
    out = evaluate_candidate_status(
        execution_complete=True,
        parent_complete=True,
        fold_execution_failed=False,
        search_bundle_deltas=search,
        holdout_bundle_deltas=holdout,
        search_incremental_parents=parents_s,
        holdout_incremental_parents=parents_h,
        search_single_parents=parents_s,
        holdout_single_parents=parents_h,
        effect_threshold=0.001,
        incremental_threshold=0.001,
        n_events=2,
    )
    assert out["candidate_scientific_status"] == "SUPPORTED"


def test_single_only_not_promoted():
    out = evaluate_candidate_status(
        execution_complete=True,
        parent_complete=True,
        fold_execution_failed=False,
        search_bundle_deltas=[0.05, 0.05],
        holdout_bundle_deltas=[0.05],
        search_incremental_parents=[{}, {}],
        holdout_incremental_parents=[{}],
        search_single_parents=[{}, {}],
        holdout_single_parents=[{}],
        effect_threshold=0.001,
        incremental_threshold=0.001,
        n_events=1,
    )
    assert out["candidate_scientific_status"] == "NOT_SUPPORTED"


def test_cohort_fraction_uses_locked_denominator():
    # 24 evaluable, 8 supported of 32 -> fraction 0.25 on locked count
    statuses = ["SUPPORTED"] * 8 + ["NOT_SUPPORTED"] * 16 + ["NOT_EVALUABLE"] * 8
    out = evaluate_cohort_status(
        statuses,
        locked_cohort_case_count=32,
        minimum_evaluable_case_count=24,
        minimum_supported_case_count=8,
        minimum_supported_fraction=0.25,
    )
    assert out["supported_fraction"] == pytest.approx(8 / 32)
    assert out["evaluable_fraction"] == pytest.approx(24 / 32)
    assert out["cohort_scientific_status"] == "SUPPORTED"


def test_development_readiness_blocked_on_unevaluable_not_on_not_supported():
    ready = development_readiness_status(
        execution_status="PASS",
        case_statuses=["NOT_SUPPORTED", "NOT_SUPPORTED", "NOT_SUPPORTED"],
        noise_ceiling_pass=True,
    )
    assert ready["development_readiness_status"] == "READY"
    blocked = development_readiness_status(
        execution_status="PASS",
        case_statuses=["NOT_SUPPORTED", "NOT_SUPPORTED", "NOT_EVALUABLE"],
        noise_ceiling_pass=True,
    )
    assert blocked["development_readiness_status"] == "BLOCKED"


def test_threshold_ceiling_and_noise_block():
    thr = derive_thresholds_from_noise(0.0005)
    assert thr["locked_min_effect_abs_delta"] == pytest.approx(0.0015)
    assert thr["locked_min_incremental_abs_gain"] == pytest.approx(0.0015)
    with pytest.raises(Exception):
        derive_thresholds_from_noise(0.002)
