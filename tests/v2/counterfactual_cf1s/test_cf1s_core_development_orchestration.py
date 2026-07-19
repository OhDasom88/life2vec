"""Development phase machine, dependency closure, and scientific oracles."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import CoreContractError
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_development import (
    DevelopmentPhase,
    PhaseStateMachine,
    apply_dependency_budget,
    build_selection_blind_closure,
    development_readiness_axes,
    evaluate_bundle_effects,
    expand_dependency_closure,
    full_abc_dependency_closure,
    migrate_legacy_holdout_closure,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_scientific import (
    evaluate_claim_levels,
    max_search_identity_reconstruction_error,
    scientific_status_from_effects,
)


def test_abc_dependency_closure_complete():
    deps = full_abc_dependency_closure(["C", "A", "B"])
    assert deps == [("A",), ("B",), ("C",), ("A", "B"), ("A", "C"), ("B", "C")]


def test_bundle_selection_auto_expands_parents():
    selected = [
        {"candidate_id": "AB", "event_ids": ["A", "B"], "kind": "PAIR"},
        {"candidate_id": "ABC", "event_ids": ["A", "B", "C"], "kind": "TRIPLE"},
    ]
    closure = expand_dependency_closure(selected)
    keys = {tuple(d["event_ids"]) for d in closure["dependencies"]}
    assert ("A",) in keys and ("A", "B") in keys and ("A", "B", "C") in keys
    assert closure["unique_dependency_count"] == 7  # A,B,C,AB,AC,BC,ABC


def test_dependency_budget_drops_trailing_bundles():
    # Craft many disjoint pairs to exceed budget=3
    selected = [
        {"candidate_id": f"P{i}", "event_ids": [f"E{i}a", f"E{i}b"], "kind": "PAIR"}
        for i in range(5)
    ]
    out = apply_dependency_budget(selected, budget=6)
    # each pair needs 3 deps (A,B,AB); budget 6 => at most 2 pairs
    assert len(out["selected_bundles"]) <= 2
    assert out["excluded_bundles"]
    assert out["excluded_bundles"][0]["exclusion_reason"] == "EXCLUDED_DEPENDENCY_BUDGET"


def test_phase_transitions_and_closure_freeze():
    sm = PhaseStateMachine()
    sm.advance(DevelopmentPhase.ATTRIBUTION)
    sm.advance(DevelopmentPhase.CANDIDATE_UNIVERSE)
    sm.advance(DevelopmentPhase.SEARCH_BASELINE_IDENTITY)
    sm.advance(DevelopmentPhase.THRESHOLD_LOCK)
    sm.advance(DevelopmentPhase.SEARCH_EFFECTS)
    sm.advance(DevelopmentPhase.SELECTION)
    closure = build_selection_blind_closure(
        case_id="c1",
        candidate_identity_hashes=["h1"],
        search_result_hash="s",
        selection_manifest_sha256="sel",
        candidate_budget_policy_sha256="b",
        threshold_policy_sha256="t",
        risk_definition_sha256="r",
        candidate_universe_hash="u",
        parent_graph_sha256="p",
        expected_effect_direction_by_candidate={"h1": "RISK_DECREASE"},
    )
    frozen = sm.freeze_closure(closure)
    assert frozen["closure_kind"] == "SELECTION_BLIND_REEVALUATION_CLOSURE"
    sm.assert_closure_immutable(frozen)
    with pytest.raises(CoreContractError):
        sm.advance(DevelopmentPhase.ATTRIBUTION)


def test_identity_error_fold_scoped_only():
    err = max_search_identity_reconstruction_error(
        identity_by_fold={"0": 0.10, "1": 0.20},
        baseline_by_fold={"0": 0.10, "1": 0.21},
    )
    assert abs(err - 0.01) < 1e-12
    with pytest.raises(CoreContractError):
        max_search_identity_reconstruction_error(
            identity_by_fold={"0": 0.1},
            baseline_by_fold={"0": 0.1, "1": 0.2},
        )


def test_claim_levels_and_scientific_separation():
    claims = evaluate_claim_levels(
        model_input_delivery_verified=True,
        stable_effect_replicated=True,
        multievent_increment_supported=False,
    )
    assert claims["highest_claim_level"] == "BUNDLE_MODEL_EFFECT_REPLICATED"
    assert claims["multievent_additional_effect_claimed"] is False
    status = scientific_status_from_effects(
        stable_effect_replicated=True,
        multievent_increment_supported=False,
    )
    assert status["development_scientific_result"] == "NOT_SUPPORTED"
    assert status["reason"] == "BUNDLE_EFFECT_REPLICATED_INCREMENT_NOT_SUPPORTED"


def test_bundle_effect_oracle_requires_search_2_of_2_and_fold2_1_of_1():
    out = evaluate_bundle_effects(
        search_fold_deltas={"0": -0.01, "1": -0.01},
        reeval_fold_deltas={"2": -0.01},
        search_parent_deltas_by_fold={"0": [-0.001, -0.001], "1": [-0.001, -0.001]},
        reeval_parent_deltas_by_fold={"2": [-0.001, -0.001]},
        expected_effect_direction="RISK_DECREASE",
        locked_threshold=0.001,
    )
    assert out["development_scientific_result"] == "SUPPORTED"
    # Search 1/2 fails → not stable
    out2 = evaluate_bundle_effects(
        search_fold_deltas={"0": -0.01, "1": 0.0},
        reeval_fold_deltas={"2": -0.01},
        search_parent_deltas_by_fold={"0": [-0.001], "1": [-0.001]},
        reeval_parent_deltas_by_fold={"2": [-0.001]},
        expected_effect_direction="RISK_DECREASE",
        locked_threshold=0.001,
    )
    assert out2["stable_effect_status"] == "NOT_REPLICATED"


def test_partial_coverage_blocks_cohort_completion_claim():
    axes = development_readiness_axes(
        execution_ready=True,
        evaluable_cases=2,
        locked_cohort_size=3,
    )
    assert axes["development_evaluation_coverage"] == "PARTIAL"
    assert axes["cohort_completion_claim_allowed"] is False


def test_legacy_closure_migration():
    legacy = {
        "closure_kind": "HOLDOUT_BLIND_EVALUATION_CLOSURE",
        "selection_manifest_sha256": "x",
    }
    migrated = migrate_legacy_holdout_closure(legacy)
    assert migrated["closure_kind"] == "SELECTION_BLIND_REEVALUATION_CLOSURE"
    assert migrated["scientifically_independent_holdout"] is False
