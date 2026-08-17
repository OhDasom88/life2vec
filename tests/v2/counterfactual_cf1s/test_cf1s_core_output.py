"""Transaction order invariance, candidate sets, readiness axes, holdout-blind closure."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import CoreContractError
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_edit_proposal import (
    AuthorityContext,
    ConstructionStatus,
    EvidenceKind,
    EvaluationCoverage,
    EvaluationStage,
    ExecutionEvidenceFlags,
    ExecutionReadiness,
    ForwardKind,
    OrderingStatus,
    ScientificStatus,
    TransactionConflictReason,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_output import (
    assert_holdout_matches_holdout_blind_closure,
    build_candidate_set_manifest,
    build_case_edit_proposal_envelope,
    build_runner_summary,
    evaluation_coverage_from_counts,
    freeze_holdout_blind_evaluation_closure,
    resolve_execution_readiness,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_production import (
    ForwardCounters,
    evaluate_scope_gate,
    lock_thresholds_from_search_baseline_noise,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_transaction import (
    check_transaction_order_invariant,
    validate_atomic_transaction,
)


def _atomic(eid, t, pos, target=1.0, tokens=None):
    return {
        "event_id": eid,
        "feature_id": "temp",
        "edit_direction": "DOWN",
        "target_raw": target,
        "schema_target": "bin1",
        "sequence_position": pos,
        "event_time_epoch": float(t),
        "to_tokens": tokens or ["TOK"],
        "sentence_tokens": tokens or ["TOK"],
    }


def test_two_event_order_invariant():
    atomics = [_atomic("A", 0, 0), _atomic("B", 1000, 1)]
    inv = check_transaction_order_invariant(atomics)
    assert inv["ok"] and inv["transaction_order_invariant"]
    assert inv["permutation_count"] == 2
    assert inv["ab_transaction_sha256"] == inv["ba_transaction_sha256"]


def test_three_event_six_permutations():
    atomics = [_atomic("A", 0, 0), _atomic("B", 1000, 1), _atomic("C", 2000, 2)]
    inv = check_transaction_order_invariant(atomics)
    assert inv["ok"]
    assert inv["permutation_count"] == 6
    assert len(set(inv["permutation_hashes"].values())) == 1


def test_duplicate_event_conflict():
    atomics = [_atomic("A", 0, 0), _atomic("A", 1000, 1)]
    v = validate_atomic_transaction(atomics)
    assert not v["ok"]
    assert TransactionConflictReason.NON_DISTINCT_EVENT_IDS.value in v["conflict_reasons"]


def test_conflicting_raw_edit():
    a = _atomic("A", 0, 0, target=1.0)
    b = _atomic("A", 0, 0, target=2.0)
    # force same event via duplicate then also conflicting targets on same cell
    v = validate_atomic_transaction([a, b])
    assert not v["ok"]


def test_scope_gate_blocks_candidate_forward():
    g = evaluate_scope_gate(scope="search", identity_pass=False, noise_ceiling_pass=True)
    assert g["allow_candidate_effect_forward"] is False
    assert "SEARCH_IDENTITY_FAILED" in g["reasons"]
    assert "CANDIDATE_FORWARD_SKIPPED_BY_SCOPE_GATE" in g["reasons"]


def test_holdout_attribution_forbidden():
    c = ForwardCounters()
    with pytest.raises(CoreContractError):
        c.record(ForwardKind.ATTRIBUTION_FORWARD, scope="holdout")


def test_threshold_lock_search_baseline_only():
    thr = lock_thresholds_from_search_baseline_noise(0.0001)
    assert thr["threshold_calibration_scope"] == "DEVELOPMENT3_SEARCH_BASELINE_REPEATS_ONLY"
    assert thr["holdout_noise_must_not_change_threshold"] is True


def test_candidate_sets_and_holdout_blind_closure():
    sets = build_candidate_set_manifest(
        constructed_candidate_ids=["c1", "c2"],
        evaluable_candidate_ids=["c1"],
        selected_candidate_ids=["c1"],
        construction_rejected_candidates={"c2": "GATE4_REJECTED"},
        budget_excluded_candidate_ids=[],
    )
    assert sets["constructed_candidate_ids"] == ["c1", "c2"]
    closure = freeze_holdout_blind_evaluation_closure(
        case_id="case1",
        candidate_identity_hashes=["h1"],
        search_result_hash="s",
        selection_manifest_sha256="sel",
        candidate_budget_policy_sha256="b",
        threshold_policy_sha256="t",
        risk_definition_sha256="r",
        candidate_universe_hash="u",
    )
    assert closure["closure_kind"] == "SELECTION_BLIND_REEVALUATION_CLOSURE"
    assert closure["legacy_enum"] == "HOLDOUT_BLIND_EVALUATION_CLOSURE"
    assert closure["holdout_effect_included"] is False
    assert closure["scientifically_independent_holdout"] is False
    assert_holdout_matches_holdout_blind_closure(
        closure=closure, selection_manifest_sha256="sel"
    )
    with pytest.raises(CoreContractError):
        assert_holdout_matches_holdout_blind_closure(
            closure=closure, selection_manifest_sha256="other"
        )


def test_readiness_axes_candidate_absence_vs_identity():
    # Pipeline OK, no evaluable candidates → READY + NONE coverage
    ready = resolve_execution_readiness(
        authority_ok=True,
        baseline_forward_ok=True,
        fold_isolation_ok=True,
        cold_build_equivalence_ok=True,
        identity_noise_gate_ok=True,
        artifact_integrity_ok=True,
    )
    assert ready == ExecutionReadiness.READY
    assert evaluation_coverage_from_counts(scientifically_evaluable_case_count=0) == EvaluationCoverage.NONE
    assert evaluation_coverage_from_counts(scientifically_evaluable_case_count=2) == EvaluationCoverage.PARTIAL

    blocked = resolve_execution_readiness(
        authority_ok=True,
        baseline_forward_ok=True,
        fold_isolation_ok=True,
        cold_build_equivalence_ok=True,
        identity_noise_gate_ok=False,
        artifact_integrity_ok=True,
    )
    assert blocked == ExecutionReadiness.BLOCKED


def test_envelope_recommendation_always_null():
    authority = AuthorityContext(
        evaluation_stage=EvaluationStage.DEVELOPMENT_PRODUCTION,
        execution_authorized=True,
        evidence_output_authorized=True,
    )
    evidence = ExecutionEvidenceFlags(
        production_input_verified=True,
        pipeline_integrity_verified=True,
        baseline_forward_complete=True,
    )
    env = build_case_edit_proposal_envelope(
        case_id="c",
        authority=authority,
        evidence=evidence,
        evidence_kind=EvidenceKind.ACTUAL_MODEL_FORWARD,
        construction_status=ConstructionStatus.NOT_CONSTRUCTIBLE,
        scientific_status=ScientificStatus.NOT_EVALUATED,
        ordering_status=OrderingStatus.NOT_ORDERABLE,
        candidate_sets=build_candidate_set_manifest(
            constructed_candidate_ids=[],
            evaluable_candidate_ids=[],
            selected_candidate_ids=[],
        ),
    )
    assert env["disposition"] == "EVIDENCE_ONLY"
    assert env["recommendation"] is None
    assert env["executable_edit"] is None
    assert env["causal_effect_claim_authorized"] is False


def test_envelope_nests_extra_not_flattens_it():
    # Regression: the verifier reads threshold/closure/selection_manifest/
    # scientific_detail via proposal.get("extra").get(...) — flattening extra
    # into the top level (the original bug) orphans those fields and fails
    # G6-G9 with "missing" errors on real (non-mocked) production output.
    authority = AuthorityContext(
        evaluation_stage=EvaluationStage.DEVELOPMENT_PRODUCTION,
        execution_authorized=True,
        evidence_output_authorized=True,
    )
    evidence = ExecutionEvidenceFlags(
        production_input_verified=True,
        pipeline_integrity_verified=True,
        baseline_forward_complete=True,
    )
    env = build_case_edit_proposal_envelope(
        case_id="c",
        authority=authority,
        evidence=evidence,
        evidence_kind=EvidenceKind.ACTUAL_MODEL_FORWARD,
        construction_status=ConstructionStatus.COMPLETE,
        scientific_status=ScientificStatus.NOT_EVALUATED,
        ordering_status=OrderingStatus.NOT_ORDERABLE,
        candidate_sets=build_candidate_set_manifest(
            constructed_candidate_ids=[],
            evaluable_candidate_ids=[],
            selected_candidate_ids=[],
        ),
        extra={"threshold": {"locked_min_effect_abs_delta": 0.001}},
    )
    assert "threshold" not in env
    assert env["extra"]["threshold"]["locked_min_effect_abs_delta"] == 0.001

    summary = build_runner_summary(
        case_envelopes=[env],
        execution_readiness=ExecutionReadiness.READY,
        evaluation_coverage=EvaluationCoverage.NONE,
        scientifically_evaluable_case_count=0,
    )
    assert summary["development_execution_readiness"] == "READY"
    assert summary["development_evaluation_coverage"] == "NONE"
    assert summary["non_null_recommendation_count"] == 0
