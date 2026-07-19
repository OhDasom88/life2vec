"""Edit proposal authority, disposition, direction consistency, and ranking-field bans."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import CoreContractError
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_edit_proposal import (
    AuthorityContext,
    ConstructionStatus,
    DirectionConsistency,
    EvidenceKind,
    EvaluationStage,
    ExecutionEvidenceFlags,
    OrderingStatus,
    OutputDisposition,
    RiskEffectDirection,
    ScientificStatus,
    assert_no_forbidden_ranking_fields,
    direction_consistency,
    execution_evidence_valid,
    resolve_disposition,
    resolve_scientific_status,
    risk_direction_from_delta,
)


def test_ordering_status_not_ranking():
    payload = {"ordering_status": OrderingStatus.NOT_ORDERABLE.value}
    assert_no_forbidden_ranking_fields(payload)
    with pytest.raises(CoreContractError):
        assert_no_forbidden_ranking_fields({"ranking_status": "NOT_RANKABLE"})


def test_attribution_only_not_evidence_only():
    authority = AuthorityContext(
        evaluation_stage=EvaluationStage.DEVELOPMENT_PRODUCTION,
        execution_authorized=True,
        evidence_output_authorized=True,
    )
    evidence = ExecutionEvidenceFlags(
        production_input_verified=True,
        pipeline_integrity_verified=True,
        attribution_forward_executed=True,
        baseline_forward_complete=False,
    )
    assert execution_evidence_valid(authority, evidence) is False
    assert (
        resolve_disposition(
            authority=authority,
            evidence=evidence,
            evidence_kind=EvidenceKind.ACTUAL_MODEL_FORWARD,
        )
        == OutputDisposition.WITHHELD
    )


def test_baseline_complete_evidence_only_even_if_not_evaluable():
    authority = AuthorityContext(
        evaluation_stage=EvaluationStage.DEVELOPMENT_PRODUCTION,
        execution_authorized=True,
        evidence_output_authorized=True,
        recommendation_authorized=False,
    )
    evidence = ExecutionEvidenceFlags(
        production_input_verified=True,
        pipeline_integrity_verified=True,
        baseline_forward_complete=True,
        identity_pass=False,
    )
    assert execution_evidence_valid(authority, evidence) is True
    assert (
        resolve_disposition(
            authority=authority,
            evidence=evidence,
            evidence_kind=EvidenceKind.ACTUAL_MODEL_FORWARD,
        )
        == OutputDisposition.EVIDENCE_ONLY
    )
    sci = resolve_scientific_status(
        construction_status=ConstructionStatus.COMPLETE,
        core_acceptance_status=None,
        consistency=DirectionConsistency.NOT_EVALUABLE,
        identity_or_noise_failed=True,
        execution_failed=False,
    )
    assert sci["scientific_status"] == ScientificStatus.NOT_EVALUABLE.value


def test_direction_consistency_enums():
    assert (
        direction_consistency(
            RiskEffectDirection.RISK_INCREASE, RiskEffectDirection.RISK_INCREASE
        )
        == DirectionConsistency.CONSISTENT
    )
    assert (
        direction_consistency(
            RiskEffectDirection.RISK_INCREASE, RiskEffectDirection.RISK_DECREASE
        )
        == DirectionConsistency.INCONSISTENT
    )
    assert (
        direction_consistency(
            RiskEffectDirection.RISK_INCREASE, RiskEffectDirection.NO_MATERIAL_CHANGE
        )
        == DirectionConsistency.ONE_SCOPE_MATERIAL_ONLY
    )
    assert (
        direction_consistency(
            RiskEffectDirection.NO_MATERIAL_CHANGE,
            RiskEffectDirection.NO_MATERIAL_CHANGE,
        )
        == DirectionConsistency.BOTH_SCOPES_NO_MATERIAL_CHANGE
    )
    assert risk_direction_from_delta(0.0, threshold=0.001) == RiskEffectDirection.NO_MATERIAL_CHANGE


def test_contract_stage_defaults():
    authority = AuthorityContext(evaluation_stage=EvaluationStage.CONTRACT)
    evidence = ExecutionEvidenceFlags()
    assert (
        resolve_disposition(
            authority=authority,
            evidence=evidence,
            evidence_kind=EvidenceKind.SYNTHETIC_CONTRACT,
        )
        == OutputDisposition.CONTRACT_ONLY
    )
