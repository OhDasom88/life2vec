"""Case-level CF-1S edit proposal output: disposition, ordering, readiness axes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .core_contract import CoreContractError, sha256_bytes
from .core_edit_proposal import (
    PROPOSAL_SCHEMA_ID,
    AuthorityContext,
    ConstructionStatus,
    EvidenceKind,
    EvaluationCoverage,
    EvaluationStage,
    ExecutionEvidenceFlags,
    ExecutionReadiness,
    OrderingStatus,
    OutputDisposition,
    ScientificStatus,
    assert_no_forbidden_ranking_fields,
    canonical_sha256,
    contract_case_defaults,
    execution_evidence_valid,
    interpretation_contract,
    resolve_disposition,
)


def build_candidate_set_manifest(
    *,
    constructed_candidate_ids: Sequence[str],
    evaluable_candidate_ids: Sequence[str],
    selected_candidate_ids: Sequence[str],
    construction_rejected_candidates: Optional[Mapping[str, str]] = None,
    budget_excluded_candidate_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "constructed_candidate_ids": list(constructed_candidate_ids),
        "evaluable_candidate_ids": list(evaluable_candidate_ids),
        "selected_candidate_ids": list(selected_candidate_ids),
        "construction_rejected_candidates": dict(construction_rejected_candidates or {}),
        "budget_excluded_candidate_ids": list(budget_excluded_candidate_ids or []),
    }


def freeze_selection_blind_reevaluation_closure(
    *,
    case_id: str,
    candidate_identity_hashes: Sequence[str],
    search_result_hash: str,
    selection_manifest_sha256: str,
    candidate_budget_policy_sha256: str,
    threshold_policy_sha256: str,
    risk_definition_sha256: str,
    candidate_universe_hash: str,
) -> Dict[str, Any]:
    """SELECTION_BLIND_REEVALUATION_CLOSURE — independent of reevaluation effects."""
    body = {
        "closure_kind": "SELECTION_BLIND_REEVALUATION_CLOSURE",
        "case_id": case_id,
        "candidate_identity_hashes": sorted(candidate_identity_hashes),
        "search_result_hash": search_result_hash,
        "selection_manifest_sha256": selection_manifest_sha256,
        "candidate_budget_policy_sha256": candidate_budget_policy_sha256,
        "threshold_policy_sha256": threshold_policy_sha256,
        "risk_definition_sha256": risk_definition_sha256,
        "candidate_universe_hash": candidate_universe_hash,
        "scientifically_independent_holdout": False,
    }
    evaluation_closure_hash = canonical_sha256(body)
    chain = {
        "candidate_universe_hash": candidate_universe_hash,
        "search_result_hash": search_result_hash,
        "selection_manifest_hash": selection_manifest_sha256,
        "evaluation_closure_hash": evaluation_closure_hash,
    }
    return {
        **body,
        "evaluation_closure_hash": evaluation_closure_hash,
        "chain": chain,
        "holdout_effect_included": False,
        "selected_boolean_included": False,
        "fold_delta_included": False,
    }


def freeze_holdout_blind_evaluation_closure(
    *,
    case_id: str,
    candidate_identity_hashes: Sequence[str],
    search_result_hash: str,
    selection_manifest_sha256: str,
    candidate_budget_policy_sha256: str,
    threshold_policy_sha256: str,
    risk_definition_sha256: str,
    candidate_universe_hash: str,
) -> Dict[str, Any]:
    """Legacy name — emits SELECTION_BLIND_REEVALUATION_CLOSURE with migration metadata."""
    closure = freeze_selection_blind_reevaluation_closure(
        case_id=case_id,
        candidate_identity_hashes=candidate_identity_hashes,
        search_result_hash=search_result_hash,
        selection_manifest_sha256=selection_manifest_sha256,
        candidate_budget_policy_sha256=candidate_budget_policy_sha256,
        threshold_policy_sha256=threshold_policy_sha256,
        risk_definition_sha256=risk_definition_sha256,
        candidate_universe_hash=candidate_universe_hash,
    )
    # Preserve legacy test/field expectations via adapter markers only when reading old artifacts.
    closure["legacy_enum"] = "HOLDOUT_BLIND_EVALUATION_CLOSURE"
    closure["semantic_role"] = "SELECTION_BLIND_REEVALUATION_CLOSURE"
    return closure


def assert_holdout_matches_holdout_blind_closure(
    *,
    closure: Mapping[str, Any],
    selection_manifest_sha256: str,
) -> None:
    if selection_manifest_sha256 != closure.get("selection_manifest_sha256"):
        raise CoreContractError(
            "reevaluation selection_manifest_sha256 does not match SELECTION_BLIND_REEVALUATION_CLOSURE"
        )
    kind = closure.get("closure_kind")
    if kind not in (
        "SELECTION_BLIND_REEVALUATION_CLOSURE",
        "HOLDOUT_BLIND_EVALUATION_CLOSURE",  # legacy read path
    ):
        raise CoreContractError("closure_kind must be SELECTION_BLIND_REEVALUATION_CLOSURE")


def empty_forward_counts() -> Dict[str, int]:
    return {
        "attribution_forward_count": 0,
        "baseline_forward_count": 0,
        "identity_forward_count": 0,
        "candidate_effect_forward_count": 0,
        "nonrequested_fold_forward_count": 0,
        "holdout_attribution_forward_count": 0,
        "holdout_candidate_forward_before_closure_count": 0,
    }


def build_case_edit_proposal_envelope(
    *,
    case_id: str,
    authority: AuthorityContext,
    evidence: ExecutionEvidenceFlags,
    evidence_kind: EvidenceKind,
    construction_status: ConstructionStatus,
    scientific_status: ScientificStatus,
    ordering_status: OrderingStatus,
    candidate_sets: Mapping[str, Any],
    candidate_results: Optional[Sequence[Mapping[str, Any]]] = None,
    core_acceptance_status: Optional[str] = None,
    direction_consistency: Optional[str] = None,
    scientific_status_reason: Optional[str] = None,
    integrity_failed: bool = False,
    untrusted_authorization_claim: bool = False,
    scope_gate_reasons: Optional[Sequence[str]] = None,
    forward_counts: Optional[Mapping[str, int]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    disposition = resolve_disposition(
        authority=authority,
        evidence=evidence,
        evidence_kind=evidence_kind,
        integrity_failed=integrity_failed,
        untrusted_authorization_claim=untrusted_authorization_claim,
    )
    envelope: Dict[str, Any] = {
        "proposal_output_schema_version": PROPOSAL_SCHEMA_ID,
        "case_id": case_id,
        "evaluation_stage": authority.evaluation_stage.value,
        "evidence_kind": evidence_kind.value,
        "disposition": disposition.value,
        "construction_status": construction_status.value,
        "scientific_status": scientific_status.value,
        "ordering_status": ordering_status.value,
        "core_acceptance_status": core_acceptance_status,
        "direction_consistency": direction_consistency,
        "scientific_status_reason": scientific_status_reason,
        "recommendation": None,
        "executable_edit": None,
        "executable": False,
        **interpretation_contract(),
        "authority_context": authority.to_dict(),
        "execution_evidence": evidence.to_dict(),
        "candidate_sets": dict(candidate_sets),
        "candidate_results": list(candidate_results or []),
        "scope_gate_reasons": list(scope_gate_reasons or []),
        "forward_counts": dict(forward_counts or empty_forward_counts()),
    }
    if disposition == OutputDisposition.CONTRACT_ONLY:
        envelope.update(
            {
                "model_effects_evaluated": False,
                "ordering_status": OrderingStatus.NOT_ORDERABLE.value,
                "scientific_status": ScientificStatus.NOT_EVALUATED.value,
            }
        )
    # Nested, not flattened: the verifier reads these back via
    # proposal.get("extra").get(...) (threshold/closure/selection_manifest/
    # scientific_detail) — flattening into the top level silently orphans
    # them and fails G6-G9 with "missing" errors on real (non-mocked) output.
    if extra:
        envelope["extra"] = dict(extra)
    assert_no_forbidden_ranking_fields(envelope)
    if envelope.get("recommendation") is not None and disposition != OutputDisposition.AUTHORIZED_RECOMMENDATION:
        raise CoreContractError("recommendation must be null without AUTHORIZED_RECOMMENDATION")
    if envelope.get("executable_edit") is not None and disposition != OutputDisposition.AUTHORIZED_RECOMMENDATION:
        raise CoreContractError("executable_edit must be null without AUTHORIZED_RECOMMENDATION")
    return envelope


def build_contract_case_envelope(case_id: str, *, candidate_results: Optional[list] = None) -> Dict[str, Any]:
    authority = AuthorityContext(
        evaluation_stage=EvaluationStage.CONTRACT,
        execution_authorized=False,
        evidence_output_authorized=False,
        recommendation_authorized=False,
    )
    evidence = ExecutionEvidenceFlags()
    return build_case_edit_proposal_envelope(
        case_id=case_id,
        authority=authority,
        evidence=evidence,
        evidence_kind=EvidenceKind.SYNTHETIC_CONTRACT,
        construction_status=ConstructionStatus.COMPLETE,
        scientific_status=ScientificStatus.NOT_EVALUATED,
        ordering_status=OrderingStatus.NOT_ORDERABLE,
        candidate_sets=build_candidate_set_manifest(
            constructed_candidate_ids=[],
            evaluable_candidate_ids=[],
            selected_candidate_ids=[],
        ),
        candidate_results=candidate_results or [],
        extra=contract_case_defaults(),
    )


def evaluation_coverage_from_counts(
    *,
    scientifically_evaluable_case_count: int,
    locked_cohort_case_count: int = 3,
) -> EvaluationCoverage:
    n = int(scientifically_evaluable_case_count)
    locked = int(locked_cohort_case_count)
    if n <= 0:
        return EvaluationCoverage.NONE
    if n >= locked:
        return EvaluationCoverage.COMPLETE
    return EvaluationCoverage.PARTIAL


def resolve_execution_readiness(
    *,
    authority_ok: bool,
    baseline_forward_ok: bool,
    fold_isolation_ok: bool,
    cold_build_equivalence_ok: bool,
    identity_noise_gate_ok: bool,
    artifact_integrity_ok: bool,
) -> ExecutionReadiness:
    if all(
        [
            authority_ok,
            baseline_forward_ok,
            fold_isolation_ok,
            cold_build_equivalence_ok,
            identity_noise_gate_ok,
            artifact_integrity_ok,
        ]
    ):
        return ExecutionReadiness.READY
    return ExecutionReadiness.BLOCKED


def build_runner_summary(
    *,
    case_envelopes: Sequence[Mapping[str, Any]],
    execution_readiness: ExecutionReadiness,
    evaluation_coverage: EvaluationCoverage,
    scientifically_evaluable_case_count: int,
    locked_cohort_case_count: int = 3,
    scientific_counts: Optional[Mapping[str, int]] = None,
    forward_counts: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    disposition_counts: Dict[str, int] = {}
    sci_counts: Dict[str, int] = {
        "supported": 0,
        "not_supported": 0,
        "inconclusive": 0,
        "not_evaluable": 0,
        "not_evaluated": 0,
    }
    if scientific_counts:
        sci_counts.update({k: int(v) for k, v in scientific_counts.items()})
    else:
        for env in case_envelopes:
            st = str(env.get("scientific_status", "")).lower()
            key = {
                "supported": "supported",
                "not_supported": "not_supported",
                "inconclusive": "inconclusive",
                "not_evaluable": "not_evaluable",
                "not_evaluated": "not_evaluated",
            }.get(st)
            if key:
                sci_counts[key] += 1
            d = str(env.get("disposition", "UNKNOWN"))
            disposition_counts[d] = disposition_counts.get(d, 0) + 1

    fc = dict(empty_forward_counts())
    if forward_counts:
        fc.update({k: int(v) for k, v in forward_counts.items()})

    summary = {
        "proposal_output_schema_version": PROPOSAL_SCHEMA_ID,
        **interpretation_contract(),
        "case_disposition_counts": disposition_counts,
        "candidate_scientific_status_counts": {},
        "candidate_ordering_status_counts": {},
        "withheld_reason_counts": {},
        "executable_proposal_count": 0,
        "non_null_recommendation_count": 0,
        "llm_boundary_passed_case_count": 0,
        **fc,
        "checkpoint_forward_attempted_case_count": 0,
        "baseline_forward_complete_case_count": sum(
            1
            for e in case_envelopes
            if (e.get("execution_evidence") or {}).get("baseline_forward_complete")
        ),
        "candidate_forward_complete_case_count": sum(
            1
            for e in case_envelopes
            if (e.get("execution_evidence") or {}).get("candidate_effect_forward_complete")
        ),
        "model_effects_evaluated_case_count": sum(
            1
            for e in case_envelopes
            if (e.get("execution_evidence") or {}).get("model_effects_evaluated")
        ),
        "development_execution_readiness": execution_readiness.value,
        "development_evaluation_coverage": evaluation_coverage.value,
        "scientifically_evaluable_case_count": int(scientifically_evaluable_case_count),
        "locked_cohort_case_count": int(locked_cohort_case_count),
        "development_scientific_result": sci_counts,
        "primary32_execution_count": 0,
        "problem20_execution_count": 0,
        "causal_action_authorized_count": 0,
    }
    assert_no_forbidden_ranking_fields(summary)
    return summary


def write_case_proposal_json(envelope: Mapping[str, Any], out_path: Path) -> Dict[str, str]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(envelope, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    out_path.write_text(text, encoding="utf-8")
    digest = sha256_bytes(text.encode("utf-8"))
    sha_path = out_path.with_suffix(out_path.suffix + ".sha256")
    sha_path.write_text(digest + "  " + out_path.name + "\n", encoding="utf-8")
    return {"path": str(out_path), "sha256": digest}
