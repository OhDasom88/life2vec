"""CF-1S edit proposal contracts: enums, authority gates, disposition, canonical hash."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .core_contract import CoreContractError, sha256_bytes, sha256_json

PROPOSAL_SCHEMA_ID = "cf1s_edit_proposal.v1"
CANONICAL_JSON_VERSION = "CF1S_CANONICAL_JSON_V1"
INTERPRETATION_SCOPE = "MODEL_RESPONSE_TO_COUNTERFACTUAL_SEQUENCE_EDIT"
PIPELINE_ID = "CF1S_COLD_FORWARD_V1"


class EvaluationStage(str, Enum):
    CONTRACT = "CONTRACT"
    DEVELOPMENT_PRODUCTION = "DEVELOPMENT_PRODUCTION"
    PRIMARY32 = "PRIMARY32"
    PROBLEM20_BLIND = "PROBLEM20_BLIND"


class EvidenceKind(str, Enum):
    SYNTHETIC_CONTRACT = "SYNTHETIC_CONTRACT"
    ACTUAL_MODEL_FORWARD = "ACTUAL_MODEL_FORWARD"
    LEGACY_INCOMPLETE = "LEGACY_INCOMPLETE"


class OutputDisposition(str, Enum):
    CONTRACT_ONLY = "CONTRACT_ONLY"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"
    WITHHELD = "WITHHELD"
    AUTHORIZED_RECOMMENDATION = "AUTHORIZED_RECOMMENDATION"


class ScientificStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ConstructionStatus(str, Enum):
    COMPLETE = "COMPLETE"
    NOT_CONSTRUCTIBLE = "NOT_CONSTRUCTIBLE"
    INCOMPLETE = "INCOMPLETE"


class RiskEffectDirection(str, Enum):
    RISK_INCREASE = "RISK_INCREASE"
    RISK_DECREASE = "RISK_DECREASE"
    NO_MATERIAL_CHANGE = "NO_MATERIAL_CHANGE"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class DirectionConsistency(str, Enum):
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"
    ONE_SCOPE_MATERIAL_ONLY = "ONE_SCOPE_MATERIAL_ONLY"
    BOTH_SCOPES_NO_MATERIAL_CHANGE = "BOTH_SCOPES_NO_MATERIAL_CHANGE"
    INSUFFICIENT_FOLDS = "INSUFFICIENT_FOLDS"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class OrderingStatus(str, Enum):
    NOT_ORDERABLE = "NOT_ORDERABLE"
    EVIDENCE_ORDERED = "EVIDENCE_ORDERED"
    ORDERING_FAILED = "ORDERING_FAILED"


class ForwardKind(str, Enum):
    ATTRIBUTION_FORWARD = "ATTRIBUTION_FORWARD"
    BASELINE_FORWARD = "BASELINE_FORWARD"
    FORCED_IDENTITY_FORWARD = "FORCED_IDENTITY_FORWARD"
    CANDIDATE_EFFECT_FORWARD = "CANDIDATE_EFFECT_FORWARD"


class EvaluationCoverage(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    NONE = "NONE"


class ExecutionReadiness(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class TransactionConflictReason(str, Enum):
    DUPLICATE_ATOMIC_TARGET = "DUPLICATE_ATOMIC_TARGET"
    CONFLICTING_RAW_EDIT = "CONFLICTING_RAW_EDIT"
    CONFLICTING_TOKEN_EDIT = "CONFLICTING_TOKEN_EDIT"
    NON_DISTINCT_EVENT_IDS = "NON_DISTINCT_EVENT_IDS"
    ORIGINAL_VALUE_MISMATCH = "ORIGINAL_VALUE_MISMATCH"
    TRANSACTION_ORDER_DEPENDENT = "TRANSACTION_ORDER_DEPENDENT"


class ScopeGateReason(str, Enum):
    SEARCH_IDENTITY_FAILED = "SEARCH_IDENTITY_FAILED"
    SEARCH_NOISE_CEILING_FAILED = "SEARCH_NOISE_CEILING_FAILED"
    HOLDOUT_IDENTITY_FAILED = "HOLDOUT_IDENTITY_FAILED"
    HOLDOUT_NOISE_CEILING_FAILED = "HOLDOUT_NOISE_CEILING_FAILED"
    CANDIDATE_FORWARD_SKIPPED_BY_SCOPE_GATE = "CANDIDATE_FORWARD_SKIPPED_BY_SCOPE_GATE"


FORBIDDEN_OUTPUT_KEYS = ("ranking_status",)
FORBIDDEN_ENUM_VALUES = ("NOT_RANKABLE",)


def canonical_json_bytes(obj: Any) -> bytes:
    """CF1S_CANONICAL_JSON_V1: UTF-8, sorted keys, no spaces, nulls preserved."""
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


@dataclass(frozen=True)
class AuthorityContext:
    """Runner-assembled authority — never trust case payload booleans alone."""

    evaluation_stage: EvaluationStage
    execution_authorized: bool = False
    evidence_output_authorized: bool = False
    recommendation_authorized: bool = False
    development_production_execution_authorized: bool = False
    primary32_execution_authorized: bool = False
    problem20_execution_authorized: bool = False
    authorization_artifact_sha256: Optional[str] = None
    policy_sha256: Optional[str] = None
    code_sha256: Optional[str] = None
    input_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evaluation_stage"] = self.evaluation_stage.value
        return d


@dataclass(frozen=True)
class ExecutionEvidenceFlags:
    production_input_verified: bool = False
    pipeline_integrity_verified: bool = False
    attribution_forward_executed: bool = False
    baseline_forward_complete: bool = False
    identity_forward_complete: bool = False
    candidate_effect_forward_complete: bool = False
    full_sequence_cold_rebuild: bool = False
    identity_pass: bool = False
    noise_ceiling_pass: bool = False
    complete_family_executed: bool = False
    all_required_folds_complete: bool = False

    @property
    def actual_checkpoint_forward(self) -> bool:
        """Derived: attribution-only does not count."""
        return bool(self.baseline_forward_complete)

    @property
    def model_effects_evaluated(self) -> bool:
        return bool(
            self.baseline_forward_complete
            and self.identity_pass
            and self.noise_ceiling_pass
            and self.complete_family_executed
            and self.all_required_folds_complete
            and self.candidate_effect_forward_complete
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["actual_checkpoint_forward"] = self.actual_checkpoint_forward
        d["model_effects_evaluated"] = self.model_effects_evaluated
        return d


def execution_evidence_valid(
    authority: AuthorityContext,
    evidence: ExecutionEvidenceFlags,
) -> bool:
    return (
        authority.execution_authorized is True
        and authority.evidence_output_authorized is True
        and evidence.production_input_verified is True
        and evidence.pipeline_integrity_verified is True
        and evidence.baseline_forward_complete is True
    )


def scientific_evaluation_complete(
    authority: AuthorityContext,
    evidence: ExecutionEvidenceFlags,
) -> bool:
    return (
        execution_evidence_valid(authority, evidence)
        and evidence.full_sequence_cold_rebuild is True
        and evidence.identity_pass is True
        and evidence.noise_ceiling_pass is True
        and evidence.complete_family_executed is True
        and evidence.all_required_folds_complete is True
        and evidence.candidate_effect_forward_complete is True
    )


def resolve_disposition(
    *,
    authority: AuthorityContext,
    evidence: ExecutionEvidenceFlags,
    evidence_kind: EvidenceKind,
    integrity_failed: bool = False,
    untrusted_authorization_claim: bool = False,
) -> OutputDisposition:
    if integrity_failed or untrusted_authorization_claim:
        return OutputDisposition.WITHHELD
    if evidence_kind == EvidenceKind.LEGACY_INCOMPLETE:
        return OutputDisposition.WITHHELD
    if authority.evaluation_stage == EvaluationStage.CONTRACT:
        if evidence.baseline_forward_complete or evidence.actual_checkpoint_forward:
            return OutputDisposition.WITHHELD
        return OutputDisposition.CONTRACT_ONLY
    if execution_evidence_valid(authority, evidence):
        if (
            authority.recommendation_authorized
            and authority.authorization_artifact_sha256
        ):
            return OutputDisposition.AUTHORIZED_RECOMMENDATION
        return OutputDisposition.EVIDENCE_ONLY
    if evidence.attribution_forward_executed and not evidence.baseline_forward_complete:
        return OutputDisposition.WITHHELD
    return OutputDisposition.WITHHELD


def risk_direction_from_delta(delta: float, *, threshold: float) -> RiskEffectDirection:
    if delta >= threshold:
        return RiskEffectDirection.RISK_INCREASE
    if delta <= -threshold:
        return RiskEffectDirection.RISK_DECREASE
    return RiskEffectDirection.NO_MATERIAL_CHANGE


def direction_consistency(
    search_direction: RiskEffectDirection,
    holdout_direction: RiskEffectDirection,
    *,
    folds_insufficient: bool = False,
) -> DirectionConsistency:
    if folds_insufficient:
        return DirectionConsistency.INSUFFICIENT_FOLDS
    if (
        search_direction == RiskEffectDirection.NOT_EVALUABLE
        or holdout_direction == RiskEffectDirection.NOT_EVALUABLE
    ):
        return DirectionConsistency.NOT_EVALUABLE
    material = {RiskEffectDirection.RISK_INCREASE, RiskEffectDirection.RISK_DECREASE}
    s_mat = search_direction in material
    h_mat = holdout_direction in material
    if not s_mat and not h_mat:
        return DirectionConsistency.BOTH_SCOPES_NO_MATERIAL_CHANGE
    if s_mat != h_mat:
        return DirectionConsistency.ONE_SCOPE_MATERIAL_ONLY
    if search_direction == holdout_direction:
        return DirectionConsistency.CONSISTENT
    return DirectionConsistency.INCONSISTENT


def resolve_scientific_status(
    *,
    construction_status: ConstructionStatus,
    core_acceptance_status: Optional[str],
    consistency: DirectionConsistency,
    identity_or_noise_failed: bool,
    execution_failed: bool,
) -> Dict[str, str]:
    if execution_failed or identity_or_noise_failed:
        return {
            "scientific_status": ScientificStatus.NOT_EVALUABLE.value,
            "scientific_status_reason": "EXECUTION_OR_IDENTITY_NOISE_FAILURE",
        }
    if construction_status == ConstructionStatus.NOT_CONSTRUCTIBLE:
        return {
            "scientific_status": ScientificStatus.NOT_EVALUATED.value,
            "scientific_status_reason": "NO_CONSTRUCTIBLE_FAMILY",
        }
    if consistency in (
        DirectionConsistency.INCONSISTENT,
        DirectionConsistency.ONE_SCOPE_MATERIAL_ONLY,
    ):
        return {
            "scientific_status": ScientificStatus.INCONCLUSIVE.value,
            "scientific_status_reason": "SEARCH_HOLDOUT_DIRECTION_MISMATCH",
        }
    if consistency == DirectionConsistency.BOTH_SCOPES_NO_MATERIAL_CHANGE:
        return {
            "scientific_status": ScientificStatus.NOT_SUPPORTED.value,
            "scientific_status_reason": "BOTH_SCOPES_NO_MATERIAL_CHANGE",
        }
    if consistency == DirectionConsistency.INSUFFICIENT_FOLDS:
        return {
            "scientific_status": ScientificStatus.NOT_EVALUABLE.value,
            "scientific_status_reason": "INSUFFICIENT_FOLDS",
        }
    if core_acceptance_status == "SUPPORTED" and consistency == DirectionConsistency.CONSISTENT:
        return {
            "scientific_status": ScientificStatus.SUPPORTED.value,
            "scientific_status_reason": "CORE_ACCEPTANCE_AND_DIRECTION_CONSISTENT",
        }
    if consistency == DirectionConsistency.CONSISTENT:
        return {
            "scientific_status": ScientificStatus.NOT_SUPPORTED.value,
            "scientific_status_reason": "ACCEPTANCE_CRITERIA_NOT_MET",
        }
    return {
        "scientific_status": ScientificStatus.NOT_EVALUATED.value,
        "scientific_status_reason": "NOT_EVALUATED",
    }


def interpretation_contract() -> Dict[str, Any]:
    return {
        "interpretation_scope": INTERPRETATION_SCOPE,
        "causal_effect_claim_authorized": False,
        "real_world_action_authorized": False,
    }


def assert_no_forbidden_ranking_fields(payload: Mapping[str, Any]) -> None:
    def _walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                if k in FORBIDDEN_OUTPUT_KEYS:
                    raise CoreContractError(f"forbidden key {k} at {path}")
                if k == "ordering_status" and v in FORBIDDEN_ENUM_VALUES:
                    raise CoreContractError(f"forbidden ordering enum {v}")
                if isinstance(v, str) and v in FORBIDDEN_ENUM_VALUES and k.endswith("status"):
                    raise CoreContractError(f"forbidden enum {v} at {path}.{k}")
                _walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}]")

    _walk(payload)


def contract_case_defaults() -> Dict[str, Any]:
    return {
        "disposition": OutputDisposition.CONTRACT_ONLY.value,
        "scientific_status": ScientificStatus.NOT_EVALUATED.value,
        "ordering_status": OrderingStatus.NOT_ORDERABLE.value,
        "recommendation": None,
        "executable_edit": None,
        **interpretation_contract(),
        "model_effects_evaluated": False,
        "actual_checkpoint_forward": False,
        "baseline_forward_complete": False,
    }
