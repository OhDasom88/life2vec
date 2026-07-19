"""Validated raw transaction: grounding → schema → retokenization → allowed change set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError


CANONICAL_SORT_KEYS = (
    "case_id",
    "event_id",
    "mg_id",
    "feature_id",
    "raw_field_path",
    "operation",
    "canonical_value",
)


@dataclass(frozen=True)
class ValidatedRawAtomicEdit:
    case_id: str
    event_id: str
    mg_id: str
    feature_id: str
    raw_field_path: str
    operation: str
    canonical_value: Any
    to_tokens: Tuple[str, ...]
    grounding_sha256: str
    schema_projection_sha256: str
    retokenization_sha256: str
    submitted_order: int = 0

    def sort_key(self) -> Tuple:
        return (
            self.case_id,
            self.event_id,
            self.mg_id,
            self.feature_id,
            self.raw_field_path,
            self.operation,
            _canonical_value_key(self.canonical_value),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "event_id": self.event_id,
            "mg_id": self.mg_id,
            "feature_id": self.feature_id,
            "raw_field_path": self.raw_field_path,
            "operation": self.operation,
            "canonical_value": self.canonical_value,
            "to_tokens": list(self.to_tokens),
            "grounding_sha256": self.grounding_sha256,
            "schema_projection_sha256": self.schema_projection_sha256,
            "retokenization_sha256": self.retokenization_sha256,
            "submitted_order": self.submitted_order,
        }


@dataclass(frozen=True)
class AllowedChangeSet:
    targeted_raw_fields: Tuple[str, ...]
    schema_dependent_fields: Tuple[str, ...]
    tokenizer_derived_tokens: Tuple[str, ...]
    permitted_event_metadata: Tuple[str, ...]
    sha256: str

    def contains_path(self, path: str) -> bool:
        return path in set(self.targeted_raw_fields) | set(self.schema_dependent_fields) | set(
            self.tokenizer_derived_tokens
        ) | set(self.permitted_event_metadata)


@dataclass(frozen=True)
class ValidatedRawTransaction:
    """Immutable production transaction; unchecked to_tokens dicts are rejected."""

    case_id: str
    atomics: Tuple[ValidatedRawAtomicEdit, ...]
    allowed_change_set: AllowedChangeSet
    submitted_transaction_sha: str
    canonical_atomic_set_sha: str
    canonical_validated_transaction_sha: str
    original_caseevents_sha: str
    edited_caseevents_sha: str
    transaction_mode: str = "CANDIDATE"
    identity: bool = False

    def edits_for_apply(self) -> List[Dict[str, Any]]:
        """Materialize apply payload from validated atomics only."""
        return [
            {
                "event_id": a.event_id,
                "to_tokens": list(a.to_tokens),
                "feature_id": a.feature_id,
                "mg_id": a.mg_id,
            }
            for a in self.atomics
        ]

    def parent_subset(self, event_ids: Sequence[str]) -> "ValidatedRawTransaction":
        wanted = {str(x) for x in event_ids}
        subset = tuple(a for a in self.atomics if a.event_id in wanted)
        if not subset:
            raise CoreContractError("parent subset empty")
        # Parent must be exact subset of bundle atomics
        return build_validated_raw_transaction(
            case_id=self.case_id,
            atomics=list(subset),
            allowed_change_set=self.allowed_change_set,
            original_caseevents_sha=self.original_caseevents_sha,
            edited_caseevents_sha="",  # recomputed by caller after materialization
            transaction_mode=self.transaction_mode,
            identity=False,
            skip_edited_sha_required=True,
        )


def _canonical_value_key(value: Any) -> str:
    return canonical_json_sha256({"v": value})


def build_allowed_change_set(
    *,
    targeted_raw_fields: Sequence[str],
    schema_dependent_fields: Sequence[str] = (),
    tokenizer_derived_tokens: Sequence[str] = (),
    permitted_event_metadata: Sequence[str] = (),
) -> AllowedChangeSet:
    body = {
        "targeted_raw_fields": sorted(set(str(x) for x in targeted_raw_fields)),
        "schema_dependent_fields": sorted(set(str(x) for x in schema_dependent_fields)),
        "tokenizer_derived_tokens": sorted(set(str(x) for x in tokenizer_derived_tokens)),
        "permitted_event_metadata": sorted(set(str(x) for x in permitted_event_metadata)),
    }
    return AllowedChangeSet(
        targeted_raw_fields=tuple(body["targeted_raw_fields"]),
        schema_dependent_fields=tuple(body["schema_dependent_fields"]),
        tokenizer_derived_tokens=tuple(body["tokenizer_derived_tokens"]),
        permitted_event_metadata=tuple(body["permitted_event_metadata"]),
        sha256=canonical_json_sha256(body),
    )


def assert_change_set_closed(
    *,
    allowed: AllowedChangeSet,
    actual_changed_paths: Sequence[str],
    required_changed_paths: Sequence[str],
) -> None:
    actual = set(str(x) for x in actual_changed_paths)
    required = set(str(x) for x in required_changed_paths)
    allowed_paths = (
        set(allowed.targeted_raw_fields)
        | set(allowed.schema_dependent_fields)
        | set(allowed.tokenizer_derived_tokens)
        | set(allowed.permitted_event_metadata)
    )
    extra = actual - allowed_paths
    if extra:
        raise CoreContractError(f"actual_changed_paths outside allowed_change_set: {sorted(extra)}")
    missing = required - actual
    if missing:
        raise CoreContractError(f"required_changed_paths missing from actual: {sorted(missing)}")


def _reject_conflicts(atomics: Sequence[ValidatedRawAtomicEdit]) -> None:
    seen_keys: Dict[Tuple, ValidatedRawAtomicEdit] = {}
    feature_targets: Dict[Tuple[str, str, str], ValidatedRawAtomicEdit] = {}
    for a in atomics:
        key = a.sort_key()
        if key in seen_keys:
            prev = seen_keys[key]
            if prev.to_dict() != a.to_dict():
                raise CoreContractError("duplicate canonical key with differing payload")
            raise CoreContractError("duplicate atomic edit")
        seen_keys[key] = a
        ft = (a.case_id, a.event_id, a.feature_id)
        if ft in feature_targets:
            raise CoreContractError(f"duplicate feature target: {ft}")
        feature_targets[ft] = a
        if not a.to_tokens:
            raise CoreContractError(f"empty to_tokens for event {a.event_id}")


def build_validated_raw_transaction(
    *,
    case_id: str,
    atomics: Sequence[ValidatedRawAtomicEdit],
    allowed_change_set: AllowedChangeSet,
    original_caseevents_sha: str,
    edited_caseevents_sha: str,
    transaction_mode: str = "CANDIDATE",
    identity: bool = False,
    skip_edited_sha_required: bool = False,
) -> ValidatedRawTransaction:
    if not identity and not atomics:
        raise CoreContractError("non-identity transaction requires atomics")
    atoms = tuple(atomics)
    _reject_conflicts(atoms)

    submitted_payload = {
        "case_id": case_id,
        "submitted_order": [a.to_dict() for a in atoms],
        "transaction_mode": transaction_mode,
    }
    submitted_sha = canonical_json_sha256(submitted_payload)

    canonical_atoms = sorted(atoms, key=lambda a: a.sort_key())
    canonical_set_payload = {
        "case_id": case_id,
        "atomics": [a.to_dict() for a in canonical_atoms],
    }
    canonical_set_sha = canonical_json_sha256(canonical_set_payload)

    validated_payload = {
        "case_id": case_id,
        "atomics": [a.to_dict() for a in canonical_atoms],
        "allowed_change_set_sha": allowed_change_set.sha256,
        "original_caseevents_sha": original_caseevents_sha,
        "transaction_mode": transaction_mode,
        "identity": bool(identity),
    }
    validated_sha = canonical_json_sha256(validated_payload)

    if not skip_edited_sha_required and not edited_caseevents_sha and not identity:
        raise CoreContractError("edited_caseevents_sha required")

    return ValidatedRawTransaction(
        case_id=str(case_id),
        atomics=atoms,
        allowed_change_set=allowed_change_set,
        submitted_transaction_sha=submitted_sha,
        canonical_atomic_set_sha=canonical_set_sha,
        canonical_validated_transaction_sha=validated_sha,
        original_caseevents_sha=original_caseevents_sha,
        edited_caseevents_sha=edited_caseevents_sha or "",
        transaction_mode=transaction_mode,
        identity=bool(identity),
    )


def build_identity_transaction(*, case_id: str, original_caseevents_sha: str) -> ValidatedRawTransaction:
    allowed = build_allowed_change_set(targeted_raw_fields=())
    return build_validated_raw_transaction(
        case_id=case_id,
        atomics=(),
        allowed_change_set=allowed,
        original_caseevents_sha=original_caseevents_sha,
        edited_caseevents_sha=original_caseevents_sha,
        transaction_mode="FORCED_IDENTITY",
        identity=True,
        skip_edited_sha_required=True,
    )


def reject_unchecked_token_dict(edits: Any) -> None:
    """Production entrypoints must not accept bare to_tokens dict sequences."""
    if isinstance(edits, ValidatedRawTransaction):
        return
    raise CoreContractError(
        "unchecked to_tokens dict rejected; require ValidatedRawTransaction or identity transaction"
    )


def verify_exact_parent_transactions(
    *,
    bundle: ValidatedRawTransaction,
    parents: Sequence[ValidatedRawTransaction],
    declared_parent_shas: Sequence[str],
) -> List[str]:
    """Require an exact canonical two-parent union for a two-event bundle."""
    if len(parents) != 2:
        raise CoreContractError("PAIR_REQUIRES_EXACTLY_TWO_PARENTS")
    if any(parent.case_id != bundle.case_id for parent in parents):
        raise CoreContractError("PARENT_CASE_MISMATCH")
    if any(len(parent.atomics) != 1 for parent in parents):
        raise CoreContractError("PAIR_PARENT_MUST_BE_ATOMIC")

    canonical_parents = sorted(
        parents,
        key=lambda parent: parent.atomics[0].sort_key(),
    )
    expected_shas = [
        parent.canonical_validated_transaction_sha for parent in canonical_parents
    ]
    if len(set(expected_shas)) != 2:
        raise CoreContractError("DUPLICATE_PARENT_TRANSACTION_SHA")
    if list(declared_parent_shas) != expected_shas:
        raise CoreContractError("PARENT_TRANSACTION_SHA_ORDER_OR_VALUE_MISMATCH")

    bundle_atoms = {canonical_json_sha256(atomic.to_dict()) for atomic in bundle.atomics}
    parent_atoms = {
        canonical_json_sha256(atomic.to_dict())
        for parent in canonical_parents
        for atomic in parent.atomics
    }
    if bundle_atoms != parent_atoms or len(bundle_atoms) != 2:
        raise CoreContractError("BUNDLE_ATOMIC_SET_NOT_EXACT_PARENT_UNION")
    return expected_shas


def build_canonical_atomic_evidence_manifest(
    atomics: Sequence[ValidatedRawAtomicEdit],
) -> Dict[str, Any]:
    """Sort by canonical atomic key; submitted order is provenance-only."""
    rows = []
    for a in sorted(atomics, key=lambda x: x.sort_key()):
        rows.append(
            {
                "atomic_id": f"{a.case_id}|{a.event_id}|{a.feature_id}|{a.raw_field_path}",
                "grounding_sha": a.grounding_sha256,
                "schema_projection_sha": a.schema_projection_sha256,
                "retokenization_sha": a.retokenization_sha256,
                "submitted_order": a.submitted_order,
            }
        )
    body = {"version": "CF1S_ATOMIC_EVIDENCE_MANIFEST_V1", "atomics": rows}
    return {
        **body,
        "canonical_atomic_evidence_manifest_sha": canonical_json_sha256(body),
    }


def transaction_evidence_chain(
    *,
    original_caseevents_sha: str,
    canonical_atomic_evidence_manifest_sha: str,
    allowed_change_set_sha: str,
    canonical_validated_transaction_sha: str,
    edited_caseevents_sha: str,
    stage_a_input_sha: str = "",
    stage_a_output_sha: str = "",
    diagnosis_batch_sha: str = "",
    semantic_critic_input_sha: str = "",
) -> Dict[str, Any]:
    body = {
        "original_caseevents_sha": original_caseevents_sha,
        "canonical_atomic_evidence_manifest_sha": canonical_atomic_evidence_manifest_sha,
        "allowed_change_set_sha": allowed_change_set_sha,
        "canonical_validated_transaction_sha": canonical_validated_transaction_sha,
        "edited_caseevents_sha": edited_caseevents_sha,
        "stage_a_input_sha": stage_a_input_sha,
        "stage_a_output_sha": stage_a_output_sha,
        "diagnosis_batch_sha": diagnosis_batch_sha,
        "semantic_critic_input_sha": semantic_critic_input_sha,
    }
    return {**body, "evidence_chain_sha": canonical_json_sha256(body)}


def permutation_canonical_invariant(
    permutations: Sequence[Sequence[ValidatedRawAtomicEdit]],
    *,
    case_id: str,
    allowed_change_set: AllowedChangeSet,
    original_caseevents_sha: str,
) -> Dict[str, Any]:
    """All permutations share canonical SHA; submitted SHA may differ."""
    submitted = []
    canonical_set = []
    validated = []
    for order in permutations:
        tx = build_validated_raw_transaction(
            case_id=case_id,
            atomics=list(order),
            allowed_change_set=allowed_change_set,
            original_caseevents_sha=original_caseevents_sha,
            edited_caseevents_sha="pending",
            skip_edited_sha_required=True,
        )
        submitted.append(tx.submitted_transaction_sha)
        canonical_set.append(tx.canonical_atomic_set_sha)
        validated.append(tx.canonical_validated_transaction_sha)
    if len(set(canonical_set)) != 1 or len(set(validated)) != 1:
        raise CoreContractError("permutation broke canonical transaction SHA invariance")
    return {
        "submitted_transaction_shas": submitted,
        "canonical_atomic_set_sha": canonical_set[0],
        "canonical_validated_transaction_sha": validated[0],
        "submitted_may_differ": len(set(submitted)) > 1 or len(permutations) == 1,
    }
