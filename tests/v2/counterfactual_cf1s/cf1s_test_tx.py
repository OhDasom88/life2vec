"""Helper to build a minimal ValidatedRawTransaction for unit tests."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import canonical_json_sha256
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_raw_transaction import (
    ValidatedRawAtomicEdit,
    ValidatedRawTransaction,
    build_allowed_change_set,
    build_identity_transaction,
    build_validated_raw_transaction,
)


def make_test_candidate_transaction(
    *,
    case_id: str = "case",
    event_id: str = "e1",
    to_tokens: Sequence[str] = ("A2",),
    feature_id: str = "feat",
) -> ValidatedRawTransaction:
    atomic = ValidatedRawAtomicEdit(
        case_id=case_id,
        event_id=event_id,
        mg_id=f"MG_{event_id}",
        feature_id=feature_id,
        raw_field_path=f"raw.{feature_id}",
        operation="SET_RAW",
        canonical_value=1.0,
        to_tokens=tuple(to_tokens),
        grounding_sha256=canonical_json_sha256({"g": event_id}),
        schema_projection_sha256=canonical_json_sha256({"s": feature_id}),
        retokenization_sha256=canonical_json_sha256({"t": list(to_tokens)}),
        submitted_order=0,
    )
    allowed = build_allowed_change_set(
        targeted_raw_fields=[atomic.raw_field_path],
        tokenizer_derived_tokens=[f"tokens.{event_id}"],
    )
    return build_validated_raw_transaction(
        case_id=case_id,
        atomics=[atomic],
        allowed_change_set=allowed,
        original_caseevents_sha=canonical_json_sha256({"orig": case_id}),
        edited_caseevents_sha=canonical_json_sha256({"edit": case_id}),
        skip_edited_sha_required=True,
    )


def make_test_identity_transaction(*, case_id: str = "case") -> ValidatedRawTransaction:
    return build_identity_transaction(
        case_id=case_id,
        original_caseevents_sha=canonical_json_sha256({"orig": case_id}),
    )
