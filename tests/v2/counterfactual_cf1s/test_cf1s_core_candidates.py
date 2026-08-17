"""Regression: a pair bundle's atomics must exactly match its standalone
dependency-parent atomics, including submitted_order — the bug this guards
against made every real (non-mocked) PAIR candidate fail
verify_exact_parent_transactions with BUNDLE_ATOMIC_SET_NOT_EXACT_PARENT_UNION
or PARENT_TRANSACTION_SHA_ORDER_OR_VALUE_MISMATCH."""

from __future__ import annotations

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_candidates import (
    _build_tx_from_atomics,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_raw_transaction import (
    verify_exact_parent_transactions,
)


def _atomic(event_id: str, feature_id: str, target_raw: float) -> dict:
    return {
        "event_id": event_id,
        "feature_id": feature_id,
        "mg_id": f"MG_{event_id}",
        "raw_field_path": f"raw.{feature_id}",
        "target_raw": target_raw,
        "to_tokens": [f"tok_{event_id}_{feature_id}"],
        "grounding_sha256": "a" * 64,
        "schema_projection_sha256": "b" * 64,
        "retokenization_sha256": "c" * 64,
    }


def test_submitted_order_offset_applies_to_singleton_build():
    atomic = _atomic("e1", "substrate_temp_c", 20.0)
    built0 = _build_tx_from_atomics(
        case_id="case-1", atomics_raw=[atomic], original_caseevents_sha="d" * 64
    )
    built1 = _build_tx_from_atomics(
        case_id="case-1",
        atomics_raw=[atomic],
        original_caseevents_sha="d" * 64,
        submitted_order_offset=1,
    )
    tx0 = built0["validated_transaction"]
    tx1 = built1["validated_transaction"]
    assert tx0.atomics[0].submitted_order == 0
    assert tx1.atomics[0].submitted_order == 1
    assert tx0.atomics[0].to_dict() != tx1.atomics[0].to_dict()


def test_pair_and_standalone_parents_verify_exact_parent_transactions():
    a = _atomic("e1", "substrate_temp_c", 20.0)
    b = _atomic("e2", "substrate_temp_c", 21.0)

    pair = _build_tx_from_atomics(
        case_id="case-1", atomics_raw=[a, b], original_caseevents_sha="d" * 64
    )
    bundle_tx = pair["validated_transaction"]

    parent_a = _build_tx_from_atomics(
        case_id="case-1",
        atomics_raw=[a],
        original_caseevents_sha="d" * 64,
        submitted_order_offset=0,
    )["validated_transaction"]
    parent_b = _build_tx_from_atomics(
        case_id="case-1",
        atomics_raw=[b],
        original_caseevents_sha="d" * 64,
        submitted_order_offset=1,
    )["validated_transaction"]

    canonical_parents = sorted(
        [parent_a, parent_b], key=lambda p: p.atomics[0].sort_key()
    )
    declared_parent_shas = [
        p.canonical_validated_transaction_sha for p in canonical_parents
    ]
    verify_exact_parent_transactions(
        bundle=bundle_tx,
        parents=canonical_parents,
        declared_parent_shas=declared_parent_shas,
    )


def test_mismatched_submitted_order_offset_fails_closed():
    a = _atomic("e1", "substrate_temp_c", 20.0)
    b = _atomic("e2", "substrate_temp_c", 21.0)

    pair = _build_tx_from_atomics(
        case_id="case-1", atomics_raw=[a, b], original_caseevents_sha="d" * 64
    )
    bundle_tx = pair["validated_transaction"]

    # Both parents built as offset=0 (the original bug): the second parent's
    # atom no longer matches its bundle-position representation.
    parent_a = _build_tx_from_atomics(
        case_id="case-1", atomics_raw=[a], original_caseevents_sha="d" * 64
    )["validated_transaction"]
    parent_b = _build_tx_from_atomics(
        case_id="case-1", atomics_raw=[b], original_caseevents_sha="d" * 64
    )["validated_transaction"]

    canonical_parents = sorted(
        [parent_a, parent_b], key=lambda p: p.atomics[0].sort_key()
    )
    declared_parent_shas = [
        p.canonical_validated_transaction_sha for p in canonical_parents
    ]
    try:
        verify_exact_parent_transactions(
            bundle=bundle_tx,
            parents=canonical_parents,
            declared_parent_shas=declared_parent_shas,
        )
        raised = False
    except Exception:
        raised = True
    assert raised
