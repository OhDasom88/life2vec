"""Validated raw transaction SHA and permutation contracts."""

from __future__ import annotations

import itertools

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import CoreContractError
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_raw_transaction import (
    ValidatedRawAtomicEdit,
    assert_change_set_closed,
    build_allowed_change_set,
    build_identity_transaction,
    build_validated_raw_transaction,
    permutation_canonical_invariant,
    reject_unchecked_token_dict,
)


def _atomic(eid, order, value=1.0):
    return ValidatedRawAtomicEdit(
        case_id="c1",
        event_id=eid,
        mg_id="mg1",
        feature_id="temp",
        raw_field_path=f"events.{eid}.temp",
        operation="SET",
        canonical_value=value,
        to_tokens=("TOK", eid),
        grounding_sha256="g",
        schema_projection_sha256="s",
        retokenization_sha256="r",
        submitted_order=order,
    )


def test_permutation_canonical_invariant_two_and_three():
    allowed = build_allowed_change_set(
        targeted_raw_fields=["events.A.temp", "events.B.temp", "events.C.temp"]
    )
    atoms2 = [_atomic("A", 0), _atomic("B", 1)]
    inv2 = permutation_canonical_invariant(
        list(itertools.permutations(atoms2)),
        case_id="c1",
        allowed_change_set=allowed,
        original_caseevents_sha="orig",
    )
    assert inv2["canonical_atomic_set_sha"]
    assert len(set(inv2["submitted_transaction_shas"])) == 2

    atoms3 = [_atomic("A", 0), _atomic("B", 1), _atomic("C", 2)]
    inv3 = permutation_canonical_invariant(
        list(itertools.permutations(atoms3)),
        case_id="c1",
        allowed_change_set=allowed,
        original_caseevents_sha="orig",
    )
    assert len(inv3["submitted_transaction_shas"]) == 6
    assert len(set(inv3["submitted_transaction_shas"])) == 6


def test_duplicate_feature_target_rejected():
    allowed = build_allowed_change_set(targeted_raw_fields=["events.A.temp"])
    with pytest.raises(CoreContractError, match="duplicate"):
        build_validated_raw_transaction(
            case_id="c1",
            atomics=[_atomic("A", 0, 1.0), _atomic("A", 1, 2.0)],
            allowed_change_set=allowed,
            original_caseevents_sha="o",
            edited_caseevents_sha="e",
        )


def test_change_set_subset_rules():
    allowed = build_allowed_change_set(
        targeted_raw_fields=["raw.a"],
        schema_dependent_fields=["dep.b"],
        tokenizer_derived_tokens=["tok.c"],
    )
    assert_change_set_closed(
        allowed=allowed,
        actual_changed_paths=["raw.a", "dep.b", "tok.c"],
        required_changed_paths=["raw.a"],
    )
    with pytest.raises(CoreContractError):
        assert_change_set_closed(
            allowed=allowed,
            actual_changed_paths=["raw.a", "outside"],
            required_changed_paths=["raw.a"],
        )


def test_unchecked_token_dict_rejected():
    with pytest.raises(CoreContractError, match="unchecked"):
        reject_unchecked_token_dict([{"event_id": "a", "to_tokens": ["x"]}])
    tx = build_identity_transaction(case_id="c1", original_caseevents_sha="o")
    reject_unchecked_token_dict(tx)
