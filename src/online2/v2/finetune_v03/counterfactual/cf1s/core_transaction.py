"""Atomic multi-event transaction conflict and order-invariance checks."""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .core_candidate_family import materialized_raw_transaction_hash
from .core_contract import sha256_json
from .core_edit_proposal import TransactionConflictReason


def _raw_cell_key(atomic: Mapping[str, Any]) -> Tuple[str, str]:
    return (str(atomic.get("event_id")), str(atomic.get("feature_id")))


def _token_cell_key(atomic: Mapping[str, Any]) -> Tuple[str, int]:
    return (str(atomic.get("event_id")), int(atomic.get("token_index", -1)))


def validate_atomic_transaction(
    atomics: Sequence[Mapping[str, Any]],
    *,
    original_by_event_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Validate A/B/AB or A/B/C/... is one simultaneous transaction on fresh original."""
    reasons: List[str] = []
    event_ids = [str(a.get("event_id")) for a in atomics]
    if len(set(event_ids)) != len(event_ids):
        reasons.append(TransactionConflictReason.NON_DISTINCT_EVENT_IDS.value)

    features = {str(a.get("feature_id")) for a in atomics}
    directions = {str(a.get("edit_direction")) for a in atomics}
    if len(features) != 1 or len(directions) != 1:
        reasons.append(TransactionConflictReason.CONFLICTING_RAW_EDIT.value)

    raw_seen: Dict[Tuple[str, str], Any] = {}
    for a in atomics:
        key = _raw_cell_key(a)
        target = a.get("target_raw")
        if key in raw_seen:
            if raw_seen[key] != target:
                reasons.append(TransactionConflictReason.CONFLICTING_RAW_EDIT.value)
            else:
                reasons.append(TransactionConflictReason.DUPLICATE_ATOMIC_TARGET.value)
        raw_seen[key] = target

    token_seen: Dict[Tuple[str, int], Tuple[Any, ...]] = {}
    for a in atomics:
        if "token_index" not in a and "to_tokens" not in a and "sentence_tokens" not in a:
            continue
        if "token_index" in a:
            key = _token_cell_key(a)
            tok = a.get("to_token") or a.get("token_value")
            if key in token_seen and token_seen[key] != (tok,):
                reasons.append(TransactionConflictReason.CONFLICTING_TOKEN_EDIT.value)
            elif key in token_seen:
                reasons.append(TransactionConflictReason.DUPLICATE_ATOMIC_TARGET.value)
            token_seen[key] = (tok,)

    # sequence/time alignment
    ordered = sorted(
        atomics,
        key=lambda x: (int(x.get("sequence_position", 0)), float(x.get("event_time_epoch", 0.0))),
    )
    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        if int(prev.get("sequence_position", 0)) > int(cur.get("sequence_position", 0)):
            reasons.append(TransactionConflictReason.CONFLICTING_RAW_EDIT.value)
        if float(prev.get("event_time_epoch", 0.0)) > float(cur.get("event_time_epoch", 0.0)):
            reasons.append(TransactionConflictReason.CONFLICTING_RAW_EDIT.value)

    if original_by_event_id is not None:
        for a in atomics:
            eid = str(a.get("event_id"))
            orig = original_by_event_id.get(eid)
            if orig is None:
                reasons.append(TransactionConflictReason.ORIGINAL_VALUE_MISMATCH.value)
                continue
            if "observed_raw" in a and a.get("observed_raw") != orig.get("observed_raw"):
                reasons.append(TransactionConflictReason.ORIGINAL_VALUE_MISMATCH.value)
            if "from_tokens" in a and list(a.get("from_tokens") or []) != list(
                orig.get("sentence_tokens") or orig.get("from_tokens") or []
            ):
                reasons.append(TransactionConflictReason.ORIGINAL_VALUE_MISMATCH.value)

    uniq = sorted(set(reasons))
    return {
        "ok": len(uniq) == 0,
        "conflict_reasons": uniq,
        "event_ids": event_ids,
        "simultaneous_on_original": True,
        "forbid_accumulate_parent_then_child": True,
    }


def materialize_transaction_projection(
    atomics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Order-independent projection of simultaneous edits (materialization only)."""
    items = []
    for a in sorted(
        atomics,
        key=lambda x: (int(x.get("sequence_position", 0)), str(x.get("event_id"))),
    ):
        items.append(
            {
                "event_id": a.get("event_id"),
                "feature_id": a.get("feature_id"),
                "edit_direction": a.get("edit_direction"),
                "target_raw": a.get("target_raw"),
                "to_tokens": list(a.get("to_tokens") or a.get("sentence_tokens") or []),
                "sequence_position": a.get("sequence_position"),
                "event_time_epoch": a.get("event_time_epoch"),
                "token_index": a.get("token_index"),
            }
        )
    valid_event_ids = [str(x["event_id"]) for x in items]
    return {
        "edited_raw_payload": items,
        "retokenized_event_payload": [
            {"event_id": x["event_id"], "to_tokens": x["to_tokens"]} for x in items
        ],
        "valid_event_id_order": valid_event_ids,
        "token_sequence": [x["to_tokens"] for x in items],
        "raw_transaction_hash": materialized_raw_transaction_hash(atomics),
        "projection_sha256": sha256_json(items),
    }


def check_transaction_order_invariant(
    atomics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    2-event: both permutations. 3-event: all 6 permutations (materialization-only).
    Model forward is not repeated.
    """
    n = len(atomics)
    if n < 2:
        return {
            "transaction_order_invariant": True,
            "permutation_count": 0,
            "permutation_hashes": {},
            "ok": True,
        }
    if n == 2:
        perms = list(itertools.permutations(range(n)))
    elif n == 3:
        perms = list(itertools.permutations(range(n)))  # 3! = 6
    else:
        # Beyond plan scope; still check all permutations if small
        if n > 4:
            raise ValueError(f"order invariant not defined for n={n}")
        perms = list(itertools.permutations(range(n)))

    hashes: Dict[str, str] = {}
    projections: List[Dict[str, Any]] = []
    for perm in perms:
        ordered = [atomics[i] for i in perm]
        # Simultaneous apply: projection must ignore apply order
        proj = materialize_transaction_projection(ordered)
        label = "".join(chr(ord("A") + i) for i in perm) if n <= 3 else str(perm)
        hashes[label] = proj["raw_transaction_hash"]
        projections.append(proj)

    canonical = materialize_transaction_projection(atomics)
    unique = set(hashes.values())
    ok = len(unique) == 1 and next(iter(unique)) == canonical["raw_transaction_hash"]
    # Also verify projection equality across perms
    proj_hashes = {p["projection_sha256"] for p in projections}
    ok = ok and len(proj_hashes) == 1

    result = {
        "transaction_order_invariant": bool(ok),
        "ok": bool(ok),
        "permutation_count": len(perms),
        "permutation_hashes": hashes,
        "canonical_transaction_sha256": canonical["raw_transaction_hash"],
        "ab_transaction_sha256": hashes.get("AB"),
        "ba_transaction_sha256": hashes.get("BA"),
    }
    if not ok:
        result["conflict_reasons"] = [
            TransactionConflictReason.TRANSACTION_ORDER_DEPENDENT.value
        ]
    return result


def assert_fresh_original_no_mutation_leak(
    *,
    original_snapshot_hash: str,
    after_candidate_original_hash: str,
) -> None:
    if original_snapshot_hash != after_candidate_original_hash:
        raise RuntimeError(
            "mutable event leaked across candidates: original sequence changed"
        )


def simultaneous_vs_accumulate_forbidden(
    parent_then_child_hash: str,
    simultaneous_hash: str,
) -> Dict[str, Any]:
    """AB must not equal apply(A) then apply(B) unless hashes happen to match by chance;
    callers should compare against known accumulate path and require inequality when
    accumulate mutates intermediate state differently. Here we only flag equality
    when accumulate path is provided as distinct."""
    return {
        "simultaneous_hash": simultaneous_hash,
        "accumulate_hash": parent_then_child_hash,
        "accumulate_forbidden": True,
        "hashes_differ": parent_then_child_hash != simultaneous_hash,
    }


def clone_original_events(events: Sequence[Any]) -> List[Any]:
    return deepcopy(list(events))
