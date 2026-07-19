"""Same-feature/same-direction family construction and exact parent projection."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .core_contract import sha256_json, temporal_arm_for_gaps


def _event_sort_key(ev: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        float(ev.get("event_time_epoch", 0.0)),
        int(ev.get("sequence_position", 0)),
        str(ev.get("event_id", "")),
    )


def canonicalize_events(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(e) for e in sorted(events, key=_event_sort_key)]


def atomic_payload_hash(atomic: Mapping[str, Any]) -> str:
    keys = (
        "event_id",
        "feature_id",
        "edit_direction",
        "target_raw",
        "schema_target",
        "sequence_position",
        "event_time_epoch",
    )
    payload = {k: atomic.get(k) for k in keys}
    return sha256_json(payload)


def materialized_raw_transaction_hash(atomics: Sequence[Mapping[str, Any]]) -> str:
    """Canonical transaction hash after sorting/grouping (no reinversion)."""
    items = []
    for a in sorted(atomics, key=lambda x: (int(x.get("sequence_position", 0)), str(x.get("event_id")))):
        items.append(
            {
                "event_id": a.get("event_id"),
                "feature_id": a.get("feature_id"),
                "edit_direction": a.get("edit_direction"),
                "target_raw": a.get("target_raw"),
                "to_tokens": list(a.get("to_tokens") or a.get("sentence_tokens") or []),
                "sequence_position": a.get("sequence_position"),
            }
        )
    return sha256_json(items)


def validate_family_membership(atomics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(atomics) < 2:
        return {"ok": False, "reason": "NEED_AT_LEAST_TWO"}
    features = {str(a.get("feature_id")) for a in atomics}
    directions = {str(a.get("edit_direction")) for a in atomics}
    event_ids = [str(a.get("event_id")) for a in atomics]
    positions = [int(a.get("sequence_position", -1)) for a in atomics]
    times = [a.get("event_time_epoch") for a in atomics]
    if None in times:
        return {"ok": False, "reason": "MISSING_EVENT_TIME", "missing_event_time_family_count": 1}
    if len(features) != 1:
        return {"ok": False, "reason": "MIXED_FEATURE"}
    if len(directions) != 1:
        return {
            "ok": False,
            "reason": "MIXED_DIRECTION",
            "mixed_direction_family_allowed": False,
        }
    if len(set(event_ids)) != len(event_ids):
        return {"ok": False, "reason": "DUPLICATE_EVENT_ID"}
    if len(set(positions)) != len(positions):
        return {"ok": False, "reason": "DUPLICATE_SEQUENCE_POSITION"}
    ordered = canonicalize_events(atomics)
    gaps = []
    for i in range(1, len(ordered)):
        gaps.append(float(ordered[i]["event_time_epoch"]) - float(ordered[i - 1]["event_time_epoch"]))
    arm, reject = temporal_arm_for_gaps(gaps)
    if reject:
        return {"ok": False, "reason": reject, "gaps_seconds": gaps}
    return {
        "ok": True,
        "feature_id": next(iter(features)),
        "edit_direction": next(iter(directions)),
        "temporal_arm": arm,
        "gaps_seconds": gaps,
        "ordered_event_ids": [str(a["event_id"]) for a in ordered],
        "same_feature_id": True,
        "same_edit_direction": True,
        "mixed_direction_family_allowed": False,
        "missing_event_time_family_count": 0,
    }


def project_parent_subset(
    bundle_atomics: Sequence[Mapping[str, Any]],
    keep_event_ids: Sequence[str],
) -> Dict[str, Any]:
    keep = set(str(x) for x in keep_event_ids)
    projected = [dict(a) for a in bundle_atomics if str(a.get("event_id")) in keep]
    # exact subset — no reinversion/retargeting
    return {
        "atomics": projected,
        "parent_payload_source": "EXACT_SUBSET_PROJECTION",
        "independently_regenerated_parent_count": 0,
        "parent_reinversion_count": 0,
        "parent_retargeting_count": 0,
        "parent_atomic_payload_hash": sha256_json(
            [atomic_payload_hash(a) for a in sorted(projected, key=lambda x: str(x.get("event_id")))]
        ),
        "parent_raw_transaction_hash": materialized_raw_transaction_hash(projected),
        "keep_event_ids": sorted(keep),
    }


def verify_parent_matches_bundle_projection(
    *,
    bundle_atomics: Sequence[Mapping[str, Any]],
    parent_atomics: Sequence[Mapping[str, Any]],
    keep_event_ids: Sequence[str],
) -> Dict[str, Any]:
    expected = project_parent_subset(bundle_atomics, keep_event_ids)
    actual_atomic_hash = sha256_json(
        [
            atomic_payload_hash(a)
            for a in sorted(parent_atomics, key=lambda x: str(x.get("event_id")))
        ]
    )
    actual_raw_hash = materialized_raw_transaction_hash(parent_atomics)
    return {
        "parent_atomic_target_hash_matches_bundle_projection": actual_atomic_hash
        == expected["parent_atomic_payload_hash"],
        "parent_raw_transaction_hash_matches_bundle_projection": actual_raw_hash
        == expected["parent_raw_transaction_hash"],
        "parent_reinversion_count": 0,
        "parent_retargeting_count": 0,
        "expected_atomic_hash": expected["parent_atomic_payload_hash"],
        "actual_atomic_hash": actual_atomic_hash,
        "expected_raw_hash": expected["parent_raw_transaction_hash"],
        "actual_raw_hash": actual_raw_hash,
    }


def build_complete_families(
    atomics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build complete 2-event and 3-event families from same-feature/direction atomics."""
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for a in atomics:
        key = (str(a.get("feature_id")), str(a.get("edit_direction")))
        by_key.setdefault(key, []).append(dict(a))

    two_event_families: List[Dict[str, Any]] = []
    three_event_families: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []

    for (_feat, _dir), group in sorted(by_key.items()):
        ordered = canonicalize_events(group)
        # 2-event
        for a, b in itertools.combinations(ordered, 2):
            mem = validate_family_membership([a, b])
            if not mem["ok"]:
                rejects.append({"size": 2, "events": [a["event_id"], b["event_id"]], **mem})
                continue
            bundle = [a, b]
            labels = ["A", "B"]
            members = {
                "A": project_parent_subset(bundle, [a["event_id"]]),
                "B": project_parent_subset(bundle, [b["event_id"]]),
                "AB": project_parent_subset(bundle, [a["event_id"], b["event_id"]]),
            }
            two_event_families.append(
                {
                    "family_size": 2,
                    "temporal_arm": mem["temporal_arm"],
                    "feature_id": mem["feature_id"],
                    "edit_direction": mem["edit_direction"],
                    "ordered_event_ids": mem["ordered_event_ids"],
                    "candidates": members,
                    "required_labels": ["A", "B", "AB"],
                    "complete": set(members) >= {"A", "B", "AB"},
                }
            )
        # 3-event
        for a, b, c in itertools.combinations(ordered, 3):
            mem = validate_family_membership([a, b, c])
            if not mem["ok"]:
                rejects.append(
                    {"size": 3, "events": [a["event_id"], b["event_id"], c["event_id"]], **mem}
                )
                continue
            bundle = [a, b, c]
            eids = [a["event_id"], b["event_id"], c["event_id"]]
            members = {
                "A": project_parent_subset(bundle, [eids[0]]),
                "B": project_parent_subset(bundle, [eids[1]]),
                "C": project_parent_subset(bundle, [eids[2]]),
                "AB": project_parent_subset(bundle, [eids[0], eids[1]]),
                "AC": project_parent_subset(bundle, [eids[0], eids[2]]),
                "BC": project_parent_subset(bundle, [eids[1], eids[2]]),
                "ABC": project_parent_subset(bundle, eids),
            }
            three_event_families.append(
                {
                    "family_size": 3,
                    "temporal_arm": mem["temporal_arm"],
                    "feature_id": mem["feature_id"],
                    "edit_direction": mem["edit_direction"],
                    "ordered_event_ids": mem["ordered_event_ids"],
                    "candidates": members,
                    "required_labels": ["A", "B", "C", "AB", "AC", "BC", "ABC"],
                    "complete": set(members)
                    >= {"A", "B", "C", "AB", "AC", "BC", "ABC"},
                }
            )

    return {
        "two_event_families": two_event_families,
        "three_event_families": three_event_families,
        "rejects": rejects,
        "two_event_family_evaluable": any(f.get("complete") for f in two_event_families),
        "three_event_family_evaluable": any(f.get("complete") for f in three_event_families),
        "independently_regenerated_parent_count": 0,
    }
