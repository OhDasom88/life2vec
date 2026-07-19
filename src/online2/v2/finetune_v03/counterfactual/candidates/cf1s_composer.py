"""Compose length-preserving multi-event CF-1S candidates (types A/B only)."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


RESULT_LABELS = (
    "MULTI_EVENT_MODEL_SENSITIVITY",
    "SEQUENCE_EDIT_FEASIBILITY",
    "NON_CAUSAL",
    "NOT_AN_ACTION",
    "NOT_RECOMMENDATION",
)


def multi_event_candidate_id(
    *,
    case_id: str,
    edits: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "case_id": str(case_id),
        "edits": sorted(
            [
                {
                    "event_time": str(e.get("event_time") or ""),
                    "event_id": str(e.get("event_id") or ""),
                    "feature": str(e.get("feature_id") or e.get("feature") or ""),
                    "target_raw": e.get("target_raw"),
                    "retokenized_mg_bundle": e.get("retokenized_mg_bundle"),
                    "mg_id": str(e.get("mg_id") or ""),
                }
                for e in edits
            ],
            key=lambda row: (
                row["event_time"],
                row["event_id"],
                row["feature"],
                str(row["mg_id"]),
            ),
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return digest


def _time_ok_pair(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    mode: str,
    min_sparse_seconds: float,
    max_contiguous_gap_seconds: float,
) -> bool:
    if str(a.get("event_id")) == str(b.get("event_id")):
        return False
    if str(a.get("event_time")) == str(b.get("event_time")):
        return False
    # Timestamps are ISO-like strings in fixtures; numeric seconds optional.
    ta = a.get("event_time_epoch")
    tb = b.get("event_time_epoch")
    if ta is None or tb is None:
        # String inequality already enforced; allow both modes without epoch.
        return True
    gap = abs(float(ta) - float(tb))
    if mode == "CONTIGUOUS":
        return gap <= float(max_contiguous_gap_seconds)
    if mode == "SPARSE":
        return gap >= float(min_sparse_seconds)
    return True


def compose_atomic_edits_for_locus(
    locus: Mapping[str, Any],
    *,
    adjacency_targets: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Create schema-adjacency atomic edits; direction not from saliency."""
    out: List[Dict[str, Any]] = []
    for target in adjacency_targets:
        out.append(
            {
                "event_id": locus.get("event_id"),
                "event_time": locus.get("event_time"),
                "event_time_epoch": locus.get("event_time_epoch"),
                "feature_id": locus.get("feature_id"),
                "mg_id": locus.get("mg_id"),
                "feature_edit_profile_id": locus.get("feature_edit_profile_id"),
                "feature_edit_profile_sha256": locus.get("feature_edit_profile_sha256"),
                "observed_raw": locus.get("observed_raw"),
                "target_raw": target,
                "direction_inferred_from_saliency": False,
                "edit_class": locus.get("edit_class"),
                "labels": list(RESULT_LABELS),
            }
        )
    return out


def compose_multi_event_bundles(
    *,
    case_id: str,
    selected_loci: Sequence[Mapping[str, Any]],
    atomic_by_locus_key: Mapping[str, Sequence[Mapping[str, Any]]],
    max_events: int = 3,
    two_event_max: int = 32,
    three_event_max: int = 24,
    min_sparse_seconds: float = 3600.0,
    max_contiguous_gap_seconds: float = 1800.0,
) -> Dict[str, Any]:
    """Build same-feature 2/3-event bundles with distinct timestamps."""
    by_feature: Dict[str, List[Mapping[str, Any]]] = {}
    for loc in selected_loci:
        by_feature.setdefault(str(loc["feature_id"]), []).append(loc)

    bundles: List[Dict[str, Any]] = []
    rejected = {
        "cross_profile_bundle_count": 0,
        "same_timestamp_rejected": 0,
        "atomic_missing": 0,
    }

    for feature_id, loci in by_feature.items():
        profiles = {str(l.get("feature_edit_profile_id")) for l in loci}
        if len(profiles) != 1:
            rejected["cross_profile_bundle_count"] += 1
            continue
        # Prefer coherent direction pairs/triples from atomic parent directions.
        for mode, limit, k in (
            ("CONTIGUOUS", two_event_max, 2),
            ("SPARSE", two_event_max, 2),
            ("CONTIGUOUS", three_event_max, 3),
            ("SPARSE", three_event_max, 3),
        ):
            if k > int(max_events):
                continue
            made = 0
            for combo in combinations(loci, k):
                if made >= limit:
                    break
                ok = True
                for a, b in combinations(combo, 2):
                    if not _time_ok_pair(
                        a,
                        b,
                        mode=mode if k > 1 else "CONTIGUOUS",
                        min_sparse_seconds=min_sparse_seconds,
                        max_contiguous_gap_seconds=max_contiguous_gap_seconds,
                    ):
                        ok = False
                        if str(a.get("event_time")) == str(b.get("event_time")):
                            rejected["same_timestamp_rejected"] += 1
                        break
                if not ok:
                    continue

                # Take first atomic edit per locus (caller supplies ordered targets).
                edits = []
                atomic_parent_ids = []
                for loc in combo:
                    key = str(loc.get("locus_key"))
                    atomics = list(atomic_by_locus_key.get(key) or [])
                    if not atomics:
                        rejected["atomic_missing"] += 1
                        edits = []
                        break
                    edits.append(dict(atomics[0]))
                    atomic_parent_ids.append(
                        str(atomics[0].get("atomic_candidate_id") or key)
                    )
                if len(edits) != k:
                    continue

                # Direction pattern
                deltas = []
                for e in edits:
                    try:
                        deltas.append(float(e["target_raw"]) - float(e["observed_raw"]))
                    except (TypeError, ValueError, KeyError):
                        deltas.append(0.0)
                if all(d > 0 for d in deltas):
                    direction_pattern = "COHERENT_UP"
                elif all(d < 0 for d in deltas):
                    direction_pattern = "COHERENT_DOWN"
                else:
                    direction_pattern = "MIXED_DIRECTION"

                cid = multi_event_candidate_id(case_id=case_id, edits=edits)
                bundles.append(
                    {
                        "multi_event_candidate_id": cid,
                        "sequence_edit_candidate_id": cid,
                        "case_id": case_id,
                        "feature_id": feature_id,
                        "n_events": k,
                        "bundle_mode": mode,
                        "direction_pattern": direction_pattern,
                        "edits": edits,
                        "atomic_parent_candidate_ids": atomic_parent_ids,
                        "labels": list(RESULT_LABELS),
                        "max_features_per_cf": 1,
                        "distinct_event_id": True,
                        "distinct_event_timestamp": True,
                    }
                )
                made += 1

    # Dedup by candidate id (order-independent).
    dedup: Dict[str, Dict[str, Any]] = {}
    for b in bundles:
        dedup[b["multi_event_candidate_id"]] = b

    return {
        "bundles": list(dedup.values()),
        "raw_bundle_count": len(bundles),
        "dedup_bundle_count": len(dedup),
        "rejected": rejected,
        "cross_profile_bundle_count": rejected["cross_profile_bundle_count"],
    }
