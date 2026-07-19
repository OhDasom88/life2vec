"""Common eligible locus universe for CF-1S (distinct raw MG before selectors)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .edit_policy import EditClass, get_feature_record, is_edit_class_allowed, load_edit_policy


def canonicalize_raw_mg_locus(
    *,
    event_id: str,
    feature_id: str,
    mg_id: str,
    feature_edit_profile_id: str,
    event_time: str = "",
    same_time_group_id: str = "",
) -> Dict[str, str]:
    """Canonical identity for a distinct raw MG locus."""
    return {
        "event_id": str(event_id),
        "feature_id": str(feature_id),
        "mg_id": str(mg_id),
        "feature_edit_profile_id": str(feature_edit_profile_id),
        "event_time": str(event_time),
        "same_time_group_id": str(same_time_group_id or event_time),
        "locus_key": "|".join(
            [
                str(event_id),
                str(feature_id),
                str(mg_id),
                str(feature_edit_profile_id),
            ]
        ),
    }


def _pool_sha256(loci: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "locus_key": loc.get("locus_key"),
            "event_time": loc.get("event_time"),
            "feature_id": loc.get("feature_id"),
            "mg_id": loc.get("mg_id"),
            "feature_edit_profile_id": loc.get("feature_edit_profile_id"),
        }
        for loc in sorted(loci, key=lambda x: str(x.get("locus_key") or ""))
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_common_eligible_universe(
    events: Sequence[Mapping[str, Any]],
    *,
    edit_policy: Mapping[str, Any],
    arm: str = "CONTROL_LIKE",
    require_raw_reconstructable: bool = True,
    require_atomic_schema_edit_possible: bool = True,
) -> Dict[str, Any]:
    """Build common eligible locus universe BEFORE selector ranking.

    Funnel steps encoded here:
      edit-class eligible → forbidden/outcome excluded → raw MG reconstructable
      → raw MG canonicalization → distinct-MG dedup → feature-edit-profile valid
      → atomic schema edit possible → feature/profile grouping
    """
    counts = {
        "n_all_events": 0,
        "n_edit_class_eligible": 0,
        "n_forbidden_outcome_excluded": 0,
        "n_raw_mg_reconstructable": 0,
        "n_after_distinct_mg_dedup": 0,
        "n_feature_edit_profile_valid": 0,
        "n_atomic_schema_edit_possible": 0,
        "n_common_eligible_loci": 0,
        "raw_mg_duplicate_locus_count": 0,
        "raw_mg_duplicate_detected_count": 0,
        "raw_mg_duplicate_remaining_count": 0,
        "forbidden_outcome_edit_count": 0,
        "cross_profile_bundle_count": 0,
    }
    seen: Dict[str, Dict[str, Any]] = {}
    feature_groups: Dict[str, List[Dict[str, Any]]] = {}

    for event in events:
        counts["n_all_events"] += 1
        feature_id = str(event.get("feature_id") or event.get("feature") or "")
        if not feature_id:
            continue
        record = get_feature_record(edit_policy, feature_id)
        if record.edit_class == EditClass.FORBIDDEN_OUTCOME:
            counts["n_forbidden_outcome_excluded"] += 1
            continue
        if not is_edit_class_allowed(record.edit_class, arm=arm):
            continue
        counts["n_edit_class_eligible"] += 1

        mg_id = str(event.get("mg_id") or event.get("measurement_group_id") or "")
        event_id = str(event.get("event_id") or "")
        event_time = str(event.get("event_time") or event.get("timestamp") or "")
        raw_ok = bool(event.get("raw_reconstructable", True))
        if event.get("observed_raw") is None and require_raw_reconstructable:
            raw_ok = False
        if not raw_ok:
            continue
        counts["n_raw_mg_reconstructable"] += 1

        canon = canonicalize_raw_mg_locus(
            event_id=event_id,
            feature_id=feature_id,
            mg_id=mg_id,
            feature_edit_profile_id=record.feature_edit_profile_id,
            event_time=event_time,
            same_time_group_id=str(event.get("same_time_group_id") or event_time),
        )
        locus_key = canon["locus_key"]
        if locus_key in seen:
            counts["raw_mg_duplicate_detected_count"] = int(
                counts.get("raw_mg_duplicate_detected_count") or 0
            ) + 1
            # Keep first occurrence; duplicates do not expand the universe.
            # Remaining duplicates after dedup must stay 0.
            continue

        schema_ok = bool(event.get("atomic_schema_edit_possible", True))
        if record.adjacency in {"none", ""}:
            schema_ok = False
        if require_atomic_schema_edit_possible and not schema_ok:
            continue
        counts["n_atomic_schema_edit_possible"] += 1
        counts["n_feature_edit_profile_valid"] += 1

        epoch = event.get("event_time_epoch")
        if epoch is None and event_time:
            try:
                import pandas as pd

                epoch = float(pd.Timestamp(event_time).timestamp())
            except Exception:
                epoch = None
        locus = {
            **canon,
            "edit_class": record.edit_class.value,
            "schema_kind": record.schema_kind,
            "adjacency": record.adjacency,
            "feature_edit_profile_sha256": record.feature_edit_profile_sha256,
            "observed_raw": event.get("observed_raw"),
            "zone_id": event.get("zone_id"),
            "event_time_epoch": epoch,
            "same_time_group_id": canon["same_time_group_id"],
            "token_indices": list(event.get("token_indices") or []),
            "absolute_saliency": float(event.get("absolute_saliency") or 0.0),
            "signed_saliency": float(event.get("signed_saliency") or 0.0),
            "fold_stable": bool(event.get("fold_stable", False)),
            "method_stable": bool(event.get("method_stable", True)),
            "perturbation_stable": bool(event.get("perturbation_stable", True)),
            "saliency_evaluable": bool(event.get("saliency_evaluable", True)),
            "saliency_rank_hint": event.get("saliency_rank_hint"),
            "member_token_count": int(event.get("member_token_count") or len(event.get("token_indices") or [])),
            "original_tokens": list(event.get("original_tokens") or []),
            "farm_id": event.get("farm_id"),
        }
        # Stability is a ranking tier later; never a common-universe filter.
        locus["stability_valid"] = bool(
            locus["fold_stable"] and locus["method_stable"] and locus["perturbation_stable"]
        )
        seen[locus_key] = locus
        feature_groups.setdefault(feature_id, []).append(locus)

    loci = list(seen.values())
    counts["n_after_distinct_mg_dedup"] = len(loci)
    counts["n_common_eligible_loci"] = len(loci)
    counts["raw_mg_duplicate_detected_count"] = int(
        counts.get("raw_mg_duplicate_detected_count") or 0
    )
    # Policy max applies to remaining duplicates after dedup.
    counts["raw_mg_duplicate_locus_count"] = 0
    counts["raw_mg_duplicate_remaining_count"] = 0

    # Group by feature/profile for selector-agnostic structure.
    grouped: Dict[str, Dict[str, Any]] = {}
    for feature_id, members in feature_groups.items():
        profiles = {m["feature_edit_profile_id"] for m in members}
        grouped[feature_id] = {
            "feature_id": feature_id,
            "feature_edit_profile_ids": sorted(profiles),
            "loci": sorted(
                members,
                key=lambda m: (
                    str(m.get("event_time") or ""),
                    str(m.get("event_id") or ""),
                    str(m.get("mg_id") or ""),
                ),
            ),
            "count": len(members),
        }

    return {
        "common_universe_unit": "DISTINCT_RAW_MG_LOCUS",
        "mg_dedup_precedes_selector": True,
        "feature_profile_grouping_precedes_selector_truncation": True,
        "common_universe_requires_saliency": False,
        "arm": arm,
        "counts": counts,
        "loci": loci,
        "feature_groups": grouped,
        "common_eligible_locus_pool_sha256": _pool_sha256(loci),
        "common_eligible_locus_count": len(loci),
        "raw_mg_duplicate_locus_count": counts["raw_mg_duplicate_locus_count"],
    }
