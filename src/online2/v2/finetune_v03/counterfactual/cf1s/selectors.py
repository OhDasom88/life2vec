"""Selector ranking and identical feature round-robin budget allocation."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SELECTOR_SALIENCY = "SALIENCY_TOP_K"
SELECTOR_RANDOM = "RANDOM_EDITABLE_LOCUS"
SELECTOR_RECENCY = "RECENCY_TOP_K"


def _pool_sha256(loci: Sequence[Mapping[str, Any]]) -> str:
    payload = [str(loc.get("locus_key") or "") for loc in loci]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _tie_key(loc: Mapping[str, Any], selector_rank: int) -> Tuple:
    return (
        int(selector_rank),
        str(loc.get("feature_edit_profile_id") or ""),
        str(loc.get("event_time") or ""),
        str(loc.get("event_id") or ""),
        str(loc.get("mg_id") or ""),
    )


def rank_loci_for_selector(
    universe: Mapping[str, Any],
    *,
    selector: str,
    random_seed: Optional[int] = None,
    cutoff_time: Optional[str] = None,
    stability_application: str = "RANKING_TIER",
) -> Dict[str, Any]:
    """Assign selector ranks ONLY; does not change common eligibility."""
    loci = [dict(x) for x in (universe.get("loci") or [])]
    selector_u = str(selector).upper()
    ranked: List[Dict[str, Any]] = []

    if selector_u == SELECTOR_SALIENCY:
        # RANKING_TIER: stable first (priority 0), unstable fallback (priority 1).
        # Never hard-filters common universe membership.
        def sal_key(loc: Mapping[str, Any]) -> Tuple:
            if stability_application == "RANKING_TIER":
                tier = 0 if loc.get("stability_valid") else 1
            else:
                tier = 0
            # Higher absolute saliency first; then deterministic ties.
            return (
                tier,
                -float(loc.get("absolute_saliency") or 0.0),
                str(loc.get("feature_edit_profile_id") or ""),
                str(loc.get("event_time") or ""),
                str(loc.get("event_id") or ""),
                str(loc.get("mg_id") or ""),
            )

        ordered = sorted(loci, key=sal_key)
        for i, loc in enumerate(ordered):
            loc = dict(loc)
            loc["selector"] = SELECTOR_SALIENCY
            loc["selector_rank"] = i
            loc["stability_tier"] = 0 if loc.get("stability_valid") else 1
            ranked.append(loc)

    elif selector_u == SELECTOR_RANDOM:
        seed = int(random_seed if random_seed is not None else 0)
        rng = random.Random(seed)
        order = list(range(len(loci)))
        rng.shuffle(order)
        for rank, idx in enumerate(order):
            loc = dict(loci[idx])
            loc["selector"] = SELECTOR_RANDOM
            loc["selector_rank"] = rank
            loc["random_seed"] = seed
            ranked.append(loc)

    elif selector_u == SELECTOR_RECENCY:
        # EVENT_TIME_DESC relative to prediction cutoff; future events forbidden.
        filtered = []
        for loc in loci:
            et = str(loc.get("event_time") or "")
            if cutoff_time is not None and et and et > str(cutoff_time):
                continue
            filtered.append(loc)
        ordered = sorted(
            filtered,
            key=lambda loc: (
                str(loc.get("event_time") or ""),
                str(loc.get("event_id") or ""),
                str(loc.get("mg_id") or ""),
            ),
            reverse=True,
        )
        for i, loc in enumerate(ordered):
            loc = dict(loc)
            loc["selector"] = SELECTOR_RECENCY
            loc["selector_rank"] = i
            ranked.append(loc)
    else:
        raise ValueError(f"unknown selector: {selector}")

    return {
        "selector": selector_u,
        "selector_changes_ranking_only": True,
        "stability_application": stability_application,
        "ranked_loci": ranked,
        "selected_pool_sha256": _pool_sha256(ranked),
        "ranked_count": len(ranked),
        "common_eligible_locus_count": int(universe.get("common_eligible_locus_count") or 0),
    }


def allocate_feature_round_robin(
    ranked: Mapping[str, Any],
    *,
    maximum_single_candidates_per_case: int = 20,
    maximum_event_loci_per_feature: int = 8,
) -> Dict[str, Any]:
    """Identical case-budget allocation across selectors."""
    by_feature: Dict[str, List[Dict[str, Any]]] = {}
    for loc in ranked.get("ranked_loci") or []:
        by_feature.setdefault(str(loc["feature_id"]), []).append(loc)

    # Within each feature keep selector order, then cap per-feature.
    feature_queues: Dict[str, List[Dict[str, Any]]] = {}
    for feature_id, members in by_feature.items():
        ordered = sorted(members, key=lambda m: int(m.get("selector_rank") or 0))
        feature_queues[feature_id] = ordered[: int(maximum_event_loci_per_feature)]

    feature_ids = sorted(feature_queues)
    selected: List[Dict[str, Any]] = []
    depth = 0
    while len(selected) < int(maximum_single_candidates_per_case):
        progressed = False
        # Round-robin over features at the same depth.
        candidates_at_depth = []
        for feature_id in feature_ids:
            queue = feature_queues[feature_id]
            if depth < len(queue):
                candidates_at_depth.append(queue[depth])
        if not candidates_at_depth:
            break
        candidates_at_depth.sort(
            key=lambda loc: _tie_key(loc, int(loc.get("selector_rank") or 0))
        )
        for loc in candidates_at_depth:
            if len(selected) >= int(maximum_single_candidates_per_case):
                break
            selected.append(loc)
            progressed = True
        if not progressed:
            break
        depth += 1

    return {
        "method": "FEATURE_ROUND_ROBIN",
        "maximum_single_candidates_per_case": int(maximum_single_candidates_per_case),
        "maximum_event_loci_per_feature": int(maximum_event_loci_per_feature),
        "selected_loci": selected,
        "selected_count": len(selected),
        "selected_pool_sha256": _pool_sha256(selected),
        "per_feature_selected_counts": {
            fid: sum(1 for s in selected if s["feature_id"] == fid) for fid in feature_ids
        },
    }
