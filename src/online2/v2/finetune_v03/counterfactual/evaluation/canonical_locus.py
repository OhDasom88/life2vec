"""Canonical locus keys and target-raw canonicalization for dedup / crossfit consensus."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def canonical_locus_key(
    *,
    feature: str,
    zone: str = "",
    measurement_group_id: str = "",
    timestamp: Any = None,
    edit_direction: str = "",
    timestamp_bucket_seconds: float = 3600.0,
) -> str:
    """Stable identity across near-duplicate timestamps within a window."""
    feat = str(feature).strip().lower()
    zn = str(zone).strip()
    mg = str(measurement_group_id).strip()
    direction = str(edit_direction).strip().lower() or "unknown"
    ts_bucket = ""
    if timestamp is not None and str(timestamp) not in {"", "None", "nan"}:
        try:
            import pandas as pd

            t = pd.Timestamp(timestamp)
            epoch = float(t.timestamp())
            bucket = int(epoch // float(timestamp_bucket_seconds)) * int(timestamp_bucket_seconds)
            ts_bucket = str(bucket)
        except Exception:
            ts_bucket = str(timestamp)
    return "|".join([feat, zn, mg, ts_bucket, direction])


def edit_direction_from_raw(observed: float, target: float, *, eps: float = 1e-9) -> str:
    d = float(target) - float(observed)
    if abs(d) <= eps:
        return "none"
    return "increase" if d > 0 else "decrease"


def canonical_target_raw(
    value: float,
    *,
    edges: Optional[Sequence[float]] = None,
    feature_resolution: Optional[float] = None,
    policy: str = "FEATURE_RESOLUTION",
) -> Dict[str, Any]:
    """Canonicalize raw for dedup keys (not plain float rounding alone)."""
    v = float(value)
    if policy == "BIN_MIDPOINT" and edges is not None and len(edges) >= 2:
        from ..grounding.raw_target import decode_interval, value_to_bin_index

        k = value_to_bin_index(v, edges)
        lo, hi = decode_interval(edges, k)
        canon = 0.5 * (lo + hi)
        return {
            "target_raw": v,
            "canonical_target_raw": float(canon),
            "canonicalization_policy": "BIN_MIDPOINT",
            "bin_index": int(k),
        }
    # FEATURE_RESOLUTION: quantize to feature-specific step (default from edges span)
    if feature_resolution is None:
        if edges is not None and len(edges) >= 2:
            span = abs(float(edges[-1]) - float(edges[0]))
            feature_resolution = max(span / 1e4, 1e-6)
        else:
            feature_resolution = 1e-4
    step = float(feature_resolution)
    canon = round(v / step) * step
    # avoid -0.0
    if abs(canon) < step * 0.5:
        canon = 0.0
    return {
        "target_raw": v,
        "canonical_target_raw": float(canon),
        "canonicalization_policy": "FEATURE_RESOLUTION",
        "feature_resolution": step,
    }


def dedup_key(
    *,
    case_id: str,
    locus_id: str,
    canonical_target_raw_value: float,
    actual_retokenized_bundle: Sequence[str],
) -> Tuple[str, str, float, Tuple[str, ...]]:
    return (
        str(case_id),
        str(locus_id),
        float(canonical_target_raw_value),
        tuple(str(t) for t in actual_retokenized_bundle),
    )


def merge_duplicate_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list:
    """Dedup by key; merge sources into candidate_sources provenance list."""
    buckets: Dict[Tuple, Dict[str, Any]] = {}
    order: list = []
    for c in candidates:
        if c.get("is_noop") or str(c.get("source") or "").startswith("noop"):
            # defer — handled by ensure_canonical_noop
            key = ("__noop__", str(c.get("case_id") or ""))
        else:
            bundle = c.get("actual_retokenized_bundle") or c.get("proposed_token_bundle") or []
            key = dedup_key(
                case_id=str(c.get("case_id") or ""),
                locus_id=str(c.get("canonical_locus_key") or c.get("locus_id") or ""),
                canonical_target_raw_value=float(
                    c.get("canonical_target_raw")
                    if c.get("canonical_target_raw") is not None
                    else c.get("target_raw")
                    or 0.0
                ),
                actual_retokenized_bundle=list(bundle),
            )
        src = str(c.get("source") or "unknown")
        if key not in buckets:
            row = dict(c)
            row["candidate_sources"] = [src]
            buckets[key] = row
            order.append(key)
        else:
            existing = buckets[key]
            sources = list(existing.get("candidate_sources") or [])
            if src not in sources:
                sources.append(src)
            existing["candidate_sources"] = sources
            # keep better (lower) search delta if present
            dr_new = c.get("delta_r_search")
            dr_old = existing.get("delta_r_search")
            if dr_new is not None and (dr_old is None or float(dr_new) < float(dr_old)):
                for k, v in c.items():
                    if k == "candidate_sources":
                        continue
                    existing[k] = v
                existing["candidate_sources"] = sources
    return [buckets[k] for k in order]


def ensure_canonical_noop(
    candidates: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    noop_template: Optional[Mapping[str, Any]] = None,
) -> list:
    """Keep exactly one canonical NO_OP per case."""
    others = [
        dict(c)
        for c in candidates
        if not (c.get("is_noop") or str(c.get("source") or "").startswith("noop"))
    ]
    noops = [
        dict(c)
        for c in candidates
        if c.get("is_noop") or str(c.get("source") or "").startswith("noop")
    ]
    if noops:
        chosen = dict(noops[0])
    elif noop_template is not None:
        chosen = dict(noop_template)
    else:
        chosen = {
            "case_id": case_id,
            "source": "noop",
            "is_noop": True,
            "candidate_id": f"noop::{case_id}",
        }
    chosen["source"] = "noop"
    chosen["is_noop"] = True
    chosen["candidate_id"] = f"noop::{case_id}"
    chosen["candidate_sources"] = ["noop"]
    return others + [chosen]
