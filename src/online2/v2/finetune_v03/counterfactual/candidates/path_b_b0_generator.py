"""Path B B0 operation candidate generator (NO_OP / truncate_* / clear_span)."""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_THRESH = {
    "positive_ratio_truncate": 0.7,
    "positive_ratio_clear": 0.9,
    "tail_mass_ratio": 0.6,
    "critical_attribution_percentile": 90,
    "min_on_events": 1,
}


def generate_b0_operations(
    span: Mapping[str, Any],
    attribution: Mapping[str, Any],
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    clear_whitelist: Optional[set] = None,
    gate1_passed: bool = True,
) -> List[Dict[str, Any]]:
    thr = dict(DEFAULT_THRESH)
    if thresholds:
        thr.update({k: float(v) for k, v in thresholds.items()})
    raw_eids = span.get("source_event_ids")
    if raw_eids is None:
        eids = []
    elif hasattr(raw_eids, "tolist"):
        eids = list(raw_eids.tolist())
    elif isinstance(raw_eids, str):
        import json
        eids = json.loads(raw_eids)
    else:
        eids = list(raw_eids)
    n = len(eids)
    sampling = float(span.get("sampling_interval_min") or 60.0)
    from_hours = float(span.get("estimate_duration_min") or (n * sampling)) / 60.0
    feature = str(span.get("feature") or attribution.get("feature") or "")
    pos_ratio = float(attribution.get("duration_weighted_positive_ratio") or attribution.get("positive_ratio") or 0.0)
    start_m = float(attribution.get("start_mass_ratio") or 0.0)
    end_m = float(attribution.get("end_mass_ratio") or 0.0)
    mean_s = float(attribution.get("signed") or attribution.get("duration_weighted_mean") or 0.0)

    cands: List[Dict[str, Any]] = [
        {
            "span_id": span.get("span_id"),
            "feature": feature,
            "operation": "NO_OP",
            "cut_events": 0,
            "from_hours": from_hours,
            "to_hours": from_hours,
            "intensity_policy": "preserve_observed",
            "capacity_status": "unknown",
        }
    ]
    if n <= 0 or not gate1_passed:
        return cands

    # truncate candidates at event boundaries down to min ON
    min_keep = int(thr["min_on_events"])
    if pos_ratio >= thr["positive_ratio_truncate"]:
        for cut in range(1, n - min_keep + 1):
            if end_m >= thr["tail_mass_ratio"] or end_m >= start_m:
                to_h = max(min_keep, n - cut) * sampling / 60.0
                cands.append(
                    {
                        "span_id": span.get("span_id"),
                        "feature": feature,
                        "operation": "truncate_end",
                        "cut_events": cut,
                        "from_hours": from_hours,
                        "to_hours": float(to_h),
                        "intensity_policy": "preserve_observed",
                        "capacity_status": "unknown",
                    }
                )
            if start_m >= thr["tail_mass_ratio"] or start_m > end_m:
                to_h = max(min_keep, n - cut) * sampling / 60.0
                cands.append(
                    {
                        "span_id": span.get("span_id"),
                        "feature": feature,
                        "operation": "truncate_start",
                        "cut_events": cut,
                        "from_hours": from_hours,
                        "to_hours": float(to_h),
                        "intensity_policy": "preserve_observed",
                        "capacity_status": "unknown",
                    }
                )

    allow_clear = True
    if clear_whitelist is not None:
        allow_clear = feature in clear_whitelist
    if (
        allow_clear
        and gate1_passed
        and pos_ratio >= thr["positive_ratio_clear"]
        and mean_s > 0
    ):
        cands.append(
            {
                "span_id": span.get("span_id"),
                "feature": feature,
                "operation": "clear_span",
                "cut_events": n,
                "from_hours": from_hours,
                "to_hours": 0.0,
                "intensity_policy": "preserve_observed",
                "capacity_status": "unknown",
            }
        )

    # dedupe by (operation, cut_events)
    seen = set()
    uniq = []
    for c in cands:
        key = (c["operation"], c["cut_events"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    # enforce only allowed ops
    allowed = {"NO_OP", "truncate_start", "truncate_end", "clear_span"}
    return [c for c in uniq if c["operation"] in allowed]


def select_best_operation(results: List[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Prefer better model_space_delta_r (more negative); ties → NO_OP."""
    if not results:
        return {"operation": "NO_OP", "status": "NO_VALID_OPERATION"}
    def key(r):
        dr = float(r.get("model_space_delta_r", 0.0))
        is_noop = 0 if r.get("operation") == "NO_OP" else 1
        return (dr, is_noop)
    return sorted(results, key=key)[0]
