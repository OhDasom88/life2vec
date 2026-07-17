"""Gate 2 — 3-layer safety bounds intersection."""

from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Tuple


def _intersect(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]):
    if a is None or b is None:
        return None
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo > hi:
        return None
    return (lo, hi)


def project_to_safe(
    value: float,
    *,
    hard: Optional[Tuple[float, float]] = None,
    agronomic: Optional[Tuple[float, float]] = None,
    operational: Optional[Tuple[float, float]] = None,
    operational_known: bool = True,
) -> Dict[str, Any]:
    inter = hard
    for b in (agronomic, operational if operational_known else None):
        if b is None:
            continue
        inter = _intersect(inter, b) if inter is not None else b
        if inter is None:
            return {
                "gate": "gate2_safety_bounds",
                "status": "REJECTED",
                "reason_codes": ["ERR_NO_SAFE_INTERSECTION"],
                "details": {"value": value, "clamped": None},
            }
    if inter is None:
        # no tables at all
        status = "PARTIAL" if not operational_known else "REJECTED"
        reasons = ["ERR_MISSING_SAFETY_TABLE"] if operational_known else ["OPERATIONAL_BOUNDS_UNKNOWN"]
        return {
            "gate": "gate2_safety_bounds",
            "status": status,
            "reason_codes": reasons,
            "details": {"value": value, "safe_interval": None, "clamped": value, "partial": True},
        }
    lo, hi = inter
    clamped = min(max(float(value), lo), hi)
    partial = (clamped != float(value)) or (not operational_known)
    status = "PARTIAL" if partial and (not operational_known or clamped != float(value)) else "PASSED"
    if not operational_known:
        status = "PARTIAL"
    return {
        "gate": "gate2_safety_bounds",
        "status": status,
        "reason_codes": ([] if status == "PASSED" else ["CLAMPED" if clamped != float(value) else "OPERATIONAL_BOUNDS_UNKNOWN"]),
        "details": {
            "value_before": float(value),
            "value_after": float(clamped),
            "safe_interval": [lo, hi],
            "operational_known": operational_known,
        },
    }
