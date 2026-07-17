"""B0 actuator span grounding from edit_scope (not token diffs)."""

from __future__ import annotations
from typing import Any, Dict, Mapping


def ground_actuator_span(edit_scope: Mapping[str, Any]) -> Dict[str, Any]:
    op = str(edit_scope.get("operation") or "NO_OP")
    from_h = float(edit_scope.get("from_hours") or 0.0)
    to_h = float(edit_scope.get("to_hours") or from_h)
    return {
        "kind": "actuator_span",
        "feature": edit_scope.get("feature"),
        "operation": op,
        "from_hours": from_h,
        "to_hours": to_h,
        "capacity": {"status": "unknown", "policy": "unspecified_positive"},
        "intensity_policy": edit_scope.get("intensity_policy", "preserve_observed"),
        "confidence": "duration_ok_capacity_weak",
        "limitation": [
            "classifier-oriented hypothesis evaluation",
            "not physical response validation",
            "not causal intervention effect",
            "not executable prescription",
        ],
    }


def apply_operation_to_event_ids(
    source_event_ids,
    operation: str,
    cut_events: int,
):
    if source_event_ids is None:
        eids = []
    elif hasattr(source_event_ids, "tolist"):
        eids = list(source_event_ids.tolist())
    elif isinstance(source_event_ids, str):
        import json
        eids = json.loads(source_event_ids)
    else:
        eids = list(source_event_ids)
    n = len(eids)
    cut = int(cut_events)
    if operation == "NO_OP":
        return eids, []
    if operation == "clear_span":
        return [], eids
    if operation == "truncate_end":
        keep = eids[: max(0, n - cut)]
        drop = eids[max(0, n - cut) :]
        return keep, drop
    if operation == "truncate_start":
        keep = eids[cut:]
        drop = eids[:cut]
        return keep, drop
    raise ValueError(f"unsupported M1 operation: {operation}")
