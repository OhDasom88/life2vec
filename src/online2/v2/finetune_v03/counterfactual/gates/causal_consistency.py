"""Gate 3 — unidirectional derived feature recompute."""

from __future__ import annotations
import math
from typing import Any, Dict, Mapping, MutableMapping, Optional


def calculate_vpd(temp_c: float, rh_pct: float) -> float:
    """Saturation vapor pressure deficit (kPa), Tetens."""
    es = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh_pct / 100.0)
    return float(es - ea)


def recompute_derived(state: MutableMapping[str, Any], parents_changed: Mapping[str, Any]) -> Dict[str, Any]:
    """Update derived keys in-place from parents. Returns gate result."""
    reasons = []
    # reject direct VPD edit without parents
    if "vpd" in parents_changed and not ({"inside_temp_c", "inside_humidity_pct"} & set(parents_changed)):
        return {
            "gate": "gate3_causal_consistency",
            "status": "REJECTED",
            "reason_codes": ["ERR_CAUSAL_CONFLICT"],
            "details": {"msg": "direct VPD edit forbidden"},
        }
    state.update({k: v for k, v in parents_changed.items() if k != "vpd"})
    if "inside_temp_c" in state and "inside_humidity_pct" in state:
        try:
            state["vpd"] = calculate_vpd(float(state["inside_temp_c"]), float(state["inside_humidity_pct"]))
        except Exception as e:
            reasons.append("ERR_CAUSAL_CONFLICT")
            return {
                "gate": "gate3_causal_consistency",
                "status": "REJECTED",
                "reason_codes": reasons,
                "details": {"error": str(e)},
            }
    if "flow" in parents_changed and "duration_min" in parents_changed:
        state["interval_quantity"] = float(parents_changed["flow"]) * float(parents_changed["duration_min"])
    return {
        "gate": "gate3_causal_consistency",
        "status": "PASSED",
        "reason_codes": [],
        "details": {"state_keys": list(state.keys())},
    }
