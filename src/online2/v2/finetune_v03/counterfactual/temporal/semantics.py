"""Temporal semantics labels for actuator features."""
from __future__ import annotations
from typing import Mapping

STATE_UNTIL_NEXT = "state_until_next_sample"
INTERVAL_Q = "interval_quantity"
POINT = "point_observation"

DEFAULT_TEMPORAL = {
    "circulation_fan": STATE_UNTIL_NEXT,
    "fcu_fan": STATE_UNTIL_NEXT,
    "fcu_pump": STATE_UNTIL_NEXT,
    "co2_supply": STATE_UNTIL_NEXT,
    "tube_rail": STATE_UNTIL_NEXT,
    "roof_vent_left": STATE_UNTIL_NEXT,
    "roof_vent_right": STATE_UNTIL_NEXT,
    "shade_screen": STATE_UNTIL_NEXT,
    "thermal_curtain": STATE_UNTIL_NEXT,
    "line_flow_rate": INTERVAL_Q,
    "total_flow_rate": INTERVAL_Q,
    "total_irrigation": INTERVAL_Q,
}

def temporal_semantics_for(feature: str, registry: Mapping | None = None) -> str:
    if registry and feature in registry:
        return str(registry[feature].get("temporal_semantics") or POINT)
    return DEFAULT_TEMPORAL.get(feature, POINT)
