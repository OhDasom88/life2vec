"""P1 — build feature/actuator semantics registry from v2 audit + schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .io_utils import write_parquet, write_yaml
from .temporal.semantics import DEFAULT_TEMPORAL, INTERVAL_Q, STATE_UNTIL_NEXT


DERIVED = {
    "vpd": {
        "feature_type": "DERIVED_STATE",
        "editable_path_a": False,
        "editable_path_b": False,
        "parents": ["inside_temp_c", "inside_humidity_pct"],
        "recompute_fn": "calculate_vpd",
        "temporal_semantics": "point_observation",
    }
}

IMMUTABLE = {"farm_id", "zone_id", "timestamp", "observation_date"}


def _map_type(row: dict) -> Tuple[str, Dict[str, Any]]:
    name = str(row["feature_name"])
    if name in DERIVED:
        return "DERIVED_STATE", dict(DERIVED[name])
    if name in IMMUTABLE:
        return "IMMUTABLE_CONTEXT", {
            "feature_type": "IMMUTABLE_CONTEXT",
            "editable_path_a": False,
            "editable_path_b": False,
            "temporal_semantics": "point_observation",
        }
    view = str(row.get("event_view") or row.get("source_file_family") or "")
    ftype = str(row.get("proposed_v2_type") or row.get("type") or "")
    unit = str(row.get("canonical_unit") or "unresolved")
    conf = str(row.get("unit_confidence") or "")
    base: Dict[str, Any] = {
        "unit": unit,
        "unit_confidence": conf,
        "source_view": view,
    }
    if view.startswith("A_") or "actuator" in view.lower():
        if ftype == "flow":
            base.update(
                {
                    "feature_type": "INTERVAL_QUANTITY",
                    "editable_path_a": False,
                    "editable_path_b": False,  # quantity not direct B0 command
                    "temporal_semantics": INTERVAL_Q,
                    "raw_semantics": "UNKNOWN_SEMANTICS" if conf == "LOW" else "interval_quantity",
                }
            )
            if conf == "LOW":
                base["feature_type"] = "UNKNOWN_SEMANTICS"
                base["editable_path_b"] = False
            return base["feature_type"], base
        # boolean / ordinal actuators
        if conf == "LOW" and ftype in {"boolean", "ordinal_actuator", "categorical"}:
            # still ACTUATOR_STATE but capacity unknown; editable_path_b true for B0 duration only
            base.update(
                {
                    "feature_type": "ACTUATOR_STATE",
                    "editable_path_a": False,
                    "editable_path_b": True,
                    "temporal_semantics": DEFAULT_TEMPORAL.get(name, STATE_UNTIL_NEXT),
                    "raw_semantics": "state_code",
                    "capacity_status": "unknown",
                }
            )
            return "ACTUATOR_STATE", base
        base.update(
            {
                "feature_type": "ACTUATOR_STATE",
                "editable_path_a": False,
                "editable_path_b": True,
                "temporal_semantics": DEFAULT_TEMPORAL.get(name, STATE_UNTIL_NEXT),
                "raw_semantics": "state_code",
            }
        )
        return "ACTUATOR_STATE", base
    if ftype in {"continuous", "circular"}:
        base.update(
            {
                "feature_type": "OBSERVED_STATE",
                "editable_path_a": True,
                "editable_path_b": False,
                "directly_controllable": False,
                "temporal_semantics": "point_observation",
            }
        )
        return "OBSERVED_STATE", base
    if ftype in {"counter", "identifier"}:
        base.update(
            {
                "feature_type": "IMMUTABLE_CONTEXT" if ftype == "identifier" else "NON_INVERTIBLE",
                "editable_path_a": False,
                "editable_path_b": False,
                "temporal_semantics": "point_observation",
            }
        )
        return base["feature_type"], base
    base.update(
        {
            "feature_type": "UNKNOWN_SEMANTICS",
            "editable_path_a": False,
            "editable_path_b": False,
            "temporal_semantics": "point_observation",
        }
    )
    return "UNKNOWN_SEMANTICS", base


def build_semantics_registry(
    audit_csv: Path,
    *,
    out_yaml: Path,
    out_feature_csv: Path,
    out_actuator_csv: Path,
) -> Dict[str, Any]:
    df = pd.read_csv(audit_csv)
    registry: Dict[str, Any] = {"meta": {"source_audit": str(audit_csv)}, "features": {}}
    feat_rows = []
    act_rows = []
    for _, row in df.iterrows():
        name = str(row["feature_name"])
        ftype, spec = _map_type(row.to_dict())
        spec = dict(spec)
        spec["feature_type"] = ftype
        if ftype == "UNKNOWN_SEMANTICS":
            spec["editable_path_a"] = False
            spec["editable_path_b"] = False
            spec["operational_eligibility"] = "BLOCKED"
        registry["features"][name] = spec
        feat_rows.append({"feature_name": name, **spec})
        if ftype in {"ACTUATOR_STATE", "ACTUATOR_COMMAND", "INTERVAL_QUANTITY"} or str(row.get("event_view", "")).startswith("A_"):
            act_rows.append({"feature_name": name, **spec})
    # ensure derived present
    for k, v in DERIVED.items():
        registry["features"].setdefault(k, v)
    write_yaml(out_yaml, registry)
    pd.DataFrame(feat_rows).to_csv(out_feature_csv, index=False)
    pd.DataFrame(act_rows).to_csv(out_actuator_csv, index=False)
    return registry
