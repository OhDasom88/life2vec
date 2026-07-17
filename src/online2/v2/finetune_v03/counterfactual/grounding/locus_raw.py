"""Locus-level EXACT raw grounding (no farm-wide median)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd

from ..evaluation.raw_join import parse_cell_raw


def ground_locus_raw(
    *,
    cells: pd.DataFrame,
    farm_id: str,
    feature: str,
    zone_id: str,
    timestamp: Any,
    event_id: str = "",
    measurement_group_id: str = "",
    tolerance_seconds: float = 0.0,
    require_exact: bool = True,
) -> Dict[str, Any]:
    """Join a single locus to cell_occurrences by zone + timestamp.

    EXACT means time_delta_seconds <= tolerance_seconds (default 0).
    """
    base = {
        "event_id": str(event_id),
        "measurement_group_id": str(measurement_group_id),
        "feature": str(feature),
        "zone_id": str(zone_id),
        "timestamp": str(timestamp),
        "observed_raw": None,
        "raw_grounding_status": "FAILED",
        "raw_source_row_id": None,
        "time_delta_seconds": None,
    }
    if cells is None or not len(cells):
        return base

    sub = cells[cells["farm_id"].astype(str) == str(farm_id)].copy()
    if "is_null" in sub.columns:
        sub = sub[~sub["is_null"].fillna(False)]
    sub = sub[sub["column_name"].astype(str) == str(feature)]
    sub = sub[sub["zone_id"].astype(str) == str(zone_id)]
    if not len(sub):
        return base

    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    sub = sub.copy()
    sub["_ts"] = pd.to_datetime(sub["observation_timestamp"], utc=True, errors="coerce")
    sub["_ts"] = sub["_ts"].dt.tz_localize(None)
    sub = sub.dropna(subset=["_ts"])
    if not len(sub):
        return base

    delta = (sub["_ts"] - ts).abs()
    best_i = delta.idxmin()
    dt_sec = float(delta.loc[best_i].total_seconds())
    row = sub.loc[best_i]
    raw = parse_cell_raw(row.get("raw_value"), row.get("raw_display"))
    try:
        raw_f = float(raw)
    except (TypeError, ValueError):
        base["time_delta_seconds"] = dt_sec
        base["raw_grounding_status"] = "NON_NUMERIC"
        base["raw_source_row_id"] = str(row.name)
        return base

    status = "EXACT" if dt_sec <= float(tolerance_seconds) + 1e-9 else "NEAREST"
    if require_exact and status != "EXACT":
        status = "NON_EXACT"

    row_id = None
    for key in ("occurrence_id", "row_id", "cell_id"):
        if key in row.index and pd.notna(row[key]):
            row_id = str(row[key])
            break
    if row_id is None:
        row_id = str(best_i)

    return {
        "event_id": str(event_id),
        "measurement_group_id": str(measurement_group_id),
        "feature": str(feature),
        "zone_id": str(zone_id),
        "timestamp": str(timestamp),
        "observed_raw": float(raw_f),
        "raw_grounding_status": status,
        "raw_source_row_id": row_id,
        "time_delta_seconds": dt_sec,
    }


def is_exact(grounding: Mapping[str, Any]) -> bool:
    return str(grounding.get("raw_grounding_status") or "") == "EXACT"
