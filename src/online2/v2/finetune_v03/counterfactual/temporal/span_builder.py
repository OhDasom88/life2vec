"""Zero-order-hold actuator span reconstruction with gap splitting."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence
import uuid

import numpy as np
import pandas as pd


def _is_on(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        s = str(value).upper()
        return s in {"POSITIVE", "ON", "1", "TRUE"}


def build_zoh_spans(
    records: pd.DataFrame,
    *,
    feature: str,
    farm_id: str,
    zone_id: str,
    case_id: Optional[str] = None,
    time_col: str = "timestamp",
    value_col: str = "raw_value",
    event_id_col: str = "event_id",
    nominal_interval_min: float = 60.0,
    max_gap_factor: float = 1.5,
) -> pd.DataFrame:
    """Build ON spans under ZOH. Do not bridge missing gaps beyond max_gap_factor * nominal."""
    if records is None or len(records) == 0:
        return pd.DataFrame()
    df = records.copy()
    df["_ts"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"]).sort_values("_ts")
    rows: List[Dict[str, Any]] = []
    cur_ids: List[str] = []
    cur_start = None
    cur_last = None
    gap_count = 0
    sampling = float(nominal_interval_min)

    def flush(end_ts, gaps: int):
        nonlocal cur_ids, cur_start, cur_last
        if not cur_ids or cur_start is None or cur_last is None:
            cur_ids, cur_start, cur_last = [], None, None
            return
        # ZOH: span ends at last_sample + sampling (upper) / last_sample (lower mid)
        estimate = (cur_last - cur_start).total_seconds() / 60.0 + sampling
        lower = max(sampling, (cur_last - cur_start).total_seconds() / 60.0)
        upper = estimate + sampling
        rows.append(
            {
                "span_id": f"span_{uuid.uuid4().hex[:10]}",
                "case_id": case_id,
                "farm_id": farm_id,
                "zone_id": str(zone_id),
                "feature": feature,
                "start_time": cur_start.isoformat(),
                "end_time": (cur_last + pd.Timedelta(minutes=sampling)).isoformat(),
                "estimate_duration_min": float(estimate),
                "lower_bound_min": float(lower),
                "upper_bound_min": float(upper),
                "sampling_interval_min": sampling,
                "missing_gap_count": int(gaps),
                "span_confidence": "low" if gaps else "medium",
                "source_event_ids": list(cur_ids),
                "assumption": "state_persists_until_next_sample",
                "state": "POSITIVE",
                "n_events": len(cur_ids),
            }
        )
        cur_ids, cur_start, cur_last = [], None, None

    prev_ts = None
    local_gaps = 0
    for _, r in df.iterrows():
        ts = r["_ts"]
        on = _is_on(r[value_col])
        eid = str(r.get(event_id_col) or f"{feature}:{ts.isoformat()}")
        if prev_ts is not None:
            gap_min = (ts - prev_ts).total_seconds() / 60.0
            if gap_min > sampling * max_gap_factor:
                # break current span without bridging
                if cur_ids:
                    flush(prev_ts, local_gaps + 1)
                local_gaps = 0
                if on:
                    cur_start = ts
                    cur_last = ts
                    cur_ids = [eid]
                prev_ts = ts
                continue
        if on:
            if not cur_ids:
                cur_start = ts
                local_gaps = 0
            cur_ids.append(eid)
            cur_last = ts
        else:
            if cur_ids:
                flush(prev_ts if prev_ts is not None else ts, local_gaps)
                local_gaps = 0
        prev_ts = ts
    if cur_ids:
        flush(prev_ts, local_gaps)
    return pd.DataFrame(rows)


def build_interval_quantities(
    records: pd.DataFrame,
    *,
    feature: str,
    farm_id: str,
    zone_id: str,
    case_id: Optional[str] = None,
    time_col: str = "timestamp",
    value_col: str = "raw_value",
) -> pd.DataFrame:
    if records is None or len(records) == 0:
        return pd.DataFrame()
    df = records.copy()
    df["_ts"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"]).sort_values("_ts")
    rows = []
    prev = None
    for _, r in df.iterrows():
        ts = r["_ts"]
        val = r[value_col]
        try:
            q = float(val)
        except (TypeError, ValueError):
            prev = (ts, None)
            continue
        if prev is not None and prev[0] is not None:
            dt_min = (ts - prev[0]).total_seconds() / 60.0
            if dt_min > 0:
                rows.append(
                    {
                        "case_id": case_id,
                        "farm_id": farm_id,
                        "zone_id": str(zone_id),
                        "feature": feature,
                        "start_time": prev[0].isoformat(),
                        "end_time": ts.isoformat(),
                        "interval_quantity": q,
                        "delta_t_min": float(dt_min),
                        "average_flux": float(q / dt_min),
                        "flux_unit": "quantity_per_min",
                        "note": "interval_average_not_instantaneous",
                    }
                )
        prev = (ts, q)
    return pd.DataFrame(rows)
