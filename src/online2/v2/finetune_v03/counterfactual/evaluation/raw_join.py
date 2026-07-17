"""Raw occurrence join metrics for Path A loci."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd


def parse_cell_raw(value: Any, display: Any = None) -> Any:
    if display is not None and str(display) not in {"", "None", "nan"}:
        try:
            return float(display)
        except (TypeError, ValueError):
            pass
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return value
    s = str(value)
    if s.startswith("{") and "value" in s:
        try:
            return json.loads(s).get("value")
        except Exception:
            return s
    return s


def join_raw_for_events(
    emb_side: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    farm_id: str,
    features: Sequence[str],
    tolerance: str = "1h",
) -> Dict[str, Any]:
    """Join cell raw values onto embedding events by zone+timestamp.

    Returns per-feature success rates and a long join table.
    """
    cells = cells.copy()
    cells = cells[cells["farm_id"].astype(str) == str(farm_id)]
    if "is_null" in cells.columns:
        cells = cells[~cells["is_null"].fillna(False)]
    cells["_ts"] = pd.to_datetime(cells["observation_timestamp"], utc=True, errors="coerce")
    cells["_raw"] = [
        parse_cell_raw(v, d)
        for v, d in zip(cells["raw_value"], cells.get("raw_display", [None] * len(cells)))
    ]
    tol = pd.Timedelta(tolerance)
    rows = []
    for feat in features:
        sub = cells[cells["column_name"].astype(str) == feat]
        n_events = 0
        n_ok = 0
        for _, ev in emb_side.iterrows():
            # ENV/ROOT features join to non-actuator views preferentially
            view = str(ev.get("view", "")).upper()
            if feat in {"circulation_fan", "fcu_fan", "fcu_pump", "co2_supply", "shade_screen", "thermal_curtain"}:
                if "ACTUATOR" not in view:
                    continue
            else:
                if "ACTUATOR" in view:
                    continue
            n_events += 1
            zone = str(ev.get("zone"))
            ts = ev.get("timestamp")
            hit = sub[(sub["zone_id"].astype(str) == zone)]
            if not len(hit) or pd.isna(ts):
                rows.append(
                    {
                        "event_id": ev.get("event_id"),
                        "feature": feat,
                        "raw_join_ok": False,
                        "raw_value": None,
                    }
                )
                continue
            delta = (hit["_ts"] - ts).abs()
            best_i = delta.idxmin()
            if delta.loc[best_i] > tol:
                rows.append(
                    {
                        "event_id": ev.get("event_id"),
                        "feature": feat,
                        "raw_join_ok": False,
                        "raw_value": None,
                    }
                )
                continue
            raw = hit.loc[best_i, "_raw"]
            try:
                raw_f = float(raw)
            except (TypeError, ValueError):
                rows.append(
                    {
                        "event_id": ev.get("event_id"),
                        "feature": feat,
                        "raw_join_ok": False,
                        "raw_value": raw,
                    }
                )
                continue
            n_ok += 1
            rows.append(
                {
                    "event_id": str(ev.get("event_id")),
                    "event_index": int(ev.get("event_index", -1)),
                    "feature": feat,
                    "raw_join_ok": True,
                    "raw_value": raw_f,
                    "zone": zone,
                    "timestamp": str(ts),
                }
            )
        # store feature-level rate in rows meta via separate dict after loop
    join_df = pd.DataFrame(rows)
    by_feat = {}
    if len(join_df):
        for feat, g in join_df.groupby("feature"):
            by_feat[str(feat)] = {
                "n": int(len(g)),
                "n_ok": int(g["raw_join_ok"].sum()),
                "success_rate": float(g["raw_join_ok"].mean()),
            }
    overall = float(join_df["raw_join_ok"].mean()) if len(join_df) else 0.0
    return {
        "overall_success_rate": overall,
        "n_rows": int(len(join_df)),
        "by_feature": by_feat,
        "join_df": join_df,
    }
