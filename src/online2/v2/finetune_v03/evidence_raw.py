"""Attach raw sensor values onto event evidence rows (P0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def load_cells_for_farm(
    cells_path: Path,
    farm_id: str,
    *,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    cols = columns or [
        "cell_id",
        "farm_id",
        "zone_id",
        "observation_timestamp",
        "column_name",
        "feature_alias",
        "raw_value",
        "raw_display",
        "modality",
        "is_null",
    ]
    df = pd.read_parquet(cells_path, columns=[c for c in cols if True])
    return df[df["farm_id"].astype(str) == str(farm_id)].copy()


def enrich_events_with_raw(
    events: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    max_cells_per_event: int = 12,
) -> List[Dict[str, Any]]:
    """Best-effort join on (farm, zone, timestamp). events need event_id, zone, timestamp."""
    if events is None or not len(events) or cells is None or not len(cells):
        return []
    cells = cells.copy()
    cells["_ts"] = pd.to_datetime(cells["observation_timestamp"], utc=True, errors="coerce")
    out: List[Dict[str, Any]] = []
    for _, ev in events.iterrows():
        row: Dict[str, Any] = {
            "event_id": str(ev.get("event_id", "")),
            "view": str(ev.get("view", "")),
            "zone": str(ev.get("zone", "")),
            "timestamp": str(ev.get("timestamp", "")),
            "raw_samples": [],
        }
        try:
            ts = pd.to_datetime(ev.get("timestamp"), utc=True)
        except Exception:
            out.append(row)
            continue
        zone = str(ev.get("zone", ""))
        hit = cells[(cells["zone_id"].astype(str) == zone) & (cells["_ts"] == ts)]
        if not len(hit):
            # nearest within 1s
            hit = cells[cells["zone_id"].astype(str) == zone]
            if len(hit):
                delta = (hit["_ts"] - ts).abs()
                hit = hit.loc[delta.nsmallest(min(max_cells_per_event, len(hit))).index]
        samples = []
        for _, c in hit.head(max_cells_per_event).iterrows():
            if bool(c.get("is_null", False)):
                continue
            samples.append(
                {
                    "feature": str(c.get("column_name") or c.get("feature_alias") or ""),
                    "raw_value": c.get("raw_value"),
                    "raw_display": c.get("raw_display"),
                    "modality": c.get("modality"),
                }
            )
        row["raw_samples"] = samples
        out.append(row)
    return out


def write_enriched_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
