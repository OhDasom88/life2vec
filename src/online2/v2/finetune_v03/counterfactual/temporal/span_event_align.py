"""Align actuator spans to Stage-A embedding event_ids via zone + timestamp."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def align_spans_to_events(
    spans: pd.DataFrame,
    emb_side: pd.DataFrame,
    *,
    view_contains: str = "ACTUATOR",
    tolerance: str = "30min",
) -> pd.DataFrame:
    """Replace synthetic source_event_ids with embedding event_ids when possible."""
    if spans is None or len(spans) == 0 or emb_side is None or len(emb_side) == 0:
        return spans.copy() if spans is not None else pd.DataFrame()

    act = emb_side[
        emb_side["view"].astype(str).str.upper().str.contains(view_contains.upper(), na=False)
    ].copy()
    if not len(act):
        out = spans.copy()
        out["alignment_status"] = "NO_ACTUATOR_EVENTS"
        out["aligned_event_count"] = 0
        out["aligned_event_indices"] = [[] for _ in range(len(out))]
        return out

    tol = pd.Timedelta(tolerance)
    rows = []
    for _, sp in spans.iterrows():
        row = dict(sp)
        start = pd.to_datetime(sp.get("start_time"), utc=True, errors="coerce")
        end = pd.to_datetime(sp.get("end_time"), utc=True, errors="coerce")
        zone = str(sp.get("zone_id"))
        cand = act[act["zone"].astype(str) == zone]
        hit = cand.iloc[0:0]
        status = "UNALIGNED"
        if len(cand) and pd.notna(start) and pd.notna(end):
            hit = cand[(cand["timestamp"] >= start) & (cand["timestamp"] < end)]
            if len(hit):
                status = "ALIGNED"
            else:
                delta = (cand["timestamp"] - start).abs()
                k = max(1, int(sp.get("n_events") or 1))
                nearest_idx = delta.nsmallest(min(k, len(cand))).index
                near = cand.loc[nearest_idx]
                near = near.loc[delta.loc[nearest_idx] <= tol]
                if len(near):
                    hit = near
                    status = "NEAREST_TS"
        eids = hit["event_id"].astype(str).tolist() if len(hit) else []
        idxs = hit["event_index"].astype(int).tolist() if len(hit) else []
        if not eids:
            raw = sp.get("source_event_ids")
            if raw is None:
                eids = []
            elif hasattr(raw, "tolist"):
                eids = [str(x) for x in raw.tolist()]
            elif isinstance(raw, str):
                import json

                eids = [str(x) for x in json.loads(raw)]
            else:
                eids = [str(x) for x in list(raw)]
        row["source_event_ids"] = eids
        row["aligned_event_indices"] = idxs
        row["aligned_event_count"] = int(len(idxs))
        row["alignment_status"] = status if idxs else "UNALIGNED"
        if idxs:
            row["n_events"] = len(idxs)
        rows.append(row)
    return pd.DataFrame(rows)


def alignment_summary(spans: pd.DataFrame) -> Dict[str, Any]:
    if spans is None or len(spans) == 0:
        return {"n_spans": 0, "aligned_rate": 0.0, "n_aligned": 0}
    if "alignment_status" not in spans.columns:
        return {"n_spans": int(len(spans)), "aligned_rate": 0.0, "n_aligned": 0}
    ok = spans["alignment_status"].isin(["ALIGNED", "NEAREST_TS"])
    return {
        "n_spans": int(len(spans)),
        "n_aligned": int(ok.sum()),
        "aligned_rate": float(ok.mean()),
        "by_status": spans["alignment_status"].value_counts().to_dict(),
    }
