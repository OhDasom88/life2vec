"""Hierarchical attribution aggregation: token → MG → event → span."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_ROLE_WEIGHTS: Dict[str, float] = {
    "abs_value": 1.0,
    "global_value": 0.5,
    "farm_relative_value": 0.5,
    "trend": 0.25,
    "feature_identity": 0.0,
    "unit": 0.0,
    "source": 0.0,
    "time_context": 0.0,
    "farm_context": 0.0,
    "value": 1.0,
    "meta": 0.1,
    "special": 0.0,
    "other": 0.1,
    "typed": 0.25,
    "view": 0.0,
}


def infer_token_role(token: str) -> str:
    t = str(token)
    if t.startswith("FEATURE|"):
        return "feature_identity"
    if t.startswith("VALUE_ABS|") or t.startswith("OBSERVED_VALUE|"):
        return "abs_value"
    if t.startswith("VALUE_GLOBAL") or "GLOBAL_REL" in t:
        return "global_value"
    if t.startswith("VALUE_FARM") or "FARM_REL" in t:
        return "farm_relative_value"
    if "TREND" in t or t.startswith("TREND|"):
        return "trend"
    if t.startswith("UNIT|"):
        return "unit"
    if t.startswith("SOURCE|"):
        return "source"
    if t.startswith(("FARM|", "ZONE|", "TIME|", "DATE|")):
        return "time_context"
    if t.startswith("VIEW|"):
        return "view"
    if t.startswith(("[",)):
        return "special"
    return "other"


def aggregate_measurement_groups(
    token_df: pd.DataFrame,
    *,
    role_weights: Optional[Mapping[str, float]] = None,
    score_col: str = "normalized_signed_attribution",
) -> pd.DataFrame:
    weights = dict(DEFAULT_ROLE_WEIGHTS)
    if role_weights:
        weights.update({str(k): float(v) for k, v in role_weights.items()})
    df = token_df.copy()
    if "token_role" not in df.columns:
        df["token_role"] = df["token_string"].map(infer_token_role)
    df["alpha"] = df["token_role"].map(lambda r: float(weights.get(str(r), 0.1)))
    rows = []
    group_cols = [c for c in ["case_id", "fold_id", "event_id", "measurement_group_id", "feature"] if c in df.columns]
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        alpha = g["alpha"].to_numpy(dtype=np.float64)
        s = g[score_col].to_numpy(dtype=np.float64)
        denom = float(alpha.sum())
        if denom <= 0:
            s_mg = 0.0
        else:
            s_mg = float((alpha * s).sum() / denom)
        abs_s = g.get("normalized_absolute_attribution", g[score_col].abs())
        row = {c: v for c, v in zip(group_cols, keys)}
        row.update(
            {
                "signed_attribution": s_mg,
                "absolute_attribution": float((alpha * np.asarray(abs_s, dtype=np.float64)).sum() / max(denom, 1e-12)),
                "token_count": int(len(g)),
                "value_token_count": int((g["token_role"].isin(["abs_value", "global_value", "farm_relative_value", "value"])).sum()),
                "signed_mean": float(s.mean()) if len(s) else 0.0,
                "absolute_mean": float(np.abs(s).mean()) if len(s) else 0.0,
                "signed_sum": float(s.sum()),
                "absolute_sum": float(np.abs(s).sum()),
            }
        )
        if "sign_agreement" in g.columns:
            abs_g = g[g["token_role"] == "abs_value"]
            if len(abs_g):
                idx = abs_g["absolute_attribution"].abs().idxmax()
                row["sign_agreement"] = bool(abs_g.loc[idx, "sign_agreement"])
            else:
                value_g = g[
                    g["token_role"].isin(
                        ["global_value", "farm_relative_value", "value"]
                    )
                ]
                if len(value_g):
                    idx = value_g["absolute_attribution"].abs().idxmax()
                    row["sign_agreement"] = bool(value_g.loc[idx, "sign_agreement"])
                else:
                    row["sign_agreement"] = bool(g["sign_agreement"].any())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_events(
    mg_df: pd.DataFrame,
    *,
    mg_weight_col: Optional[str] = None,
) -> pd.DataFrame:
    df = mg_df.copy()
    if mg_weight_col and mg_weight_col in df.columns:
        df["w_mg"] = df[mg_weight_col].astype(float)
    else:
        df["w_mg"] = df.get("value_token_count", 1).astype(float).clip(lower=1.0)
    rows = []
    group_cols = [c for c in ["case_id", "fold_id", "event_id"] if c in df.columns]
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        w = g["w_mg"].to_numpy(dtype=np.float64)
        s = g["signed_attribution"].to_numpy(dtype=np.float64)
        denom = float(w.sum())
        s_e = float((w * s).sum() / denom) if denom > 0 else 0.0
        row = {c: v for c, v in zip(group_cols, keys)}
        row.update(
            {
                "signed_attribution": s_e,
                "absolute_attribution": float((w * np.abs(s)).sum() / max(denom, 1e-12)),
                "n_measurement_groups": int(len(g)),
            }
        )
        if "feature" in g.columns:
            # keep top feature by |score|
            top = g.reindex(g["signed_attribution"].abs().sort_values(ascending=False).index).iloc[0]
            row["top_feature"] = str(top.get("feature", ""))
        rows.append(row)
    return pd.DataFrame(rows)


def _mass_ratios(times: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(times) == 0:
        return {
            "attribution_mass_center": 0.0,
            "start_mass_ratio": 0.0,
            "middle_mass_ratio": 0.0,
            "end_mass_ratio": 0.0,
            "peak_locus_index": -1,
        }
    pos = np.clip(scores, a_min=0.0, a_max=None)
    mass = pos.copy()
    if mass.sum() <= 1e-12:
        mass = np.abs(scores)
    if mass.sum() <= 1e-12:
        mass = np.ones_like(scores)
    mass = mass / mass.sum()
    n = len(times)
    # thirds by index (equal event thirds); duration handled by caller via Δt weights
    a, b = max(1, n // 3), max(1, (2 * n) // 3)
    start = float(mass[:a].sum())
    mid = float(mass[a:b].sum())
    end = float(mass[b:].sum())
    center = float(np.sum(np.arange(n) * mass) / max(n - 1, 1))
    peak = int(np.argmax(pos)) if pos.sum() > 0 else int(np.argmax(np.abs(scores)))
    return {
        "attribution_mass_center": center,
        "start_mass_ratio": start,
        "middle_mass_ratio": mid,
        "end_mass_ratio": end,
        "peak_locus_index": peak,
    }


def aggregate_spans(
    event_df: pd.DataFrame,
    span_df: pd.DataFrame,
    *,
    event_id_col: str = "event_id",
) -> pd.DataFrame:
    """Duration-weighted span attribution. Positive ratio uses duration, not event count."""
    rows = []
    ev = event_df.set_index(["case_id", "fold_id", event_id_col], drop=False) if "fold_id" in event_df.columns else event_df
    for _, span in span_df.iterrows():
        eids = list(span.get("source_event_ids") or [])
        if isinstance(eids, str):
            import json
            eids = json.loads(eids)
        case_id = span.get("case_id")
        fold_id = span.get("fold_id")
        dts = []
        scores = []
        for i, eid in enumerate(eids):
            # duration share: equal if no per-event dt provided
            dt = float(span.get("sampling_interval_min") or 60.0)
            if "event_durations_min" in span and span["event_durations_min"]:
                durs = span["event_durations_min"]
                if isinstance(durs, str):
                    import json
                    durs = json.loads(durs)
                dt = float(durs[i]) if i < len(durs) else dt
            # lookup score
            score = 0.0
            sub = event_df
            if case_id is not None and "case_id" in event_df.columns:
                sub = sub[sub["case_id"] == case_id]
            if fold_id is not None and "fold_id" in event_df.columns:
                sub = sub[sub["fold_id"] == fold_id]
            hit = sub[sub[event_id_col].astype(str) == str(eid)]
            if len(hit):
                score = float(hit.iloc[0]["signed_attribution"])
            dts.append(dt)
            scores.append(score)
        dts_a = np.asarray(dts, dtype=np.float64)
        scores_a = np.asarray(scores, dtype=np.float64)
        denom = float(dts_a.sum()) if dts_a.size else 0.0
        mean = float((dts_a * scores_a).sum() / denom) if denom > 0 else 0.0
        pos_dur = float(dts_a[scores_a > 0].sum()) if dts_a.size else 0.0
        pos_ratio = pos_dur / denom if denom > 0 else 0.0
        mass = _mass_ratios(np.arange(len(scores_a)), scores_a)
        row = {
            "case_id": case_id,
            "fold_id": fold_id,
            "span_id": span.get("span_id"),
            "feature": span.get("feature"),
            "farm_id": span.get("farm_id"),
            "zone_id": span.get("zone_id"),
            "source_event_ids": eids,
            "signed_total_attribution": float(scores_a.sum()),
            "absolute_total_attribution": float(np.abs(scores_a).sum()),
            "duration_weighted_mean": mean,
            "duration_weighted_positive_ratio": pos_ratio,
            "max_positive_attribution": float(scores_a.max()) if scores_a.size else 0.0,
            "max_absolute_attribution": float(np.abs(scores_a).max()) if scores_a.size else 0.0,
            "estimate_duration_min": span.get("estimate_duration_min"),
            "missing_gap_count": span.get("missing_gap_count", 0),
        }
        row.update(mass)
        rows.append(row)
    return pd.DataFrame(rows)
