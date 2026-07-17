"""Intervention locus selection (P0-C)."""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ..gates.context_abac import check_path_a, check_path_b
from ..attribution.fold_consensus import aggregate_fold_frame


NO_VALID = "NO_VALID_INTERVENTION_LOCUS"


def select_path_a_loci(
    mg_attr: pd.DataFrame,
    *,
    semantics: Mapping[str, Mapping[str, Any]],
    raw_join_ok_features: Optional[set] = None,
    min_agree: int = 2,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    if mg_attr is None or len(mg_attr) == 0:
        return []
    # fold consensus on MG keys
    keys = [c for c in ["case_id", "event_id", "measurement_group_id", "feature"] if c in mg_attr.columns]
    cons = aggregate_fold_frame(mg_attr, keys, value_col="signed_attribution", min_agree=min_agree)
    cons = cons[cons["agreement_pass"] > 0].copy()
    loci = []
    for _, row in cons.sort_values("median", key=lambda s: s.abs(), ascending=False).head(top_k).iterrows():
        feat = str(row.get("feature", ""))
        g1 = check_path_a(feat, semantics)
        if g1["status"] != "PASSED":
            continue
        ftype = (semantics.get(feat) or {}).get("feature_type")
        if ftype != "OBSERVED_STATE":
            continue
        raw_ok = True if raw_join_ok_features is None else feat in raw_join_ok_features
        if not raw_ok:
            continue
        loci.append(
            {
                "case_id": row.get("case_id"),
                "level": "mg",
                "path": "A",
                "event_ids": [str(row.get("event_id"))],
                "span_id": None,
                "measurement_group_id": str(row.get("measurement_group_id")),
                "feature": feat,
                "attribution": {
                    "signed": float(row["median"]),
                    "iqr": float(row["iqr"]),
                    "folds_agree": int(row["folds_agree_sign"]),
                },
                "content_hint": {"path": "A", "editable": True},
                "raw_join_ok": bool(raw_ok),
                "gate1": g1,
            }
        )
    return loci


def select_path_b_loci(
    span_attr: pd.DataFrame,
    *,
    semantics: Mapping[str, Mapping[str, Any]],
    whitelist: Optional[set] = None,
    min_agree: int = 2,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    if span_attr is None or len(span_attr) == 0:
        return []
    df = span_attr.copy()
    if "missing_gap_count" in df.columns:
        df = df[df["missing_gap_count"].fillna(0).astype(int) == 0]
    # consensus across folds if present
    if "fold_id" in df.columns:
        keys = [c for c in ["case_id", "span_id", "feature"] if c in df.columns]
        # use duration_weighted_mean
        tmp = df.rename(columns={"duration_weighted_mean": "signed_attribution"})
        cons = aggregate_fold_frame(tmp, keys, value_col="signed_attribution", min_agree=min_agree)
        cons = cons[cons["agreement_pass"] > 0]
        # join back mass ratios from median fold row
        merged = cons.merge(df, on=[c for c in keys if c in df.columns], how="left")
        # dedupe keeping first
        merged = merged.drop_duplicates(subset=keys)
        work = merged
        score_col = "median"
    else:
        work = df
        score_col = "duration_weighted_mean"
    loci = []
    work = work.sort_values(score_col, key=lambda s: s.abs(), ascending=False).head(top_k * 3)
    for _, row in work.iterrows():
        feat = str(row.get("feature", ""))
        g1 = check_path_b(feat, semantics, whitelist=whitelist)
        if g1["status"] != "PASSED":
            continue
        eids = row.get("source_event_ids")
        if eids is None:
            eids = []
        elif hasattr(eids, "tolist"):
            eids = list(eids.tolist())
        elif isinstance(eids, str):
            import json
            eids = json.loads(eids)
        else:
            eids = list(eids)
        loci.append(
            {
                "case_id": row.get("case_id"),
                "level": "span",
                "path": "B",
                "event_ids": list(eids),
                "span_id": str(row.get("span_id")),
                "measurement_group_id": None,
                "feature": feat,
                "attribution": {
                    "signed": float(row.get(score_col) or row.get("duration_weighted_mean") or 0.0),
                    "positive_ratio": float(row.get("duration_weighted_positive_ratio") or 0.0),
                    "start_mass_ratio": float(row.get("start_mass_ratio") or 0.0),
                    "end_mass_ratio": float(row.get("end_mass_ratio") or 0.0),
                    "folds_agree": int(row.get("folds_agree_sign") or 0),
                },
                "content_hint": {"path": "B", "editable": True},
                "raw_join_ok": True,
                "gate1": g1,
            }
        )
        if len(loci) >= top_k:
            break
    return loci
