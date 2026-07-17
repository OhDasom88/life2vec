"""Observed bundle bank loader for constrained MLM (training_corpus_plus_case_past)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
import pyarrow.parquet as pq

from ..attribution.aggregation import infer_token_role
from .constrained_mlm import build_observed_bundle_bank, filter_bundles, role_signature
from .mlm_window_lift import annotate_sentence_tokens, extract_value_bundle


def bundle_fingerprint(feature: str, tokens: Sequence[str]) -> str:
    payload = json.dumps(
        {"feature": str(feature), "tokens": list(tokens)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sentence_to_tokens(sentence: str) -> List[str]:
    return [t for t in str(sentence or "").split(" ") if t]


def extract_mg_rows_from_tokens(
    tokens: Sequence[str],
    *,
    event_id: str,
    vocab_token2index: Optional[dict] = None,
    unk_id: int = 0,
    continuous_features: Optional[Set[str]] = None,
    timestamp: Optional[pd.Timestamp] = None,
    case_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """One row per measurement group with value tokens."""
    ann = annotate_sentence_tokens(
        tokens, vocab_token2index=vocab_token2index, unk_id=unk_id
    )
    by_mg: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(ann.group_ids):
        by_mg[str(g)].append(i)
    rows: List[Dict[str, Any]] = []
    for mg_id, idxs in by_mg.items():
        if mg_id in {"mg:none", "mg:unknown"}:
            continue
        feat = ann.features[idxs[0]] if idxs else "unknown"
        if continuous_features is not None and str(feat).lower() not in continuous_features:
            # also allow exact case match
            if str(feat) not in continuous_features:
                continue
        v_toks, v_ids, v_roles = extract_value_bundle(ann, measurement_group_id=mg_id)
        if not v_toks:
            continue
        rows.append(
            {
                "feature": str(feat),
                "event_id": str(event_id),
                "case_id": str(case_id or ""),
                "measurement_group_id": str(mg_id),
                "tokens": list(v_toks),
                "token_ids": list(v_ids),
                "roles": list(v_roles),
                "signature": role_signature(v_roles),
                "timestamp": timestamp,
                "fingerprint": bundle_fingerprint(feat, v_toks),
            }
        )
    return rows


def load_continuous_feature_names(feature_schema_path: Path) -> Set[str]:
    import yaml

    raw = yaml.safe_load(Path(feature_schema_path).read_text(encoding="utf-8")) or {}
    feats = raw.get("features") or {}
    out: Set[str] = set()
    for name, spec in feats.items():
        if str((spec or {}).get("type") or "").lower() == "continuous":
            out.add(str(name))
            out.add(str(name).lower())
    return out


def _parse_case_period(case_id: str) -> Tuple[str, pd.Timestamp, pd.Timestamp]:
    parts = str(case_id).split("_")
    if len(parts) < 3:
        raise ValueError(f"bad case_id={case_id}")
    farm = parts[0]
    start = pd.Timestamp(parts[1])
    end = pd.Timestamp(parts[2])
    return farm, start, end


def iter_event_bundle_rows(
    events_path: Path,
    *,
    case_ids: Sequence[str],
    vocab_token2index: Optional[dict] = None,
    unk_id: int = 0,
    continuous_features: Optional[Set[str]] = None,
    include_image: bool = False,
) -> List[Dict[str, Any]]:
    """Extract MG value bundles from events parquet for given cases."""
    farms = {str(c).split("_")[0] for c in case_ids}
    cols = [
        "event_id",
        "event_kind",
        "observation_timestamp",
        "farm_id",
        "SENTENCE",
    ]
    table = pq.read_table(events_path, columns=cols)
    df = table.to_pandas()
    df = df[df["farm_id"].astype(str).isin(farms)].copy()
    df["_ts"] = pd.to_datetime(df["observation_timestamp"], utc=True).dt.tz_localize(None)
    df = df.rename(columns={"_ts": "obs_ts"})

    exclude = {"INTERPRETATION"}
    if not include_image:
        exclude.add("IMAGE")
    df = df[~df["event_kind"].astype(str).isin(exclude)].copy()

    # map farm -> list of (case_id, start, end)
    by_farm: Dict[str, List[Tuple[str, pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for cid in case_ids:
        farm, start, end = _parse_case_period(cid)
        by_farm[farm].append((cid, start, end + pd.Timedelta(days=1)))

    rows: List[Dict[str, Any]] = []
    for farm, windows in by_farm.items():
        sub = df[df["farm_id"].astype(str) == str(farm)]
        for cid, start, end_excl in windows:
            hit = sub[(sub["obs_ts"] >= start) & (sub["obs_ts"] < end_excl)]
            for r in hit.itertuples(index=False):
                toks = _sentence_to_tokens(r.SENTENCE)
                rows.extend(
                    extract_mg_rows_from_tokens(
                        toks,
                        event_id=str(r.event_id),
                        vocab_token2index=vocab_token2index,
                        unk_id=unk_id,
                        continuous_features=continuous_features,
                        timestamp=pd.Timestamp(r.obs_ts),
                        case_id=cid,
                    )
                )
    return rows


def build_bundle_bank_for_locus(
    rows: Iterable[Mapping[str, Any]],
    *,
    feature: str,
    expected_roles: Sequence[str],
    target_event_id: str,
    target_timestamp: Optional[pd.Timestamp] = None,
    case_id: Optional[str] = None,
    exclude_target_event: bool = True,
    exclude_same_case_future: bool = True,
    original_tokens: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter bank to feature/signature with leave-out + future exclude; unique fingerprints."""
    feat = str(feature)
    work = []
    for row in rows:
        if str(row.get("feature")) != feat and str(row.get("feature")).lower() != feat.lower():
            continue
        eid = str(row.get("event_id") or "")
        if exclude_target_event and eid == str(target_event_id):
            continue
        if exclude_same_case_future and case_id and target_timestamp is not None:
            if str(row.get("case_id") or "") == str(case_id):
                ts = row.get("timestamp")
                if ts is not None and pd.Timestamp(ts) >= pd.Timestamp(target_timestamp):
                    continue
        work.append(dict(row))

    # unique by fingerprint
    seen = set()
    unique = []
    for r in work:
        fp = r.get("fingerprint") or bundle_fingerprint(r["feature"], r["tokens"])
        if fp in seen:
            continue
        seen.add(fp)
        r["fingerprint"] = fp
        unique.append(r)

    bank_by_feat = build_observed_bundle_bank(unique, exclude_event_id=None)
    feat_rows = []
    for k, v in bank_by_feat.items():
        if str(k).lower() == feat.lower():
            feat_rows.extend(v)
    if not feat_rows:
        feat_rows = unique
    filtered = filter_bundles(
        feat_rows,
        feature=feat,
        expected_signature=expected_roles,
        original_tokens=original_tokens,
    )
    # ensure originals marked
    if original_tokens is not None:
        for b in filtered:
            b["is_original"] = list(b["tokens"]) == list(original_tokens)

    meta = {
        "feature": feat,
        "target_event_id": str(target_event_id),
        "case_id": str(case_id or ""),
        "exclude_target_event": bool(exclude_target_event),
        "exclude_same_case_future": bool(exclude_same_case_future),
        "n_source_rows": len(work),
        "n_unique_fingerprints": len(unique),
        "n_filtered_bundles": len(filtered),
        "n_original_marked": sum(1 for b in filtered if b.get("is_original")),
        "bundle_bank_scope": "training_corpus_plus_case_past",
        "expected_signature": list(expected_roles),
    }
    return filtered, meta


def resolve_training_case_ids(
    labels_path: Path,
    *,
    example_cohort_values: Sequence[str] = ("example_set", "example", "EXAMPLE"),
) -> List[str]:
    df = pd.read_csv(labels_path)
    if "cohort" in df.columns:
        mask = df["cohort"].astype(str).isin(set(str(x) for x in example_cohort_values))
        # also accept diagnosis-based example if cohort missing values
        ids = df.loc[mask, "case_id"].astype(str).tolist()
        if ids:
            return sorted(set(ids))
    # fallback: all cases except explicit problem_set
    if "cohort" in df.columns:
        ids = df.loc[df["cohort"].astype(str) != "problem_set", "case_id"].astype(str).tolist()
        if ids:
            return sorted(set(ids))
    return sorted(set(df["case_id"].astype(str).tolist()))
