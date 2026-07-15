"""v0.1 post-hoc layered saliency (factor strata + Stage A token IxG).

Does not modify v0.1 freeze training modules or v0.2 train path.
Reuses Stage A window helpers via importlib (read-only).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.online2.v2.diagnosis_dataset import (
    collate_diagnosis_batch,
    view_to_id,
    zone_to_id,
)
from src.online2.v2.event_pooling_finetune import EventPoolingDiagnosisModel

ROOT = Path(__file__).resolve().parents[3]
V01_CACHE = ROOT / "scripts/online2_v2/cache_stage_a_event_embeddings.py"

AGE_BINS = [0, 24, 72, 168, 336, 10_000]
AGE_LABELS = ["0-24h", "24-72h", "72-168h", "168-336h", "336h+"]
HOUR_BINS = [0, 6, 12, 18, 24]
HOUR_LABELS = ["00-05", "06-11", "12-17", "18-23"]


def load_stage_a_module():
    name = "cache_stage_a_v01_readonly"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, V01_CACHE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required before exec for @dataclass on Py3.10
    spec.loader.exec_module(mod)
    return mod


def classify_token_type(token: str) -> str:
    t = str(token)
    if t in {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"} or t.startswith("["):
        return "special"
    if t.startswith("VIEW|"):
        return "view"
    if t.startswith("FEATURE|"):
        return "feature"
    if t.startswith(("OBSERVED_VALUE|", "VALUE_", "EC_SENSOR|", "RAW_CODE")):
        return "value"
    if t.startswith(("QUALITY|", "STATE_SEMANTICS|", "EVENT_KIND|")):
        return "meta"
    if "|" in t:
        return "typed"
    return "other"


def age_bin_label(hours: float) -> str:
    h = float(hours) if hours is not None and not (isinstance(hours, float) and np.isnan(hours)) else 0.0
    for lo, hi, lab in zip(AGE_BINS[:-1], AGE_BINS[1:], AGE_LABELS):
        if lo <= h < hi:
            return lab
    return AGE_LABELS[-1]


def hour_bin_label(hour: Any) -> str:
    try:
        h = int(hour)
    except (TypeError, ValueError):
        h = 0
    h = max(0, min(23, h))
    for lo, hi, lab in zip(HOUR_BINS[:-1], HOUR_BINS[1:], HOUR_LABELS):
        if lo <= h < hi:
            return lab
    return HOUR_LABELS[-1]


def summarize_factor_distributions(
    consensus_df: pd.DataFrame,
    *,
    case_id: str,
    primary_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Layer-1: saliency score distributions by view / zone / hour / age."""
    df = consensus_df.copy()
    if df.empty:
        return {
            "case_id": case_id,
            "n_events": 0,
            "by_view": [],
            "by_zone": [],
            "by_view_zone": [],
            "by_hour_bin": [],
            "by_age_bin": [],
            "primary_top": [],
        }

    df["abs_saliency"] = df["median_saliency"].astype(float).abs()
    df["hour_bin"] = df["local_hour"].map(hour_bin_label)
    df["age_bin"] = df["case_age_hours"].map(age_bin_label)

    def _agg(g: pd.DataFrame) -> Dict[str, Any]:
        s = g["median_saliency"].astype(float)
        return {
            "n": int(len(g)),
            "mean_saliency": float(s.mean()),
            "median_saliency": float(s.median()),
            "p90_abs": float(g["abs_saliency"].quantile(0.9)),
            "sum_abs": float(g["abs_saliency"].sum()),
            "mean_agreement": float(g["positive_agreement_count"].astype(float).mean())
            if "positive_agreement_count" in g.columns
            else None,
            "frac_agreement_ge4": float((g["positive_agreement_count"] >= 4).mean())
            if "positive_agreement_count" in g.columns
            else None,
        }

    def _group_rows(keys: List[str]) -> List[Dict[str, Any]]:
        rows = []
        for key_vals, g in df.groupby(keys, dropna=False):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in zip(keys, key_vals)}
            row.update(_agg(g))
            rows.append(row)
        rows.sort(key=lambda r: -float(r["sum_abs"]))
        return rows

    primary_top: List[Dict[str, Any]] = []
    src = primary_df if primary_df is not None and len(primary_df) else df
    if len(src):
        top = src.sort_values("median_saliency", ascending=False).head(15)
        for _, r in top.iterrows():
            primary_top.append(
                {
                    "event_id": str(r["event_id"]),
                    "view": str(r.get("view")),
                    "zone": r.get("zone"),
                    "median_saliency": float(r["median_saliency"]),
                    "positive_agreement_count": int(r.get("positive_agreement_count", 0)),
                    "local_hour": int(r["local_hour"]) if pd.notna(r.get("local_hour")) else None,
                    "case_age_hours": float(r["case_age_hours"])
                    if pd.notna(r.get("case_age_hours"))
                    else None,
                }
            )

    return {
        "case_id": case_id,
        "n_events": int(len(df)),
        "by_view": _group_rows(["view"]),
        "by_zone": _group_rows(["zone"]),
        "by_view_zone": _group_rows(["view", "zone"]),
        "by_hour_bin": _group_rows(["hour_bin"]),
        "by_age_bin": _group_rows(["age_bin"]),
        "primary_top": primary_top,
        "global": _agg(df),
    }


def factor_summary_rows(dist: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten distribution dict into CSV-friendly rows."""
    rows: List[Dict[str, Any]] = []
    case_id = dist["case_id"]
    for axis, items in [
        ("view", dist.get("by_view") or []),
        ("zone", dist.get("by_zone") or []),
        ("view_zone", dist.get("by_view_zone") or []),
        ("hour_bin", dist.get("by_hour_bin") or []),
        ("age_bin", dist.get("by_age_bin") or []),
    ]:
        for it in items:
            factor_parts = []
            for k in ("view", "zone", "hour_bin", "age_bin"):
                if k in it and it[k] is not None:
                    factor_parts.append(f"{k}={it[k]}")
            rows.append(
                {
                    "case_id": case_id,
                    "axis": axis,
                    "factor": "|".join(factor_parts) if factor_parts else "ALL",
                    "n": it.get("n"),
                    "mean_saliency": it.get("mean_saliency"),
                    "median_saliency": it.get("median_saliency"),
                    "p90_abs": it.get("p90_abs"),
                    "sum_abs": it.get("sum_abs"),
                    "mean_agreement": it.get("mean_agreement"),
                    "frac_agreement_ge4": it.get("frac_agreement_ge4"),
                }
            )
    return rows


def render_factor_report_md(
    dist: Dict[str, Any],
    *,
    diagnosis: Optional[str] = None,
    token_summary: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        f"# Layer saliency 요인 분석: {dist['case_id']}",
        "",
    ]
    if diagnosis:
        lines.append(f"- 진단: **{diagnosis}**")
    lines.append(f"- 이벤트 수: {dist.get('n_events', 0)}")
    g = dist.get("global") or {}
    if g:
        lines.append(
            f"- 전체 median_saliency 평균={g.get('mean_saliency')}, "
            f"합의≥4 비율={g.get('frac_agreement_ge4')}"
        )
    lines += ["", "## Layer 1 — view / zone 기여 (sum |saliency|)", ""]
    for it in (dist.get("by_view_zone") or [])[:12]:
        lines.append(
            f"- `{it.get('view')}` zone{it.get('zone')}: "
            f"n={it['n']}, sum_abs={it['sum_abs']:.3f}, "
            f"mean={it['mean_saliency']:.3f}, agr≥4={it.get('frac_agreement_ge4')}"
        )
    lines += ["", "## Layer 1 — 시간대 / case_age", ""]
    for it in dist.get("by_hour_bin") or []:
        lines.append(
            f"- hour `{it.get('hour_bin')}`: n={it['n']}, sum_abs={it['sum_abs']:.3f}"
        )
    for it in dist.get("by_age_bin") or []:
        lines.append(
            f"- age `{it.get('age_bin')}`: n={it['n']}, sum_abs={it['sum_abs']:.3f}"
        )
    lines += ["", "## Primary top events", ""]
    for it in (dist.get("primary_top") or [])[:10]:
        lines.append(
            f"- {it['view']}|zone{it['zone']} sal={it['median_saliency']:.3f} "
            f"agr={it['positive_agreement_count']} `{it['event_id'][:20]}…`"
        )
    if token_summary:
        lines += ["", "## Layer 2 — token IxG (consensus events)", ""]
        lines.append(
            f"- events_scored={token_summary.get('n_events')}, "
            f"tokens={token_summary.get('n_tokens')}, "
            f"mean_pool_vs_cache_l2={token_summary.get('mean_pool_l2_err')}"
        )
        for t, s in (token_summary.get("by_token_type") or {}).items():
            lines.append(f"- type `{t}`: sum_abs={s:.4f}")
        lines += ["", "### Top tokens", ""]
        for it in (token_summary.get("top_tokens") or [])[:15]:
            lines.append(
                f"- `{it['token']}` ({it['token_type']}) score={it['score']:.4f} "
                f"event={it['event_id'][:16]}…"
            )
    lines.append("")
    return "\n".join(lines)


@dataclass
class TokenIxGResult:
    event_id: str
    tokens: List[str]
    scores: np.ndarray  # [L_target]
    token_types: List[str]
    pool_l2_err: float
    span_start: int
    span_end: int
    fold_scores: Optional[np.ndarray] = None  # [F, L]


def load_case_batch_tensors(
    emb_path: Path,
    *,
    max_events: int = 4096,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[str], pd.DataFrame]:
    df = pd.read_parquet(emb_path).sort_values("event_order").reset_index(drop=True)
    if len(df) > max_events:
        df = df.iloc[:max_events].copy()
    means = np.stack([np.asarray(x, dtype=np.float32) for x in df["event_mean"]])
    maxes = np.stack([np.asarray(x, dtype=np.float32) for x in df["event_max"]])
    if "case_age_hours" in df.columns:
        ages = df["case_age_hours"].astype(np.float32).to_numpy()
    else:
        ts = pd.to_datetime(df["timestamp"])
        ages = ((ts - ts.iloc[0]) / pd.Timedelta(hours=1)).astype(np.float32).to_numpy()
    if "local_hour" in df.columns:
        hours = df["local_hour"].astype(np.int64).to_numpy()
    else:
        hours = pd.to_datetime(df["timestamp"]).dt.hour.astype(np.int64).to_numpy()
    views = np.asarray([view_to_id(v) for v in df["view"]], dtype=np.int64)
    zones = np.asarray([zone_to_id(z) for z in df["zone"]], dtype=np.int64)
    sample = {
        "case_id": emb_path.stem,
        "farm_id": emb_path.stem.split("_")[0],
        "label": 0,
        "diagnosis_normalized": "",
        "event_mean": means,
        "event_max": maxes,
        "case_age_hours": ages,
        "view_id": views,
        "zone_id": zones,
        "local_hour": hours,
        "length": int(means.shape[0]),
        "hidden_size": int(means.shape[1]),
    }
    batch = collate_diagnosis_batch([sample], max_events=max_events)
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    event_ids = [str(x) for x in df["event_id"].tolist()]
    return batch, event_ids, df


def _pool_target_span(hidden: torch.Tensor, s: int, e: int) -> Tuple[torch.Tensor, torch.Tensor]:
    span = hidden[0, s:e, :]
    if span.numel() == 0:
        z = hidden.new_zeros(hidden.size(-1))
        return z, z
    return span.mean(dim=0), span.max(dim=0).values


def token_ixg_for_event(
    *,
    encoder: nn.Module,
    diagnosis_models: Sequence[EventPoolingDiagnosisModel],
    stage_a_mod: Any,
    events: list,
    target_idx: int,
    case_t0: pd.Timestamp,
    abspos_reference: pd.Timestamp,
    vocab: Any,
    batch: Dict[str, torch.Tensor],
    event_index: int,
    class_id: int,
    device: torch.device,
    max_length: int = 1024,
) -> TokenIxGResult:
    """Re-encode one event via Stage A, stitch into case, IxG on target tokens."""
    window = stage_a_mod.construct_target_window(events, target_idx, max_length=max_length)
    x, pad_mask, _L = stage_a_mod.window_to_tensors(
        events,
        window,
        vocab=vocab,
        abspos_reference=abspos_reference,
        case_t0=case_t0,
        max_length=max_length,
    )
    x = x.unsqueeze(0).to(device)  # [1,4,L]
    pad = pad_mask.unsqueeze(0).to(device).long()
    s, e = int(window.target_token_start), int(window.target_token_end)
    target_tokens = events[target_idx].sentence_tokens
    # Truncation can shorten target span
    span_len = e - s
    tokens_out = target_tokens[:span_len]

    # Consistency check (no grad)
    with torch.no_grad():
        H0 = encoder.transformer.forward_finetuning(x=x, padding_mask=pad)
        mean0, max0 = _pool_target_span(H0, s, e)
        cache_mean = batch["event_mean"][0, event_index].detach()
        pool_l2 = float(torch.norm(mean0 - cache_mean).item())

    fold_score_list: List[torch.Tensor] = []
    for model in diagnosis_models:
        model.eval()
        emb, _ = encoder.transformer.get_sequence_embedding(x.long())
        emb = emb.detach().requires_grad_(True)
        H = encoder.transformer.forward_finetuning_with_embeddings(emb, pad)
        mean_v, max_v = _pool_target_span(H, s, e)

        event_mean = batch["event_mean"].detach().clone()
        event_max = batch["event_max"].detach().clone()
        event_mean = event_mean.clone()
        event_max = event_max.clone()
        # replace target event with differentiable pool
        event_mean_t = event_mean.clone()
        event_max_t = event_max.clone()
        event_mean_t[0, event_index] = mean_v
        event_max_t[0, event_index] = max_v

        out = model(
            event_mean=event_mean_t,
            event_max=event_max_t,
            case_age_hours=batch["case_age_hours"],
            view_id=batch["view_id"],
            zone_id=batch["zone_id"],
            local_hour=batch["local_hour"],
            padding_mask=batch["padding_mask"],
        )
        objective = out["logits"][0, int(class_id)]
        model.zero_grad(set_to_none=True)
        if emb.grad is not None:
            emb.grad = None
        objective.backward()
        assert emb.grad is not None
        tok_score = (emb.grad * emb).sum(dim=-1)[0, s:e].detach()
        fold_score_list.append(tok_score)

    stack = torch.stack(fold_score_list, dim=0)  # [F, L]
    median_scores = stack.median(dim=0).values.cpu().numpy().astype(np.float32)
    types = [classify_token_type(t) for t in tokens_out]
    # pad types/tokens if span shorter mismatch
    if len(types) < len(median_scores):
        types += ["other"] * (len(median_scores) - len(types))
        tokens_out += ["<?>"] * (len(median_scores) - len(tokens_out))
    elif len(types) > len(median_scores):
        types = types[: len(median_scores)]
        tokens_out = tokens_out[: len(median_scores)]

    return TokenIxGResult(
        event_id=str(events[target_idx].event_id),
        tokens=list(tokens_out),
        scores=median_scores,
        token_types=types,
        pool_l2_err=pool_l2,
        span_start=s,
        span_end=e,
        fold_scores=stack.cpu().numpy().astype(np.float32),
    )


def token_results_to_frame(
    results: Sequence[TokenIxGResult],
    *,
    case_id: str,
    meta: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    rows = []
    meta = meta or {}
    for r in results:
        m = meta.get(r.event_id, {})
        order = np.argsort(-np.abs(r.scores))
        for rank, j in enumerate(order):
            rows.append(
                {
                    "case_id": case_id,
                    "event_id": r.event_id,
                    "token_index": int(j),
                    "rank_in_event": int(rank),
                    "token": r.tokens[j],
                    "token_type": r.token_types[j],
                    "saliency": float(r.scores[j]),
                    "abs_saliency": float(abs(r.scores[j])),
                    "pool_l2_err": r.pool_l2_err,
                    "view": m.get("view"),
                    "zone": m.get("zone"),
                    "median_event_saliency": m.get("median_saliency"),
                    "positive_agreement_count": m.get("positive_agreement_count"),
                }
            )
    return pd.DataFrame(rows)


def summarize_token_frame(tok_df: pd.DataFrame) -> Dict[str, Any]:
    if tok_df is None or tok_df.empty:
        return {"n_events": 0, "n_tokens": 0, "by_token_type": {}, "top_tokens": [], "mean_pool_l2_err": None}
    by_type = (
        tok_df.groupby("token_type")["abs_saliency"].sum().sort_values(ascending=False).to_dict()
    )
    by_type = {str(k): float(v) for k, v in by_type.items()}
    top = (
        tok_df.sort_values("abs_saliency", ascending=False)
        .head(20)[["token", "token_type", "saliency", "event_id", "abs_saliency"]]
        .to_dict(orient="records")
    )
    for t in top:
        t["score"] = float(t.pop("saliency"))
        t["abs_saliency"] = float(t["abs_saliency"])
    return {
        "n_events": int(tok_df["event_id"].nunique()),
        "n_tokens": int(len(tok_df)),
        "by_token_type": by_type,
        "top_tokens": top,
        "mean_pool_l2_err": float(tok_df.groupby("event_id")["pool_l2_err"].first().mean()),
    }


def pick_primary_events(
    primary_df: pd.DataFrame,
    *,
    max_events: int,
) -> pd.DataFrame:
    if primary_df is None or primary_df.empty:
        return pd.DataFrame()
    df = primary_df.sort_values(
        ["positive_agreement_count", "median_saliency"], ascending=[False, False]
    )
    return df.head(int(max_events)).copy()


def write_cross_case_aggregates(
    out_root: Path,
    *,
    case_infos: Sequence[Dict[str, Any]],
    factor_summary_csv: Path,
) -> Dict[str, str]:
    """Build diagnosis×factor and primary-event catalogs for offline factor analysis."""
    out_root = Path(out_root)
    paths: Dict[str, str] = {}

    # 1) factor_summary joined with diagnosis
    fs = pd.read_csv(factor_summary_csv) if factor_summary_csv.exists() else pd.DataFrame()
    diag_map = {
        str(c["case_id"]): c.get("diagnosis") for c in case_infos if c.get("case_id")
    }
    if len(fs):
        fs["diagnosis"] = fs["case_id"].astype(str).map(diag_map)
        by_diag = (
            fs.groupby(["diagnosis", "axis", "factor"], dropna=False)
            .agg(
                n_cases=("case_id", "nunique"),
                n_rows=("n", "sum"),
                mean_of_mean_saliency=("mean_saliency", "mean"),
                mean_sum_abs=("sum_abs", "mean"),
                mean_frac_agr4=("frac_agreement_ge4", "mean"),
            )
            .reset_index()
            .sort_values(["diagnosis", "axis", "mean_sum_abs"], ascending=[True, True, False])
        )
        p1 = out_root / "factor_by_diagnosis.csv"
        by_diag.to_csv(p1, index=False)
        paths["factor_by_diagnosis"] = str(p1)

        view_zone = fs[fs["axis"] == "view_zone"].copy()
        if len(view_zone):
            pivot = (
                view_zone.groupby(["diagnosis", "factor"], dropna=False)["sum_abs"]
                .mean()
                .reset_index()
                .sort_values(["diagnosis", "sum_abs"], ascending=[True, False])
            )
            p2 = out_root / "view_zone_sumabs_by_diagnosis.csv"
            pivot.to_csv(p2, index=False)
            paths["view_zone_sumabs_by_diagnosis"] = str(p2)

    # 2) primary event catalog across cases
    prim_rows: List[Dict[str, Any]] = []
    for c in case_infos:
        cid = c.get("case_id")
        diag = c.get("diagnosis")
        for it in (c.get("dist") or {}).get("primary_top") or []:
            prim_rows.append(
                {
                    "case_id": cid,
                    "diagnosis": diag,
                    **it,
                }
            )
    prim = pd.DataFrame(prim_rows)
    if len(prim):
        p3 = out_root / "primary_event_catalog.csv"
        prim.to_csv(p3, index=False)
        paths["primary_event_catalog"] = str(p3)
        top_feat = (
            prim.groupby(["diagnosis", "view"], dropna=False)
            .agg(
                n_primary=("event_id", "count"),
                mean_saliency=("median_saliency", "mean"),
                mean_agreement=("positive_agreement_count", "mean"),
            )
            .reset_index()
            .sort_values(["diagnosis", "mean_saliency"], ascending=[True, False])
        )
        p4 = out_root / "primary_view_by_diagnosis.csv"
        top_feat.to_csv(p4, index=False)
        paths["primary_view_by_diagnosis"] = str(p4)

    # 3) token type rollup if any token parquet exist
    tok_files = sorted((out_root / "cases").glob("*/token_saliency.parquet"))
    if tok_files:
        toks = pd.concat([pd.read_parquet(p) for p in tok_files], ignore_index=True)
        toks["diagnosis"] = toks["case_id"].astype(str).map(diag_map)
        type_roll = (
            toks.groupby(["diagnosis", "token_type"], dropna=False)["abs_saliency"]
            .sum()
            .reset_index()
            .sort_values(["diagnosis", "abs_saliency"], ascending=[True, False])
        )
        p5 = out_root / "token_type_by_diagnosis.csv"
        type_roll.to_csv(p5, index=False)
        paths["token_type_by_diagnosis"] = str(p5)

        feat_tok = toks[toks["token_type"].isin(["feature", "value"])].copy()
        if len(feat_tok):
            top_tok = (
                feat_tok.groupby(["diagnosis", "token", "token_type"], dropna=False)
                .agg(
                    n=("event_id", "count"),
                    mean_abs=("abs_saliency", "mean"),
                    sum_abs=("abs_saliency", "sum"),
                )
                .reset_index()
                .sort_values(["diagnosis", "sum_abs"], ascending=[True, False])
            )
            # keep top 30 per diagnosis
            keep = []
            for _, g in top_tok.groupby("diagnosis", dropna=False):
                keep.append(g.head(30))
            top_tok = pd.concat(keep, ignore_index=True) if keep else top_tok
            p6 = out_root / "top_feature_tokens_by_diagnosis.csv"
            top_tok.to_csv(p6, index=False)
            paths["top_feature_tokens_by_diagnosis"] = str(p6)

        p7 = out_root / "token_saliency_all.parquet"
        toks.to_parquet(p7, index=False)
        paths["token_saliency_all"] = str(p7)

    return paths
