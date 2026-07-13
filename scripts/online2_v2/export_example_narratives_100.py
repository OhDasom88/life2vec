#!/usr/bin/env python3
"""Export 100 example narratives with analysis-friendly columns (V2 corpus)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/online2/v2_build/example_100_narratives_v2.csv"
CATALOG = ROOT / "datasets/agrichallenge/online2/narratives/normalized_catalog_v8.csv"
SEQ_PATH = ROOT / "outputs/online2/v2_build/sequences_v2.parquet"
EVENTS_PATH = ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet"

SPECIAL = {"[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"}
STRUCT_TOKENS = {"[GROUP_SEP]", "[EVENT_SEP]", "[MEAS_SEP]"}


def parse_sentence(sentence: str) -> dict:
    tokens = sentence.split()
    n_feature = sum(1 for t in tokens if t.startswith("FEATURE|"))
    n_abs = sum(1 for t in tokens if t.startswith("VALUE_ABS|"))
    n_global = sum(1 for t in tokens if t.startswith("VALUE_GLOBAL_REL|"))
    n_farm = sum(1 for t in tokens if t.startswith("VALUE_FARM_REL|"))
    n_view = sum(1 for t in tokens if t.startswith("VIEW|"))
    n_image = sum(1 for t in tokens if t.startswith("IMAGE_"))
    n_text = sum(1 for t in tokens if t.startswith("TEXT_"))
    n_events = sentence.count("[EVENT_SEP]") + (1 if "EVENT_KIND|" in sentence else 0)
    n_groups = sentence.count("[GROUP_SEP]")
    n_meas = sentence.count("[MEAS_SEP]")
    bg = []
    for tok in tokens:
        if tok in STRUCT_TOKENS or tok.startswith("EVENT_KIND|"):
            break
        if tok.startswith(("FEATURE|", "VALUE_", "VIEW|", "IMAGE_", "TEXT_")):
            break
        bg.append(tok)
    preview = " ".join(tokens[:16])
    if len(tokens) > 16:
        preview += " ..."
    return {
        "n_tokens": len(tokens),
        "n_events_est": n_events,
        "n_groups": n_groups,
        "n_measurement_groups": n_meas,
        "n_feature_tokens": n_feature,
        "n_value_abs": n_abs,
        "n_value_global_rel": n_global,
        "n_value_farm_rel": n_farm,
        "n_view_tokens": n_view,
        "n_image_tokens": n_image,
        "n_text_tokens": n_text,
        "background_tokens": " ".join(bg[:8]),
        "token_preview": preview,
    }


def structure_summary(row: pd.Series, parsed: dict) -> str:
    dup = row.get("duplicate_class", "")
    views = row.get("context_view_count", 1)
    parts = [
        f"서사 {row['narrative_id']}({row.get('narrative_name_ko', '')})의 학습 입력 1건.",
        f"이벤트≈{parsed['n_events_est']}개, SameTimeGroup≈{parsed['n_groups']}개, 토큰 {parsed['n_tokens']}개.",
        f"FEATURE {parsed['n_feature_tokens']} / ABS {parsed['n_value_abs']} / GLOBAL_REL {parsed['n_value_global_rel']} / FARM_REL {parsed['n_value_farm_rel']}.",
    ]
    if parsed["n_image_tokens"] or row.get("contains_image"):
        parts.append("이미지 슬롯 포함.")
    if dup == "SAME_CONTEXT_DIFFERENT_NARRATIVE":
        parts.append(f"동일 맥락의 다른 narrative view(공유 {views}건) → sampling_weight={row.get('sampling_weight', 1):.3f}.")
    elif dup == "UNIQUE":
        parts.append("고유 맥락 시퀀스.")
    return " ".join(parts)


def how_materialized(row: pd.Series) -> str:
    return (
        f"카탈로그 {row['narrative_id']}({row.get('narrative_name_ko', '')}) / "
        f"matcher={row.get('materialization_key', '')}; "
        f"trigger={row.get('start_condition', '')}; "
        f"window={row.get('window_definition', '')}; "
        f"공개풀(example+problem 관측) → V2 registry/binning → tokenize → sequence materialize"
    )


def how_used() -> str:
    return (
        "RegistryVocabulary(v2, 876)로 encode; grouped MLM/SOP; "
        "transductive_public_pretraining; metadata(NARRATIVE/CATEGORY)는 입력 제외; "
        "sampling_weight·duplicate_class 반영 가능"
    )


def sample_100(df: pd.DataFrame, seed: int = 2023) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    # cover all narratives once
    for nid, grp in df.groupby("narrative_id", sort=True):
        med = grp["token_count_stored"].median()
        idx = (grp["token_count_stored"] - med).abs().idxmin()
        picked.append(idx)
    remaining = 100 - len(picked)
    if remaining > 0:
        used = set(picked)
        pool = df[~df.index.isin(used)].copy()
        # stratify by narrative volume
        weights = pool["narrative_id"].map(pool["narrative_id"].value_counts()).astype(float)
        weights = weights / weights.sum()
        extra = pool.sample(n=remaining, weights=weights, random_state=seed, replace=False)
        picked.extend(extra.index.tolist())
    out = df.loc[picked].copy()
    if len(out) > 100:
        out = out.head(100)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    catalog = pd.read_csv(CATALOG, dtype=str).set_index("narrative_id")
    light_cols = [
        "sequence_id",
        "narrative_id",
        "event_ids",
        "same_time_group_ids",
        "farm_ids",
        "contains_image",
        "token_count_stored",
        "duplicate_class",
        "context_view_count",
        "sampling_weight",
        "shadow_split",
        "training_mode",
        "tokenization_version",
        "canonical_context_id",
    ]
    meta = pd.read_parquet(SEQ_PATH, columns=light_cols)
    meta["n_events"] = meta["event_ids"].map(lambda s: len(json.loads(s)) if isinstance(s, str) else 0)
    meta["n_groups"] = meta["same_time_group_ids"].map(
        lambda s: len(json.loads(s)) if isinstance(s, str) and s else 0
    )
    meta["farm_ids_list"] = meta["farm_ids"].map(
        lambda s: json.loads(s) if isinstance(s, str) and s else []
    )
    meta["n_farms"] = meta["farm_ids_list"].map(len)
    meta["farm_ids_str"] = meta["farm_ids_list"].map(lambda xs: ",".join(xs))

    narr_counts = meta["narrative_id"].value_counts()
    meta["narrative_sequence_count"] = meta["narrative_id"].map(narr_counts)
    meta["narrative_share_pct"] = (meta["narrative_sequence_count"] / len(meta) * 100).round(3)
    rank = narr_counts.rank(method="dense", ascending=False).astype(int)
    meta["narrative_rank_by_volume"] = meta["narrative_id"].map(rank)

    sample_meta = sample_100(meta, seed=args.seed)
    ids = set(sample_meta["sequence_id"].tolist())

    # load SENTENCE only for sampled rows via row-group scan (memory safe)
    sentences: dict[str, str] = {}
    pf = pq.ParquetFile(SEQ_PATH)
    for i in range(pf.metadata.num_row_groups):
        frame = pf.read_row_group(i, columns=["sequence_id", "SENTENCE"]).to_pandas()
        sub = frame[frame["sequence_id"].isin(ids)]
        for rec in sub.itertuples(index=False):
            sentences[rec.sequence_id] = rec.SENTENCE
        if len(sentences) == len(ids):
            break

    rows = []
    for i, rec in enumerate(sample_meta.itertuples(index=False), start=1):
        nid = rec.narrative_id
        cat = catalog.loc[nid] if nid in catalog.index else pd.Series(dtype=str)
        sent = sentences.get(rec.sequence_id, "")
        parsed = parse_sentence(sent) if sent else {}
        base = {
            "row_id": i,
            "narrative_id": nid,
            "category": cat.get("category", ""),
            "narrative_name_ko": cat.get("narrative_name_ko", ""),
            "purpose": cat.get("purpose", ""),
            "agronomic_interpretation": cat.get("agronomic_interpretation", ""),
            "diagnosis_level": cat.get("diagnosis_level", ""),
            "data_sources": cat.get("data_sources", ""),
            "materialization_key": cat.get("materialization_key", ""),
            "start_condition": cat.get("start_condition", ""),
            "window_definition": cat.get("window_definition", ""),
            "order_semantics": cat.get("order_semantics", ""),
            "op_eligible": cat.get("op_eligible", ""),
            "catalog_trigger_rows": cat.get("trigger_row_count", ""),
            "catalog_unique_instances": cat.get("unique_instance_count", ""),
            "sequence_id": rec.sequence_id,
            "sequence_id_short": rec.sequence_id[:16],
            "narrative_sequence_count_total": int(rec.narrative_sequence_count),
            "narrative_rank_by_volume": int(rec.narrative_rank_by_volume),
            "narrative_share_pct": float(rec.narrative_share_pct),
            "n_events": int(rec.n_events),
            "n_same_time_groups": int(rec.n_groups),
            "token_count_stored": int(rec.token_count_stored),
            "duplicate_class": rec.duplicate_class,
            "context_view_count": int(rec.context_view_count),
            "sampling_weight": float(rec.sampling_weight),
            "contains_image": bool(rec.contains_image),
            "farm_ids": rec.farm_ids_str,
            "n_farms": int(rec.n_farms),
            "shadow_split": rec.shadow_split,
            "training_mode": rec.training_mode,
            "tokenization_version": rec.tokenization_version,
            "canonical_context_id": rec.canonical_context_id,
            "background_tokens": parsed.get("background_tokens", ""),
            "n_feature_tokens": parsed.get("n_feature_tokens", 0),
            "n_value_abs": parsed.get("n_value_abs", 0),
            "n_value_global_rel": parsed.get("n_value_global_rel", 0),
            "n_value_farm_rel": parsed.get("n_value_farm_rel", 0),
            "n_view_tokens": parsed.get("n_view_tokens", 0),
            "n_image_tokens": parsed.get("n_image_tokens", 0),
            "n_text_tokens": parsed.get("n_text_tokens", 0),
            "token_preview": parsed.get("token_preview", ""),
            "catalog_sequence_example": cat.get("sequence_example", ""),
            "catalog_token_example": cat.get("token_example", ""),
            "structure_summary_ko": "",
            "how_materialized_ko": "",
            "how_used_in_training_ko": how_used(),
            "analysis_note_ko": "",
        }
        ser = pd.Series(base)
        ser["structure_summary_ko"] = structure_summary(ser, parsed)
        ser["how_materialized_ko"] = how_materialized(ser)
        if cat.get("expected_pattern"):
            ser["analysis_note_ko"] = (
                f"기대 패턴: {cat.get('expected_pattern', '')}. "
                f"해석 수준: {cat.get('diagnosis_level', '')}. "
                f"교란요인: {cat.get('confounders', '')}."
            )
        rows.append(ser)

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": len(out_df),
                "unique_narratives": int(out_df["narrative_id"].nunique()),
                "duplicate_class_counts": out_df["duplicate_class"].value_counts().to_dict(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
