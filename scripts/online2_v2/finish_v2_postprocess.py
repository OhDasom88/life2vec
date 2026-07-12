#!/usr/bin/env python3
"""Finish V2 post-processing from an existing streamed sequences_v2.parquet.

Avoids reloading full SENTENCE payloads into memory at once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.online2.v2.vocab import VocabV2


ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def load_light_index(seq_path: Path) -> pd.DataFrame:
    cols = [
        "sequence_id",
        "narrative_id",
        "PERSON_ID",
        "event_ids",
        "same_time_group_ids",
        "farm_ids",
        "serialization_signature",
        "event_set_signature",
        "canonical_context_id",
        "split_group_id",
        "measurement_group_ids",
        "token_roles",
        "SENTENCE",
        "sampling_weight",
        "shadow_split",
        "duplicate_class",
        "context_view_count",
    ]
    pf = pq.ParquetFile(seq_path)
    available = set(pf.schema_arrow.names)
    use = [c for c in cols if c in available]
    # First pass without SENTENCE / large JSON for frequency maps
    light_cols = [
        c
        for c in use
        if c
        not in {
            "SENTENCE",
            "measurement_group_ids",
            "token_roles",
            "event_ids",
            "same_time_group_ids",
        }
    ]
    print(f"reading light index cols={light_cols}", flush=True)
    light = pd.read_parquet(seq_path, columns=light_cols)
    print({"light_rows": len(light)}, flush=True)
    return light


def annotate_duplicate_classes(light: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    context_freq = light.groupby("event_set_signature").size()
    light = light.copy()
    light["duplicate_class"] = np.where(
        light["event_set_signature"].map(context_freq) > 1,
        "SAME_CONTEXT_DIFFERENT_NARRATIVE",
        "UNIQUE",
    )
    view_counts = light.groupby("canonical_context_id").size().rename("context_view_count")
    light = light.drop(columns=[c for c in ["context_view_count"] if c in light.columns])
    light = light.join(view_counts, on="canonical_context_id")
    light["sampling_weight"] = 1.0 / np.sqrt(light["context_view_count"].clip(lower=1))
    report = {
        "after": int(len(light)),
        "class_counts": light["duplicate_class"].value_counts().to_dict(),
        "note": "exact duplicates already discarded online during materialize",
        "removed_true_exact_estimate": int(967012 - len(light)) if len(light) < 967012 else 0,
        "before_materialize_target": 967012,
    }
    return light, report


def rewrite_sequences_with_annotations(
    seq_path: Path, annotations: pd.DataFrame, out_path: Path
) -> None:
    ann = annotations.set_index("sequence_id")[
        ["duplicate_class", "context_view_count", "sampling_weight"]
    ]
    pf = pq.ParquetFile(seq_path)
    writer: Optional[pq.ParquetWriter] = None
    tmp = out_path.with_suffix(".tmp.parquet")
    if tmp.exists():
        tmp.unlink()
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i)
        frame = table.to_pandas()
        frame = frame.drop(
            columns=[
                c
                for c in ["duplicate_class", "context_view_count", "sampling_weight", "shadow_split"]
                if c in frame.columns
            ],
            errors="ignore",
        )
        frame = frame.join(ann, on="sequence_id", how="left")
        # fill if missing
        frame["duplicate_class"] = frame["duplicate_class"].fillna("UNIQUE")
        frame["context_view_count"] = frame["context_view_count"].fillna(1).astype(int)
        frame["sampling_weight"] = frame["sampling_weight"].fillna(1.0)
        out_table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, out_table.schema, compression="zstd")
        writer.write_table(out_table)
        if i == 0 or (i + 1) % 20 == 0 or i + 1 == pf.metadata.num_row_groups:
            print(f"rewrite_sequences rg {i+1}/{pf.metadata.num_row_groups}", flush=True)
    if writer is not None:
        writer.close()
    tmp.replace(out_path)


def shadow_split_light(seq_path: Path, annotations: pd.DataFrame) -> dict[str, Any]:
    # Need event_ids + split_group_id + sequence_id only
    print("shadow_split: load event_ids", flush=True)
    ids = pd.read_parquet(
        seq_path, columns=["sequence_id", "split_group_id", "event_ids"]
    )
    ids = ids.merge(
        annotations[["sequence_id", "duplicate_class", "sampling_weight", "context_view_count"]],
        on="sequence_id",
        how="left",
    )
    seed = 2023
    ratios = np.asarray([0.8, 0.1, 0.1])
    thr = np.cumsum(ratios)

    def bucket(key: str) -> str:
        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        if value < thr[0]:
            return "train"
        if value < thr[1]:
            return "val"
        return "test"

    ids["shadow_split"] = ids["split_group_id"].map(bucket)
    exploded = ids[["shadow_split", "event_ids", "sequence_id", "split_group_id"]].copy()
    exploded["event_id"] = exploded["event_ids"].map(json.loads)
    exploded = exploded.explode("event_id")
    overlap = (
        exploded.groupby("event_id")["shadow_split"].nunique().reset_index(name="n_splits")
    )
    bad = overlap[overlap["n_splits"] > 1]
    if len(bad):
        bad_events = set(bad["event_id"])
        touched = exploded[exploded["event_id"].isin(bad_events)]["sequence_id"].unique()
        groups = ids.loc[ids["sequence_id"].isin(touched), "split_group_id"].unique()
        ids.loc[ids["split_group_id"].isin(groups), "shadow_split"] = "train"
        exploded = ids[["shadow_split", "event_ids"]].copy()
        exploded["event_id"] = exploded["event_ids"].map(json.loads)
        exploded = exploded.explode("event_id")
        overlap = (
            exploded.groupby("event_id")["shadow_split"].nunique().reset_index(name="n_splits")
        )
        bad = overlap[overlap["n_splits"] > 1]

    # Rewrite sequences again adding shadow_split via row groups
    split_map = ids.set_index("sequence_id")["shadow_split"]
    pf = pq.ParquetFile(seq_path)
    tmp = seq_path.with_suffix(".split.tmp.parquet")
    writer: Optional[pq.ParquetWriter] = None
    for i in range(pf.metadata.num_row_groups):
        frame = pf.read_row_group(i).to_pandas()
        frame = frame.drop(columns=["shadow_split"], errors="ignore")
        frame["shadow_split"] = frame["sequence_id"].map(split_map)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
        if i == 0 or (i + 1) % 20 == 0 or i + 1 == pf.metadata.num_row_groups:
            print(f"shadow rewrite rg {i+1}/{pf.metadata.num_row_groups}", flush=True)
    if writer is not None:
        writer.close()
    tmp.replace(seq_path)

    report = {
        "event_overlap_after_repair": int(len(bad)),
        "split_counts": ids["shadow_split"].value_counts().to_dict(),
        "sequence_count": int(len(ids)),
    }
    out = seq_path.parent
    write_json(out / "shadow_split_report_v2.json", report)
    overlap.to_csv(out / "shadow_split_event_overlap_v2.csv", index=False)
    return report


def export_training_chunked(out: Path, seq_path: Path, vocab: VocabV2) -> dict[str, Any]:
    pf = pq.ParquetFile(seq_path)
    export_path = out / "training_events_v2.parquet"
    tmp = export_path.with_suffix(".tmp.parquet")
    if tmp.exists():
        tmp.unlink()
    writer: Optional[pq.ParquetWriter] = None
    n_rows = 0
    for i in range(pf.metadata.num_row_groups):
        frame = pf.read_row_group(i).to_pandas()
        rows = []
        for rec in frame.itertuples(index=False):
            event_ids = json.loads(rec.event_ids)
            stgs = json.loads(rec.same_time_group_ids) if rec.same_time_group_ids else []
            farms = json.loads(rec.farm_ids) if rec.farm_ids else []
            rows.append(
                {
                    "PERSON_ID": int(rec.PERSON_ID),
                    "sequence_id": rec.sequence_id,
                    "event_id": event_ids[0] if event_ids else "",
                    "event_position": 0,
                    "time_group_rank": 0,
                    "same_time_group_id": stgs[0] if stgs else "",
                    "START_DATE": pd.Timestamp("2024-01-01"),
                    "AGE": 0.0,
                    "SENTENCE": rec.SENTENCE,
                    "event_kind": "SEQUENCE",
                    "narrative_id": rec.narrative_id,
                    "order_semantics": "STRICT_CHRONOLOGICAL",
                    "op_eligible": True,
                    "modality_ref": "[]",
                    "SEGMENT": 1,
                    "farm_id": farms[0] if farms else "",
                    "zone_id": "",
                    "BACKGROUND_TOKENS": "[]",
                    "build_id": "v2_transductive",
                    "registry_version": "v2",
                    "embedding_status": "ready",
                    "sampling_weight": float(getattr(rec, "sampling_weight", 1.0)),
                    "shadow_split": getattr(rec, "shadow_split", "train"),
                    "canonical_context_id": rec.canonical_context_id,
                    "split_group_id": rec.split_group_id,
                    "measurement_group_ids": rec.measurement_group_ids,
                    "token_roles": rec.token_roles,
                    "training_mode": "transductive_public_pretraining",
                    "contains_problem_observations": True,
                    "contains_problem_images": True,
                    "contains_problem_hidden_targets": False,
                }
            )
        export_df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(export_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
        n_rows += len(export_df)
        if i == 0 or (i + 1) % 20 == 0 or i + 1 == pf.metadata.num_row_groups:
            print(f"export rg {i+1}/{pf.metadata.num_row_groups} rows={n_rows}", flush=True)
    if writer is not None:
        writer.close()
    tmp.replace(export_path)

    tokens = [
        {
            "token_id": i,
            "token": vocab.id_to_token[i],
            "TOKEN": vocab.id_to_token[i],
            "ID": i,
            "category": "GENERAL"
            if vocab.id_to_token[i] in {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}
            else vocab.id_to_token[i].split("|", 1)[0],
            "CATEGORY": "GENERAL"
            if vocab.id_to_token[i] in {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}
            else vocab.id_to_token[i].split("|", 1)[0],
            "registry_version": "v2",
        }
        for i in range(vocab.size())
    ]
    write_json(
        out / "life2vec_token_registry_v2.json",
        {"tokens": tokens, "registry_version": "v2", "schema_version": "online2-v2"},
    )
    return {"rows": n_rows, "path": str(export_path), "vocab_size": vocab.size()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/online2/v2_build")
    args = parser.parse_args()
    out = args.out
    seq_path = out / "sequences_v2.parquet"
    assert seq_path.exists(), seq_path

    print("== annotate duplicates ==")
    light = load_light_index(seq_path)
    annotated, dup_report = annotate_duplicate_classes(light)
    write_json(out / "duplicate_report_v2.json", dup_report)
    annotated[
        [
            "sequence_id",
            "narrative_id",
            "serialization_signature",
            "event_set_signature",
            "duplicate_class",
            "context_view_count",
            "sampling_weight",
        ]
    ].to_parquet(out / "sequences_v2_pre_dedup.parquet", index=False)
    print(dup_report, flush=True)

    print("== rewrite sequences annotations ==")
    rewrite_sequences_with_annotations(seq_path, annotated, seq_path)

    print("== shadow split ==")
    split_report = shadow_split_light(seq_path, annotated)
    print(split_report, flush=True)

    print("== export training ==")
    vocab = VocabV2.load(out / "vocab_v2.json")
    export = export_training_chunked(out, seq_path, vocab)
    print(export, flush=True)

    reg = {}
    if (out / "registry_build_report.json").exists():
        reg = json.loads((out / "registry_build_report.json").read_text())
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit(),
        "training_mode": "transductive_public_pretraining",
        "contains_problem_observations": True,
        "contains_problem_images": True,
        "contains_problem_hidden_targets": False,
        "registries": reg,
        "duplicate_report": dup_report,
        "shadow_split": split_report,
        "export": export,
        "out": str(out),
        "status": "PASS",
    }
    write_json(out / "build_manifest_v2.json", manifest)
    print("DONE", manifest, flush=True)


if __name__ == "__main__":
    main()
