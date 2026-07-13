#!/usr/bin/env python3
"""Build Online2 V2 registries, tokenization, and training export (transductive)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from src.online2.v2.binning import BinningPolicy, fit_binning_v2
from src.online2.v2.feature_schema import build_feature_schema_from_audit
from src.online2.v2.masking import GroupedMLMMasker
from src.online2.v2.provenance import training_mode_meta
from src.online2.v2.tokenizer import TokenizerV2, wind_compass
from src.online2.v2.vocab import VocabV2, build_vocab_v2, save_token_usage_policy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = ROOT / "outputs/online2/build-v8-active80-r3"
DEFAULT_AUDIT = ROOT / "outputs/online2/v2_audit/feature_semantics_and_units.csv"
DEFAULT_OUT = ROOT / "outputs/online2/v2_build"
DEFAULT_BINNING_POLICY = DEFAULT_OUT / "binning_policy_v2.yaml"
DATA = ROOT / "datasets/agrichallenge/online2"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def case_sets() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in ("example_set", "problem_set"):
        path = DATA / name / "case_list.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for farm in frame["farm_id"].astype(str):
            mapping[farm] = name
    return mapping


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def local_spatial_maps(farms: list[str], zones: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    farm_map = {farm: f"FARM_LOCAL|{idx}" for idx, farm in enumerate(sorted(set(farms)))}
    zone_map = {zone: f"ZONE_LOCAL|{idx}" for idx, zone in enumerate(sorted(set(z for z in zones if z)))}
    return farm_map, zone_map


def delta_t_bucket(hours: float) -> str:
    if hours < 1.5:
        return "DELTA_T|1H"
    if hours < 4:
        return "DELTA_T|3H"
    if hours < 9:
        return "DELTA_T|6H"
    if hours < 18:
        return "DELTA_T|12H"
    if hours < 36:
        return "DELTA_T|1D"
    if hours < 84:
        return "DELTA_T|3D"
    if hours < 168:
        return "DELTA_T|7D"
    return "DELTA_T|GT_7D"


def day_from_start_bucket(days: float) -> str:
    if days <= 2:
        return "DAY_FROM_START|D00_02"
    if days <= 7:
        return "DAY_FROM_START|D03_07"
    if days <= 14:
        return "DAY_FROM_START|D08_14"
    if days <= 21:
        return "DAY_FROM_START|D15_21"
    return "DAY_FROM_START|D22_PLUS"


def build_registries(out: Path, legacy_build: Path, audit: Path) -> dict[str, Any]:
    schema = build_feature_schema_from_audit(audit)
    schema.save(out / "feature_schema_v2.yaml")
    farms = case_sets()
    cells = pd.read_parquet(
        legacy_build / "cell_occurrences.parquet",
        columns=["cell_id", "column_name", "raw_display", "farm_id", "zone_id", "is_null"],
    )
    # Fit rows
    # Deterministic row order for streaming population hash.
    cells = cells.sort_values(
        ["column_name", "farm_id", "zone_id", "cell_id"], kind="mergesort"
    )
    rows = []
    example_n = problem_n = 0
    for rec in cells.itertuples(index=False):
        source_set = farms.get(str(rec.farm_id), "unknown")
        if source_set == "example_set":
            example_n += 1
        elif source_set == "problem_set":
            problem_n += 1
        raw = "" if bool(getattr(rec, "is_null", False)) else str(rec.raw_display)
        try:
            value = float(raw) if raw not in {"", "nan", "None"} else None
        except ValueError:
            value = None
        rows.append((str(rec.column_name), str(rec.farm_id), value, source_set))

    source_hashes = {}
    for path in sorted((legacy_build).glob("*.json")):
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    policy_path = out / "binning_policy_v2.yaml"
    if not policy_path.exists():
        policy_path = DEFAULT_BINNING_POLICY
    policy = BinningPolicy.load(policy_path if policy_path.exists() else None)
    binning = fit_binning_v2(
        schema,
        rows,
        source_file_hashes=source_hashes,
        example_count=example_n,
        problem_count=problem_n,
        code_commit_hash=git_commit(),
        policy=policy,
    )
    binning.save(out / "binning_registry_v2_transductive.json")
    # Compact machine-readable policy snapshot (hash + counts).
    write_json(
        out / "binning_policy_v2.json",
        {
            "edge_policy": binning.meta.get("edge_policy"),
            "abs_policy": binning.meta.get("abs_policy"),
            "global_rel_policy": binning.meta.get("global_rel_policy"),
            "farm_rel_policy": binning.meta.get("farm_rel_policy"),
            "zero_mass_threshold": binning.meta.get("zero_mass_threshold"),
            "policy_version": binning.meta.get("policy_version"),
            "registry_hash": binning.meta.get("registry_hash"),
            "feature_count": len(binning.meta.get("feature_summaries") or {}),
            "farm_rule_count": binning.meta.get("farm_rule_count"),
            "policy_yaml": str(policy_path) if policy_path.exists() else None,
        },
    )
    vocab = build_vocab_v2(schema, binning, code_commit_hash=git_commit())
    vocab.save(out / "vocab_v2.json")
    save_token_usage_policy(out / "token_usage_policy_v2.json")
    write_json(
        out / "measurement_group_policy_v2.json",
        {
            "grouped_masking": {
                "enabled": True,
                "mask_feature_identity": False,
                "mask_value_tokens_together": True,
                "mask_directly_revealing_derived_tokens": True,
            },
            "farm_relative": {
                "enabled": True,
                "fit_scope": "full_public_farm_distribution",
                "causal": False,
            },
            "spatial_identity_mode": "local_ordinal",
            **training_mode_meta(git_commit_hash=git_commit(), config_hash=schema.config_hash()),
        },
    )
    # Circular unit test artifact
    assert wind_compass(1) == "N"
    assert wind_compass(359) == "N"
    assert wind_compass(45) == "NE"
    return {
        "schema_hash": schema.config_hash(),
        "binning_hash": binning.meta["registry_hash"],
        "vocab_size": vocab.size(),
        "example_observation_count": example_n,
        "problem_observation_count": problem_n,
        "rule_count": len(binning.rules),
    }


def tokenize_events(
    out: Path, legacy_build: Path, max_events: Optional[int] = None
) -> pd.DataFrame:
    schema = build_feature_schema_from_audit  # noqa - load from saved
    from src.online2.v2.feature_schema import FeatureSchema
    from src.online2.v2.binning import BinningRegistryV2

    schema = FeatureSchema.load(out / "feature_schema_v2.yaml")
    binning = BinningRegistryV2.load(out / "binning_registry_v2_transductive.json")
    vocab = VocabV2.load(out / "vocab_v2.json")
    tokenizer = TokenizerV2(schema, binning, vocab)

    cells = pd.read_parquet(
        legacy_build / "cell_occurrences.parquet",
        columns=[
            "cell_id",
            "column_name",
            "raw_display",
            "farm_id",
            "zone_id",
            "observation_timestamp",
            "modality",
            "is_null",
        ],
    )
    mappings = pd.read_parquet(
        legacy_build / "cell_token_mappings.parquet",
        columns=["cell_id", "atomic_value_id"],
    ).drop_duplicates("cell_id")
    cells = cells.merge(mappings, on="cell_id", how="left")
    events = pd.read_parquet(
        legacy_build / "events.parquet",
        columns=["event_id", "event_kind", "event_view", "observation_timestamp", "same_time_group_id", "farm_ids", "zone_ids", "embedding_status"],
    )
    # Map cells to events via farm/zone/timestamp/modality
    cells = cells.copy()
    cells["farm"] = cells["farm_id"].astype(str)
    cells["zone"] = cells["zone_id"].astype(str)
    cells["ts"] = cells["observation_timestamp"].astype(str)
    cells["view"] = cells["modality"].astype(str)

    events = events.copy()
    def _first_id(raw: Any) -> str:
        if raw is None or raw == "":
            return ""
        values = json.loads(raw) if isinstance(raw, str) else list(raw)
        return str(values[0]) if values else ""

    events["farm"] = events["farm_ids"].map(_first_id)
    events["zone"] = events["zone_ids"].map(_first_id)
    events["ts"] = events["observation_timestamp"].astype(str)
    events["view"] = events["event_view"].astype(str)

    # Restrict identity columns
    skip_cols = {"farm_id", "zone_id", "timestamp", "observation_date", "hour"}
    if max_events:
        keep_ids = set(events["event_id"].head(max_events))
        events = events[events["event_id"].isin(keep_ids)]

    # Compact cell index: avoid materializing 200k+ DataFrame group objects.
    cells = cells.sort_values(
        ["farm", "zone", "ts", "view", "column_name", "cell_id"], kind="mergesort"
    )
    cell_records = list(
        cells[
            [
                "farm",
                "zone",
                "ts",
                "view",
                "column_name",
                "cell_id",
                "farm_id",
                "raw_display",
                "is_null",
                "atomic_value_id",
            ]
        ].itertuples(index=False, name=None)
    )
    cell_groups: dict[tuple[str, str, str, str], list[tuple]] = defaultdict(list)
    for rec in cell_records:
        cell_groups[(rec[0], rec[1], rec[2], rec[3])].append(rec)

    rows = []
    total = len(events)
    for idx, ev in enumerate(events.itertuples(index=False), start=1):
        if idx == 1 or idx % 5000 == 0 or idx == total:
            print(f"tokenize_events {idx}/{total}", flush=True)
        key = (ev.farm, ev.zone, ev.ts, ev.view)
        grp = cell_groups.get(key)
        tokens: list[str] = []
        group_ids: list[str] = []
        roles: list[str] = []
        if ev.event_kind == "IMAGE":
            tokens = ["[IMAGE_SLOT]", "IMAGE_ROLE|UNRESOLVED", "VIEW|IMAGE", "EVENT_KIND|IMAGE"]
            group_ids = ["NONE"] * len(tokens)
            roles = ["slot", "meta", "meta", "meta"]
        elif ev.event_kind == "INTERPRETATION":
            tokens = ["[TEXT_SLOT]", "VIEW|INTERPRETATION", "EVENT_KIND|INTERPRETATION"]
            group_ids = ["NONE"] * len(tokens)
            roles = ["slot", "meta", "meta"]
        elif grp is not None:
            for cell in grp:
                column_name = cell[4]
                if column_name in skip_cols:
                    continue
                raw = "" if bool(cell[8]) else str(cell[7])
                measured = tokenizer.tokenize_value(
                    str(column_name),
                    raw,
                    farm_id=str(cell[6]),
                    cell_id=str(cell[5]),
                    atomic_value_id=str(cell[9] or ""),
                )
                tokens.append("[MEAS_SEP]")
                group_ids.append(measured.measurement_group_id)
                roles.append("sep")
                for tok, role in zip(measured.tokens, measured.roles):
                    tokens.append(tok)
                    group_ids.append(measured.measurement_group_id)
                    roles.append(role)
            view_name = str(ev.view)
            view_tok = view_name.split("_")[-1].upper() if "_" in view_name else view_name.upper()
            tokens = ["[EVENT_SEP]", f"VIEW|{view_tok}", "EVENT_KIND|OBSERVATION"] + tokens
            group_ids = ["NONE", "NONE", "NONE"] + group_ids
            roles = ["sep", "meta", "meta"] + roles
        else:
            tokens = ["[EVENT_SEP]", "EVENT_KIND|OBSERVATION", "[MISSING]"]
            group_ids = ["NONE"] * 3
            roles = ["sep", "meta", "quality"]

        token_ids = [vocab.get(t) for t in tokens]
        rows.append(
            {
                "event_id": ev.event_id,
                "same_time_group_id": ev.same_time_group_id,
                "event_kind": ev.event_kind,
                "observation_timestamp": ev.observation_timestamp,
                "farm_id": ev.farm,
                "zone_id": ev.zone,
                "SENTENCE": " ".join(tokens),
                "measurement_group_ids": json.dumps(group_ids),
                "token_roles": json.dumps(roles),
                "token_ids": json.dumps(token_ids),
                "embedding_status": getattr(ev, "embedding_status", ""),
                "tokenization_version": "v2",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "events_tokenized_v2.parquet", index=False)
    return frame


def materialize_sequences(out: Path, legacy_build: Path, events_v2: pd.DataFrame) -> dict[str, Any]:
    """Stream sequence materialization to avoid holding ~1M full sentences in RAM."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    segments = pd.read_parquet(
        legacy_build / "sequence_segments.parquet",
        columns=["sequence_id", "event_id", "position"],
    )
    sequences = pd.read_parquet(
        legacy_build / "sequences.parquet",
        columns=["sequence_id", "narrative_id", "contains_image"],
    )
    seq_meta = sequences.set_index("sequence_id")
    segments = segments.sort_values(["sequence_id", "position"], kind="mergesort")

    sent_map = events_v2.set_index("event_id")["SENTENCE"].astype(str).to_dict()
    farm_map = events_v2.set_index("event_id")["farm_id"].astype(str).to_dict()
    stg_map = events_v2.set_index("event_id")["same_time_group_id"].astype(str).to_dict()
    role_map = events_v2.set_index("event_id")["token_roles"].astype(str).to_dict()
    mg_map = events_v2.set_index("event_id")["measurement_group_ids"].astype(str).to_dict()
    ts_map = {
        eid: ts
        for eid, ts in zip(
            events_v2["event_id"],
            pd.to_datetime(events_v2["observation_timestamp"], utc=True),
        )
    }

    max_store_tokens = 2048
    chunk_size = 5000
    parts_dir = out / "_seq_parts"
    if parts_dir.exists():
        for old in parts_dir.glob("*.parquet"):
            old.unlink()
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Online dedup: keep first (serialization_signature, narrative_id)
    seen_exact: set[tuple[str, str]] = set()
    event_set_counts: dict[str, int] = defaultdict(int)
    class_counts = {
        "TRUE_EXACT_DUPLICATE": 0,
        "SAME_CONTEXT_DIFFERENT_NARRATIVE": 0,
        "UNIQUE": 0,
    }
    before = 0
    kept = 0
    chunk: list[dict[str, Any]] = []
    part_idx = 0
    writer: Optional[pq.ParquetWriter] = None
    final_path = out / "sequences_v2.parquet"

    def flush_chunk() -> None:
        nonlocal chunk, part_idx, writer
        if not chunk:
            return
        frame = pd.DataFrame(chunk)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(final_path, table.schema, compression="zstd")
        writer.write_table(table)
        part_idx += 1
        chunk = []

    total = int(segments["sequence_id"].nunique())
    idx = 0
    for sequence_id, group in segments.groupby("sequence_id", sort=False):
        idx += 1
        if idx == 1 or idx % 20000 == 0 or idx == total:
            print(f"materialize_sequences {idx}/{total} kept={kept}", flush=True)
        event_ids = group["event_id"].tolist()
        if any(eid not in sent_map for eid in event_ids):
            continue
        before += 1
        meta = seq_meta.loc[sequence_id]
        narrative_id = str(meta["narrative_id"])
        farms = [farm_map[eid] for eid in event_ids]
        stgs: list[str] = []
        parts: list[str] = []
        flat_groups: list[str] = []
        flat_roles: list[str] = []
        prev_ts = None
        start_ts = ts_map[event_ids[0]]
        for eid in event_ids:
            ts = ts_map[eid]
            prefix: list[str] = []
            if prev_ts is not None:
                hours = (ts - prev_ts).total_seconds() / 3600.0
                prefix.append(delta_t_bucket(hours))
            days = (ts - start_ts).total_seconds() / 86400.0
            prefix.append(day_from_start_bucket(days))
            prefix.append(f"LOCAL_HOUR|H{ts.tz_convert('Asia/Seoul').hour:02d}")
            prefix.append("[GROUP_SEP]")
            prev_ts = ts
            body = str(sent_map[eid]).split()
            gids = json.loads(mg_map[eid])
            roles = json.loads(role_map[eid])
            parts.extend(prefix + body + ["[SEQ_SEP]"])
            flat_groups.extend(["NONE"] * len(prefix) + gids + ["NONE"])
            flat_roles.extend(
                ["temporal"] * (len(prefix) - 1) + ["sep"] + roles + ["sep"]
            )
            stgs.append(stg_map[eid])
        farm_local, _ = local_spatial_maps(farms, [])
        spatial = list(dict.fromkeys(farm_local[f] for f in farms))
        full_tokens = spatial + parts
        serialization = " ".join(full_tokens)
        ordered_sig = hashlib.sha256("|".join(event_ids).encode()).hexdigest()
        event_set_sig = hashlib.sha256("|".join(sorted(event_ids)).encode()).hexdigest()
        sig = hashlib.sha256(serialization.encode()).hexdigest()
        window_sig = hashlib.sha256(
            f"{min(farms)}|{start_ts.date()}|{ts_map[event_ids[-1]].date()}".encode()
        ).hexdigest()

        exact_key = (sig, narrative_id)
        is_exact_dup = exact_key in seen_exact
        if is_exact_dup:
            class_counts["TRUE_EXACT_DUPLICATE"] += 1
            continue
        # First occurrence of exact key — may still be SAME_CONTEXT later; classify after pass
        seen_exact.add(exact_key)
        event_set_counts[event_set_sig] += 1
        stored_tokens = full_tokens[:max_store_tokens]
        aligned_groups = (["NONE"] * len(spatial) + flat_groups)[:max_store_tokens]
        aligned_roles = (["meta"] * len(spatial) + flat_roles)[:max_store_tokens]
        chunk.append(
            {
                "sequence_id": sequence_id,
                "narrative_id": narrative_id,
                "PERSON_ID": int(hashlib.sha256(str(sequence_id).encode()).hexdigest()[:16], 16)
                & ((1 << 63) - 1),
                "SENTENCE": " ".join(stored_tokens),
                "event_ids": json.dumps(event_ids),
                "same_time_group_ids": json.dumps(list(dict.fromkeys(stgs))),
                "measurement_group_ids": json.dumps(aligned_groups),
                "token_roles": json.dumps(aligned_roles),
                "farm_ids": json.dumps(sorted(set(map(str, farms)))),
                "ordered_event_signature": ordered_sig,
                "event_set_signature": event_set_sig,
                "serialization_signature": sig,
                "source_window_signature": window_sig,
                "canonical_context_id": event_set_sig,
                "split_group_id": window_sig,
                "contains_image": bool(meta["contains_image"]),
                "tokenization_version": "v2",
                "training_mode": "transductive_public_pretraining",
                "token_count_full": len(full_tokens),
                "token_count_stored": len(stored_tokens),
                "duplicate_class": "PENDING",
            }
        )
        kept += 1
        if len(chunk) >= chunk_size:
            flush_chunk()

    flush_chunk()
    if writer is not None:
        writer.close()

    # Assign duplicate classes + sampling weights without loading SENTENCE columns fully.
    if not final_path.exists():
        write_json(
            out / "duplicate_report_v2.json",
            {"before": before, "after": 0, "removed_true_exact": before, "class_counts_pre": class_counts},
        )
        return {"before": before, "after": 0}

    light_cols = [
        "sequence_id",
        "narrative_id",
        "serialization_signature",
        "event_set_signature",
        "canonical_context_id",
    ]
    index = pd.read_parquet(final_path, columns=light_cols)
    context_freq = index.groupby("event_set_signature").size()
    view_counts = index.groupby("canonical_context_id").size()
    dup_class = np.where(
        index["event_set_signature"].map(context_freq) > 1,
        "SAME_CONTEXT_DIFFERENT_NARRATIVE",
        "UNIQUE",
    )
    context_view_count = index["canonical_context_id"].map(view_counts).astype(int)
    sampling_weight = 1.0 / np.sqrt(context_view_count.clip(lower=1).astype(float))
    annotate = pd.DataFrame(
        {
            "sequence_id": index["sequence_id"].to_numpy(),
            "duplicate_class": dup_class,
            "context_view_count": context_view_count.to_numpy(),
            "sampling_weight": sampling_weight.to_numpy(),
        }
    ).set_index("sequence_id")

    pre_index = index[
        ["sequence_id", "narrative_id", "serialization_signature", "event_set_signature"]
    ].copy()
    pre_index["duplicate_class"] = dup_class
    pre_index.to_parquet(out / "sequences_v2_pre_dedup.parquet", index=False)

    # Row-group rewrite to attach annotation columns (keeps peak RAM ~1 group).
    pf = pq.ParquetFile(final_path)
    tmp_path = out / "sequences_v2.annotated.tmp.parquet"
    ann_writer: Optional[pq.ParquetWriter] = None
    for rg in range(pf.metadata.num_row_groups):
        frame = pf.read_row_group(rg).to_pandas()
        drop_cols = [c for c in ("duplicate_class", "context_view_count", "sampling_weight") if c in frame.columns]
        if drop_cols:
            frame = frame.drop(columns=drop_cols)
        joined = frame.join(annotate, on="sequence_id", how="left")
        table = pa.Table.from_pandas(joined, preserve_index=False)
        if ann_writer is None:
            ann_writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
        ann_writer.write_table(table)
        del frame, joined, table
    if ann_writer is not None:
        ann_writer.close()
        tmp_path.replace(final_path)

    class_counts["SAME_CONTEXT_DIFFERENT_NARRATIVE"] = int((dup_class == "SAME_CONTEXT_DIFFERENT_NARRATIVE").sum())
    class_counts["UNIQUE"] = int((dup_class == "UNIQUE").sum())
    after_n = int(len(index))
    del index, annotate, context_freq, view_counts
    write_json(
        out / "duplicate_report_v2.json",
        {
            "before": before,
            "after": after_n,
            "removed_true_exact": int(before - after_n),
            "class_counts_pre": class_counts,
            "note": "pre_dedup parquet stores kept index only; exact-dup rows discarded online; annotate via row-groups",
        },
    )
    return {"before": before, "after": after_n}


def shadow_split(out: Path) -> dict[str, Any]:
    """Assign shadow_split using light columns + row-group rewrite (memory safe)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    final_path = out / "sequences_v2.parquet"
    light = pd.read_parquet(
        final_path,
        columns=["sequence_id", "split_group_id", "event_ids"],
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

    light = light.copy()
    light["shadow_split"] = light["split_group_id"].map(bucket)
    exploded = light[["shadow_split", "event_ids", "sequence_id", "split_group_id"]].copy()
    exploded["event_id"] = exploded["event_ids"].map(json.loads)
    exploded = exploded.explode("event_id")
    overlap = (
        exploded.groupby("event_id")["shadow_split"].nunique().reset_index(name="n_splits")
    )
    bad = overlap[overlap["n_splits"] > 1]
    if len(bad):
        bad_events = set(bad["event_id"])
        touched = exploded[exploded["event_id"].isin(bad_events)]["sequence_id"].unique()
        groups = light.loc[light["sequence_id"].isin(touched), "split_group_id"].unique()
        light.loc[light["split_group_id"].isin(groups), "shadow_split"] = "train"
        exploded = light[["shadow_split", "event_ids"]].copy()
        exploded["event_id"] = exploded["event_ids"].map(json.loads)
        exploded = exploded.explode("event_id")
        overlap = exploded.groupby("event_id")["shadow_split"].nunique().reset_index(name="n_splits")
        bad = overlap[overlap["n_splits"] > 1]

    split_map = light.set_index("sequence_id")["shadow_split"]
    pf = pq.ParquetFile(final_path)
    tmp_path = out / "sequences_v2.shadow.tmp.parquet"
    writer: Optional[pq.ParquetWriter] = None
    for rg in range(pf.metadata.num_row_groups):
        frame = pf.read_row_group(rg).to_pandas()
        frame["shadow_split"] = frame["sequence_id"].map(split_map)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
        writer.write_table(table)
        del frame, table
    if writer is not None:
        writer.close()
        tmp_path.replace(final_path)

    report = {
        "event_overlap_after_repair": int(len(bad)),
        "split_counts": light["shadow_split"].value_counts().to_dict(),
        "sequence_count": int(len(light)),
    }
    write_json(out / "shadow_split_report_v2.json", report)
    overlap.to_csv(out / "shadow_split_event_overlap_v2.csv", index=False)
    return report


def export_training(out: Path, vocab: VocabV2) -> dict[str, Any]:
    """Export one training row per sequence, streaming by parquet row-group."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    seq_path = out / "sequences_v2.parquet"
    export_path = out / "training_events_v2.parquet"
    tmp_path = out / "training_events_v2.tmp.parquet"
    pf = pq.ParquetFile(seq_path)
    writer: Optional[pq.ParquetWriter] = None
    n_rows = 0
    for rg in range(pf.metadata.num_row_groups):
        seq = pf.read_row_group(rg).to_pandas()
        seq_rows = []
        for rec in seq.itertuples(index=False):
            event_ids = json.loads(rec.event_ids)
            stgs = json.loads(rec.same_time_group_ids) if rec.same_time_group_ids else []
            farm_ids = json.loads(rec.farm_ids) if rec.farm_ids else []
            seq_rows.append(
                {
                    "PERSON_ID": rec.PERSON_ID,
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
                    "farm_id": farm_ids[0] if farm_ids else "",
                    "zone_id": "",
                    "BACKGROUND_TOKENS": "[]",
                    "build_id": "v2_transductive",
                    "registry_version": "v2",
                    "embedding_status": "pending",
                    "sampling_weight": getattr(rec, "sampling_weight", 1.0),
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
        chunk = pd.DataFrame(seq_rows)
        n_rows += len(chunk)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
        writer.write_table(table)
        del seq, seq_rows, chunk, table
    if writer is not None:
        writer.close()
        tmp_path.replace(export_path)
    else:
        pd.DataFrame([]).to_parquet(export_path, index=False)
    # life2vec compatible registry
    tokens = [
        {
            "token_id": i,
            "token": vocab.id_to_token[i],
            "category": "GENERAL"
            if vocab.id_to_token[i] in {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}
            else vocab.id_to_token[i].split("|", 1)[0],
            "registry_version": "v2",
        }
        for i in range(vocab.size())
    ]
    # Map new specials into categories; ensure PAD=0
    write_json(out / "life2vec_token_registry_v2.json", {"tokens": tokens, "registry_version": "v2", "schema_version": "online2-v2"})
    return {"rows": int(n_rows), "path": str(export_path), "vocab_size": vocab.size()}


def validate_grouped_masking(out: Path) -> dict[str, Any]:
    vocab = VocabV2.load(out / "vocab_v2.json")
    masker = GroupedMLMMasker(vocab, mask_ratio=1.0, seed=2023)
    abs_tok = next(t for t in vocab.token_to_id if t.startswith("VALUE_ABS|ABS_B"))
    glob_tok = next(t for t in vocab.token_to_id if t.startswith("VALUE_GLOBAL_REL|GLOBAL_REL_B"))
    farm_tok = next(t for t in vocab.token_to_id if t.startswith("VALUE_FARM_REL|FARM_REL_B"))
    tokens = ["FEATURE|INSIDE_TEMP_C", abs_tok, glob_tok, farm_tok, "QUALITY|OK"]
    for tok in tokens:
        if tok not in vocab.token_to_id:
            raise AssertionError(f"missing token {tok}")
    roles = ["feature_identity", "value_abs", "value_global", "value_farm", "quality"]
    ids = [vocab.get(t) for t in tokens]
    groups = ["mg1"] * len(tokens)
    masked, pos, tgt, report = masker.mask(ids, groups, roles)
    assert masked[0] == ids[0]
    assert report.masked_group_count == 1
    assert set(pos.tolist()) == {1, 2, 3, 4}
    narr = [i for t, i in vocab.token_to_id.items() if t.startswith("NARRATIVE|")]
    assert not any(int(x) in narr for x in masked)
    payload = {
        "report": report.__dict__,
        "feature_identity_preserved": bool(masked[0] == ids[0]),
        "masked_positions": pos.tolist(),
        "pass": True,
    }
    write_json(out / "grouped_masking_validation.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-events", type=int, default=0, help="0=all")
    parser.add_argument("--skip-sequences", action="store_true")
    parser.add_argument("--skip-registries", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.skip_registries and (out / "registry_build_report.json").exists():
        reg = json.loads((out / "registry_build_report.json").read_text())
        print("== registries (cached) ==")
        print(reg)
    else:
        print("== registries ==")
        reg = build_registries(out, args.legacy_build, args.audit)
        write_json(out / "registry_build_report.json", reg)
        print(reg)

    print("== grouped masking validation ==")
    print(validate_grouped_masking(out))

    max_events = args.max_events or None
    if args.skip_tokenize and (out / "events_tokenized_v2.parquet").exists():
        print("== tokenize events (cached) ==")
        events = pd.read_parquet(out / "events_tokenized_v2.parquet")
        print({"events": len(events)})
    else:
        print("== tokenize events ==")
        events = tokenize_events(out, args.legacy_build, max_events=max_events)
        print({"events": len(events)})

    if not args.skip_sequences:
        print("== materialize sequences ==")
        # For full run this is heavy; if max_events set, filter segments
        if max_events:
            # build mini sequences from tokenized events only
            mini = []
            for farm, grp in events.groupby("farm_id"):
                grp = grp.sort_values("observation_timestamp", kind="mergesort").head(24)
                if len(grp) < 2:
                    continue
                event_ids = grp["event_id"].tolist()
                serialization = " [SEQ_SEP] ".join(grp["SENTENCE"].tolist())
                sig = hashlib.sha256(serialization.encode()).hexdigest()
                mini.append(
                    {
                        "sequence_id": f"mini_{farm}_{sig[:12]}",
                        "narrative_id": "MINI_SMOKE",
                        "PERSON_ID": int(sig[:16], 16) & ((1 << 63) - 1),
                        "SENTENCE": serialization,
                        "event_ids": json.dumps(event_ids),
                        "same_time_group_ids": json.dumps(grp["same_time_group_id"].tolist()),
                        "measurement_group_ids": grp.iloc[0]["measurement_group_ids"],
                        "token_roles": grp.iloc[0]["token_roles"],
                        "farm_ids": json.dumps([farm]),
                        "ordered_event_signature": hashlib.sha256("|".join(event_ids).encode()).hexdigest(),
                        "event_set_signature": hashlib.sha256("|".join(sorted(event_ids)).encode()).hexdigest(),
                        "serialization_signature": sig,
                        "source_window_signature": hashlib.sha256(f"{farm}".encode()).hexdigest(),
                        "canonical_context_id": hashlib.sha256("|".join(sorted(event_ids)).encode()).hexdigest(),
                        "split_group_id": hashlib.sha256(f"{farm}".encode()).hexdigest(),
                        "contains_image": bool((grp["event_kind"] == "IMAGE").any()),
                        "tokenization_version": "v2",
                        "training_mode": "transductive_public_pretraining",
                        "duplicate_class": "UNIQUE",
                        "context_view_count": 1,
                        "sampling_weight": 1.0,
                    }
                )
            seq = pd.DataFrame(mini)
            seq.to_parquet(out / "sequences_v2_pre_dedup.parquet", index=False)
            seq.to_parquet(out / "sequences_v2.parquet", index=False)
            write_json(out / "duplicate_report_v2.json", {"before": len(seq), "after": len(seq), "mode": "smoke_mini"})
            split_report = shadow_split(out)
        else:
            dup = materialize_sequences(out, args.legacy_build, events)
            print(dup)
            split_report = shadow_split(out)
        print(split_report)
        vocab = VocabV2.load(out / "vocab_v2.json")
        export = export_training(out, vocab)
        print(export)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit(),
        "training_mode": "transductive_public_pretraining",
        "contains_problem_observations": True,
        "contains_problem_images": True,
        "contains_problem_hidden_targets": False,
        "registries": reg,
        "out": str(out),
    }
    write_json(out / "build_manifest_v2.json", manifest)
    print("DONE", manifest)


if __name__ == "__main__":
    main()
