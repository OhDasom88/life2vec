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

from src.online2.v2.binning import fit_binning_v2
from src.online2.v2.feature_schema import build_feature_schema_from_audit
from src.online2.v2.masking import GroupedMLMMasker
from src.online2.v2.provenance import training_mode_meta
from src.online2.v2.tokenizer import TokenizerV2, wind_compass
from src.online2.v2.vocab import VocabV2, build_vocab_v2, save_token_usage_policy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = ROOT / "outputs/online2/build-v8-active80-r3"
DEFAULT_AUDIT = ROOT / "outputs/online2/v2_audit/feature_semantics_and_units.csv"
DEFAULT_OUT = ROOT / "outputs/online2/v2_build"
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

    binning = fit_binning_v2(
        schema,
        rows,
        source_file_hashes=source_hashes,
        example_count=example_n,
        problem_count=problem_n,
        code_commit_hash=git_commit(),
    )
    binning.save(out / "binning_registry_v2_transductive.json")
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

    # Build event key index
    key_cols = ["farm", "zone", "ts", "view"]
    cell_groups = {key: grp for key, grp in cells.groupby(key_cols, sort=False)}

    rows = []
    for ev in events.itertuples(index=False):
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
            # deterministic column order
            for cell in grp.sort_values(["column_name", "cell_id"], kind="mergesort").itertuples(index=False):
                if cell.column_name in skip_cols:
                    continue
                raw = "" if bool(cell.is_null) else str(cell.raw_display)
                measured = tokenizer.tokenize_value(
                    str(cell.column_name),
                    raw,
                    farm_id=str(cell.farm_id),
                    cell_id=str(cell.cell_id),
                    atomic_value_id=str(getattr(cell, "atomic_value_id", "") or ""),
                )
                tokens.append("[MEAS_SEP]")
                group_ids.append(measured.measurement_group_id)
                roles.append("sep")
                for tok, role in zip(measured.tokens, measured.roles):
                    tokens.append(tok)
                    group_ids.append(measured.measurement_group_id)
                    roles.append(role)
            tokens = ["[EVENT_SEP]", f"VIEW|{str(ev.view).split('_')[-1].upper() if '_' in str(ev.view) else str(ev.view).upper()}", "EVENT_KIND|OBSERVATION"] + tokens
            group_ids = ["NONE", "NONE", "NONE"] + group_ids
            roles = ["sep", "meta", "meta"] + roles
        else:
            tokens = ["[EVENT_SEP]", "EVENT_KIND|OBSERVATION", "[MISSING]"]
            group_ids = ["NONE"] * 3
            roles = ["sep", "meta", "quality"]

        # map unknown tokens to UNK for serialization ids, but keep strings for audit
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
    segments = pd.read_parquet(legacy_build / "sequence_segments.parquet")
    sequences = pd.read_parquet(
        legacy_build / "sequences.parquet",
        columns=[
            "sequence_id",
            "narrative_id",
            "same_time_group_ids",
            "contains_image",
            "contains_interpretation",
            "farm_count",
        ],
    )
    events_v2 = events_v2.set_index("event_id", drop=False)
    segments = segments.merge(
        sequences[["sequence_id", "narrative_id", "contains_image", "contains_interpretation"]],
        on="sequence_id",
        how="left",
    )
    segments = segments.sort_values(["sequence_id", "position"], kind="mergesort")

    # Precompute event sentence lookup
    sent_map = events_v2["SENTENCE"].to_dict()
    farm_map = events_v2["farm_id"].to_dict()
    stg_map = events_v2["same_time_group_id"].to_dict()
    role_map = events_v2["token_roles"].to_dict()
    mg_map = events_v2["measurement_group_ids"].to_dict()
    ts_map = {
        eid: ts
        for eid, ts in zip(events_v2["event_id"], pd.to_datetime(events_v2["observation_timestamp"], utc=True))
    }

    records = []
    for sequence_id, group in segments.groupby("sequence_id", sort=False):
        event_ids = group["event_id"].tolist()
        if any(eid not in sent_map for eid in event_ids):
            continue
        farms = [farm_map[eid] for eid in event_ids]
        stgs = []
        parts = []
        flat_groups = []
        flat_roles = []
        prev_ts = None
        start_ts = ts_map[event_ids[0]]
        for eid in event_ids:
            ts = ts_map[eid]
            prefix = []
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
            flat_roles.extend(["temporal"] * (len(prefix) - 1) + ["sep"] + roles + ["sep"])
            stgs.append(stg_map[eid])
        farm_local, zone_local = local_spatial_maps(farms, [])
        spatial = list(dict.fromkeys(farm_local[f] for f in farms))
        serialization = " ".join(spatial + parts)
        ordered_sig = hashlib.sha256("|".join(event_ids).encode()).hexdigest()
        event_set_sig = hashlib.sha256("|".join(sorted(event_ids)).encode()).hexdigest()
        sig = hashlib.sha256(serialization.encode()).hexdigest()
        window_sig = hashlib.sha256(
            f"{min(farms)}|{start_ts.date()}|{ts_map[event_ids[-1]].date()}".encode()
        ).hexdigest()
        records.append(
            {
                "sequence_id": sequence_id,
                "narrative_id": group["narrative_id"].iloc[0],
                "PERSON_ID": int(hashlib.sha256(str(sequence_id).encode()).hexdigest()[:16], 16)
                & ((1 << 63) - 1),
                "SENTENCE": serialization,
                "event_ids": json.dumps(event_ids),
                "same_time_group_ids": json.dumps(list(dict.fromkeys(stgs))),
                "measurement_group_ids": json.dumps(flat_groups),
                "token_roles": json.dumps(flat_roles),
                "farm_ids": json.dumps(sorted(set(map(str, farms)))),
                "ordered_event_signature": ordered_sig,
                "event_set_signature": event_set_sig,
                "serialization_signature": sig,
                "source_window_signature": window_sig,
                "canonical_context_id": event_set_sig,
                "split_group_id": window_sig,
                "contains_image": bool(group["contains_image"].iloc[0]),
                "tokenization_version": "v2",
                "training_mode": "transductive_public_pretraining",
            }
        )

    seq_df = pd.DataFrame(records)
    before = len(seq_df)
    seq_df["duplicate_class"] = np.where(
        seq_df.duplicated(["serialization_signature", "narrative_id"], keep=False),
        "TRUE_EXACT_DUPLICATE",
        np.where(
            seq_df.duplicated(["event_set_signature"], keep=False),
            "SAME_CONTEXT_DIFFERENT_NARRATIVE",
            "UNIQUE",
        ),
    )
    dedup = seq_df.sort_values(["sequence_id"]).drop_duplicates(
        ["serialization_signature", "narrative_id"], keep="first"
    )
    view_counts = dedup.groupby("canonical_context_id").size().rename("context_view_count")
    dedup = dedup.join(view_counts, on="canonical_context_id")
    dedup["sampling_weight"] = 1.0 / np.sqrt(dedup["context_view_count"].clip(lower=1))
    dedup.to_parquet(out / "sequences_v2.parquet", index=False)
    seq_df.to_parquet(out / "sequences_v2_pre_dedup.parquet", index=False)
    write_json(
        out / "duplicate_report_v2.json",
        {
            "before": before,
            "after": int(len(dedup)),
            "removed_true_exact": int(before - len(dedup)),
            "class_counts_pre": seq_df["duplicate_class"].value_counts().to_dict(),
        },
    )
    return {"before": before, "after": int(len(dedup))}


def shadow_split(out: Path) -> dict[str, Any]:
    seq = pd.read_parquet(out / "sequences_v2.parquet")
    # Connected components for multi-farm via shared events already collapsed into split_group_id=window
    # Use split_group_id hash for farm-held-out style buckets without cross overlap.
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

    seq = seq.copy()
    seq["shadow_split"] = seq["split_group_id"].map(bucket)
    # Ensure no event_id overlap across splits
    exploded = seq[["shadow_split", "event_ids", "sequence_id"]].copy()
    exploded["event_id"] = exploded["event_ids"].map(json.loads)
    exploded = exploded.explode("event_id")
    overlap = (
        exploded.groupby("event_id")["shadow_split"].nunique().reset_index(name="n_splits")
    )
    bad = overlap[overlap["n_splits"] > 1]
    # Repair: assign all sequences sharing bad events to majority split of their split_group
    if len(bad):
        # force by split_group_id already; if still bad, move entire split_group to train
        bad_events = set(bad["event_id"])
        touched = exploded[exploded["event_id"].isin(bad_events)]["sequence_id"].unique()
        groups = seq.loc[seq["sequence_id"].isin(touched), "split_group_id"].unique()
        seq.loc[seq["split_group_id"].isin(groups), "shadow_split"] = "train"
        exploded = seq[["shadow_split", "event_ids"]].copy()
        exploded["event_id"] = exploded["event_ids"].map(json.loads)
        exploded = exploded.explode("event_id")
        overlap = exploded.groupby("event_id")["shadow_split"].nunique().reset_index(name="n_splits")
        bad = overlap[overlap["n_splits"] > 1]

    seq.to_parquet(out / "sequences_v2.parquet", index=False)
    report = {
        "event_overlap_after_repair": int(len(bad)),
        "split_counts": seq["shadow_split"].value_counts().to_dict(),
        "sequence_count": int(len(seq)),
    }
    write_json(out / "shadow_split_report_v2.json", report)
    overlap.to_csv(out / "shadow_split_event_overlap_v2.csv", index=False)
    return report


def export_training(out: Path, vocab: VocabV2) -> dict[str, Any]:
    seq = pd.read_parquet(out / "sequences_v2.parquet")
    # Flatten to event-level rows for Online2ParquetTokenSource compatibility: one row per event occurrence
    # Simpler path: store sequence-level documents for a dedicated V2 source.
    rows = []
    for rec in seq.itertuples(index=False):
        event_ids = json.loads(rec.event_ids)
        stgs = json.loads(rec.same_time_group_ids)
        for pos, event_id in enumerate(event_ids):
            rows.append(
                {
                    "PERSON_ID": rec.PERSON_ID,
                    "sequence_id": rec.sequence_id,
                    "event_id": event_id,
                    "event_position": pos,
                    "time_group_rank": min(pos, max(0, len(stgs) - 1)),
                    "same_time_group_id": stgs[min(pos, len(stgs) - 1)] if stgs else "",
                    "START_DATE": pd.Timestamp("2024-01-01"),
                    "AGE": float(pos),
                    "SENTENCE": rec.SENTENCE if pos == 0 else "[EVENT_SEP]",
                    "event_kind": "SEQUENCE" if pos == 0 else "EVENT",
                    "narrative_id": rec.narrative_id,
                    "order_semantics": "STRICT_CHRONOLOGICAL",
                    "op_eligible": True,
                    "modality_ref": "[]",
                    "SEGMENT": (pos % 3) + 1,
                    "farm_id": json.loads(rec.farm_ids)[0] if rec.farm_ids else "",
                    "zone_id": "",
                    "BACKGROUND_TOKENS": "[]",
                    "build_id": "v2_transductive",
                    "registry_version": "v2",
                    "embedding_status": "",
                    "sampling_weight": getattr(rec, "sampling_weight", 1.0),
                    "shadow_split": getattr(rec, "shadow_split", "train"),
                    "canonical_context_id": rec.canonical_context_id,
                    "split_group_id": rec.split_group_id,
                    "measurement_group_ids": rec.measurement_group_ids if pos == 0 else "[]",
                    "token_roles": rec.token_roles if pos == 0 else "[]",
                    "training_mode": "transductive_public_pretraining",
                    "contains_problem_observations": True,
                    "contains_problem_images": True,
                    "contains_problem_hidden_targets": False,
                }
            )
    # More efficient: one row per sequence for V2 MLM path
    seq_rows = []
    for rec in seq.itertuples(index=False):
        seq_rows.append(
            {
                "PERSON_ID": rec.PERSON_ID,
                "sequence_id": rec.sequence_id,
                "event_id": json.loads(rec.event_ids)[0],
                "event_position": 0,
                "time_group_rank": 0,
                "same_time_group_id": json.loads(rec.same_time_group_ids)[0]
                if json.loads(rec.same_time_group_ids)
                else "",
                "START_DATE": pd.Timestamp("2024-01-01"),
                "AGE": 0.0,
                "SENTENCE": rec.SENTENCE,
                "event_kind": "SEQUENCE",
                "narrative_id": rec.narrative_id,
                "order_semantics": "STRICT_CHRONOLOGICAL",
                "op_eligible": True,
                "modality_ref": "[]",
                "SEGMENT": 1,
                "farm_id": json.loads(rec.farm_ids)[0] if rec.farm_ids else "",
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
    export = pd.DataFrame(seq_rows)
    export_path = out / "training_events_v2.parquet"
    export.to_parquet(export_path, index=False)
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
    return {"rows": int(len(export)), "path": str(export_path), "vocab_size": vocab.size()}


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
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("== registries ==")
    reg = build_registries(out, args.legacy_build, args.audit)
    write_json(out / "registry_build_report.json", reg)
    print(reg)

    print("== grouped masking validation ==")
    print(validate_grouped_masking(out))

    print("== tokenize events ==")
    max_events = args.max_events or None
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
