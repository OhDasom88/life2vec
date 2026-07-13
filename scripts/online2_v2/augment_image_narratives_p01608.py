#!/usr/bin/env python3
"""Augment V2 sequences with P01/P06/P08 image-context narratives.

P01: lookback sensors → IMAGE (IMAGE_ROLE|UNRESOLVED)
P08: same lookback → IMAGE (IMAGE_ROLE|QUERY)
P06: same lookback only (no IMAGE) — photo-preceding context / contrast window

Does not rebuild the full 945k corpus; emits new rows then concatenates into
sequences_v2.parquet and re-runs annotate / shadow_split / training export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.online2_v2.build_v2 import (
    day_from_start_bucket,
    delta_t_bucket,
    local_spatial_maps,
)
from scripts.online2_v2.finish_v2_postprocess import (
    annotate_duplicate_classes,
    export_training_chunked,
    rewrite_sequences_with_annotations,
    shadow_split_light,
    write_json,
)
from src.online2.v2.vocab import VocabV2


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/online2/v2_build"

LOOKBACKS = (8, 16, 24, 32, 48)
VIEW_MIXES: tuple[tuple[str, ...], ...] = (
    ("E",),
    ("R",),
    ("A",),
    ("E", "R"),
    ("E", "R", "A"),
)
MIN_PAST = 4
MAX_STORE_TOKENS = 2048


def classify_view(sentence: str) -> str:
    if "VIEW|IMAGE" in sentence or "EVENT_KIND|IMAGE" in sentence:
        return "I"
    if "VIEW|INTERPRETATION" in sentence or "EVENT_KIND|INTERPRETATION" in sentence:
        return "T"
    if "VIEW|ENVIRONMENT" in sentence:
        return "E"
    if "VIEW|ROOTZONE" in sentence:
        return "R"
    if "VIEW|ACTUATOR" in sentence:
        return "A"
    if "VIEW|GROWTH" in sentence or "EVENT_KIND|GROWTH" in sentence:
        return "G"
    return "U"


def stable_sequence_id(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "sequence_" + hashlib.sha256(blob.encode()).hexdigest()


def image_sentence(role: str) -> str:
    return f"[IMAGE_SLOT] IMAGE_ROLE|{role} VIEW|IMAGE EVENT_KIND|IMAGE"


def image_roles(role: str) -> list[str]:
    return ["slot", "meta", "meta", "meta"]


def assemble_row(
    *,
    narrative_id: str,
    event_ids: list[str],
    farms: list[str],
    sentences: list[str],
    stgs: list[str],
    mg_jsons: list[str],
    role_jsons: list[str],
    timestamps: list[pd.Timestamp],
    contains_image: bool,
) -> dict[str, Any]:
    farm_local, _ = local_spatial_maps(farms, [])
    spatial = list(dict.fromkeys(farm_local[f] for f in farms))
    parts: list[str] = []
    flat_groups: list[str] = []
    flat_roles: list[str] = []
    prev_ts: Optional[pd.Timestamp] = None
    start_ts = timestamps[0]
    for sent, mg_raw, role_raw, ts in zip(sentences, mg_jsons, role_jsons, timestamps):
        prefix: list[str] = []
        if prev_ts is not None:
            hours = (ts - prev_ts).total_seconds() / 3600.0
            prefix.append(delta_t_bucket(hours))
        days = (ts - start_ts).total_seconds() / 86400.0
        prefix.append(day_from_start_bucket(days))
        prefix.append(f"LOCAL_HOUR|H{ts.tz_convert('Asia/Seoul').hour:02d}")
        prefix.append("[GROUP_SEP]")
        prev_ts = ts
        body = sent.split()
        gids = json.loads(mg_raw)
        roles = json.loads(role_raw)
        if len(gids) != len(body):
            gids = (gids + ["NONE"] * len(body))[: len(body)]
        if len(roles) != len(body):
            roles = (roles + ["meta"] * len(body))[: len(body)]
        parts.extend(prefix + body + ["[SEQ_SEP]"])
        flat_groups.extend(["NONE"] * len(prefix) + gids + ["NONE"])
        flat_roles.extend(["temporal"] * (len(prefix) - 1) + ["sep"] + roles + ["sep"])

    full_tokens = spatial + parts
    serialization = " ".join(full_tokens)
    ordered_sig = hashlib.sha256("|".join(event_ids).encode()).hexdigest()
    event_set_sig = hashlib.sha256("|".join(sorted(event_ids)).encode()).hexdigest()
    sig = hashlib.sha256(serialization.encode()).hexdigest()
    window_sig = hashlib.sha256(
        f"{min(farms)}|{start_ts.date()}|{timestamps[-1].date()}|{narrative_id}".encode()
    ).hexdigest()
    aligned_groups_full = ["NONE"] * len(spatial) + flat_groups
    aligned_roles_full = ["meta"] * len(spatial) + flat_roles
    # Keep IMAGE terminal: truncate oldest tokens from the left when over budget.
    if contains_image and len(full_tokens) > MAX_STORE_TOKENS:
        stored = full_tokens[-MAX_STORE_TOKENS:]
        aligned_groups = aligned_groups_full[-MAX_STORE_TOKENS:]
        aligned_roles = aligned_roles_full[-MAX_STORE_TOKENS:]
    else:
        stored = full_tokens[:MAX_STORE_TOKENS]
        aligned_groups = aligned_groups_full[:MAX_STORE_TOKENS]
        aligned_roles = aligned_roles_full[:MAX_STORE_TOKENS]
    sequence_id = stable_sequence_id(
        {
            "narrative_id": narrative_id,
            "event_ids": event_ids,
            "role": "QUERY" if "P08" in narrative_id else "UNRESOLVED",
            "contains_image": contains_image,
        }
    )
    return {
        "sequence_id": sequence_id,
        "narrative_id": narrative_id,
        "PERSON_ID": int(hashlib.sha256(sequence_id.encode()).hexdigest()[:16], 16)
        & ((1 << 63) - 1),
        "SENTENCE": " ".join(stored),
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
        "contains_image": bool(contains_image),
        "tokenization_version": "v2",
        "training_mode": "transductive_public_pretraining",
        "token_count_full": len(full_tokens),
        "token_count_stored": len(stored),
        "duplicate_class": "PENDING",
        "context_view_count": 1,
        "sampling_weight": 1.0,
        "shadow_split": "train",
    }


def load_events(path: Path) -> pd.DataFrame:
    cols = [
        "event_id",
        "event_kind",
        "observation_timestamp",
        "farm_id",
        "zone_id",
        "SENTENCE",
        "same_time_group_id",
        "token_roles",
        "measurement_group_ids",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["ts"] = pd.to_datetime(df["observation_timestamp"], utc=True)
    df["view"] = df["SENTENCE"].map(classify_view)
    return df.sort_values(["farm_id", "ts", "event_id"], kind="mergesort").reset_index(drop=True)


def past_candidates(farm_df: pd.DataFrame, image_ts: pd.Timestamp, views: Iterable[str]) -> pd.DataFrame:
    view_set = set(views)
    mask = (farm_df["ts"] <= image_ts) & (farm_df["view"].isin(view_set))
    return farm_df.loc[mask]


def generate_augment_rows(events: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = events[events["event_kind"] == "IMAGE"].copy()
    by_farm = {farm: g for farm, g in events.groupby("farm_id", sort=False)}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    stats = defaultdict(int)

    img_mg = json.dumps(["NONE"] * 4)
    img_roles_unresolved = json.dumps(image_roles("UNRESOLVED"))
    img_roles_query = json.dumps(image_roles("QUERY"))

    for img in images.itertuples(index=False):
        farm_df = by_farm[str(img.farm_id)]
        for mix in VIEW_MIXES:
            mix_key = "".join(mix)
            past_all = past_candidates(farm_df, img.ts, mix)
            if len(past_all) < MIN_PAST:
                stats["skip_short_farm_mix"] += 1
                continue
            for n_lb in LOOKBACKS:
                past = past_all.tail(n_lb)
                if len(past) < MIN_PAST:
                    stats["skip_short_lookback"] += 1
                    continue
                past_ids = past["event_id"].tolist()
                past_sents = past["SENTENCE"].astype(str).tolist()
                past_stgs = past["same_time_group_id"].astype(str).tolist()
                past_mgs = past["measurement_group_ids"].astype(str).tolist()
                past_roles = past["token_roles"].astype(str).tolist()
                past_ts = list(past["ts"])
                past_farms = past["farm_id"].astype(str).tolist()

                # P06: lookback only (contrast / preceding context)
                row_p06 = assemble_row(
                    narrative_id="P06",
                    event_ids=past_ids,
                    farms=past_farms,
                    sentences=past_sents,
                    stgs=past_stgs,
                    mg_jsons=past_mgs,
                    role_jsons=past_roles,
                    timestamps=past_ts,
                    contains_image=False,
                )
                key06 = (row_p06["serialization_signature"], "P06")
                if key06 not in seen:
                    seen.add(key06)
                    rows.append(row_p06)
                    stats["P06"] += 1
                else:
                    stats["dup_P06"] += 1

                # P01: lookback + IMAGE UNRESOLVED
                row_p01 = assemble_row(
                    narrative_id="P01",
                    event_ids=past_ids + [str(img.event_id)],
                    farms=past_farms + [str(img.farm_id)],
                    sentences=past_sents + [image_sentence("UNRESOLVED")],
                    stgs=past_stgs + [str(img.same_time_group_id)],
                    mg_jsons=past_mgs + [img_mg],
                    role_jsons=past_roles + [img_roles_unresolved],
                    timestamps=past_ts + [img.ts],
                    contains_image=True,
                )
                key01 = (row_p01["serialization_signature"], "P01")
                if key01 not in seen:
                    seen.add(key01)
                    rows.append(row_p01)
                    stats["P01"] += 1
                else:
                    stats["dup_P01"] += 1

                # P08: lookback + IMAGE QUERY
                row_p08 = assemble_row(
                    narrative_id="P08",
                    event_ids=past_ids + [str(img.event_id)],
                    farms=past_farms + [str(img.farm_id)],
                    sentences=past_sents + [image_sentence("QUERY")],
                    stgs=past_stgs + [str(img.same_time_group_id)],
                    mg_jsons=past_mgs + [img_mg],
                    role_jsons=past_roles + [img_roles_query],
                    timestamps=past_ts + [img.ts],
                    contains_image=True,
                )
                key08 = (row_p08["serialization_signature"], "P08")
                if key08 not in seen:
                    seen.add(key08)
                    rows.append(row_p08)
                    stats["P08"] += 1
                else:
                    stats["dup_P08"] += 1

                stats[f"mix_{mix_key}"] += 1
                stats[f"lb_{n_lb}"] += 1

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_images": int(len(images)),
        "lookbacks": list(LOOKBACKS),
        "view_mixes": [list(m) for m in VIEW_MIXES],
        "counts": dict(stats),
        "n_rows": len(rows),
        "n_with_image": sum(1 for r in rows if r["contains_image"]),
        "n_p06_no_image": sum(1 for r in rows if not r["contains_image"]),
        "n_image_slot_in_stored": sum(
            1 for r in rows if r["contains_image"] and "[IMAGE_SLOT]" in r["SENTENCE"]
        ),
        "n_left_truncated_image": sum(
            1
            for r in rows
            if r["contains_image"] and r["token_count_full"] > r["token_count_stored"]
        ),
    }
    return rows, report


def write_augment_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def concat_sequences(base_path: Path, augment_path: Path, out_path: Path) -> dict[str, Any]:
    """Stream-concat base + augment into out_path (zstd)."""
    tmp = out_path.with_suffix(".concat.tmp.parquet")
    if tmp.exists():
        tmp.unlink()
    writer: Optional[pq.ParquetWriter] = None
    n = 0
    existing_ids: set[str] = set()

    # Collect augment ids to skip collisions if re-run
    aug = pq.read_table(augment_path)
    aug_ids = set(aug.column("sequence_id").to_pylist())

    base_pf = pq.ParquetFile(base_path)
    for i in range(base_pf.metadata.num_row_groups):
        table = base_pf.read_row_group(i)
        # drop any previous P01/P06/P08 if re-running on already-augmented base
        narr = table.column("narrative_id").to_pylist()
        sids = table.column("sequence_id").to_pylist()
        keep = [
            j
            for j, (nid, sid) in enumerate(zip(narr, sids))
            if nid not in {"P01", "P06", "P08"} and sid not in aug_ids
        ]
        if len(keep) != table.num_rows:
            table = table.take(keep)
        if table.num_rows == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        # align schema to writer if needed
        if table.schema.equals(writer.schema):
            writer.write_table(table)
        else:
            frame = table.to_pandas()
            # ensure all writer columns exist
            for name in writer.schema.names:
                if name not in frame.columns:
                    frame[name] = None
            frame = frame[list(writer.schema.names)]
            writer.write_table(pa.Table.from_pandas(frame, preserve_index=False, schema=writer.schema))
        n += table.num_rows
        existing_ids.update(table.column("sequence_id").to_pylist())
        if i % 20 == 0:
            print(f"concat base rg {i}/{base_pf.metadata.num_row_groups} n={n}", flush=True)

    # append augment (skip ids already present)
    aug_df = aug.to_pandas()
    aug_df = aug_df[~aug_df["sequence_id"].isin(existing_ids)]
    if len(aug_df):
        # match base schema column order if writer exists
        if writer is None:
            table = pa.Table.from_pandas(aug_df, preserve_index=False)
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
        else:
            for name in writer.schema.names:
                if name not in aug_df.columns:
                    if name in {"context_view_count"}:
                        aug_df[name] = 1
                    elif name in {"sampling_weight"}:
                        aug_df[name] = 1.0
                    elif name in {"shadow_split"}:
                        aug_df[name] = "train"
                    elif name in {"duplicate_class"}:
                        aug_df[name] = "PENDING"
                    else:
                        aug_df[name] = None
            aug_df = aug_df[list(writer.schema.names)]
            writer.write_table(pa.Table.from_pandas(aug_df, preserve_index=False, schema=writer.schema))
        n += len(aug_df)
    if writer is not None:
        writer.close()
    tmp.replace(out_path)
    return {"n_sequences": n, "n_augment_added": int(len(aug_df)), "path": str(out_path)}


def postprocess(out: Path, seq_path: Path) -> dict[str, Any]:
    print("== annotate duplicates ==", flush=True)
    light = pd.read_parquet(
        seq_path,
        columns=[
            "sequence_id",
            "narrative_id",
            "serialization_signature",
            "event_set_signature",
            "canonical_context_id",
        ],
    )
    annotated, dup_report = annotate_duplicate_classes(light)
    write_json(out / "duplicate_report_v2.json", dup_report)
    annotated_path = out / "sequences_v2.annotated.tmp.parquet"
    rewrite_sequences_with_annotations(seq_path, annotated, annotated_path)
    annotated_path.replace(seq_path)

    print("== shadow split ==", flush=True)
    split_report = shadow_split_light(seq_path, annotated)

    print("== export training ==", flush=True)
    vocab = VocabV2.load(out / "vocab_v2.json")
    export_report = export_training_chunked(out, seq_path, vocab)
    return {"duplicate": dup_report, "shadow_split": split_report, "export": export_report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument("--skip-concat", action="store_true", help="Only write augment parquet")
    args = parser.parse_args()
    out = args.out
    events_path = out / "events_tokenized_v2.parquet"
    seq_path = out / "sequences_v2.parquet"
    assert events_path.exists(), events_path
    assert seq_path.exists(), seq_path

    print("load events", flush=True)
    events = load_events(events_path)
    print({"n_events": len(events), "n_images": int((events.event_kind == "IMAGE").sum())}, flush=True)

    rows, report = generate_augment_rows(events)
    aug_path = out / "sequences_image_augment_p01608.parquet"
    write_augment_parquet(rows, aug_path)
    write_json(out / "image_augment_p01608_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    # catalog note
    write_json(
        out / "image_augment_narratives.json",
        {
            "P01": {
                "name_ko": "촬영 전 센서 궤적 → 이미지 단면",
                "shape": "[lookback E/R/A] → IMAGE_ROLE|UNRESOLVED",
                "trigger": "image exists (period_end anchor)",
            },
            "P08": {
                "name_ko": "촬영 전 센서 궤적 → 이미지 QUERY 단면",
                "shape": "[lookback] → IMAGE_ROLE|QUERY",
                "trigger": "same as P01; role marks terminal phenotype query",
            },
            "P06": {
                "name_ko": "촬영 직전 창(사진 없음) 대조",
                "shape": "[lookback] only",
                "trigger": "same lookback as P01 without IMAGE terminal",
            },
        },
    )

    if args.skip_concat:
        return

    bak_ptr = out / "sequences_v2.before_p01608.parquet"
    if bak_ptr.exists():
        base_seq = bak_ptr
        print(f"using clean base {base_seq.name}", flush=True)
    else:
        base_seq = seq_path

    merged_tmp = out / "sequences_v2.with_image_augment.tmp.parquet"
    concat_report = concat_sequences(base_seq, aug_path, merged_tmp)
    # replace current sequences_v2
    if seq_path.exists():
        doomed = out / "sequences_v2.prev_augment.parquet"
        if doomed.exists():
            doomed.unlink()
        shutil.move(str(seq_path), str(doomed))
    if not bak_ptr.exists() and base_seq == seq_path:
        # first run already moved away; keep doomed as before if needed
        pass
    elif not bak_ptr.exists():
        shutil.copy2(base_seq, bak_ptr)
    shutil.move(str(merged_tmp), str(seq_path))
    # drop bulky prev if we have before_p01608
    prev = out / "sequences_v2.prev_augment.parquet"
    if prev.exists() and bak_ptr.exists():
        prev.unlink()
    write_json(out / "image_augment_concat_report.json", concat_report)
    print(json.dumps(concat_report, ensure_ascii=False, indent=2), flush=True)

    if args.skip_postprocess:
        return
    post = postprocess(out, seq_path)
    write_json(out / "image_augment_postprocess_report.json", post)
    print("done", flush=True)


if __name__ == "__main__":
    main()
