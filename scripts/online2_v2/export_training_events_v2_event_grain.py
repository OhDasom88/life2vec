#!/usr/bin/env python3
"""Re-export training_events_v2 at event grain (1 row = sequence×event occurrence).

Recovers from sequences_v2.event_ids ⨝ events_tokenized_v2 — does not synthesize
events or regex-split flat SENTENCE. Streams sequences by row-group.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _loads_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return list(value)
    return list(json.loads(str(value)))


def _modality_ref(event_kind: str, sentence: str) -> str:
    kind = str(event_kind or "")
    if kind == "IMAGE" or "[IMAGE_SLOT]" in sentence:
        return json.dumps(["IMAGE_EMBED_SLOT"])
    if kind == "INTERPRETATION" or "[TEXT_SLOT]" in sentence:
        return json.dumps(["TEXT_EMBED_SLOT"])
    return "[]"


def load_event_lookup(events_path: Path) -> dict[str, dict[str, Any]]:
    """Load ~222k events into a dict (small vs sequences)."""
    cols = [
        "event_id",
        "same_time_group_id",
        "event_kind",
        "observation_timestamp",
        "farm_id",
        "zone_id",
        "SENTENCE",
        "measurement_group_ids",
        "token_roles",
        "embedding_status",
    ]
    table = pq.read_table(events_path, columns=cols)
    df = table.to_pandas()
    lookup: dict[str, dict[str, Any]] = {}
    for rec in df.itertuples(index=False):
        lookup[str(rec.event_id)] = {
            "same_time_group_id": str(rec.same_time_group_id or ""),
            "event_kind": str(rec.event_kind or ""),
            "observation_timestamp": rec.observation_timestamp,
            "farm_id": str(rec.farm_id or ""),
            "zone_id": str(rec.zone_id or ""),
            "SENTENCE": str(rec.SENTENCE or ""),
            "measurement_group_ids": rec.measurement_group_ids
            if rec.measurement_group_ids is not None
            else "[]",
            "token_roles": rec.token_roles if rec.token_roles is not None else "[]",
            "embedding_status": str(rec.embedding_status or ""),
        }
    del df, table
    return lookup


def expand_sequence_rows(
    rec: Any,
    event_lookup: dict[str, dict[str, Any]],
    build_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Expand one sequence record into event-grain training rows."""
    stats = Counter()
    event_ids = [str(x) for x in _loads_list(rec.event_ids)]
    if not event_ids:
        stats["empty_event_ids"] += 1
        return [], stats

    farms = [str(x) for x in _loads_list(getattr(rec, "farm_ids", "[]"))]
    background = json.dumps([f"FARM|{f}" for f in farms] if farms else [])

    # Resolve per-event STG from events table (sequences store unique STGs only).
    resolved: list[dict[str, Any]] = []
    for eid in event_ids:
        ev = event_lookup.get(eid)
        if ev is None:
            stats["missing_event"] += 1
            continue
        if not ev["SENTENCE"].strip():
            stats["empty_sentence"] += 1
            continue
        resolved.append({"event_id": eid, **ev})
        stats[f"kind_{ev['event_kind']}"] += 1

    if not resolved:
        stats["all_events_unresolved"] += 1
        return [], stats

    # time_group_rank = order of first appearance of each same_time_group_id
    rank_map: dict[str, int] = {}
    for ev in resolved:
        gid = ev["same_time_group_id"] or f"solo_{ev['event_id']}"
        if gid not in rank_map:
            rank_map[gid] = len(rank_map)

    timestamps = []
    for ev in resolved:
        ts = pd.to_datetime(ev["observation_timestamp"], utc=True, errors="coerce")
        timestamps.append(ts)
    valid_ts = [t for t in timestamps if pd.notna(t)]
    first_ts = min(valid_ts) if valid_ts else pd.Timestamp("2024-01-01", tz="UTC")

    rows: list[dict[str, Any]] = []
    for pos, (ev, ts) in enumerate(zip(resolved, timestamps)):
        if pd.isna(ts):
            start = pd.Timestamp("2024-01-01")
            age = float(pos)
        else:
            start = ts.tz_localize(None) if ts.tzinfo is not None else ts
            age = float((ts - first_ts).total_seconds() / 3600.0)
        gid = ev["same_time_group_id"] or f"solo_{ev['event_id']}"
        rows.append(
            {
                "PERSON_ID": int(rec.PERSON_ID),
                "sequence_id": str(rec.sequence_id),
                "event_id": ev["event_id"],
                "event_position": int(pos),
                "time_group_rank": int(rank_map[gid]),
                "same_time_group_id": gid,
                "START_DATE": start,
                "AGE": age,
                "SENTENCE": ev["SENTENCE"],
                "event_kind": ev["event_kind"],
                "narrative_id": str(rec.narrative_id),
                "order_semantics": "STRICT_CHRONOLOGICAL",
                "op_eligible": True,
                "modality_ref": _modality_ref(ev["event_kind"], ev["SENTENCE"]),
                "SEGMENT": int(pos % 3 + 1),
                "farm_id": ev["farm_id"] or (farms[0] if farms else ""),
                "zone_id": ev["zone_id"],
                "BACKGROUND_TOKENS": background,
                "build_id": build_id,
                "registry_version": "v2",
                "embedding_status": ev["embedding_status"] or "pending",
                "sampling_weight": float(getattr(rec, "sampling_weight", 1.0) or 1.0),
                "shadow_split": str(getattr(rec, "shadow_split", "train") or "train"),
                "canonical_context_id": str(getattr(rec, "canonical_context_id", "") or ""),
                "split_group_id": str(getattr(rec, "split_group_id", "") or ""),
                "measurement_group_ids": ev["measurement_group_ids"]
                if isinstance(ev["measurement_group_ids"], str)
                else json.dumps(ev["measurement_group_ids"]),
                "token_roles": ev["token_roles"]
                if isinstance(ev["token_roles"], str)
                else json.dumps(ev["token_roles"]),
                "training_mode": str(
                    getattr(rec, "training_mode", "transductive_public_pretraining")
                ),
                "contains_problem_observations": True,
                "contains_problem_images": True,
                "contains_problem_hidden_targets": False,
            }
        )
    stats["rows_emitted"] += len(rows)
    stats["sequences_emitted"] += 1
    n_stg = len(rank_map)
    stats["sop_eligible_seq"] += int(n_stg >= 2)
    stats["sop_ineligible_seq"] += int(n_stg < 2)
    return rows, stats


def export_event_grain(
    build_dir: Path,
    *,
    smoke_sequences: int = 64,
    backup_flat: bool = True,
) -> dict[str, Any]:
    build_dir = build_dir.resolve()
    seq_path = build_dir / "sequences_v2.parquet"
    events_path = build_dir / "events_tokenized_v2.parquet"
    out_path = build_dir / "training_events_v2.parquet"
    tmp_path = build_dir / "training_events_v2.event_grain.tmp.parquet"
    smoke_path = build_dir / "training_events_v2_smoke.parquet"
    report_path = build_dir / "training_events_v2_export_report.json"

    if not seq_path.exists() or not events_path.exists():
        raise FileNotFoundError("sequences_v2 / events_tokenized_v2 required")

    if backup_flat and out_path.exists():
        backup = build_dir / "training_events_v2.flat_backup.parquet"
        if not backup.exists():
            print(f"backing up flat export → {backup}", flush=True)
            shutil.move(str(out_path), str(backup))
        else:
            alt = build_dir / f"training_events_v2.flat_backup.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.parquet"
            print(f"flat backup exists; moving current → {alt}", flush=True)
            shutil.move(str(out_path), str(alt))

    print("loading event lookup…", flush=True)
    event_lookup = load_event_lookup(events_path)
    print(f"event_lookup_size={len(event_lookup)}", flush=True)

    build_id = "v2_event_grain"
    manifest_path = build_dir / "build_manifest.json"
    if manifest_path.exists():
        try:
            build_id = str(json.loads(manifest_path.read_text()).get("build_id", build_id))
        except json.JSONDecodeError:
            pass

    pf = pq.ParquetFile(seq_path)
    writer: Optional[pq.ParquetWriter] = None
    smoke_rows: list[dict[str, Any]] = []
    smoke_seq_count = 0
    total_stats: Counter = Counter()
    rows_per_seq: list[int] = []
    narrative_counts: Counter = Counter()
    n_seq_seen = 0
    sentence_check: dict[str, Any] = {}

    for rg in range(pf.metadata.num_row_groups):
        seq = pf.read_row_group(rg).to_pandas()
        chunk_rows: list[dict[str, Any]] = []
        for rec in seq.itertuples(index=False):
            n_seq_seen += 1
            rows, stats = expand_sequence_rows(rec, event_lookup, build_id)
            total_stats.update(stats)
            if not rows:
                continue
            rows_per_seq.append(len(rows))
            narrative_counts[str(rec.narrative_id)] += 1
            chunk_rows.extend(rows)
            if smoke_seq_count < smoke_sequences:
                smoke_rows.extend(rows)
                smoke_seq_count += 1
            if not sentence_check and len(rows) >= 2:
                # integrity sample: position i matches events_tokenized SENTENCE
                eid0 = rows[0]["event_id"]
                assert rows[0]["SENTENCE"] == event_lookup[eid0]["SENTENCE"]
                sentence_check = {
                    "sequence_id": rows[0]["sequence_id"],
                    "n_events": len(rows),
                    "event_ids_head": [r["event_id"] for r in rows[:3]],
                    "sentence_match_pos0": True,
                    "n_same_time_groups": len({r["same_time_group_id"] for r in rows}),
                }
        if chunk_rows:
            table = pa.Table.from_pandas(pd.DataFrame(chunk_rows), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
            writer.write_table(table)
            del table
        del seq, chunk_rows
        if (rg + 1) % 20 == 0 or rg + 1 == pf.metadata.num_row_groups:
            print(
                {
                    "row_group": rg + 1,
                    "of": pf.metadata.num_row_groups,
                    "rows_so_far": int(total_stats["rows_emitted"]),
                    "seqs_so_far": int(total_stats["sequences_emitted"]),
                },
                flush=True,
            )

    if writer is not None:
        writer.close()
        tmp_path.replace(out_path)
    else:
        raise RuntimeError("No training rows emitted")

    smoke_df = pd.DataFrame(smoke_rows)
    smoke_df.to_parquet(smoke_path, index=False, compression="zstd")

    arr = np.array(rows_per_seq, dtype=np.int64) if rows_per_seq else np.array([0])
    kind_counts = {
        k.replace("kind_", ""): int(v)
        for k, v in total_stats.items()
        if k.startswith("kind_")
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grain": "event",
        "source_sequences": str(seq_path),
        "source_events": str(events_path),
        "output": str(out_path),
        "smoke_output": str(smoke_path),
        "flat_backup": str(build_dir / "training_events_v2.flat_backup.parquet"),
        "event_lookup_size": len(event_lookup),
        "sequences_seen": n_seq_seen,
        "sequences_emitted": int(total_stats["sequences_emitted"]),
        "rows": int(total_stats["rows_emitted"]),
        "unique_sequences": int(total_stats["sequences_emitted"]),
        "rows_per_seq": {
            "min": int(arr.min()),
            "p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
        },
        "sop_eligible_sequences_ge2_stg": int(total_stats["sop_eligible_seq"]),
        "sop_ineligible_sequences": int(total_stats["sop_ineligible_seq"]),
        "missing_event": int(total_stats["missing_event"]),
        "empty_sentence": int(total_stats["empty_sentence"]),
        "event_kind_counts": kind_counts,
        "narrative_id_top": narrative_counts.most_common(30),
        "p01_p06_p08": {
            "P01": int(narrative_counts.get("P01", 0)),
            "P06": int(narrative_counts.get("P06", 0)),
            "P08": int(narrative_counts.get("P08", 0)),
        },
        "image_event_rows": int(kind_counts.get("IMAGE", 0)),
        "growth_event_rows": int(kind_counts.get("GROWTH", 0)),
        "interpretation_event_rows": int(kind_counts.get("INTERPRETATION", 0)),
        "integrity_sample": sentence_check,
        "notes": [
            "SENTENCE is per-event from events_tokenized_v2 (not flat sequence).",
            "same_time_group_id comes from events table (sequences store unique STGs only).",
            "BACKGROUND_TOKENS holds farm markers; NARRATIVE is not repeated into event SENTENCE.",
        ],
    }
    write_json(report_path, report)
    print(json.dumps({k: report[k] for k in ("rows", "unique_sequences", "rows_per_seq", "p01_p06_p08")}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_build",
    )
    parser.add_argument("--smoke-sequences", type=int, default=64)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    export_event_grain(
        args.build_dir,
        smoke_sequences=args.smoke_sequences,
        backup_flat=not args.no_backup,
    )


if __name__ == "__main__":
    main()
