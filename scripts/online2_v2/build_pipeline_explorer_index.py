#!/usr/bin/env python3
"""pipeline_explorer UI가 즉시 로드할 수 있는 소형 시퀀스 인덱스를 만든다.

`training_events_v2.parquet`(1,500만 행, event-grain — 시퀀스당 여러 행)를
매 클릭마다 스트리밍하면 UI가 느려지므로, 80개 narrative_id마다 소수(기본
15개) 시퀀스씩만 뽑아 **그 시퀀스에 속한 모든 이벤트 행**을 담은 작은
parquet를 미리 만들어 둔다.

주의: `event_id`만으로 필터링하면 안 된다 — 같은 물리적 raw 이벤트가 narrative
기반 재윈도잉 때문에 여러 다른 시퀀스에 걸쳐 재사용되므로, event_id 하나가
수십 개 시퀀스에 나타날 수 있다(실제로 처음 이 방식으로 시도했다가 1,519개
타깃 이벤트에서 97,829행이 나와서 발견함). 그래서 여기서는 `sequence_id`
단위로 표본을 뽑고, 그 sequence_id에 속한 행만 남긴다.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

INDEX_COLUMNS = [
    "sequence_id",
    "event_id",
    "event_position",
    "farm_id",
    "zone_id",
    "narrative_id",
    "START_DATE",
    "AGE",
    "SENTENCE",
    "measurement_group_ids",
    "token_roles",
    "event_kind",
    "SEGMENT",
]


def scan_sequences_by_narrative(parquet_path: Path, *, batch_size: int = 500_000) -> dict[str, list[str]]:
    """narrative_id -> 그 narrative가 만든 sequence_id 목록(중복 없이)."""
    pf = pq.ParquetFile(parquet_path)
    seen: dict[str, set[str]] = defaultdict(set)
    n_seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=["sequence_id", "narrative_id"]):
        seq_ids = batch.column("sequence_id").to_pylist()
        narratives = batch.column("narrative_id").to_pylist()
        for sid, narr in zip(seq_ids, narratives):
            seen[narr].add(sid)
        n_seen += len(seq_ids)
        if n_seen % 4_000_000 < batch_size:
            print(f"  scanned {n_seen} rows ...")
    return {narr: sorted(ids) for narr, ids in seen.items()}


def sample_sequence_ids(by_narrative: dict[str, list[str]], *, per_narrative_cap: int, seed: int) -> set[str]:
    rng = np.random.RandomState(seed)
    chosen: set[str] = set()
    for narr in sorted(by_narrative):
        ids = by_narrative[narr]
        if len(ids) > per_narrative_cap:
            ids = list(rng.choice(ids, size=per_narrative_cap, replace=False))
        chosen.update(ids)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-events", type=Path, default=ROOT / "outputs/online2/v2_build/training_events_v2.parquet"
    )
    parser.add_argument("--per-narrative-cap", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/online2/sae_pilot/pipeline_explorer_index.parquet"
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("pass 1/2: scanning sequence_id -> narrative_id ...")
    by_narrative = scan_sequences_by_narrative(args.training_events)
    print(f"  {len(by_narrative)} distinct narrative_id, "
          f"{sum(len(v) for v in by_narrative.values())} distinct sequences total")

    chosen_sequence_ids = sample_sequence_ids(by_narrative, per_narrative_cap=args.per_narrative_cap, seed=args.seed)
    print(f"  sampled {len(chosen_sequence_ids)} sequences "
          f"(cap {args.per_narrative_cap}/narrative x up to {len(by_narrative)} narratives)")

    print("pass 2/2: pulling full sequence rows for the chosen sequence_ids ...")
    pf = pq.ParquetFile(args.training_events)
    kept_batches = []
    n_found = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=INDEX_COLUMNS):
        seq_ids = batch.column("sequence_id").to_pylist()
        mask = pa.array([sid in chosen_sequence_ids for sid in seq_ids])
        filtered = batch.filter(mask)
        if filtered.num_rows:
            kept_batches.append(filtered)
            n_found += filtered.num_rows

    table = pa.Table.from_batches(kept_batches)
    pq.write_table(table, args.out)
    print(f"wrote {n_found} rows ({len(chosen_sequence_ids)} sequences) -> {args.out}")


if __name__ == "__main__":
    main()
