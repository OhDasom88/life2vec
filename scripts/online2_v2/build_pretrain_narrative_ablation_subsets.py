#!/usr/bin/env python3
"""서사(narrative) 다양성이 사전학습에 미치는 영향을 검증하기 위한 4종 부분 코퍼스 생성.

`training_events_v2.parquet`(1,500만 행, event-grain — 시퀀스당 여러 행)를
`narrative_id` 컬럼으로 필터링해 아래 3종 부분 코퍼스를 만든다(원시 데이터
재빌드 없이 기존 parquet를 스트리밍 필터링만 하므로 몇 분 안에 끝난다 — 3단계
원시 빌드 파이프라인을 다시 돌리면 ~1.5시간 걸린다):

1. narrow_subset — 실제 등장 빈도 상위 8개 narrative_id만 사용("일부 narrative만
   사용한 경우").
2. single_table_only — `data_sources`가 테이블 1개뿐인 34개 narrative(전체 80개 중,
   `outputs/online2/build-v8-active80-r3/normalized_catalog.csv` 기준).
3. multi_table_only — 테이블 2개 이상을 넘나드는 46개 narrative("다양한 테이블을
   오가는 서사").

그리고 서사 기반 재윈도잉이 만드는 ~16배 중복(같은 raw event가 여러 narrative
템플릿에 여러 번 걸쳐 재사용되는 것) 자체의 효과를 보기 위한 4번째 세트:

4. dedup_reduced — 같은 (farm_id, target_event_id) 쌍에 대해 여러 narrative가
   중복으로 만들어낸 시퀀스 중 하나만 남긴다("작은 데이터셋을 재구성"과 가장
   가까운 비교 — 원본 raw event 규모에 근접한, 중복 제거된 학습 세트).

전부 스트리밍 2-pass로 처리한다(1,500만 행을 한 번에 메모리에 안 올림).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

COLUMNS_FOR_STATS = ["sequence_id", "narrative_id", "farm_id", "event_position", "event_id"]


def single_and_multi_table_narratives(catalog_path: Path) -> tuple[set[str], set[str]]:
    df = pd.read_csv(catalog_path)
    n_sources = df["data_sources"].fillna("").apply(lambda s: len([x for x in s.split(";") if x.strip()]))
    single = set(df.loc[n_sources <= 1, "narrative_id"])
    multi = set(df.loc[n_sources >= 2, "narrative_id"])
    return single, multi


def scan_stats(parquet_path: Path, *, batch_size: int = 500_000):
    """1-pass: narrative_id별 행 수 집계 + 시퀀스별 (farm_id, 최대 event_position) 타깃 이벤트 파악."""
    pf = pq.ParquetFile(parquet_path)
    narrative_row_count: Counter[str] = Counter()
    seq_narrative: dict[str, str] = {}
    seq_farm: dict[str, str] = {}
    seq_max_pos: dict[str, int] = {}
    seq_target_event: dict[str, str] = {}

    n_seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS_FOR_STATS):
        seq_ids = batch.column("sequence_id").to_pylist()
        narratives = batch.column("narrative_id").to_pylist()
        farms = batch.column("farm_id").to_pylist()
        positions = batch.column("event_position").to_pylist()
        event_ids = batch.column("event_id").to_pylist()
        for sid, narr, farm, pos, eid in zip(seq_ids, narratives, farms, positions, event_ids):
            narrative_row_count[narr] += 1
            seq_narrative[sid] = narr
            seq_farm[sid] = farm
            if sid not in seq_max_pos or pos > seq_max_pos[sid]:
                seq_max_pos[sid] = pos
                seq_target_event[sid] = eid
        n_seen += len(seq_ids)
        if n_seen % 4_000_000 < batch_size:
            print(f"  scanned {n_seen} rows ...")

    return narrative_row_count, seq_narrative, seq_farm, seq_target_event


def pick_kept_sequences_for_dedup(
    seq_narrative: dict[str, str], seq_farm: dict[str, str], seq_target_event: dict[str, str]
) -> set[str]:
    """(farm_id, target_event_id)마다 처음 만난 sequence_id 하나만 남긴다(narrative_id 알파벳순 결정적)."""
    by_key: dict[tuple[str, str], str] = {}
    for sid in sorted(seq_narrative.keys(), key=lambda s: (seq_narrative[s], s)):
        key = (seq_farm[sid], seq_target_event[sid])
        by_key.setdefault(key, sid)
    return set(by_key.values())


def filter_parquet_by_sequence_predicate(
    parquet_path: Path, out_path: Path, keep_sequence_id) -> int:
    """sequence_id 단위 predicate로 전체 parquet를 스트리밍 필터링해서 새 parquet에 쓴다."""
    pf = pq.ParquetFile(parquet_path)
    writer = None
    n_kept = 0
    try:
        for batch in pf.iter_batches(batch_size=500_000):
            seq_ids = batch.column("sequence_id").to_pylist()
            mask = pa.array([keep_sequence_id(s) for s in seq_ids])
            filtered = batch.filter(mask)
            if filtered.num_rows == 0:
                continue
            table = pa.Table.from_batches([filtered])
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            n_kept += filtered.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n_kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-events", type=Path, default=ROOT / "outputs/online2/v2_build/training_events_v2.parquet"
    )
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "outputs/online2/build-v8-active80-r3/normalized_catalog.csv"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/online2/v2_build/narrative_ablation")
    parser.add_argument("--narrow-subset-size", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("pass 1/2: scanning for narrative frequency + per-sequence target events ...")
    narrative_row_count, seq_narrative, seq_farm, seq_target_event = scan_stats(args.training_events)
    print(f"  {len(seq_narrative)} distinct sequences, {len(narrative_row_count)} distinct narrative_id")

    single_table, multi_table = single_and_multi_table_narratives(args.catalog)
    print(f"  single-table narratives: {len(single_table)}, multi-table narratives: {len(multi_table)}")

    narrow_ids = {n for n, _ in narrative_row_count.most_common(args.narrow_subset_size)}
    print(f"  narrow subset ({args.narrow_subset_size} most frequent narrative_id): {sorted(narrow_ids)}")

    dedup_keep = pick_kept_sequences_for_dedup(seq_narrative, seq_farm, seq_target_event)
    print(f"  dedup_reduced: {len(dedup_keep)} sequences kept out of {len(seq_narrative)} "
          f"(dedup ratio {len(dedup_keep) / max(len(seq_narrative), 1):.3f})")

    conditions = {
        "narrow_subset": lambda sid: seq_narrative.get(sid) in narrow_ids,
        "single_table_only": lambda sid: seq_narrative.get(sid) in single_table,
        "multi_table_only": lambda sid: seq_narrative.get(sid) in multi_table,
        "dedup_reduced": lambda sid: sid in dedup_keep,
    }

    print("pass 2/2: writing filtered parquet files ...")
    for name, predicate in conditions.items():
        out_path = args.out_dir / f"{name}.parquet"
        n_kept = filter_parquet_by_sequence_predicate(args.training_events, out_path, predicate)
        print(f"  {name}: {n_kept} rows -> {out_path}")


if __name__ == "__main__":
    main()
