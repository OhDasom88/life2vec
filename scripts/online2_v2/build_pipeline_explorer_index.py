#!/usr/bin/env python3
"""pipeline_explorer UI가 즉시 로드할 수 있는 소형 시퀀스 인덱스를 만든다.

2026-07-26: `training_events_v2.parquet`(사전에 260배 중복 펼쳐 저장한 flat
파일)를 더 이상 요구하지 않는다 — `sequences_v2.parquet`(시퀀스당 1행,
event_ids만 참조) + `events_tokenized_v2.parquet`(고유 이벤트당 1행)을
직접 조인해서 필요한 표본만 그때그때 펼친다(`run_v2_pretrain_loop.py`가
학습 로딩 시점에 하는 것과 정확히 같은 방식, 같은 함수 재사용). 이러면
9백만 시퀀스를 5,800만 행으로 미리 펼쳐 쓰는 다단계(수 시간) 없이도 인덱스를
바로 만들 수 있다.

주의: `event_id`만으로 필터링하면 안 된다 — 같은 물리적 raw 이벤트가 narrative
기반 재윈도잉 때문에 여러 다른 시퀀스에 걸쳐 재사용되므로, event_id 하나가
수십 개 시퀀스에 나타날 수 있다(실제로 처음 이 방식으로 시도했다가 1,519개
타깃 이벤트에서 97,829행이 나와서 발견함). 그래서 여기서는 `sequence_id`
단위로 표본을 뽑는다.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

LIGHT_SEQ_COLUMNS = [
    "sequence_id",
    "narrative_id",
    "PERSON_ID",
    "event_ids",
    "farm_ids",
    "canonical_context_id",
    "split_group_id",
    "training_mode",
    "sampling_weight",
    "shadow_split",
]


def scan_sequences_by_narrative(sequences_parquet: Path, *, batch_size: int = 500_000) -> dict[str, list[str]]:
    """narrative_id -> 그 narrative가 만든 sequence_id 목록. sequences_v2.parquet는 이미
    시퀀스당 1행이라(training_events_v2와 달리 260배 중복이 없음) dedup이 필요 없다."""
    pf = pq.ParquetFile(sequences_parquet)
    by_narrative: dict[str, list[str]] = defaultdict(list)
    n_seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=["sequence_id", "narrative_id"]):
        seq_ids = batch.column("sequence_id").to_pylist()
        narratives = batch.column("narrative_id").to_pylist()
        for sid, narr in zip(seq_ids, narratives):
            by_narrative[narr].append(sid)
        n_seen += len(seq_ids)
        if n_seen % 2_000_000 < batch_size:
            print(f"  scanned {n_seen} sequences ...", flush=True)
    return dict(by_narrative)


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
        "--build-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_build_expanded_full1517",
        help="sequences_v2.parquet / events_tokenized_v2.parquet가 있는 디렉토리.",
    )
    parser.add_argument("--per-narrative-cap", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/online2/sae_pilot/pipeline_explorer_index.parquet"
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sequences_parquet = args.build_dir / "sequences_v2.parquet"
    events_parquet = args.build_dir / "events_tokenized_v2.parquet"
    if not sequences_parquet.exists() or not events_parquet.exists():
        raise FileNotFoundError(f"{sequences_parquet}와 {events_parquet}가 둘 다 있어야 합니다.")

    print("pass 1/2: scanning sequence_id -> narrative_id ...", flush=True)
    by_narrative = scan_sequences_by_narrative(sequences_parquet)
    print(
        f"  {len(by_narrative)} distinct narrative_id, "
        f"{sum(len(v) for v in by_narrative.values())} sequences total",
        flush=True,
    )

    chosen_sequence_ids = sample_sequence_ids(by_narrative, per_narrative_cap=args.per_narrative_cap, seed=args.seed)
    print(
        f"  sampled {len(chosen_sequence_ids)} sequences "
        f"(cap {args.per_narrative_cap}/narrative x up to {len(by_narrative)} narratives)",
        flush=True,
    )

    print("pass 2/2: joining event content for the chosen sequences ...", flush=True)
    from export_training_events_v2_event_grain import expand_sequence_rows, load_event_lookup  # noqa: E402

    event_lookup = load_event_lookup(events_parquet)
    print(f"  event_lookup_size={len(event_lookup)}", flush=True)

    pf = pq.ParquetFile(sequences_parquet)
    available = set(pf.schema_arrow.names)
    cols = [c for c in LIGHT_SEQ_COLUMNS if c in available]
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        part = batch.to_pandas()
        sub = part[part["sequence_id"].isin(chosen_sequence_ids)]
        if sub.empty:
            continue
        for rec in sub.itertuples(index=False):
            expanded, _stats = expand_sequence_rows(rec, event_lookup, build_id="pipeline_explorer_index")
            rows.extend(expanded)

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows ({len(chosen_sequence_ids)} sequences) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
