#!/usr/bin/env python3
"""이벤트(=raw 테이블 한 행)에 번들된 여러 measurement group을 각자의 독립된
"micro-event"로 쪼갠 대안 학습 코퍼스를 만든다.

## 왜

`build_v2.py`의 `tokenize_events()`를 보면, 하나의 raw 관측 행(예: 한 시각의
E_environment 판독값)에 있는 여러 컬럼(co2_ppm, inside_humidity_pct, ...)이
전부 **하나의 event 토큰 스팬**에 `[MEAS_SEP]`로만 구분돼 번들된다. 그런데
`GroupedMLMMasker`(src/online2/v2/masking.py)는 마스킹할 group은 올바르게
고르지만(measurement_group_id 단위), 고른 group **안에서는** 각 토큰(FEATURE/
VALUE_ABS/VALUE_ABS_COMBINED/VALUE_GLOBAL_REL/...)마다 독립적으로 80/10/10을
추첨한다 — 그 결과 절대/전역상대/농장상대 3개 척도로 같은 물리량을 중복
표현한 토큰들이 서로 다른 bin으로 랜덤 치환돼 물리적으로 불가능한 조합이
학습 데이터에 섞일 수 있다.

이 스크립트는 그 중 "이벤트 경계 자체가 여러 물리량을 부적절하게 묶고
있다"는 절반의 문제를 해결한다 — 원래 life2vec처럼 원자값 하나(=한
measurement group)를 그 자체로 독립된 event로 만든다. **주의**: 이것만으로
위에서 설명한 group-내부 토큰별 독립 치환 문제가 전부 없어지진 않는다(한
measurement group 안에도 여전히 FEATURE/VALUE_ABS/VALUE_GLOBAL_REL 등 여러
토큰이 있고, masking.py의 per-token 추첨 로직 자체는 안 건드렸다) — 그건
별도 개선(마스킹 알고리즘 자체 수정)이 필요하다.

## 어떻게

`training_events_v2.parquet`을 다시 원시 빌드(`build_v2.py`의 3단계, ~1.5시간)
하지 않고, 이미 만들어진 그 parquet을 스트리밍으로 읽어 각 행의 SENTENCE를
`measurement_group_ids`/`token_roles`로 파싱한 뒤 group별로 쪼갠다. 각 새
micro-event는:
  - event_id: `{원래 event_id}#{순번}` (실제로 쪼개진 경우만; 안 쪼개지면 원래 그대로)
  - event_position: `원래 position * 100 + 순번` — 이러면 전체 파일을 sequence_id별로
    다시 정렬하거나 그룹핑하지 않고 한 번의 스트리밍 패스로 순서를 보존할 수 있다
    (원래 정렬 방식과 똑같이 max event_position이 그 시퀀스의 "타깃"이 된다).
  - same_time_group_id/farm_id/zone_id/START_DATE/AGE/SEGMENT/narrative_id 등은
    원래 이벤트에서 그대로 상속(이 값들은 raw 행 단위지 컬럼 단위가 아니므로
    쪼갠다고 달라질 이유가 없다 — 특히 same_time_group_id를 그대로 물려받으므로
    SOP의 `_group_blocks`가 자동으로 "같은 원래 이벤트에서 나온 micro-event들은
    같은 블록"으로 묶는다, 코드 변경 없이).
  - SENTENCE: `[EVENT_SEP] VIEW|xxx EVENT_KIND|OBSERVATION [MEAS_SEP] <그 group의 토큰만>`

IMAGE/INTERPRETATION 이벤트나 measurement group이 아예 없는 이벤트(예:
`[MISSING]`)는 쪼갤 게 없으므로 원본 그대로 한 행 유지한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_SCHEMA_COLUMNS = None  # 입력과 동일한 컬럼셋을 그대로 씀(값만 바뀜)


def split_row_into_microevents(row: dict) -> list[dict]:
    """한 행(원래 이벤트)을 measurement_group_id별 micro-event 행 목록으로 쪼갠다.

    쪼갤 게 없으면(그룹이 0개 또는 1개) 원본 그대로 1개짜리 리스트를 반환한다
    (불필요하게 event_id에 접미사를 붙이지 않는다).
    """
    sentence = row["SENTENCE"]
    tokens = sentence.split()
    groups = json.loads(row["measurement_group_ids"]) if row["measurement_group_ids"] else []
    roles = json.loads(row["token_roles"]) if row["token_roles"] else []
    n = len(tokens)
    groups = groups + ["NONE"] * (n - len(groups))
    roles = roles + ["meta"] * (n - len(roles))

    # 앞쪽 wrapper(EVENT_SEP/VIEW/EVENT_KIND, group="NONE")를 걷어내고, 그 뒤부터
    # measurement_group_id가 바뀔 때마다 새 group으로 나눈다. wrapper가 없는(예:
    # IMAGE_SLOT처럼 group이 전부 NONE인) 이벤트는 통째로 그대로 둔다.
    distinct_groups = [g for g in dict.fromkeys(groups) if g != "NONE"]
    if len(distinct_groups) <= 1:
        return [dict(row)]

    # wrapper prefix: 맨 앞부터 group != "NONE"이 처음 나오기 전까지.
    first_group_idx = next(i for i, g in enumerate(groups) if g != "NONE")
    prefix_tokens = tokens[:first_group_idx]  # 보통 [EVENT_SEP] VIEW|xxx EVENT_KIND|OBSERVATION

    by_group: dict[str, list[int]] = {}
    for i in range(first_group_idx, n):
        by_group.setdefault(groups[i], []).append(i)

    out_rows = []
    for sub_idx, group_id in enumerate(distinct_groups):
        idxs = by_group[group_id]
        # prefix_tokens는 이미 마지막 원소로 [MEAS_SEP]를 포함한다(첫 group 직전 구분자) —
        # 여기서 또 추가하면 [MEAS_SEP] [MEAS_SEP]가 중복된다.
        sub_tokens = prefix_tokens + [tokens[i] for i in idxs]
        sub_groups = groups[:first_group_idx] + [groups[i] for i in idxs]
        sub_roles = roles[:first_group_idx] + [roles[i] for i in idxs]

        new_row = dict(row)
        new_row["event_id"] = f"{row['event_id']}#{sub_idx}"
        new_row["event_position"] = int(row["event_position"]) * 100 + sub_idx
        new_row["SENTENCE"] = " ".join(sub_tokens)
        new_row["measurement_group_ids"] = json.dumps(sub_groups)
        new_row["token_roles"] = json.dumps(sub_roles)
        out_rows.append(new_row)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-events", type=Path, default=ROOT / "outputs/online2/v2_build/training_events_v2.parquet"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs/online2/v2_build/training_events_v2_microevent.parquet"
    )
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--limit-rows", type=int, default=0, help="0=전체, >0이면 스모크 테스트용으로 앞부분만")
    args = parser.parse_args()

    pf = pq.ParquetFile(args.training_events)
    writer: pq.ParquetWriter | None = None
    n_in = 0
    n_out = 0
    n_split_events = 0

    for batch in pf.iter_batches(batch_size=args.batch_size):
        df = batch.to_pandas()
        out_records: list[dict] = []
        for row in df.to_dict("records"):
            split = split_row_into_microevents(row)
            if len(split) > 1:
                n_split_events += 1
            out_records.extend(split)
            n_in += 1
            if args.limit_rows and n_in >= args.limit_rows:
                break

        out_df = pa.Table.from_pylist(out_records, schema=batch.schema)
        if writer is None:
            writer = pq.ParquetWriter(args.out, out_df.schema)
        writer.write_table(out_df)
        n_out += len(out_records)
        print(f"  processed {n_in} input rows -> {n_out} output rows so far "
              f"(split so far: {n_split_events})", flush=True)

        if args.limit_rows and n_in >= args.limit_rows:
            break

    if writer is not None:
        writer.close()
    print(f"done: {n_in} input rows -> {n_out} output rows "
          f"({n_split_events} original events were split, "
          f"{n_out / max(n_in, 1):.2f}x row multiplier) -> {args.out}")


if __name__ == "__main__":
    main()
