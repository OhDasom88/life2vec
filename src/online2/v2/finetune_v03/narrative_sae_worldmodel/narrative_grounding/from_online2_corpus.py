"""Adapter: 기존 online2 코퍼스 빌드 결과 -> 계획서 Narrative/DataWindow.

시계열→서사(계획서 §5.2)의 규칙 기반 생성 자체는 이미 구현돼 있다:
``src/online2/materializers.py``의 ``NarrativeTemplate``/``MaterializerRegistry``가
``normalized_catalog.csv``(80개 ACTIVE 템플릿)를 원시 데이터에 매칭하고,
``src/online2/builder.py``의 ``CorpusBuilder``가 raw point부터 event, sequence까지
stable_id 해시 사슬로 물질화한다. 실제로 ``outputs/online2/build-v8-active80-r3/``에
967,012개 sequence가 이미 빌드돼 있다.

이 모듈은 그 매칭·해시 로직을 다시 만들지 않는다. ``sequences.parquet`` 한 행과
``normalized_catalog.csv``의 대응 템플릿 행을 계획서 §5.2 ``Narrative`` 객체로
감싸는 어댑터만 담당한다.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Mapping

from .schemas import DataWindow, Narrative

_DOWNGRADE_FLAGS = frozenset(
    {
        "CATALOG_MAX_EVENTS_APPLIED",
        "MAX_TOKEN_WINDOW_APPLIED",
        "OP_ELIGIBILITY_DOWNGRADED",
    }
)


def load_catalog(path: Path) -> dict[str, dict[str, str]]:
    """``normalized_catalog.csv``를 ``narrative_id``로 색인한다."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["narrative_id"]: row for row in reader}


def _tokens_with_prefix(background_tokens: list[str], prefix: str) -> tuple[str, ...]:
    values = {
        token.split("|", 1)[1]
        for token in background_tokens
        if token.startswith(prefix) and "|" in token
    }
    return tuple(sorted(values))


def window_from_sequence_row(row: Mapping[str, object]) -> DataWindow:
    """``sequences.parquet`` 한 행을 ``DataWindow``로 변환한다.

    ``narrative_center``는 builder.py에서 ``events[-1]["timestamp"]``(마지막 이벤트
    시각)로 정의된다. window 시작은 ``covered_time_span_hours``를 빼서 역산한다 —
    개별 이벤트 timestamp 목록(sequence_segments.parquet)까지 조인하지 않는 1차
    근사이며, 정밀한 시작 시각이 필요하면 ``sequence_segments`` 조인으로 보강한다.
    """
    background = json.loads(str(row["background_tokens"]))
    center = datetime.fromisoformat(str(row["narrative_center"]).replace("Z", "+00:00"))
    span_hours = float(row["covered_time_span_hours"])
    start = center - timedelta(hours=span_hours)
    return DataWindow(
        window_id=str(row["sequence_id"]),
        narrative_template_id=str(row["narrative_id"]),
        start_timestamp=start.isoformat().replace("+00:00", "Z"),
        end_timestamp=str(row["narrative_center"]),
        farm_ids=_tokens_with_prefix(background, "FARM|"),
        zone_ids=_tokens_with_prefix(background, "ZONE|"),
        segment_ids=tuple(json.loads(str(row["segment_ids"]))),
        event_views=tuple(json.loads(str(row["event_views"]))),
        covered_time_span_hours=span_hours,
        quality_flags=tuple(json.loads(str(row["quality_flags"]))),
    )


def narrative_from_sequence_row(
    row: Mapping[str, object], catalog: Mapping[str, dict[str, str]]
) -> Narrative:
    """``sequences.parquet`` 한 행 + catalog 템플릿 행 -> ``Narrative``.

    ``interpretation``은 템플릿 저자(전문가)가 미리 써 둔 ``agronomic_interpretation``을
    그대로 옮긴 것이다 — 이 인스턴스에서 재확인된 해석이 아니라 "가능한" 해석이라는
    계획서 §5.2 정의를 그대로 유지하며, ``causal_status``는 항상 기본값
    (``NOT_ESTABLISHED``)에서 시작한다.
    """
    narrative_id = str(row["narrative_id"])
    template = catalog.get(narrative_id)
    if template is None:
        raise KeyError(f"narrative_id {narrative_id!r} not found in catalog")

    window = window_from_sequence_row(row)
    downgraded = bool(_DOWNGRADE_FLAGS.intersection(window.quality_flags))

    observation = (
        f"[{narrative_id}] {template.get('narrative_name_ko', '')} 템플릿이 "
        f"farm={','.join(window.farm_ids) or 'UNKNOWN'} "
        f"zone={','.join(window.zone_ids) or 'UNKNOWN'} 구간 "
        f"{window.start_timestamp}~{window.end_timestamp} "
        f"({window.covered_time_span_hours:.1f}h, 이벤트 {len(window.segment_ids)}개, "
        f"event_views={','.join(window.event_views)})에서 매칭됨. "
        f"window_definition={template.get('window_definition', '')} / "
        f"start_condition={template.get('start_condition', '')} / "
        f"end_condition={template.get('end_condition', '')}"
    )
    derived_state = (
        f"threshold_type={template.get('threshold_type', '')}; "
        f"threshold_or_rule={template.get('threshold_or_rule', '')}; "
        f"expected_pattern={template.get('expected_pattern', '')}"
    )
    interpretation = template.get("agronomic_interpretation", "")

    confidence = {
        "template_expert_confidence": template.get("confidence", ""),
        "template_implementation_priority": template.get("implementation_priority", ""),
        "template_expert_evidence": template.get("expert_evidence", ""),
        "op_eligible": bool(row["op_eligible"]),
        "distinct_time_group_count": int(row["distinct_time_group_count"]),  # type: ignore[arg-type]
        "quality_flags": list(window.quality_flags),
    }

    return Narrative(
        narrative_instance_id=str(row["sequence_id"]),
        observation=observation,
        derived_state=derived_state,
        interpretation=interpretation,
        supporting_windows=(window,),
        contradicting_windows=(),
        confidence=confidence,
        sequence_recommendation="REVIEW" if downgraded else "INCLUDE",
    )


def iter_narratives_from_build(
    build_dir: Path, batch_size: int = 10_000
) -> Iterator[Narrative]:
    """빌드 산출물 디렉토리(예: ``outputs/online2/build-v8-active80-r3/``)에서
    ``sequences.parquet``을 스트리밍으로 읽어 ``Narrative``를 생성한다.
    """
    import pyarrow.parquet as pq

    catalog = load_catalog(build_dir / "normalized_catalog.csv")
    table = pq.read_table(build_dir / "sequences.parquet")
    for batch in table.to_batches(max_chunksize=batch_size):
        for row in batch.to_pylist():
            yield narrative_from_sequence_row(row, catalog)
