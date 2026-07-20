"""narrative_grounding 스키마와 online2 코퍼스 어댑터 회귀 테스트.

실제 967,012행 규모 outputs/online2/build-v8-active80-r3/sequences.parquet에
narrative_from_sequence_row가 예외 없이 동작함은 대화형으로 확인했다(3000개 무작위
샘플, 67/80 템플릿 커버). 여기서는 CI에서 빠르게 도는 합성 fixture로 어댑터
로직(윈도우 시각 역산, 다운그레이드 플래그 -> REVIEW, NO_SUPPORT 불변식)만 검증한다.
"""

from __future__ import annotations

import json

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.from_online2_corpus import (
    narrative_from_sequence_row,
    window_from_sequence_row,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.schemas import (
    CAUSAL_STATUS_DEFAULT,
    DataWindow,
    Narrative,
)

_CATALOG = {
    "A01": {
        "narrative_name_ko": "실내온도 급격한 점프",
        "window_definition": "점프 전후 각 3시간",
        "start_condition": "차분 robust z 6 초과",
        "end_condition": "시작 후 3시간",
        "threshold_type": "empirical",
        "threshold_or_rule": "농장 구역별 차분 median과 MAD 기반 robust z",
        "expected_pattern": "단일 시점 점프 후 복귀",
        "agronomic_interpretation": "sensor_anomaly",
        "confidence": "높음",
        "implementation_priority": "상",
        "expert_evidence": "고정 전문가 임계값 없음",
    }
}


def _sequence_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sequence_id": "sequence_test0001",
        "narrative_id": "A01",
        "narrative_center": "2024-12-08T01:00:00Z",
        "background_tokens": json.dumps(["FARM|F130230", "ZONE|1"]),
        "segment_ids": json.dumps(["event_a", "event_b"]),
        "event_views": json.dumps(["E_environment"]),
        "covered_time_span_hours": 6.0,
        "quality_flags": json.dumps([]),
        "op_eligible": True,
        "distinct_time_group_count": 2,
    }
    row.update(overrides)
    return row


def test_window_from_sequence_row_reconstructs_start_from_span() -> None:
    window = window_from_sequence_row(_sequence_row())
    assert window.window_id == "sequence_test0001"
    assert window.end_timestamp == "2024-12-08T01:00:00Z"
    assert window.start_timestamp == "2024-12-07T19:00:00Z"
    assert window.farm_ids == ("F130230",)
    assert window.zone_ids == ("1",)


def test_narrative_from_sequence_row_defaults_to_not_established() -> None:
    narrative = narrative_from_sequence_row(_sequence_row(), _CATALOG)
    assert narrative.causal_status == CAUSAL_STATUS_DEFAULT
    assert narrative.recommendation_status == "NOT_GENERATED"
    assert narrative.has_support
    assert narrative.interpretation == "sensor_anomaly"
    assert narrative.sequence_recommendation == "INCLUDE"


def test_downgrade_flag_forces_review() -> None:
    row = _sequence_row(quality_flags=json.dumps(["CATALOG_MAX_EVENTS_APPLIED"]))
    narrative = narrative_from_sequence_row(row, _CATALOG)
    assert narrative.sequence_recommendation == "REVIEW"


def test_unknown_narrative_id_raises() -> None:
    row = _sequence_row(narrative_id="ZZZ_UNKNOWN")
    with pytest.raises(KeyError):
        narrative_from_sequence_row(row, _CATALOG)


def test_narrative_without_support_cannot_claim_established_causality() -> None:
    with pytest.raises(ValueError):
        Narrative(
            narrative_instance_id="x",
            observation="o",
            derived_state="d",
            interpretation="i",
            causal_status="E1_ASSOCIATED",
            supporting_windows=(),
            sequence_recommendation="REVIEW",
        )


def test_invalid_sequence_recommendation_rejected() -> None:
    window = DataWindow(
        window_id="w",
        narrative_template_id="A01",
        start_timestamp="t0",
        end_timestamp="t1",
        farm_ids=(),
        zone_ids=(),
        segment_ids=(),
        event_views=(),
        covered_time_span_hours=0.0,
        quality_flags=(),
    )
    with pytest.raises(ValueError):
        Narrative(
            narrative_instance_id="x",
            observation="o",
            derived_state="d",
            interpretation="i",
            supporting_windows=(window,),
            sequence_recommendation="MAYBE",
        )
