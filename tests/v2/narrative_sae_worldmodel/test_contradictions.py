"""반례(contradicting evidence) 탐지 회귀 테스트."""

from __future__ import annotations

import json

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.contradictions import (
    augment_with_contradictions,
    find_contradicting_windows,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.from_online2_corpus import (
    narrative_from_sequence_row,
)

_CATALOG = {
    "A01": {
        "narrative_name_ko": "실내온도 급격한 점프",
        "category": "센서운영이상",
        "purpose": "작물 반응과 센서 이상 분리",
        "window_definition": "점프 전후 각 3시간",
        "start_condition": "차분 robust z 6 초과",
        "end_condition": "시작 후 3시간",
        "threshold_type": "empirical",
        "threshold_or_rule": "농장 구역별 차분 median과 MAD 기반 robust z",
        "expected_pattern": "단일 시점 점프 후 복귀",
        "agronomic_interpretation": "sensor_anomaly",
        "confidence": "높음",
        "implementation_priority": "상",
        "expert_evidence": "",
        "confounders": "실제 급변;환기",
        "data_sources": "E_environment",
    },
    # 환기(ventilation) 관련 실제 작업 이벤트 - A01의 confounders "환기"와 매칭돼야 함.
    "A10": {
        "narrative_name_ko": "환기창 개방 반응",
        "category": "농작업반응",
        "purpose": "환기 작업 전후 온습도 변화 확인",
        "agronomic_interpretation": "ventilation_actuation",
        "confounders": "",
        "data_sources": "A_actuator",
    },
    # 관수 관련 - 키워드가 안 겹쳐서 매칭되면 안 됨.
    "B01": {
        "narrative_name_ko": "관수 후 EC 회복",
        "category": "원인반응",
        "purpose": "관수 반응 정상 패턴 확인",
        "agronomic_interpretation": "irrigation_response",
        "confounders": "",
        "data_sources": "A_actuator",
    },
}


def _row(narrative_id: str, sequence_id: str, farm: str, zone: str, center: str, span_hours: float) -> dict[str, object]:
    return {
        "sequence_id": sequence_id,
        "narrative_id": narrative_id,
        "narrative_center": center,
        "background_tokens": json.dumps([f"FARM|{farm}", f"ZONE|{zone}"]),
        "segment_ids": json.dumps(["event_a"]),
        "event_views": json.dumps(["E_environment"]),
        "covered_time_span_hours": span_hours,
        "quality_flags": json.dumps([]),
        "op_eligible": True,
        "distinct_time_group_count": 1,
    }


def _base_narrative():
    row = _row("A01", "seq_base", "F130230", "1", "2024-12-08T01:00:00Z", 6.0)
    return narrative_from_sequence_row(row, _CATALOG)


def test_finds_confounder_matched_overlapping_window() -> None:
    narrative = _base_narrative()
    candidates = [
        # 시간 겹침 + farm 동일 + confounder "환기" 매칭 -> 반례로 잡혀야 함.
        _row("A10", "seq_vent", "F130230", "1", "2024-12-08T02:00:00Z", 2.0),
    ]
    matches = find_contradicting_windows(narrative, _CATALOG, candidates)
    assert len(matches) == 1
    assert matches[0].window.window_id == "seq_vent"
    assert matches[0].matched_keyword in {"실제 급변", "환기"}


def test_ignores_same_template_instances() -> None:
    narrative = _base_narrative()
    candidates = [_row("A01", "seq_same_template", "F130230", "1", "2024-12-08T02:00:00Z", 2.0)]
    assert find_contradicting_windows(narrative, _CATALOG, candidates) == ()


def test_ignores_non_overlapping_farm() -> None:
    narrative = _base_narrative()
    candidates = [_row("A10", "seq_other_farm", "F999999", "9", "2024-12-08T02:00:00Z", 2.0)]
    assert find_contradicting_windows(narrative, _CATALOG, candidates) == ()


def test_ignores_non_overlapping_time() -> None:
    narrative = _base_narrative()
    # base window: 2024-12-07T19:00 ~ 2024-12-08T01:00. 아래 후보는 다음날이라 안 겹침.
    candidates = [_row("A10", "seq_far_future", "F130230", "1", "2024-12-09T12:00:00Z", 1.0)]
    assert find_contradicting_windows(narrative, _CATALOG, candidates) == ()


def test_ignores_keyword_mismatch() -> None:
    narrative = _base_narrative()
    # B01은 confounders 키워드("실제 급변", "환기")와 무관한 텍스트라 안 잡혀야 함.
    candidates = [_row("B01", "seq_irrigation", "F130230", "1", "2024-12-08T02:00:00Z", 2.0)]
    assert find_contradicting_windows(narrative, _CATALOG, candidates) == ()


def test_augment_with_contradictions_fills_field_and_keeps_narrative_frozen() -> None:
    narrative = _base_narrative()
    assert narrative.contradicting_windows == ()
    candidates = [_row("A10", "seq_vent", "F130230", "1", "2024-12-08T02:00:00Z", 2.0)]
    augmented = augment_with_contradictions(narrative, _CATALOG, candidates)
    assert augmented is not narrative
    assert narrative.contradicting_windows == ()  # 원본 불변
    assert len(augmented.contradicting_windows) == 1
    assert augmented.contradicting_windows[0].window_id == "seq_vent"


def test_augment_returns_same_object_when_no_match() -> None:
    narrative = _base_narrative()
    augmented = augment_with_contradictions(narrative, _CATALOG, [])
    assert augmented is narrative
