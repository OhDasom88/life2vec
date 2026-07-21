"""ui/data_grounding_curation/review_batch.py 회귀 테스트 (Streamlit 의존성 없음)."""

from __future__ import annotations

import json

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.ui.data_grounding_curation.review_batch import (
    append_decision,
    build_batch_queue,
    load_decisions,
    make_decision_record,
    sample_per_template,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.review_queue import (
    ReviewQueueEntry,
    ReviewReason,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.schemas import Narrative

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
    "A10": {
        "narrative_name_ko": "환기창 개방 반응",
        "category": "농작업반응",
        "purpose": "환기 작업 전후 온습도 변화 확인",
        "agronomic_interpretation": "ventilation_actuation",
        "confidence": "중간",
        "confounders": "",
        "data_sources": "A_actuator",
    },
}


def _row(narrative_id: str, sequence_id: str, farm: str, zone: str, center: str, span_hours: float) -> dict:
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


def test_sample_per_template_caps_each_group_independently() -> None:
    rows = [_row("A01", f"a{i}", "F1", "1", "2024-12-08T01:00:00Z", 1.0) for i in range(5)]
    rows += [_row("A10", f"b{i}", "F1", "1", "2024-12-08T01:00:00Z", 1.0) for i in range(2)]
    sampled = sample_per_template(rows, per_template=3, seed=0)
    by_template: dict[str, int] = {}
    for row in sampled:
        by_template[row["narrative_id"]] = by_template.get(row["narrative_id"], 0) + 1
    assert by_template == {"A01": 3, "A10": 2}


def test_sample_per_template_is_deterministic_given_seed() -> None:
    rows = [_row("A01", f"a{i}", "F1", "1", "2024-12-08T01:00:00Z", 1.0) for i in range(10)]
    first = sample_per_template(rows, per_template=3, seed=42)
    second = sample_per_template(rows, per_template=3, seed=42)
    assert [r["sequence_id"] for r in first] == [r["sequence_id"] for r in second]


def test_build_batch_queue_surfaces_contradiction_and_low_frequency() -> None:
    farm_rows = [
        _row("A01", "seq_base", "F130230", "1", "2024-12-08T01:00:00Z", 6.0),
        _row("A10", "seq_vent", "F130230", "1", "2024-12-08T02:00:00Z", 2.0),
    ]
    queue, stats = build_batch_queue(
        farm_rows, _CATALOG, per_template=5, seed=0, low_frequency_threshold=50
    )
    assert stats == {"farm_row_count": 2, "sampled_count": 2, "queue_length": len(queue)}

    entry_by_id = {e.narrative.narrative_instance_id: e for e in queue}
    assert "seq_base" in entry_by_id  # A01: 반례(seq_vent) + 저빈도(count=1 < 50) 둘 다 걸림
    base_entry = entry_by_id["seq_base"]
    codes = {r.code for r in base_entry.reasons}
    assert "SUPPORT_AND_CONTRADICTION_BOTH_PRESENT" in codes
    assert "LOW_FREQUENCY_CONCEPT" in codes
    assert len(base_entry.narrative.contradicting_windows) == 1


def test_decision_roundtrip(tmp_path) -> None:
    path = tmp_path / "decisions.jsonl"
    assert load_decisions(path) == {}

    narrative = Narrative(
        narrative_instance_id="seq_1", observation="o", derived_state="d", interpretation="i"
    )
    entry = ReviewQueueEntry(
        narrative=narrative,
        reasons=(ReviewReason("LOW_FREQUENCY_CONCEPT", "count=1"),),
        priority_score=1.0,
    )
    record = make_decision_record(entry, "ACCEPT", "tester")
    append_decision(path, record)

    decisions = load_decisions(path)
    assert decisions["seq_1"]["decision"] == "ACCEPT"
    assert decisions["seq_1"]["reviewer"] == "tester"

    # 같은 narrative를 다시 검토 -> 기존 줄을 덮어쓰지 않고 추가, 마지막 결정이 우선.
    append_decision(path, make_decision_record(entry, "REJECT", "tester2"))
    decisions_after = load_decisions(path)
    assert decisions_after["seq_1"]["decision"] == "REJECT"
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_make_decision_record_rejects_invalid_decision() -> None:
    narrative = Narrative(
        narrative_instance_id="seq_1", observation="o", derived_state="d", interpretation="i"
    )
    entry = ReviewQueueEntry(narrative=narrative, reasons=(), priority_score=0.0)
    with pytest.raises(ValueError):
        make_decision_record(entry, "MAYBE", "tester")
