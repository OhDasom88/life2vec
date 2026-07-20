"""§5.1 서사→시계열 검색 회귀 테스트.

Qwen3EmbeddingProvider(실제 모델)는 GPU/네트워크가 필요해 CI에서 쓰지 않는다.
대신 결정론적 bag-of-words 해시 기반 FakeEmbeddingProvider로 랭킹·결합점수·
분기(threshold) 로직만 검증한다 — 실제 모델 품질 평가는 §5.4 evaluation 몫이다.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.text_to_window import (
    GROUNDING_DECISIONS,
    StructuredHints,
    TemplateEmbeddingIndex,
    extract_structured_hints,
    search_text_to_window,
)

_CATALOG = {
    "A01": {
        "narrative_name_ko": "실내온도 급격한 점프",
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
    },
    "B01": {
        "narrative_name_ko": "관수 후 EC 회복",
        "purpose": "관수 반응 정상 패턴 확인",
        "window_definition": "관수 이벤트 전후 2시간",
        "start_condition": "관수 시작 감지",
        "end_condition": "관수 종료 후 2시간",
        "threshold_type": "rule",
        "threshold_or_rule": "EC 변화량 기준",
        "expected_pattern": "EC 일시 하락 후 회복",
        "agronomic_interpretation": "irrigation_response",
        "confidence": "중간",
        "implementation_priority": "중",
        "expert_evidence": "",
    },
}


class FakeEmbeddingProvider:
    """narrative_id 텍스트에 등장하는 키워드를 그대로 원-핫 축으로 쓰는 해시 임베딩.

    실제 의미를 이해하진 못하지만, "온도/점프" 질의가 A01(온도 점프) 템플릿과
    "관수/EC" 질의가 B01과 더 가깝게 나오는지 정도는 검증할 수 있다.
    """

    _VOCAB = ["온도", "점프", "센서", "관수", "ec", "회복", "구역", "차분"]

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vec = np.array([1.0 if word in lowered else 0.0 for word in self._VOCAB], dtype=np.float32)
            vectors.append(vec)
        return np.stack(vectors)


def _sequence_row(narrative_id: str, sequence_id: str, farm: str, zone: str) -> dict[str, object]:
    return {
        "sequence_id": sequence_id,
        "narrative_id": narrative_id,
        "narrative_center": "2024-12-08T01:00:00Z",
        "background_tokens": json.dumps([f"FARM|{farm}", f"ZONE|{zone}"]),
        "segment_ids": json.dumps(["event_a"]),
        "event_views": json.dumps(["E_environment"]),
        "covered_time_span_hours": 6.0,
        "quality_flags": json.dumps([]),
        "op_eligible": True,
        "distinct_time_group_count": 1,
    }


def test_extract_structured_hints_parses_farm_and_zone() -> None:
    hints = extract_structured_hints("F130230 zone 1에서 관측된 온도 점프")
    assert hints.farm_ids == ("F130230",)
    assert hints.zone_ids == ("1",)


def test_extract_structured_hints_empty_when_absent() -> None:
    hints = extract_structured_hints("온도가 급격히 점프했다")
    assert hints == StructuredHints(farm_ids=(), zone_ids=())


def test_template_index_ranks_matching_keywords_first() -> None:
    provider = FakeEmbeddingProvider()
    index = TemplateEmbeddingIndex(_CATALOG, provider)
    ranked = index.rank(provider.embed(["온도 점프 센서 이상"])[0])
    assert ranked[0][0] == "A01"


def test_search_prefers_matching_farm_and_high_similarity() -> None:
    provider = FakeEmbeddingProvider()
    index = TemplateEmbeddingIndex(_CATALOG, provider)
    rows = [
        _sequence_row("A01", "seq_1", "F130230", "1"),
        _sequence_row("A01", "seq_2", "F999999", "9"),
        _sequence_row("B01", "seq_3", "F130230", "1"),
    ]
    candidates = search_text_to_window(
        "F130230 zone 1 온도 점프 관측",
        catalog=_CATALOG,
        template_index=index,
        provider=provider,
        corpus_rows=rows,
        top_k_templates=2,
    )
    assert candidates
    assert all(c.decision in GROUNDING_DECISIONS for c in candidates)
    # farm이 일치하는 seq_1이 farm 불일치인 seq_2보다 결합 점수가 높아야 한다.
    seq1 = next(c for c in candidates if c.narrative.narrative_instance_id == "seq_1")
    seq2 = next(c for c in candidates if c.narrative.narrative_instance_id == "seq_2")
    assert seq1.combined_score > seq2.combined_score


def test_search_respects_max_windows_per_template() -> None:
    provider = FakeEmbeddingProvider()
    index = TemplateEmbeddingIndex(_CATALOG, provider)
    rows = [_sequence_row("A01", f"seq_{i}", "F130230", "1") for i in range(10)]
    candidates = search_text_to_window(
        "온도 점프",
        catalog=_CATALOG,
        template_index=index,
        provider=provider,
        corpus_rows=rows,
        top_k_templates=1,
        max_windows_per_template=3,
    )
    assert len(candidates) == 3


def test_grounding_candidate_rejects_invalid_decision() -> None:
    from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.text_to_window import (
        GroundingCandidate,
    )
    from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.schemas import (
        Narrative,
    )

    narrative = Narrative(
        narrative_instance_id="x",
        observation="o",
        derived_state="d",
        interpretation="i",
    )
    with pytest.raises(ValueError):
        GroundingCandidate(
            narrative=narrative,
            template_similarity=0.5,
            structured_score=0.5,
            combined_score=0.5,
            decision="MAYBE",
            decision_reason="bad",
        )
