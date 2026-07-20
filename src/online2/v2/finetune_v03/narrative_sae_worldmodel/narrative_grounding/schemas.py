"""Narrative / DataWindow objects (계획서 §4.1, §5.2).

DataWindow와 Narrative는 새로 관측을 만들지 않는다. ``src/online2`` 코퍼스
빌더(``builder.py``/``materializers.py``)가 이미 raw point부터 event, sequence까지
stable_id 해시 사슬로 만들어 놓았다(``outputs/online2/build-v8-active80-r3/``,
967,012 sequence 실측). 이 모듈은 그 결과를 계획서 스키마로 감싸는 read-only
뷰이며, ``window_id``는 신규로 발급하지 않고 기존 ``sequence_id``를 그대로
재사용한다 — 계획서 §4.2 추적 키 사슬을 두 벌로 만들지 않기 위함이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CAUSAL_STATUS_DEFAULT = "NOT_ESTABLISHED"
RECOMMENDATION_STATUS_DEFAULT = "NOT_GENERATED"
SEQUENCE_RECOMMENDATIONS = frozenset({"INCLUDE", "EXCLUDE", "REVIEW"})


@dataclass(frozen=True)
class DataWindow:
    """계획서 §4.1 ``DataWindow``. ``window_id``는 online2 ``sequence_id`` 재사용."""

    window_id: str
    narrative_template_id: str
    start_timestamp: str
    end_timestamp: str
    farm_ids: tuple[str, ...]
    zone_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    event_views: tuple[str, ...]
    covered_time_span_hours: float
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class Narrative:
    """계획서 §5.2 JSON 구조. 인과/추천 상태는 명시적으로 확정하기 전까지 기본값을 유지한다."""

    narrative_instance_id: str
    observation: str
    derived_state: str
    interpretation: str
    causal_status: str = CAUSAL_STATUS_DEFAULT
    recommendation_status: str = RECOMMENDATION_STATUS_DEFAULT
    supporting_windows: tuple[DataWindow, ...] = field(default_factory=tuple)
    contradicting_windows: tuple[DataWindow, ...] = field(default_factory=tuple)
    confidence: dict[str, Any] = field(default_factory=dict)
    sequence_recommendation: str = "REVIEW"

    def __post_init__(self) -> None:
        if self.sequence_recommendation not in SEQUENCE_RECOMMENDATIONS:
            raise ValueError(
                f"invalid sequence_recommendation: {self.sequence_recommendation!r}"
            )
        # §17-B1: 모든 narrative가 supporting window 또는 NO_SUPPORT를 가져야 한다.
        # 여기서는 빈 supporting_windows 자체를 NO_SUPPORT로 취급하고,
        # 근거 없는 narrative에 인과성을 암묵적으로 부여하지 않도록 강제한다.
        if not self.supporting_windows and self.causal_status != CAUSAL_STATUS_DEFAULT:
            raise ValueError(
                "narrative without supporting_windows must keep "
                f"causal_status == {CAUSAL_STATUS_DEFAULT!r}"
            )

    @property
    def has_support(self) -> bool:
        return bool(self.supporting_windows)
