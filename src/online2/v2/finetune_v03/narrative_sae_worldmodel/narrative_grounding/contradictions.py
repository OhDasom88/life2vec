"""반례(contradicting evidence) 탐지 (계획서 §5.1 "반대 근거", §5.3 기준 4).

전문가가 이미 각 narrative 템플릿에 붙여 둔 ``confounders`` 텍스트
(``normalized_catalog.csv``)를 근거로 쓴다 — 새로운 규칙을 발명하지 않는다.
어떤 window가 narrative_id가 다른 템플릿에 속하면서 (a) 같은 farm에서 (b) 이
narrative의 window와 시간이 겹치고 (c) 그 템플릿의 설명 텍스트에 이 narrative
템플릿의 confounders 키워드가 등장하면 "경쟁 설명(competing explanation)"으로
보고 ``contradicting_windows`` 후보로 올린다.

이건 의미(semantic) 매칭이 아니라 부분 문자열 매칭이다 — 오탐·누락이 있는 1차
신호이며, 그 자체로 "이 narrative는 틀렸다"는 결론이 아니라 §5.3 검토 큐로
넘기기 위한 신호다("근거·반례 동시 존재" 기준을 실제로 발동시킨다).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .evaluation import temporal_iou
from .from_online2_corpus import window_from_sequence_row
from .schemas import DataWindow, Narrative

_CANDIDATE_TEXT_FIELDS = (
    "category",
    "narrative_name_ko",
    "purpose",
    "agronomic_interpretation",
    "data_sources",
)


def _confounder_keywords(template: Mapping[str, str]) -> tuple[str, ...]:
    raw = template.get("confounders", "")
    return tuple(part.strip() for part in raw.split(";") if part.strip())


def _candidate_text(template: Mapping[str, str]) -> str:
    return " ".join(template.get(field, "") for field in _CANDIDATE_TEXT_FIELDS)


@dataclass(frozen=True)
class ContradictionMatch:
    window: DataWindow
    matched_keyword: str
    temporal_iou: float


def find_contradicting_windows(
    narrative: Narrative,
    catalog: Mapping[str, Mapping[str, str]],
    candidate_rows: Iterable[Mapping[str, object]],
    *,
    min_temporal_iou: float = 0.0,
) -> tuple[ContradictionMatch, ...]:
    """narrative와 farm이 겹치고 시간이 겹치며 confounder 키워드가 매칭되는,
    narrative_id가 다른 템플릿의 window들을 찾는다.

    ``candidate_rows``는 호출자가 이미 같은 farm으로 좁혀 놓은
    ``sequences.parquet`` 행 iterable이어야 한다 — ``search_text_to_window``와
    동일한 설계 원칙으로, 967K행 전체를 여기서 스캔하지 않는다.
    """
    if not narrative.supporting_windows:
        return ()
    base_window = narrative.supporting_windows[0]
    base_template = catalog.get(base_window.narrative_template_id)
    if base_template is None:
        return ()
    keywords = _confounder_keywords(base_template)
    if not keywords:
        return ()

    matches: list[ContradictionMatch] = []
    for row in candidate_rows:
        other_template_id = str(row["narrative_id"])
        if other_template_id == base_window.narrative_template_id:
            continue
        other_template = catalog.get(other_template_id)
        if other_template is None:
            continue
        other_window = window_from_sequence_row(row)
        if not (set(base_window.farm_ids) & set(other_window.farm_ids)):
            continue
        iou = temporal_iou(
            base_window.start_timestamp,
            base_window.end_timestamp,
            other_window.start_timestamp,
            other_window.end_timestamp,
        )
        if iou <= min_temporal_iou:
            continue
        haystack = _candidate_text(other_template)
        hit = next((keyword for keyword in keywords if keyword and keyword in haystack), None)
        if hit is None:
            continue
        matches.append(ContradictionMatch(window=other_window, matched_keyword=hit, temporal_iou=iou))

    matches.sort(key=lambda match: match.temporal_iou, reverse=True)
    return tuple(matches)


def augment_with_contradictions(
    narrative: Narrative,
    catalog: Mapping[str, Mapping[str, str]],
    candidate_rows: Iterable[Mapping[str, object]],
    *,
    min_temporal_iou: float = 0.0,
) -> Narrative:
    """발견된 반례로 ``contradicting_windows``를 채운 새 ``Narrative``를 반환한다.

    ``Narrative``는 frozen dataclass이므로 원본은 바뀌지 않는다. 반례가 없으면
    원본 객체를 그대로 돌려준다(불필요한 복사 방지).
    """
    matches = find_contradicting_windows(
        narrative, catalog, candidate_rows, min_temporal_iou=min_temporal_iou
    )
    if not matches:
        return narrative
    return replace(narrative, contradicting_windows=tuple(match.window for match in matches))
