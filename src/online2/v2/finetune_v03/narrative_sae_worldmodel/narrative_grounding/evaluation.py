"""Grounding 평가 지표 (계획서 §5.4).

여기 있는 함수들은 이미 계산된 랭킹·라벨을 받아 지표만 계산하는 순수 함수다.
사람이 라벨링한 정답 pair(예: "이 서사는 실제로 이 window들을 근거로 한다")는
아직 이 저장소에 없다 — 그건 ``../ui/data_grounding_curation/``에서 §5.3
검토 큐를 사람이 실제로 처리해야 나온다. 즉 이 모듈은 "지표를 계산할 준비"이지
"실제 grounding 품질이 이 정도다"라는 결과가 아니다. 라벨이 쌓이기 전까지는
합성/소규모 데이터로만 검증한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Sequence

from .schemas import DataWindow, Narrative


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Narrative→Window Recall@K. ``relevant``가 비어 있으면 정의되지 않아 0.0."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Window→Narrative Precision@K. ``retrieved``가 k보다 짧으면 실제 길이로 나눈다."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / len(top_k)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def temporal_iou(a_start: str, a_end: str, b_start: str, b_end: str) -> float:
    """두 시간 구간의 Intersection-over-Union. 겹치지 않으면 0.0."""
    a0, a1 = _parse(a_start), _parse(a_end)
    b0, b1 = _parse(b_start), _parse(b_end)
    if a0 > a1 or b0 > b1:
        raise ValueError("interval start must not be after end")
    intersection = max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds())
    union = (max(a1, b1) - min(a0, b0)).total_seconds()
    if union <= 0:
        return 0.0
    return intersection / union


def feature_set_overlap(a: set[str], b: set[str]) -> float:
    """Jaccard 유사도. 둘 다 비어 있으면 정의되지 않아 0.0."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def farm_zone_scope_accuracy(predicted: DataWindow, gold: DataWindow) -> float:
    """farm/zone 두 축의 일치 비율(0.0/0.5/1.0). gold가 두 축 다 비어 있으면 0.0."""
    dims = 0
    matched = 0
    if gold.farm_ids:
        dims += 1
        if set(predicted.farm_ids) & set(gold.farm_ids):
            matched += 1
    if gold.zone_ids:
        dims += 1
        if set(predicted.zone_ids) & set(gold.zone_ids):
            matched += 1
    return (matched / dims) if dims else 0.0


def cycle_consistency(
    original_window_id: str,
    text_from_window_fn: Callable[[str], str],
    window_from_text_fn: Callable[[str], Sequence[str]],
    top_k: int = 5,
) -> bool:
    """window -> narrative -> window round-trip이 원래 window로 돌아오는지.

    ``text_from_window_fn``: window_id -> 그 window로 생성한 narrative 텍스트
    (예: ``Narrative.observation``). ``window_from_text_fn``: 그 텍스트로 §5.1
    검색을 실행해 얻은 랭킹된 window_id 목록(top_k까지). 원래 window_id가
    그 안에 있으면 사이클이 일관적이라고 본다.
    """
    text = text_from_window_fn(original_window_id)
    retrieved = window_from_text_fn(text)
    return original_window_id in retrieved[:top_k]


def unsupported_claim_rate(narratives: Iterable[Narrative]) -> float:
    """지지 window가 하나도 없는(NO_SUPPORT) narrative의 비율."""
    narratives = list(narratives)
    if not narratives:
        return 0.0
    unsupported = sum(1 for n in narratives if not n.has_support)
    return unsupported / len(narratives)


def expert_acceptance_rate(expert_decisions: Sequence[bool]) -> float:
    """검토 큐를 거친 항목 중 전문가가 실제로 승인(accept=True)한 비율."""
    if not expert_decisions:
        return 0.0
    return sum(1 for accepted in expert_decisions if accepted) / len(expert_decisions)


def auto_accept_error_and_review_rate(
    decisions: Sequence[str], correct: Sequence[bool]
) -> dict[str, float]:
    """§5.1 4분기 분류 결과의 사후 품질.

    ``decisions[i]``는 ``AUTO_ACCEPT_CANDIDATE|REVIEW|QUARANTINE|REJECT`` 중 하나,
    ``correct[i]``는 그 후보가 실제로 맞았는지(사람 검증 결과)에 대한 라벨.
    반환값:
      - ``auto_accept_error_rate``: AUTO_ACCEPT_CANDIDATE로 분류된 것 중 틀린 비율
      - ``human_review_rate``: REVIEW+QUARANTINE으로 넘어가 사람 손이 필요한 비율
    """
    if len(decisions) != len(correct):
        raise ValueError("decisions and correct must have the same length")
    total = len(decisions)
    if total == 0:
        return {"auto_accept_error_rate": 0.0, "human_review_rate": 0.0}

    auto_accept_indices = [i for i, d in enumerate(decisions) if d == "AUTO_ACCEPT_CANDIDATE"]
    auto_accept_error_rate = (
        sum(1 for i in auto_accept_indices if not correct[i]) / len(auto_accept_indices)
        if auto_accept_indices
        else 0.0
    )
    human_review_count = sum(1 for d in decisions if d in {"REVIEW", "QUARANTINE"})
    return {
        "auto_accept_error_rate": auto_accept_error_rate,
        "human_review_rate": human_review_count / total,
    }
