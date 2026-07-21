"""§5.1 검색 결과 -> §5.3 검토 큐 연결부.

``search_text_to_window(_over_table)``이 ``REVIEW``/``QUARANTINE``으로 분류한
``GroundingCandidate``는 자동 승인도 자동 거부도 아니다 — 사람이 확인해야
한다. ``AUTO_ACCEPT_CANDIDATE``는 이미 충분히 확신하는 판정이라 사람 검토를
요구하지 않고, ``REJECT``는 §5.1 검색 자체가 이미 후보에서 배제한 것이라
검토 큐에 다시 올릴 이유가 없다.

이 모듈은 그 두 등급(REVIEW/QUARANTINE)만 ``review_queue.ReviewQueueEntry``로
감싸서 §5.3 큐에 그대로 흘려보낸다 — §5.3 UI/결정 기록 코드를 포크하지
않는다. 이렇게 넘어온 항목의 사람 결정(ACCEPT/REJECT)은 §5.1 판정이 맞았는지의
실측 신호가 되고, ``ReviewQueueEntry.grounding_search_decision``을 통해
``decisions.jsonl``까지 구조적으로 실려간다(문자열 파싱이 아니라 필드로).

주의: 이걸로 ``evaluation.auto_accept_error_and_review_rate``가 완전히
연결되는 것은 아니다. 그 지표는 AUTO_ACCEPT_CANDIDATE/REJECT까지 포함한
전체 후보 모집단 위에서 계산해야 human_review_rate가 의미를 가지는데,
여기서는 애초에 REVIEW/QUARANTINE만 큐에 올리므로 그 서브셋만 라벨이
생긴다. ``decision_metrics.py``의 ``not_connected`` 설명 참조.
"""

from __future__ import annotations

from .review_queue import ReviewQueueEntry, ReviewReason
from .text_to_window import GroundingCandidate

# 사람 확인이 필요한 §5.1 판정. AUTO_ACCEPT_CANDIDATE(이미 확신)와
# REJECT(§5.1이 이미 배제)는 검토 큐에 올리지 않는다.
NEEDS_CONFIRMATION_DECISIONS = frozenset({"REVIEW", "QUARANTINE"})

REVIEW_REASON_CODE = "GROUNDING_SEARCH_REVIEW_CANDIDATE"
QUARANTINE_REASON_CODE = "GROUNDING_SEARCH_QUARANTINE_CANDIDATE"

# review_queue._REASON_WEIGHTS(최대 3.0, SPLIT_ACCESS_AMBIGUOUS)와 같은 척도.
# QUARANTINE이 REVIEW보다 더 의심스러운 판정이므로 우선순위를 더 높게 둔다.
_PRIORITY_BY_DECISION = {
    "REVIEW": 1.5,
    "QUARANTINE": 2.5,
}
_REASON_CODE_BY_DECISION = {
    "REVIEW": REVIEW_REASON_CODE,
    "QUARANTINE": QUARANTINE_REASON_CODE,
}


def needs_human_confirmation(candidate: GroundingCandidate) -> bool:
    return candidate.decision in NEEDS_CONFIRMATION_DECISIONS


def entry_from_grounding_candidate(candidate: GroundingCandidate) -> ReviewQueueEntry:
    """§5.1 후보 하나를 §5.3 큐 항목으로 감싼다.

    호출자가 ``needs_human_confirmation``으로 먼저 걸러야 한다 — 이 함수는
    AUTO_ACCEPT_CANDIDATE/REJECT도 그대로 감싸버리므로(가드 없음), 무분별하게
    전체를 큐에 올리지 않도록 ``queue_from_candidates``를 쓰는 걸 권장한다.
    """
    reason = ReviewReason(
        code=_REASON_CODE_BY_DECISION.get(candidate.decision, "GROUNDING_SEARCH_CANDIDATE"),
        detail=(
            f"combined_score={candidate.combined_score:.3f}, "
            f"template_similarity={candidate.template_similarity:.3f}, "
            f"structured_score={candidate.structured_score:.3f} — {candidate.decision_reason}"
        ),
    )
    return ReviewQueueEntry(
        narrative=candidate.narrative,
        reasons=(reason,),
        priority_score=_PRIORITY_BY_DECISION.get(candidate.decision, 1.0),
        grounding_search_decision=candidate.decision,
    )


def queue_from_candidates(candidates: list[GroundingCandidate]) -> list[ReviewQueueEntry]:
    """REVIEW/QUARANTINE 후보만 추려 §5.3 큐 항목(우선순위 내림차순)으로 변환."""
    entries = [
        entry_from_grounding_candidate(candidate)
        for candidate in candidates
        if needs_human_confirmation(candidate)
    ]
    entries.sort(key=lambda entry: entry.priority_score, reverse=True)
    return entries
