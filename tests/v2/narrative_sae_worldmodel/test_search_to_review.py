"""§5.1 검색 결과 -> §5.3 검토 큐 연결부 회귀 테스트."""

from __future__ import annotations

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.schemas import (
    DataWindow,
    Narrative,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.search_to_review import (
    QUARANTINE_REASON_CODE,
    REVIEW_REASON_CODE,
    entry_from_grounding_candidate,
    needs_human_confirmation,
    queue_from_candidates,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.text_to_window import (
    GroundingCandidate,
)

_WINDOW = DataWindow(
    window_id="w1",
    narrative_template_id="A01",
    start_timestamp="t0",
    end_timestamp="t1",
    farm_ids=("F1",),
    zone_ids=("1",),
    segment_ids=(),
    event_views=(),
    covered_time_span_hours=1.0,
    quality_flags=(),
)


def _candidate(instance_id: str, decision: str) -> GroundingCandidate:
    narrative = Narrative(
        narrative_instance_id=instance_id,
        observation="o",
        derived_state="d",
        interpretation="i",
        supporting_windows=(_WINDOW,),
    )
    return GroundingCandidate(
        narrative=narrative,
        template_similarity=0.8,
        structured_score=0.6,
        combined_score=0.7,
        decision=decision,
        decision_reason="test",
    )


def test_needs_human_confirmation_only_for_review_and_quarantine() -> None:
    assert needs_human_confirmation(_candidate("a", "REVIEW"))
    assert needs_human_confirmation(_candidate("b", "QUARANTINE"))
    assert not needs_human_confirmation(_candidate("c", "AUTO_ACCEPT_CANDIDATE"))
    assert not needs_human_confirmation(_candidate("d", "REJECT"))


def test_entry_from_grounding_candidate_carries_decision_and_reason() -> None:
    entry = entry_from_grounding_candidate(_candidate("a", "QUARANTINE"))
    assert entry.grounding_search_decision == "QUARANTINE"
    assert len(entry.reasons) == 1
    assert entry.reasons[0].code == QUARANTINE_REASON_CODE
    assert "combined_score=0.700" in entry.reasons[0].detail


def test_quarantine_has_higher_priority_than_review() -> None:
    review_entry = entry_from_grounding_candidate(_candidate("a", "REVIEW"))
    quarantine_entry = entry_from_grounding_candidate(_candidate("b", "QUARANTINE"))
    assert quarantine_entry.priority_score > review_entry.priority_score


def test_queue_from_candidates_filters_and_sorts() -> None:
    candidates = [
        _candidate("auto", "AUTO_ACCEPT_CANDIDATE"),
        _candidate("review", "REVIEW"),
        _candidate("quarantine", "QUARANTINE"),
        _candidate("reject", "REJECT"),
    ]
    queue = queue_from_candidates(candidates)
    ids = [entry.narrative.narrative_instance_id for entry in queue]
    assert ids == ["quarantine", "review"]  # QUARANTINE 우선순위가 더 높음
    assert all(entry.grounding_search_decision in {"REVIEW", "QUARANTINE"} for entry in queue)


def test_batch_origin_entry_has_no_grounding_search_decision() -> None:
    from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.review_queue import (
        ReviewQueueEntry,
    )

    narrative = Narrative(
        narrative_instance_id="x", observation="o", derived_state="d", interpretation="i"
    )
    entry = ReviewQueueEntry(narrative=narrative, reasons=(), priority_score=0.0)
    assert entry.grounding_search_decision is None
