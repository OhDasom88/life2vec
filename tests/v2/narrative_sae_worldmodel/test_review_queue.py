"""§5.3 사람 검토 큐 회귀 테스트."""

from __future__ import annotations

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.review_queue import (
    ReviewContext,
    build_review_queue,
    evaluate_review_reasons,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.schemas import (
    DataWindow,
    Narrative,
)

_WINDOW = DataWindow(
    window_id="w1",
    narrative_template_id="A01",
    start_timestamp="t0",
    end_timestamp="t1",
    farm_ids=("F130230",),
    zone_ids=("1",),
    segment_ids=(),
    event_views=(),
    covered_time_span_hours=6.0,
    quality_flags=(),
)


def _narrative(**overrides: object) -> Narrative:
    fields: dict[str, object] = dict(
        narrative_instance_id="n1",
        observation="o",
        derived_state="d",
        interpretation="i",
        supporting_windows=(_WINDOW,),
        confidence={"template_expert_confidence": "높음"},
        sequence_recommendation="INCLUDE",
    )
    fields.update(overrides)
    return Narrative(**fields)  # type: ignore[arg-type]


def test_no_reasons_when_nothing_flagged() -> None:
    narrative = _narrative()
    reasons = evaluate_review_reasons(narrative)
    assert reasons == ()


def test_expert_high_confidence_but_review_recommendation_flags_disagreement() -> None:
    narrative = _narrative(
        confidence={"template_expert_confidence": "높음"}, sequence_recommendation="REVIEW"
    )
    reasons = evaluate_review_reasons(narrative)
    codes = {r.code for r in reasons}
    assert "MODEL_EXPERT_DISAGREEMENT" in codes


def test_expert_low_confidence_but_include_recommendation_flags_disagreement() -> None:
    narrative = _narrative(
        confidence={"template_expert_confidence": "낮음"}, sequence_recommendation="INCLUDE"
    )
    reasons = evaluate_review_reasons(narrative)
    codes = {r.code for r in reasons}
    assert "MODEL_EXPERT_DISAGREEMENT" in codes


def test_low_frequency_concept_requires_frequency_table() -> None:
    narrative = _narrative(confidence={})
    # frequency table 없으면 판정 자체를 안 한다.
    assert evaluate_review_reasons(narrative, template_frequency=None) == ()
    reasons = evaluate_review_reasons(
        narrative, template_frequency={"A01": 5}, low_frequency_threshold=50
    )
    codes = {r.code for r in reasons}
    assert "LOW_FREQUENCY_CONCEPT" in codes


def test_causal_status_beyond_default_flags_review() -> None:
    narrative = _narrative(confidence={}, causal_status="E1_ASSOCIATED")
    reasons = evaluate_review_reasons(narrative)
    codes = {r.code for r in reasons}
    assert "CAUSAL_OR_RECOMMENDATION_PRESENT" in codes


def test_external_context_signals_require_explicit_injection() -> None:
    narrative = _narrative(confidence={})
    empty_context_reasons = evaluate_review_reasons(narrative, ReviewContext())
    assert empty_context_reasons == ()

    filled_context = ReviewContext(
        prediction_impact=0.9,
        concept_split_merge_candidate=True,
        split_ambiguous=True,
    )
    reasons = evaluate_review_reasons(narrative, filled_context)
    codes = {r.code for r in reasons}
    assert codes == {
        "HIGH_PREDICTION_IMPACT",
        "CONCEPT_SPLIT_MERGE_CANDIDATE",
        "SPLIT_ACCESS_AMBIGUOUS",
    }


def test_build_review_queue_sorts_by_priority_and_drops_clean_entries() -> None:
    frequent_window = DataWindow(
        window_id="w2",
        narrative_template_id="A02",
        start_timestamp="t0",
        end_timestamp="t1",
        farm_ids=(),
        zone_ids=(),
        segment_ids=(),
        event_views=(),
        covered_time_span_hours=1.0,
        quality_flags=(),
    )
    clean = _narrative(
        narrative_instance_id="clean", confidence={}, supporting_windows=(frequent_window,)
    )
    split_ambiguous = _narrative(narrative_instance_id="risky", confidence={})
    low_freq = _narrative(narrative_instance_id="rare", confidence={})

    items = [
        (clean, ReviewContext()),
        (split_ambiguous, ReviewContext(split_ambiguous=True)),
        (low_freq, ReviewContext()),
    ]
    queue = build_review_queue(
        items,
        template_frequency={"A01": 1, "A02": 500},
        low_frequency_threshold=50,
    )

    ids = [entry.narrative.narrative_instance_id for entry in queue]
    assert "clean" not in ids
    # SPLIT_ACCESS_AMBIGUOUS(가중치 3.0)가 LOW_FREQUENCY_CONCEPT(가중치 1.0)보다 우선.
    assert ids[0] == "risky"
    assert ids[1] == "rare"
    assert queue[0].priority_score > queue[1].priority_score
