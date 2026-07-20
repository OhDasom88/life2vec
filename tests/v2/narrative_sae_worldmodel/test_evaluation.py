"""§5.4 grounding 평가 지표 회귀 테스트 (합성 데이터)."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.evaluation import (
    auto_accept_error_and_review_rate,
    cycle_consistency,
    expert_acceptance_rate,
    farm_zone_scope_accuracy,
    feature_set_overlap,
    precision_at_k,
    recall_at_k,
    temporal_iou,
    unsupported_claim_rate,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.schemas import (
    DataWindow,
    Narrative,
)


def test_recall_at_k_basic() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c", "d"}, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == pytest.approx(1.0)


def test_recall_at_k_empty_relevant_is_zero() -> None:
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_precision_at_k_basic() -> None:
    assert precision_at_k(["a", "x", "c"], {"a", "c"}, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_short_retrieved_uses_actual_length() -> None:
    assert precision_at_k(["a"], {"a", "b"}, k=5) == pytest.approx(1.0)


def test_temporal_iou_full_overlap() -> None:
    iou = temporal_iou(
        "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z",
        "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z",
    )
    assert iou == pytest.approx(1.0)


def test_temporal_iou_partial_overlap() -> None:
    iou = temporal_iou(
        "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z",
        "2024-01-01T01:00:00Z", "2024-01-01T03:00:00Z",
    )
    # intersection 1h, union 3h
    assert iou == pytest.approx(1 / 3)


def test_temporal_iou_no_overlap() -> None:
    iou = temporal_iou(
        "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z",
        "2024-01-01T05:00:00Z", "2024-01-01T06:00:00Z",
    )
    assert iou == 0.0


def test_temporal_iou_rejects_inverted_interval() -> None:
    with pytest.raises(ValueError):
        temporal_iou(
            "2024-01-01T02:00:00Z", "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z",
        )


def test_feature_set_overlap_jaccard() -> None:
    assert feature_set_overlap({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert feature_set_overlap(set(), set()) == 0.0


def _window(farm_ids: tuple[str, ...], zone_ids: tuple[str, ...]) -> DataWindow:
    return DataWindow(
        window_id="w",
        narrative_template_id="A01",
        start_timestamp="t0",
        end_timestamp="t1",
        farm_ids=farm_ids,
        zone_ids=zone_ids,
        segment_ids=(),
        event_views=(),
        covered_time_span_hours=1.0,
        quality_flags=(),
    )


def test_farm_zone_scope_accuracy_full_match() -> None:
    predicted = _window(("F1",), ("1",))
    gold = _window(("F1",), ("1",))
    assert farm_zone_scope_accuracy(predicted, gold) == pytest.approx(1.0)


def test_farm_zone_scope_accuracy_partial_match() -> None:
    predicted = _window(("F1",), ("9",))
    gold = _window(("F1",), ("1",))
    assert farm_zone_scope_accuracy(predicted, gold) == pytest.approx(0.5)


def test_farm_zone_scope_accuracy_gold_without_scope_is_undefined_zero() -> None:
    predicted = _window(("F1",), ("1",))
    gold = _window((), ())
    assert farm_zone_scope_accuracy(predicted, gold) == 0.0


def test_cycle_consistency_round_trip() -> None:
    text_of = {"w1": "온도 점프"}
    retrieval_of = {"온도 점프": ["w1", "w2"]}
    assert cycle_consistency("w1", lambda wid: text_of[wid], lambda t: retrieval_of[t])
    assert not cycle_consistency("w3", lambda wid: "미등록", lambda t: [])


def test_unsupported_claim_rate() -> None:
    supported = Narrative(
        narrative_instance_id="a", observation="o", derived_state="d", interpretation="i",
        supporting_windows=(_window(("F1",), ("1",)),),
    )
    unsupported = Narrative(
        narrative_instance_id="b", observation="o", derived_state="d", interpretation="i",
    )
    assert unsupported_claim_rate([supported, unsupported]) == pytest.approx(0.5)
    assert unsupported_claim_rate([]) == 0.0


def test_expert_acceptance_rate() -> None:
    assert expert_acceptance_rate([True, True, False, True]) == pytest.approx(0.75)
    assert expert_acceptance_rate([]) == 0.0


def test_auto_accept_error_and_review_rate() -> None:
    decisions = ["AUTO_ACCEPT_CANDIDATE", "AUTO_ACCEPT_CANDIDATE", "REVIEW", "QUARANTINE", "REJECT"]
    correct =   [True,                    False,                   True,     True,          False]
    result = auto_accept_error_and_review_rate(decisions, correct)
    assert result["auto_accept_error_rate"] == pytest.approx(0.5)
    assert result["human_review_rate"] == pytest.approx(2 / 5)


def test_auto_accept_error_and_review_rate_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        auto_accept_error_and_review_rate(["REVIEW"], [])
