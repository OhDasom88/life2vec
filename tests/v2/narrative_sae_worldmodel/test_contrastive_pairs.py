"""§7.3 대조학습 pair 품질 등급/hard negative/shortcut 전처리 회귀 테스트."""

from __future__ import annotations

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.multimodal_pretrain.contrastive_pairs import (
    TIER_WEIGHTS,
    find_hard_negatives,
    pair_quality_tier,
    strip_identifying_tokens,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.schemas import (
    DataWindow,
    Narrative,
)


def _window(template_id: str, farm: str) -> DataWindow:
    return DataWindow(
        window_id=f"w-{template_id}-{farm}",
        narrative_template_id=template_id,
        start_timestamp="t0",
        end_timestamp="t1",
        farm_ids=(farm,),
        zone_ids=("1",),
        segment_ids=(),
        event_views=(),
        covered_time_span_hours=1.0,
        quality_flags=(),
    )


def _narrative(instance_id: str, template_id: str, farm: str) -> Narrative:
    return Narrative(
        narrative_instance_id=instance_id,
        observation="o",
        derived_state="d",
        interpretation="i",
        supporting_windows=(_window(template_id, farm),),
    )


def test_human_accept_always_wins_gold() -> None:
    assert pair_quality_tier(human_decision="ACCEPT", grounding_search_decision="REJECT") == "GOLD_EXPERT"


def test_human_reject_always_wins_contradicted() -> None:
    assert (
        pair_quality_tier(human_decision="REJECT", grounding_search_decision="AUTO_ACCEPT_CANDIDATE")
        == "CONTRADICTED"
    )


def test_grounding_search_decision_used_when_no_human_decision() -> None:
    assert pair_quality_tier(grounding_search_decision="AUTO_ACCEPT_CANDIDATE") == "SILVER_MODEL_GROUNDED"
    assert pair_quality_tier(grounding_search_decision="REVIEW") == "WEAK_TEMPORAL_MATCH"
    assert pair_quality_tier(grounding_search_decision="QUARANTINE") == "WEAK_TEMPORAL_MATCH"
    assert pair_quality_tier(grounding_search_decision="REJECT") == "CONTRADICTED"


def test_default_falls_back_to_rule_grounded() -> None:
    assert pair_quality_tier() == "SILVER_RULE_GROUNDED"
    assert pair_quality_tier(is_rule_grounded=False) == "UNVERIFIED"


def test_skip_decision_does_not_override_search_decision() -> None:
    # SKIP은 "아직 결정 안 됨"이지 승인/거부가 아니므로 human_decision 분기를 안 탄다.
    assert pair_quality_tier(human_decision="SKIP", grounding_search_decision="AUTO_ACCEPT_CANDIDATE") == (
        "SILVER_MODEL_GROUNDED"
    )


def test_tier_weights_cover_all_tiers_and_contradicted_is_zero() -> None:
    from src.online2.v2.finetune_v03.narrative_sae_worldmodel.multimodal_pretrain.contrastive_pairs import (
        PAIR_QUALITY_TIERS,
    )

    assert set(TIER_WEIGHTS.keys()) == set(PAIR_QUALITY_TIERS)
    assert TIER_WEIGHTS["CONTRADICTED"] == 0.0
    assert TIER_WEIGHTS["GOLD_EXPERT"] == max(TIER_WEIGHTS.values())


def test_find_hard_negatives_requires_shared_farm_and_different_template() -> None:
    anchor = _narrative("anchor", "A01", "F1")
    candidates = [
        _narrative("same_template", "A01", "F1"),  # 같은 템플릿 -> 제외
        _narrative("other_farm", "B01", "F2"),  # farm 안 겹침 -> 제외
        _narrative("good_negative", "B01", "F1"),  # farm 겹침 + 다른 템플릿 -> 채택
    ]
    negatives = find_hard_negatives(anchor, candidates)
    assert [n.narrative_instance_id for n in negatives] == ["good_negative"]


def test_find_hard_negatives_respects_max_negatives() -> None:
    anchor = _narrative("anchor", "A01", "F1")
    candidates = [_narrative(f"n{i}", "B01", "F1") for i in range(10)]
    negatives = find_hard_negatives(anchor, candidates, max_negatives=3)
    assert len(negatives) == 3


def test_find_hard_negatives_no_support_returns_empty() -> None:
    anchor = Narrative(narrative_instance_id="x", observation="o", derived_state="d", interpretation="i")
    negatives = find_hard_negatives(anchor, [_narrative("a", "A01", "F1")])
    assert negatives == []


def test_strip_identifying_tokens_removes_farm_and_date() -> None:
    stripped = strip_identifying_tokens("F130230 zone 1에서 2024-12-08 관측됨")
    assert "F130230" not in stripped
    assert "2024-12-08" not in stripped
    assert "[FARM]" in stripped
    assert "[DATE]" in stripped


def test_strip_identifying_tokens_leaves_other_text_untouched() -> None:
    assert strip_identifying_tokens("온도가 급격히 올랐다") == "온도가 급격히 올랐다"
