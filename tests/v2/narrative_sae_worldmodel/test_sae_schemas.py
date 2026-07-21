"""§9.1/§9.7 SAE 상태 머신·인과 등급 회귀 테스트."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.schemas import (
    CausalGrade,
    FeatureState,
    SAEFeatureRef,
    assert_causal_claim_allowed,
    assert_valid_promotion,
)


def test_single_step_promotion_allowed() -> None:
    assert_valid_promotion(FeatureState.SAE_LATENT, FeatureState.CANDIDATE_SEMANTIC_FEATURE)
    assert_valid_promotion(
        FeatureState.CANDIDATE_SEMANTIC_FEATURE, FeatureState.EVALUATED_SEMANTIC_FEATURE
    )


def test_skipping_a_stage_rejected() -> None:
    with pytest.raises(ValueError, match="must pass through"):
        assert_valid_promotion(FeatureState.SAE_LATENT, FeatureState.INTERVENTION_SUPPORTED_FEATURE)


def test_demotion_always_allowed() -> None:
    # 재평가로 신뢰도가 떨어지는 것(강등)은 언제나 허용 - 한 단계 초과 승격만 막는다.
    assert_valid_promotion(FeatureState.CIRCUIT_SUPPORTED_FEATURE, FeatureState.SAE_LATENT)


def test_causal_claim_below_e3_rejected() -> None:
    for grade in (CausalGrade.E0_OBSERVED, CausalGrade.E1_ASSOCIATED, CausalGrade.E2_PREDICTIVE):
        with pytest.raises(ValueError, match="below E3_INTERVENTION"):
            assert_causal_claim_allowed(grade)


def test_causal_claim_at_or_above_e3_allowed() -> None:
    for grade in (CausalGrade.E3_INTERVENTION, CausalGrade.E4_MEDIATED, CausalGrade.E5_CIRCUIT):
        assert_causal_claim_allowed(grade)  # 예외 없이 통과해야 함


def test_sae_feature_ref_negative_latent_id_rejected() -> None:
    with pytest.raises(ValueError, match="latent_id"):
        SAEFeatureRef(sae_version="SAE_PRETRAIN", latent_id=-1, state=FeatureState.SAE_LATENT)


def test_intervention_supported_requires_causal_evidence() -> None:
    with pytest.raises(ValueError, match="INTERVENTION_SUPPORTED_FEATURE"):
        SAEFeatureRef(
            sae_version="SAE_PRETRAIN",
            latent_id=1,
            state=FeatureState.INTERVENTION_SUPPORTED_FEATURE,
            causal_grade=CausalGrade.E1_ASSOCIATED,  # E3 미만인데 개입 지지 상태를 주장
        )


def test_intervention_supported_with_sufficient_evidence_succeeds() -> None:
    ref = SAEFeatureRef(
        sae_version="SAE_PRETRAIN",
        latent_id=1,
        state=FeatureState.INTERVENTION_SUPPORTED_FEATURE,
        causal_grade=CausalGrade.E3_INTERVENTION,
    )
    assert ref.state is FeatureState.INTERVENTION_SUPPORTED_FEATURE


def test_candidate_state_does_not_require_causal_evidence() -> None:
    # CANDIDATE/EVALUATED 단계는 기본 causal_grade(E0)로도 문제없이 만들 수 있다.
    ref = SAEFeatureRef(
        sae_version="SAE_PRETRAIN", latent_id=1, state=FeatureState.EVALUATED_SEMANTIC_FEATURE
    )
    assert ref.causal_grade == CausalGrade.E0_OBSERVED
