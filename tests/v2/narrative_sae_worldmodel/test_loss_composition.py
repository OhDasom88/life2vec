"""L_total = Σλ·L 결합 로직 회귀 테스트."""

from __future__ import annotations

import pytest
import torch

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.multimodal_pretrain.loss_composition import (
    PretrainLossWeights,
    compose_total_loss,
)


def test_default_weights_only_use_mlm_and_sop() -> None:
    weights = PretrainLossWeights()
    result = compose_total_loss(weights, mlm_loss=torch.tensor(2.0), sop_loss=torch.tensor(3.0))
    assert result.total.item() == pytest.approx(5.0)
    assert set(result.breakdown.keys()) == {"mlm", "sop"}


def test_weighted_sum_is_correct() -> None:
    weights = PretrainLossWeights(mlm=1.0, sop=0.5, time_reconstruction=2.0)
    result = compose_total_loss(
        weights,
        mlm_loss=torch.tensor(4.0),
        sop_loss=torch.tensor(2.0),
        time_reconstruction_loss=torch.tensor(1.0),
    )
    # 1*4 + 0.5*2 + 2*1 = 4 + 1 + 2 = 7
    assert result.total.item() == pytest.approx(7.0)


def test_missing_loss_for_positive_weight_raises() -> None:
    weights = PretrainLossWeights(time_text_contrastive=0.5)
    with pytest.raises(ValueError, match="time_text_contrastive"):
        compose_total_loss(weights, mlm_loss=torch.tensor(1.0), sop_loss=torch.tensor(1.0))


def test_zero_weight_does_not_require_loss_value() -> None:
    weights = PretrainLossWeights(mlm=1.0, sop=0.0)
    result = compose_total_loss(weights, mlm_loss=torch.tensor(3.0))  # sop_loss 안 줘도 됨
    assert result.total.item() == pytest.approx(3.0)
    assert "sop" not in result.breakdown


def test_all_zero_weights_raises() -> None:
    weights = PretrainLossWeights(mlm=0.0, sop=0.0)
    with pytest.raises(ValueError, match="all loss weights are 0"):
        compose_total_loss(weights)


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError):
        PretrainLossWeights(mlm=-1.0)


def test_breakdown_reports_unweighted_values() -> None:
    weights = PretrainLossWeights(mlm=10.0)
    result = compose_total_loss(weights, mlm_loss=torch.tensor(1.5), sop_loss=torch.tensor(999.0), )
    # sop 가중치가 기본 1.0(양수)이므로 sop_loss도 필요 - breakdown은 가중치 적용 전 원값이어야 함
    assert result.breakdown["mlm"] == pytest.approx(1.5)


def test_all_five_losses_combine() -> None:
    weights = PretrainLossWeights(
        mlm=1.0, sop=1.0, time_reconstruction=1.0, time_text_contrastive=1.0, cross_modal_matching=1.0
    )
    result = compose_total_loss(
        weights,
        mlm_loss=torch.tensor(1.0),
        sop_loss=torch.tensor(1.0),
        time_reconstruction_loss=torch.tensor(1.0),
        time_text_contrastive_loss=torch.tensor(1.0),
        cross_modal_matching_loss=torch.tensor(1.0),
    )
    assert result.total.item() == pytest.approx(5.0)
    assert set(result.breakdown.keys()) == {
        "mlm", "sop", "time_reconstruction", "time_text_contrastive", "cross_modal_matching",
    }
