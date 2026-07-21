"""§9.3 dead feature resampling 회귀 테스트."""

from __future__ import annotations

import pytest
import torch

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.evaluation import (
    compute_activation_frequency,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.model import SparseAutoencoder
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.training import (
    resample_dead_features,
)


def test_no_dead_features_returns_zero() -> None:
    model = SparseAutoencoder(input_dim=8, dict_size=16, sparsity_mode="L1")
    frequency = torch.ones(16)  # 전부 항상 활성화 -> 죽은 게 없음
    assert resample_dead_features(model, frequency) == 0


def test_dead_features_get_reinitialized() -> None:
    torch.manual_seed(0)
    model = SparseAutoencoder(input_dim=8, dict_size=16, sparsity_mode="L1")
    before_encoder = model.encoder_weight.detach().clone()
    before_decoder = model.decoder_weight.detach().clone()

    frequency = torch.ones(16)
    frequency[[2, 5, 9]] = 0.0  # 세 feature가 죽어있다고 가정

    n_resampled = resample_dead_features(model, frequency)
    assert n_resampled == 3

    # 죽은 feature의 가중치만 바뀌어야 한다.
    for i in range(16):
        changed_encoder = not torch.allclose(model.encoder_weight[i], before_encoder[i])
        changed_decoder = not torch.allclose(model.decoder_weight[:, i], before_decoder[:, i])
        if i in {2, 5, 9}:
            assert changed_encoder and changed_decoder
        else:
            assert not changed_encoder and not changed_decoder


def test_resampled_features_become_reusable() -> None:
    # resample 후 해당 feature가 다시 활성화될 수 있어야 한다(0으로 죽여둔 게 아니라
    # 재초기화한 것이므로) - encoder_bias가 0이 아닌 임의의 큰 입력을 넣어 확인.
    torch.manual_seed(1)
    model = SparseAutoencoder(input_dim=8, dict_size=16, sparsity_mode="L1")
    frequency = torch.ones(16)
    frequency[3] = 0.0
    resample_dead_features(model, frequency)

    x = torch.randn(200, 8) * 5  # 다양한 입력으로 최소 한 번은 feature 3이 켜지길 기대
    _, codes = model(x)
    assert (codes[:, 3] > 0).any()


def test_threshold_controls_which_features_are_dead() -> None:
    model = SparseAutoencoder(input_dim=4, dict_size=8, sparsity_mode="L1")
    frequency = torch.tensor([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    # threshold=0.05면 0.0/0.01/0.05 세 개가 "죽음" 판정(<=)
    n_resampled = resample_dead_features(model, frequency, threshold=0.05)
    assert n_resampled == 3


def test_shape_mismatch_raises() -> None:
    model = SparseAutoencoder(input_dim=4, dict_size=8, sparsity_mode="L1")
    with pytest.raises(ValueError, match="shape"):
        resample_dead_features(model, torch.ones(5))


def test_resample_integrates_with_activation_frequency_computation() -> None:
    torch.manual_seed(2)
    model = SparseAutoencoder(input_dim=6, dict_size=10, sparsity_mode="L1")
    # 절반 정도가 절대 안 켜지도록, 입력을 encoder bias에 비해 매우 작게 만든다.
    x = torch.zeros(50, 6)
    _, codes = model(x)
    frequency = compute_activation_frequency(codes)
    n_resampled = resample_dead_features(model, frequency)
    assert n_resampled >= 0  # 그냥 예외 없이 끝까지 도는지 확인(실제 죽은 개수는 초기화에 따라 다름)
