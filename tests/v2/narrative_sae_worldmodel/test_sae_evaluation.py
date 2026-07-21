"""§9.4 SAE 평가(복원/희소성/downstream fidelity) 회귀 테스트."""

from __future__ import annotations

import pytest
import torch

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.evaluation import (
    compute_activation_frequency,
    downstream_fidelity,
    reconstruction_metrics,
    sparsity_metrics,
)


def test_perfect_reconstruction_has_zero_mse_and_full_explained_variance() -> None:
    x = torch.randn(20, 8)
    metrics = reconstruction_metrics(x, x.clone())
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["explained_variance"] == pytest.approx(1.0, abs=1e-4)


def test_reconstruction_metrics_worsen_with_noise() -> None:
    x = torch.randn(50, 8)
    slightly_off = x + 0.01 * torch.randn(50, 8)
    very_off = x + 2.0 * torch.randn(50, 8)
    slight = reconstruction_metrics(x, slightly_off)
    bad = reconstruction_metrics(x, very_off)
    assert slight["mse"] < bad["mse"]
    assert slight["explained_variance"] > bad["explained_variance"]


def test_activation_frequency_shape_and_range() -> None:
    codes = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [1.0, 0.0, 0.0]])
    frequency = compute_activation_frequency(codes)
    assert frequency.shape == (3,)
    assert torch.allclose(frequency, torch.tensor([2 / 3, 0.0, 2 / 3]))


def test_sparsity_metrics_reports_dead_features() -> None:
    codes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )  # feature 0만 항상 켜지고 1,2는 죽어있음
    metrics = sparsity_metrics(codes)
    assert metrics["mean_l0"] == pytest.approx(1.0)
    assert metrics["dead_feature_ratio"] == pytest.approx(2 / 3)


def test_sparsity_metrics_no_dead_features() -> None:
    codes = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    metrics = sparsity_metrics(codes)
    assert metrics["dead_feature_ratio"] == 0.0


def test_downstream_fidelity_zero_for_identical_outputs() -> None:
    output = torch.randn(10)
    result = downstream_fidelity(output, output.clone())
    assert result["mean_absolute_delta"] == pytest.approx(0.0, abs=1e-6)
    assert result["correlation"] == pytest.approx(1.0, abs=1e-4)


def test_downstream_fidelity_nonzero_for_different_outputs() -> None:
    original = torch.tensor([1.0, 2.0, 3.0, 4.0])
    reconstructed = torch.tensor([1.0, 2.5, 2.0, 5.0])
    result = downstream_fidelity(original, reconstructed)
    assert result["mean_absolute_delta"] > 0


def test_downstream_fidelity_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        downstream_fidelity(torch.randn(4), torch.randn(5))
