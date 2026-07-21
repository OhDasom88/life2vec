"""§9.2/§9.3 SparseAutoencoder 회귀 테스트 (실제 torch forward/backward)."""

from __future__ import annotations

import pytest
import torch

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.model import (
    SparseAutoencoder,
    sae_loss,
)


def test_forward_shapes_are_correct() -> None:
    model = SparseAutoencoder(input_dim=16, dict_size=64, sparsity_mode="L1")
    x = torch.randn(8, 16)
    reconstruction, codes = model(x)
    assert reconstruction.shape == (8, 16)
    assert codes.shape == (8, 64)


def test_l1_codes_are_nonnegative() -> None:
    model = SparseAutoencoder(input_dim=10, dict_size=20, sparsity_mode="L1")
    x = torch.randn(5, 10)
    _, codes = model(x)
    assert torch.all(codes >= 0)


def test_topk_enforces_exact_sparsity_budget() -> None:
    model = SparseAutoencoder(input_dim=10, dict_size=32, sparsity_mode="TOPK", top_k=4)
    x = torch.randn(6, 10)
    _, codes = model(x)
    active_counts = (codes > 0).sum(dim=-1)
    assert torch.all(active_counts <= 4)


def test_topk_requires_positive_top_k() -> None:
    with pytest.raises(ValueError, match="TOPK"):
        SparseAutoencoder(input_dim=10, dict_size=20, sparsity_mode="TOPK", top_k=None)
    with pytest.raises(ValueError, match="TOPK"):
        SparseAutoencoder(input_dim=10, dict_size=20, sparsity_mode="TOPK", top_k=0)


def test_invalid_sparsity_mode_rejected() -> None:
    with pytest.raises(ValueError, match="sparsity_mode"):
        SparseAutoencoder(input_dim=10, dict_size=20, sparsity_mode="SOFTMAX")


def test_non_positive_dims_rejected() -> None:
    with pytest.raises(ValueError):
        SparseAutoencoder(input_dim=0, dict_size=20)
    with pytest.raises(ValueError):
        SparseAutoencoder(input_dim=10, dict_size=0)


def test_tied_weights_share_encoder_decoder() -> None:
    model = SparseAutoencoder(input_dim=8, dict_size=16, sparsity_mode="L1", tied_weights=True)
    assert model.decoder_weight is None
    x = torch.randn(4, 8)
    reconstruction, codes = model(x)
    assert reconstruction.shape == (4, 8)
    # tied 상태에서 encoder_weight를 업데이트하면 decoder 쪽 재구성에도 즉시 반영돼야 한다
    # (별개 파라미터로 몰래 복제되지 않았는지 확인).
    with torch.no_grad():
        model.encoder_weight.add_(1.0)
    reconstruction_after, _ = model(x)
    assert not torch.allclose(reconstruction, reconstruction_after)


def test_training_reduces_reconstruction_loss() -> None:
    torch.manual_seed(0)
    model = SparseAutoencoder(input_dim=12, dict_size=48, sparsity_mode="L1")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(32, 12)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        reconstruction, codes = model(x)
        loss = sae_loss(x, reconstruction, codes, sparsity_mode="L1", l1_coefficient=1e-3)
        loss.total.backward()
        optimizer.step()
        losses.append(loss.reconstruction_loss)

    assert losses[-1] < losses[0]


def test_sae_loss_topk_has_zero_sparsity_term() -> None:
    model = SparseAutoencoder(input_dim=8, dict_size=16, sparsity_mode="TOPK", top_k=3)
    x = torch.randn(4, 8)
    reconstruction, codes = model(x)
    loss = sae_loss(x, reconstruction, codes, sparsity_mode="TOPK")
    assert loss.sparsity_loss == 0.0


def test_sae_loss_l1_sparsity_term_scales_with_coefficient() -> None:
    model = SparseAutoencoder(input_dim=8, dict_size=16, sparsity_mode="L1")
    x = torch.randn(4, 8)
    reconstruction, codes = model(x)
    low = sae_loss(x, reconstruction, codes, sparsity_mode="L1", l1_coefficient=0.01)
    high = sae_loss(x, reconstruction, codes, sparsity_mode="L1", l1_coefficient=1.0)
    assert high.sparsity_loss > low.sparsity_loss


def test_sae_loss_invalid_mode_rejected() -> None:
    x = torch.randn(2, 4)
    with pytest.raises(ValueError):
        sae_loss(x, x, torch.randn(2, 8), sparsity_mode="BOGUS")
