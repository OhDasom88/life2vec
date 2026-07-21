"""§7.2 신규 손실 3종(time reconstruction, time-text contrastive, cross-modal) 회귀 테스트."""

from __future__ import annotations

import pytest
import torch

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.multimodal_pretrain.losses import (
    cross_modal_matching_loss,
    multi_positive_info_nce,
    time_reconstruction_loss,
    time_text_contrastive_loss,
)


def test_time_reconstruction_loss_zero_when_perfect() -> None:
    values = torch.randn(8)
    assert time_reconstruction_loss(values, values.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_time_reconstruction_loss_matches_manual_mse() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.0, 4.0])
    expected = ((pred - target) ** 2).mean()
    assert time_reconstruction_loss(pred, target).item() == pytest.approx(expected.item())


def test_time_reconstruction_loss_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        time_reconstruction_loss(torch.randn(3), torch.randn(4))


def test_time_reconstruction_loss_respects_valid_mask() -> None:
    pred = torch.tensor([10.0, 1.0, 10.0])
    target = torch.tensor([0.0, 1.0, 0.0])  # 앞뒤는 카테고리컬(유효하지 않음), 가운데만 유효
    mask = torch.tensor([False, True, False])
    assert time_reconstruction_loss(pred, target, valid_mask=mask).item() == pytest.approx(0.0, abs=1e-6)


def test_time_reconstruction_loss_all_invalid_returns_zero() -> None:
    pred = torch.randn(4)
    target = torch.randn(4)
    mask = torch.zeros(4, dtype=torch.bool)
    assert time_reconstruction_loss(pred, target, valid_mask=mask).item() == 0.0


def test_info_nce_prefers_aligned_embeddings() -> None:
    torch.manual_seed(0)
    n = 6
    anchors = torch.randn(n, 16)
    # candidates를 anchor와 거의 동일하게(대각선이 강한 양성) 만들면 손실이 작아야 한다.
    aligned_candidates = anchors + 0.01 * torch.randn(n, 16)
    misaligned_candidates = torch.randn(n, 16)
    positive_mask = torch.eye(n, dtype=torch.bool)

    aligned_loss = multi_positive_info_nce(anchors, aligned_candidates, positive_mask)
    misaligned_loss = multi_positive_info_nce(anchors, misaligned_candidates, positive_mask)
    assert aligned_loss.item() < misaligned_loss.item()


def test_info_nce_multi_positive_allows_several_true_per_row() -> None:
    anchors = torch.randn(4, 8)
    candidates = torch.randn(5, 8)
    positive_mask = torch.zeros(4, 5, dtype=torch.bool)
    positive_mask[0, [1, 2]] = True
    positive_mask[1, 0] = True
    positive_mask[2, [3, 4]] = True
    # 3번 앵커는 양성이 없음 -> 제외되어야 함(에러 없이).
    loss = multi_positive_info_nce(anchors, candidates, positive_mask)
    assert torch.isfinite(loss)


def test_info_nce_no_positives_anywhere_returns_zero() -> None:
    anchors = torch.randn(3, 4)
    candidates = torch.randn(3, 4)
    positive_mask = torch.zeros(3, 3, dtype=torch.bool)
    loss = multi_positive_info_nce(anchors, candidates, positive_mask)
    assert loss.item() == 0.0


def test_info_nce_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        multi_positive_info_nce(torch.randn(3, 4), torch.randn(5, 4), torch.zeros(3, 5, dtype=torch.bool)[:2])


def test_info_nce_sample_weights_downweight_low_quality_pairs() -> None:
    torch.manual_seed(1)
    anchors = torch.randn(4, 8)
    candidates = torch.randn(4, 8)
    positive_mask = torch.eye(4, dtype=torch.bool)
    uniform_weights = torch.ones(4)
    skewed_weights = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 사실상 첫 anchor만 반영
    loss_uniform = multi_positive_info_nce(anchors, candidates, positive_mask, sample_weights=uniform_weights)
    loss_skewed = multi_positive_info_nce(anchors, candidates, positive_mask, sample_weights=skewed_weights)
    assert not torch.isclose(loss_uniform, loss_skewed)


def test_time_text_contrastive_loss_is_info_nce_alias() -> None:
    anchors = torch.randn(3, 4)
    candidates = torch.randn(3, 4)
    positive_mask = torch.eye(3, dtype=torch.bool)
    direct = multi_positive_info_nce(anchors, candidates, positive_mask, temperature=0.1)
    via_alias = time_text_contrastive_loss(anchors, candidates, positive_mask, temperature=0.1)
    assert torch.isclose(direct, via_alias)


def test_cross_modal_matching_loss_averages_available_pairs() -> None:
    torch.manual_seed(2)
    n = 4
    positive_mask = torch.eye(n, dtype=torch.bool)
    embeddings = {
        "time": torch.randn(n, 8),
        "text": torch.randn(n, 8),
        "image": torch.randn(n, 8),
    }
    loss = cross_modal_matching_loss(embeddings, positive_mask)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_cross_modal_matching_loss_skips_missing_modalities() -> None:
    n = 3
    positive_mask = torch.eye(n, dtype=torch.bool)
    embeddings = {"time": torch.randn(n, 4), "text": torch.randn(n, 4)}
    # cube가 없는데 cube가 낀 쌍을 요청하면 그 쌍은 건너뛰고 나머지로만 계산돼야 한다.
    loss = cross_modal_matching_loss(
        embeddings, positive_mask, modality_pairs=[("time", "text"), ("time", "cube")]
    )
    expected = multi_positive_info_nce(embeddings["time"], embeddings["text"], positive_mask)
    assert torch.isclose(loss, expected)


def test_cross_modal_matching_loss_no_available_pairs_returns_zero() -> None:
    n = 2
    positive_mask = torch.eye(n, dtype=torch.bool)
    embeddings = {"time": torch.randn(n, 4)}
    loss = cross_modal_matching_loss(embeddings, positive_mask, modality_pairs=[("time", "cube")])
    assert loss.item() == 0.0
