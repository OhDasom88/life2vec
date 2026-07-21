"""§10 neighborhood preservation(trustworthiness) 회귀 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_explorer.neighborhood_preservation import (
    compute_neighborhood_preservation,
)


def test_identity_projection_has_perfect_trustworthiness() -> None:
    rng = np.random.default_rng(0)
    original = rng.normal(size=(30, 10))
    score = compute_neighborhood_preservation(original, original.copy(), n_neighbors=5)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_random_unrelated_projection_scores_lower_than_identity() -> None:
    rng = np.random.default_rng(1)
    original = rng.normal(size=(40, 20))
    identity_score = compute_neighborhood_preservation(original, original.copy(), n_neighbors=5)
    shuffled = original.copy()
    rng.shuffle(shuffled)  # 행 순서를 섞어 이웃 관계를 원공간과 무관하게 만듦
    shuffled_score = compute_neighborhood_preservation(original, shuffled, n_neighbors=5)
    assert shuffled_score < identity_score


def test_score_is_within_unit_interval() -> None:
    rng = np.random.default_rng(2)
    original = rng.normal(size=(25, 8))
    projected = rng.normal(size=(25, 3))
    score = compute_neighborhood_preservation(original, projected, n_neighbors=5)
    assert 0.0 <= score <= 1.0


def test_point_count_mismatch_raises() -> None:
    original = np.random.default_rng(3).normal(size=(10, 5))
    projected = np.random.default_rng(3).normal(size=(9, 3))
    with pytest.raises(ValueError, match="point count mismatch"):
        compute_neighborhood_preservation(original, projected, n_neighbors=3)


def test_too_few_points_for_n_neighbors_raises() -> None:
    original = np.random.default_rng(4).normal(size=(4, 5))
    projected = np.random.default_rng(4).normal(size=(4, 3))
    with pytest.raises(ValueError, match="n_neighbors"):
        compute_neighborhood_preservation(original, projected, n_neighbors=5)
