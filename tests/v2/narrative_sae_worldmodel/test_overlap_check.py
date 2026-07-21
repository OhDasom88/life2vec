"""§10 split overlap(projection 공간) 회귀 테스트."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_explorer.metadata import (
    ProjectionArtifact,
    ProjectionMetadata,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_explorer.overlap_check import (
    split_overlap_in_projection_space,
)


def _metadata() -> ProjectionMetadata:
    return ProjectionMetadata(
        original_dimensionality=8,
        method="PCA",
        params={},
        seed=0,
        checkpoint_id="ckpt",
        layer="final",
        color_label_source="split",
        neighborhood_preservation=0.9,
    )


def _artifact(coordinates: list[tuple[float, float, float]]) -> ProjectionArtifact:
    point_ids = tuple(f"p{i}" for i in range(len(coordinates)))
    return ProjectionArtifact(metadata=_metadata(), point_ids=point_ids, coordinates=tuple(coordinates))


def test_perfectly_separated_clusters_have_zero_overlap() -> None:
    # 두 클러스터가 아주 멀리 떨어져 있고, 각 클러스터가 정확히 하나의 split.
    train_cluster = [(0.0 + i * 0.01, 0.0, 0.0) for i in range(6)]
    holdout_cluster = [(100.0 + i * 0.01, 0.0, 0.0) for i in range(6)]
    artifact = _artifact(train_cluster + holdout_cluster)
    split_by_id = {
        **{f"p{i}": "TRAIN" for i in range(6)},
        **{f"p{i}": "HOLDOUT" for i in range(6, 12)},
    }
    result = split_overlap_in_projection_space(artifact, split_by_id, k=3)
    assert result["TRAIN"]["mean_other_split_neighbor_fraction"] == pytest.approx(0.0)
    assert result["HOLDOUT"]["mean_other_split_neighbor_fraction"] == pytest.approx(0.0)


def test_interleaved_points_have_high_overlap() -> None:
    # 두 split이 한 줄에 번갈아 배치됨 -> 이웃 대부분이 다른 split.
    coordinates = [(float(i), 0.0, 0.0) for i in range(12)]
    split_by_id = {f"p{i}": ("TRAIN" if i % 2 == 0 else "HOLDOUT") for i in range(12)}
    artifact = _artifact(coordinates)
    result = split_overlap_in_projection_space(artifact, split_by_id, k=2)
    assert result["TRAIN"]["mean_other_split_neighbor_fraction"] > 0.5
    assert result["HOLDOUT"]["mean_other_split_neighbor_fraction"] > 0.5


def test_missing_split_assignment_raises() -> None:
    artifact = _artifact([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)])
    with pytest.raises(ValueError, match="missing entries"):
        split_overlap_in_projection_space(artifact, {"p0": "TRAIN"}, k=1)


def test_k_too_large_for_point_count_raises() -> None:
    artifact = _artifact([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
    split_by_id = {"p0": "TRAIN", "p1": "HOLDOUT"}
    with pytest.raises(ValueError, match="need more than"):
        split_overlap_in_projection_space(artifact, split_by_id, k=5)


def test_result_reports_n_per_split() -> None:
    coordinates = [(float(i), 0.0, 0.0) for i in range(8)]
    split_by_id = {f"p{i}": ("TRAIN" if i < 5 else "HOLDOUT") for i in range(8)}
    artifact = _artifact(coordinates)
    result = split_overlap_in_projection_space(artifact, split_by_id, k=2)
    assert result["TRAIN"]["n"] == 5.0
    assert result["HOLDOUT"]["n"] == 3.0
