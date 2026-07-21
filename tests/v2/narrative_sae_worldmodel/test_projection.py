"""§10 project_activations 회귀 테스트 (실제 scikit-learn/umap-learn 사용)."""

from __future__ import annotations

import numpy as np
import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_explorer.projection import (
    project_activations,
)


def _synthetic_activations(n: int = 40, dim: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float64)


def test_pca_projection_produces_three_dimensional_coordinates() -> None:
    activations = _synthetic_activations()
    point_ids = [f"p{i}" for i in range(activations.shape[0])]
    artifact = project_activations(
        activations,
        point_ids,
        method="PCA",
        checkpoint_id="ckpt_v1",
        layer="stage_a_pooled_output",
        color_label_source="split",
    )
    assert len(artifact.coordinates) == activations.shape[0]
    assert all(len(c) == 3 for c in artifact.coordinates)
    assert artifact.metadata.method == "PCA"
    assert artifact.metadata.original_dimensionality == 16


def test_umap_projection_produces_three_dimensional_coordinates() -> None:
    activations = _synthetic_activations(n=30, dim=12)
    point_ids = [f"p{i}" for i in range(activations.shape[0])]
    artifact = project_activations(
        activations,
        point_ids,
        method="UMAP",
        checkpoint_id="ckpt_v1",
        layer="stage_a_pooled_output",
        color_label_source="split",
        method_params={"n_neighbors": 5, "min_dist": 0.05},
    )
    assert artifact.metadata.method == "UMAP"
    assert artifact.metadata.params == {"n_neighbors": 5, "min_dist": 0.05}
    assert len(artifact.coordinates) == 30


def test_neighborhood_preservation_is_populated_and_valid() -> None:
    activations = _synthetic_activations()
    point_ids = [f"p{i}" for i in range(activations.shape[0])]
    artifact = project_activations(
        activations,
        point_ids,
        method="PCA",
        checkpoint_id="ckpt_v1",
        layer="final",
        color_label_source="split",
    )
    assert 0.0 <= artifact.metadata.neighborhood_preservation <= 1.0


def test_seed_reproducibility_for_pca() -> None:
    activations = _synthetic_activations()
    point_ids = [f"p{i}" for i in range(activations.shape[0])]
    kwargs = dict(
        method="PCA", checkpoint_id="ckpt_v1", layer="final", color_label_source="split", seed=42
    )
    first = project_activations(activations, point_ids, **kwargs)
    second = project_activations(activations, point_ids, **kwargs)
    assert first.coordinates == second.coordinates


def test_unsupported_method_rejected() -> None:
    activations = _synthetic_activations(n=10, dim=4)
    point_ids = [f"p{i}" for i in range(10)]
    with pytest.raises(ValueError, match="unsupported method"):
        project_activations(
            activations, point_ids, method="TSNE", checkpoint_id="c", layer="l", color_label_source="s"
        )


def test_point_id_length_mismatch_rejected() -> None:
    activations = _synthetic_activations(n=10, dim=4)
    with pytest.raises(ValueError, match="length mismatch"):
        project_activations(
            activations,
            ["p0", "p1"],  # 10개인데 2개만 줌
            method="PCA",
            checkpoint_id="c",
            layer="l",
            color_label_source="s",
        )


def test_metadata_checkpoint_and_layer_recorded() -> None:
    activations = _synthetic_activations(n=10, dim=4)
    point_ids = [f"p{i}" for i in range(10)]
    artifact = project_activations(
        activations,
        point_ids,
        method="PCA",
        checkpoint_id="ckpt_abc",
        layer="layer_6",
        color_label_source="narrative_template_id",
        trustworthiness_n_neighbors=2,  # n=10이라 기본 5는 sklearn 제약(< n/2) 위반
    )
    assert artifact.metadata.checkpoint_id == "ckpt_abc"
    assert artifact.metadata.layer == "layer_6"
    assert artifact.metadata.color_label_source == "narrative_template_id"
