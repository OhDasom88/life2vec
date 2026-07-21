"""§10 ProjectionMetadata/ProjectionArtifact 강제 규칙 회귀 테스트."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_explorer.metadata import (
    DISCLAIMER,
    ProjectionArtifact,
    ProjectionMetadata,
)


def _metadata(**overrides) -> ProjectionMetadata:
    fields = dict(
        original_dimensionality=384,
        method="UMAP",
        params={"n_neighbors": 15, "min_dist": 0.1},
        seed=0,
        checkpoint_id="ckpt_v1",
        layer="stage_a_pooled_output",
        color_label_source="split",
        neighborhood_preservation=0.8,
    )
    fields.update(overrides)
    return ProjectionMetadata(**fields)


def test_valid_metadata_constructs() -> None:
    meta = _metadata()
    assert meta.method == "UMAP"


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("original_dimensionality", 0),
        ("method", ""),
        ("checkpoint_id", ""),
        ("layer", ""),
        ("color_label_source", ""),
        ("neighborhood_preservation", 1.5),
        ("neighborhood_preservation", -0.1),
    ],
)
def test_missing_or_invalid_required_field_rejected(field_name: str, bad_value) -> None:
    with pytest.raises(ValueError):
        _metadata(**{field_name: bad_value})


def test_projection_artifact_requires_matching_lengths() -> None:
    meta = _metadata()
    with pytest.raises(ValueError, match="length mismatch"):
        ProjectionArtifact(
            metadata=meta,
            point_ids=("a", "b"),
            coordinates=((0.0, 0.0, 0.0),),
        )


def test_projection_artifact_requires_three_dimensional_coordinates() -> None:
    meta = _metadata()
    with pytest.raises(ValueError, match="3 components"):
        ProjectionArtifact(
            metadata=meta,
            point_ids=("a",),
            coordinates=((0.0, 0.0),),
        )


def test_projection_artifact_disclaimer_cannot_be_overridden() -> None:
    meta = _metadata()
    with pytest.raises(ValueError, match="disclaimer"):
        ProjectionArtifact(
            metadata=meta,
            point_ids=("a",),
            coordinates=((0.0, 0.0, 0.0),),
            disclaimer="이 projection은 개념을 확정적으로 보여준다",
        )


def test_projection_artifact_valid_construction() -> None:
    meta = _metadata()
    artifact = ProjectionArtifact(
        metadata=meta,
        point_ids=("a", "b"),
        coordinates=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )
    assert artifact.disclaimer == DISCLAIMER
    assert len(artifact.point_ids) == 2
