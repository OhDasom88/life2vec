"""§8.2 modality ablation 집계 회귀 테스트."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_tracking.attribution import (
    ModalityAblationRecord,
    find_dead_modalities,
    rank_modalities_by_contribution,
    summarize_modality_ablation,
)


def test_delta_and_absolute_delta() -> None:
    record = ModalityAblationRecord(
        modality="image", sequence_id="s1", prediction_with_modality=0.8, prediction_without_modality=0.5
    )
    assert record.delta == pytest.approx(0.3)
    assert record.absolute_delta == pytest.approx(0.3)


def test_absolute_delta_handles_negative_direction() -> None:
    record = ModalityAblationRecord(
        modality="image", sequence_id="s1", prediction_with_modality=0.2, prediction_without_modality=0.5
    )
    assert record.delta == pytest.approx(-0.3)
    assert record.absolute_delta == pytest.approx(0.3)


def test_summarize_groups_by_modality() -> None:
    records = [
        ModalityAblationRecord("image", "s1", 0.8, 0.5),
        ModalityAblationRecord("image", "s2", 0.6, 0.5),
        ModalityAblationRecord("text", "s1", 0.9, 0.89),
    ]
    summary = summarize_modality_ablation(records)
    assert summary["image"]["n"] == 2
    assert summary["image"]["mean_absolute_delta"] == pytest.approx((0.3 + 0.1) / 2)
    assert summary["text"]["n"] == 1
    assert summary["text"]["mean_absolute_delta"] == pytest.approx(0.01)


def test_summarize_empty_records_returns_empty_dict() -> None:
    assert summarize_modality_ablation([]) == {}


def test_find_dead_modalities_uses_threshold() -> None:
    summary = {
        "image": {"n": 5.0, "mean_delta": 0.0, "mean_absolute_delta": 1e-9},
        "text": {"n": 5.0, "mean_delta": 0.2, "mean_absolute_delta": 0.2},
    }
    assert find_dead_modalities(summary) == ["image"]
    assert find_dead_modalities(summary, threshold=0.0) == []


def test_rank_modalities_by_contribution_descending() -> None:
    summary = {
        "image": {"n": 1.0, "mean_delta": 0.0, "mean_absolute_delta": 0.1},
        "text": {"n": 1.0, "mean_delta": 0.0, "mean_absolute_delta": 0.5},
        "cube": {"n": 1.0, "mean_delta": 0.0, "mean_absolute_delta": 0.3},
    }
    ranking = rank_modalities_by_contribution(summary)
    assert [name for name, _ in ranking] == ["text", "cube", "image"]
