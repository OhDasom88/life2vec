"""representation_tracking.capture 회귀 테스트 (합성 parquet, 실제 967K/캐시 데이터 불필요).

capture.activation_refs_from_stage_a_parquet가 실제
outputs/online2/v2_finetune_v02/event_embeddings/*.parquet(384차원
event_mean/event_max, 캐시된 55 case)에 대해서도 예외 없이 동작함을
대화형으로 확인했다(README 참조) — 여기서는 로직만 빠르게 검증한다.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_tracking.capture import (
    activation_refs_from_stage_a_parquet,
    load_activation_vector,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_tracking.schemas import (
    RepresentationKind,
)


@pytest.fixture()
def stage_a_parquet(tmp_path):
    table = pa.table(
        {
            "case_id": ["case_1", "case_1"],
            "event_id": ["event_a", "event_b"],
            "target_token_start": [0, 10],
            "target_token_end": [5, 15],
            "event_mean": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "event_max": [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
        }
    )
    path = tmp_path / "stage_a.parquet"
    pq.write_table(table, path)
    return path


def test_yields_mean_and_max_ref_per_row(stage_a_parquet) -> None:
    pairs = list(
        activation_refs_from_stage_a_parquet(stage_a_parquet, checkpoint_id="ckpt_v1")
    )
    assert len(pairs) == 2
    mean_ref, max_ref = pairs[0]
    assert mean_ref.pooling == "mean"
    assert max_ref.pooling == "max"
    assert mean_ref.kind is RepresentationKind.CONTEXTUAL_ACTIVATION
    assert mean_ref.checkpoint_id == "ckpt_v1"
    assert mean_ref.sequence_id == "case_1"
    assert mean_ref.position == "event_a:0-5"


def test_mean_and_max_ref_have_different_identity(stage_a_parquet) -> None:
    mean_ref, max_ref = next(
        activation_refs_from_stage_a_parquet(stage_a_parquet, checkpoint_id="ckpt_v1")
    )
    assert mean_ref.identity_key != max_ref.identity_key


def test_different_checkpoint_id_changes_identity(stage_a_parquet) -> None:
    mean_ref_a, _ = next(
        activation_refs_from_stage_a_parquet(stage_a_parquet, checkpoint_id="ckpt_a")
    )
    mean_ref_b, _ = next(
        activation_refs_from_stage_a_parquet(stage_a_parquet, checkpoint_id="ckpt_b")
    )
    assert mean_ref_a.identity_key != mean_ref_b.identity_key


def test_load_activation_vector_roundtrips(stage_a_parquet) -> None:
    mean_ref, max_ref = next(
        activation_refs_from_stage_a_parquet(stage_a_parquet, checkpoint_id="ckpt_v1")
    )
    assert load_activation_vector(mean_ref) == [1.0, 2.0, 3.0]
    assert load_activation_vector(max_ref) == [1.5, 2.5, 3.5]


def test_load_activation_vector_second_row(stage_a_parquet) -> None:
    pairs = list(
        activation_refs_from_stage_a_parquet(stage_a_parquet, checkpoint_id="ckpt_v1")
    )
    mean_ref, _ = pairs[1]
    assert load_activation_vector(mean_ref) == [4.0, 5.0, 6.0]


def test_load_activation_vector_rejects_bad_format() -> None:
    from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_tracking.schemas import (
        ActivationRef,
    )

    bad_ref = ActivationRef(
        kind=RepresentationKind.CONTEXTUAL_ACTIVATION,
        checkpoint_id="c",
        layer="l",
        sequence_id="s",
        position="p",
        pooling="mean",
        artifact_ref="no_hash_separator_here",
    )
    with pytest.raises(ValueError, match="unsupported artifact_ref format"):
        load_activation_vector(bad_ref)


def test_load_activation_vector_missing_event_raises(stage_a_parquet) -> None:
    mean_ref, _ = next(
        activation_refs_from_stage_a_parquet(stage_a_parquet, checkpoint_id="ckpt_v1")
    )
    from dataclasses import replace

    tampered = replace(
        mean_ref,
        artifact_ref=mean_ref.artifact_ref.replace("event_a", "event_does_not_exist"),
    )
    with pytest.raises(KeyError):
        load_activation_vector(tampered)
