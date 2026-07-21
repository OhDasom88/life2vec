"""§8.3 ActivationRef 타입 회귀 테스트."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_tracking.schemas import (
    ActivationRef,
    RepresentationKind,
    sequence_position_ref,
    token_embedding_ref,
)


def test_token_embedding_ref_uses_none_pooling() -> None:
    ref = token_embedding_ref(vocab_version="v2", token_id="IN_TEMP|b03")
    assert ref.kind is RepresentationKind.TOKEN_EMBEDDING
    assert ref.pooling == "none"
    assert ref.checkpoint_id == "v2"


def test_sequence_position_ref_uses_none_pooling() -> None:
    ref = sequence_position_ref(sequence_id="seq_1", position_index=3)
    assert ref.kind is RepresentationKind.SEQUENCE_POSITION
    assert ref.position == "3"


def test_token_embedding_with_pooling_rejected() -> None:
    with pytest.raises(ValueError, match="TOKEN_EMBEDDING"):
        ActivationRef(
            kind=RepresentationKind.TOKEN_EMBEDDING,
            checkpoint_id="v2",
            layer="vocab_embedding_table",
            sequence_id="",
            position="tok",
            pooling="mean",
            artifact_ref="x",
        )


def test_sequence_position_with_pooling_rejected() -> None:
    with pytest.raises(ValueError, match="SEQUENCE_POSITION"):
        ActivationRef(
            kind=RepresentationKind.SEQUENCE_POSITION,
            checkpoint_id="",
            layer="",
            sequence_id="seq_1",
            position="0",
            pooling="max",
            artifact_ref="x",
        )


def test_invalid_pooling_rejected() -> None:
    with pytest.raises(ValueError, match="invalid pooling"):
        ActivationRef(
            kind=RepresentationKind.CONTEXTUAL_ACTIVATION,
            checkpoint_id="ckpt1",
            layer="final",
            sequence_id="seq_1",
            position="e1:0-4",
            pooling="sum",  # 지원 안 하는 값
            artifact_ref="x",
        )


def test_contextual_activation_allows_pooling() -> None:
    ref = ActivationRef(
        kind=RepresentationKind.CONTEXTUAL_ACTIVATION,
        checkpoint_id="ckpt1",
        layer="stage_a_final",
        sequence_id="seq_1",
        position="e1:0-4",
        pooling="mean",
        artifact_ref="path.parquet#event_mean",
    )
    assert ref.pooling == "mean"


def test_identity_key_differs_when_checkpoint_differs() -> None:
    base = dict(
        kind=RepresentationKind.CONTEXTUAL_ACTIVATION,
        layer="final",
        sequence_id="seq_1",
        position="e1:0-4",
        pooling="mean",
        artifact_ref="x",
    )
    ref_a = ActivationRef(checkpoint_id="ckpt_a", **base)
    ref_b = ActivationRef(checkpoint_id="ckpt_b", **base)
    # 계획서 §8.3: 같은 토큰이라도 checkpoint가 다르면 별도 activation.
    assert ref_a.identity_key != ref_b.identity_key


def test_identity_key_differs_when_layer_differs() -> None:
    base = dict(
        kind=RepresentationKind.CONTEXTUAL_ACTIVATION,
        checkpoint_id="ckpt1",
        sequence_id="seq_1",
        position="e1:0-4",
        pooling="mean",
        artifact_ref="x",
    )
    ref_a = ActivationRef(layer="layer_6", **base)
    ref_b = ActivationRef(layer="layer_12", **base)
    assert ref_a.identity_key != ref_b.identity_key


def test_identity_key_same_for_identical_fields() -> None:
    kwargs = dict(
        kind=RepresentationKind.CONTEXTUAL_ACTIVATION,
        checkpoint_id="ckpt1",
        layer="final",
        sequence_id="seq_1",
        position="e1:0-4",
        pooling="mean",
        artifact_ref="different_path_does_not_matter",
    )
    ref_a = ActivationRef(**kwargs)
    kwargs2 = dict(kwargs, artifact_ref="another_path")
    ref_b = ActivationRef(**kwargs2)
    # artifact_ref(저장 위치)는 정체성에 안 들어간다 - 같은 activation을 다른
    # 곳에 다시 저장해도 논리적으로는 같은 것.
    assert ref_a.identity_key == ref_b.identity_key
