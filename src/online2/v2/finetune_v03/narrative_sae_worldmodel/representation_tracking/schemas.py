"""§8.3 표현 위치 4종을 혼동하지 않는 타입 (계획서 §4.1 ``ActivationRef``).

착수 전 조사 결과, 이 4종 구분의 재료가 되는 하위 primitive는 이미 있었다:

- ``src/transformer/transformer.py``의 ``get_sequence_embedding()``이
  ``TOKEN_EMBEDDING``(고정 vocab lookup)을, ``forward_finetuning()``/
  ``forward_finetuning_with_embeddings()``가 ``CONTEXTUAL_ACTIVATION``(문맥별
  hidden state)을 이미 분리해서 계산한다 — 이 모듈이 그 구분을 재발명하지
  않고 타입으로만 강제한다.
- ``scripts/online2_v2/cache_stage_a_event_embeddings.py``가
  ``CONTEXTUAL_ACTIVATION``을 이미 pooling(``event_mean``/``event_max``)해서
  parquet로 캐시하고 있다. 다만 checkpoint/layer는 데이터 컬럼이 아니라
  실행 인자(``--ckpt``)에 박혀 있어서 기록되지 않는다 — 이게 이 모듈이 실제로
  메꾸는 구멍이다(``capture.py`` 참조).

여기서 새로 만드는 건 activation 재계산이 아니라, "이게 어느 종류의
표현이고 어느 (checkpoint, layer, sequence, position)에서 나왔는지"를 잘못
섞을 수 없게 만드는 타입이다. 계획서 §8.3 원문: "동일 토큰이라도 sequence,
checkpoint, layer가 다르면 별도 activation으로 기록한다."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.online2.canonical import stable_id

POOLING_KINDS = frozenset({"none", "mean", "max", "mean_max"})


class RepresentationKind(str, Enum):
    SEQUENCE_POSITION = "SEQUENCE_POSITION"
    TOKEN_EMBEDDING = "TOKEN_EMBEDDING"
    CONTEXTUAL_ACTIVATION = "CONTEXTUAL_ACTIVATION"
    SAE_FEATURE_ACTIVATION = "SAE_FEATURE_ACTIVATION"  # ../sae/가 아직 없어 생성처는 없음, 스키마만 대비


@dataclass(frozen=True)
class ActivationRef:
    """실제 activation 값을 들고 다니지 않는다 — ``artifact_ref``로 저장 위치만 가리킨다.

    (checkpoint, layer, sequence, position, pooling)의 해시가 정체성이다 —
    이 다섯 중 하나라도 다르면 "동일 토큰"이라도 다른 ``ActivationRef``다.
    """

    kind: RepresentationKind
    checkpoint_id: str  # TOKEN_EMBEDDING이면 vocab version, 그 외엔 model checkpoint 식별자
    layer: str
    sequence_id: str
    position: str  # 예: "event_id:token_start-token_end"
    pooling: str
    artifact_ref: str  # 값이 실제로 저장된 곳(parquet 경로#컬럼 등)

    def __post_init__(self) -> None:
        if self.pooling not in POOLING_KINDS:
            raise ValueError(f"invalid pooling: {self.pooling!r}")
        if self.kind is RepresentationKind.TOKEN_EMBEDDING and self.pooling != "none":
            # 고정 vocab 벡터는 pooling 대상이 아니다 - "pooled activation을
            # token embedding으로 착각"하는 버그를 타입 레벨에서 막는다.
            raise ValueError("TOKEN_EMBEDDING must use pooling='none' (it is not pooled)")
        if self.kind is RepresentationKind.SEQUENCE_POSITION and self.pooling != "none":
            raise ValueError("SEQUENCE_POSITION must use pooling='none' (it is an index, not a value)")

    @property
    def identity_key(self) -> str:
        """(kind, checkpoint, layer, sequence, position, pooling) 전체의 해시."""
        return stable_id(
            "activation_ref",
            {
                "kind": self.kind.value,
                "checkpoint_id": self.checkpoint_id,
                "layer": self.layer,
                "sequence_id": self.sequence_id,
                "position": self.position,
                "pooling": self.pooling,
            },
        )


def token_embedding_ref(*, vocab_version: str, token_id: str) -> ActivationRef:
    """고정 vocab embedding table 조회 — ``get_sequence_embedding()``이 계산하는 것."""
    return ActivationRef(
        kind=RepresentationKind.TOKEN_EMBEDDING,
        checkpoint_id=vocab_version,
        layer="vocab_embedding_table",
        sequence_id="",  # 문맥과 무관한 고정 벡터라 특정 시퀀스에 속하지 않는다
        position=token_id,
        pooling="none",
        artifact_ref=f"vocab:{vocab_version}#{token_id}",
    )


def sequence_position_ref(*, sequence_id: str, position_index: int) -> ActivationRef:
    """시퀀스 내 위치 — 값이 아니라 인덱스 그 자체다."""
    return ActivationRef(
        kind=RepresentationKind.SEQUENCE_POSITION,
        checkpoint_id="",
        layer="",
        sequence_id=sequence_id,
        position=str(position_index),
        pooling="none",
        artifact_ref=f"sequence:{sequence_id}#position:{position_index}",
    )
