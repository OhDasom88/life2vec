"""기존 Stage-A activation 캐시(``scripts/online2_v2/cache_stage_a_event_embeddings.py``의
``event_embeddings/*.parquet``)를 ``ActivationRef``로 감싸는 어댑터.

activation을 다시 계산하지 않는다 — 이미 캐시된 ``event_mean``/``event_max``
(384차원, mean/max pooled contextual activation)를 읽기만 한다. 실측 확인:
``outputs/online2/v2_finetune_v02/event_embeddings/*.parquet``에 실제로
55개 case 전부 캐시돼 있다(예: ``F354848_2025-03-25_2025-04-07.parquet``
3,928행, 컬럼 ``case_id/event_id/target_token_start/target_token_end/
event_mean/event_max`` 등).

이 parquet에 없는 것: ``checkpoint_id``. ``cache_stage_a_event_embeddings.py``는
체크포인트를 ``--ckpt`` 실행 인자로만 받고 컬럼에 남기지 않는다 — 그래서
호출자가 "이 parquet을 만든 실행에 쓰인 체크포인트가 뭐였는지"를 직접
넘겨야 한다. 이게 이 모듈이 실제로 메꾸는 구멍이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from .schemas import ActivationRef, RepresentationKind

_DEFAULT_LAYER = "stage_a_pooled_output"


def _position_of(row: dict) -> str:
    return f"{row['event_id']}:{row['target_token_start']}-{row['target_token_end']}"


def activation_refs_from_stage_a_parquet(
    parquet_path: Path, *, checkpoint_id: str, layer: str = _DEFAULT_LAYER
) -> Iterator[tuple[ActivationRef, ActivationRef]]:
    """parquet 한 행마다 (mean pooling ref, max pooling ref) 튜플을 yield한다.

    ``checkpoint_id``는 이 parquet을 만든 실행에서 실제로 쓰인 체크포인트를
    호출자가 알고 있어야 한다(예: ``cache_stage_a_event_embeddings.py --ckpt``에
    넘긴 값과 동일한 식별자). 잘못된 checkpoint_id를 넘겨도 이 함수는 검증할
    방법이 없다 — parquet 자체에 그 정보가 없기 때문이다(위 docstring 참조).
    """
    table = pq.read_table(
        parquet_path,
        columns=["case_id", "event_id", "target_token_start", "target_token_end"],
    )
    for row in table.to_pylist():
        position = _position_of(row)
        mean_ref = ActivationRef(
            kind=RepresentationKind.CONTEXTUAL_ACTIVATION,
            checkpoint_id=checkpoint_id,
            layer=layer,
            sequence_id=row["case_id"],
            position=position,
            pooling="mean",
            artifact_ref=f"{parquet_path}#{row['event_id']}:event_mean",
        )
        max_ref = ActivationRef(
            kind=RepresentationKind.CONTEXTUAL_ACTIVATION,
            checkpoint_id=checkpoint_id,
            layer=layer,
            sequence_id=row["case_id"],
            position=position,
            pooling="max",
            artifact_ref=f"{parquet_path}#{row['event_id']}:event_max",
        )
        yield mean_ref, max_ref


def load_activation_vector(activation_ref: ActivationRef) -> list[float]:
    """``ActivationRef.artifact_ref``(``path#event_id:column`` 형식)에서 실제 벡터를 읽는다.

    ``activation_refs_from_stage_a_parquet``가 만든 ref만 지원한다(다른 형식의
    ``artifact_ref``를 넣으면 ``ValueError``).
    """
    if "#" not in activation_ref.artifact_ref:
        raise ValueError(f"unsupported artifact_ref format: {activation_ref.artifact_ref!r}")
    path_part, locator = activation_ref.artifact_ref.rsplit("#", 1)
    event_id, column = locator.split(":", 1)
    if column not in {"event_mean", "event_max"}:
        raise ValueError(f"unsupported column in artifact_ref: {column!r}")

    table = pq.read_table(Path(path_part), columns=["event_id", column])
    for row in table.to_pylist():
        if row["event_id"] == event_id:
            return list(row[column])
    raise KeyError(f"event_id {event_id!r} not found in {path_part}")
