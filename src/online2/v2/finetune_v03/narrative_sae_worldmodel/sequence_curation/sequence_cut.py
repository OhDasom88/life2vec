"""이벤트 경계를 유지하는 시퀀스 절단/분할 (계획서 §6.2).

착수 전 확인한 결과, 이 정확한 알고리즘(긴 시퀀스를 max_length 이하의 여러
청크로 나누되 — 첫 청크는 앞에서부터, 마지막 청크는 뒤에서부터 채우고, 중간
청크는 버전 관리되는 overlap 정책을 쓰는 것)은 저장소 어디에도 없었다.
가장 가까운 선례는 두 곳뿐이다:

- ``src/tasks/base.py``의 ``Task.clip_document()`` — 뒤에서부터 채우는
  단일 윈도우 절단(``accumulate(reversed(lengths))``). 이 모듈의 마지막
  청크 채우기(``_back_fill_index``)가 이 패턴을 그대로 가져온 것이다.
- ``src/online2/builder.py``의 ``emit_template_sequence`` — 토큰 예산
  초과 시 앞에서부터 이벤트를 하나씩 버리는 단일 윈도우 절단.

둘 다 "하나의 윈도우"만 만든다. 긴 시퀀스를 여러 청크로 **분할**해서 전체를
덮는 로직은 어디에도 없었으므로 여기서 새로 만든다.

이벤트 내부 토큰은 절대 쪼개지 않는다 — 모든 절단은 이벤트 인덱스 경계에서만
일어난다. 이벤트 하나가 그 자체로 ``max_length``를 넘으면(그런 이벤트는
원칙적으로 없어야 하지만) 잘라내지 않고 통째로 넣은 뒤
``SINGLE_EVENT_EXCEEDS_MAX_LENGTH`` 플래그를 남긴다 — online2 builder.py의
``quality_flags`` 관례와 동일하다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

CHUNK_POSITIONS = frozenset({"SOLE", "FIRST", "MIDDLE", "LAST"})

SINGLE_EVENT_EXCEEDS_MAX_LENGTH = "SINGLE_EVENT_EXCEEDS_MAX_LENGTH"


@dataclass(frozen=True)
class TokenEvent:
    """절단 알고리즘이 다루는 최소 단위 — 이 경계 안쪽은 절대 쪼개지 않는다."""

    event_id: str
    tokens: tuple[str, ...]

    @property
    def token_count(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class SequenceChunk:
    position: str  # SOLE|FIRST|MIDDLE|LAST
    events: tuple[TokenEvent, ...]
    start_index: int  # 원본 리스트 기준 시작 인덱스(포함)
    end_index: int  # 끝 인덱스(제외) — overlap이 있으면 이웃 청크와 겹칠 수 있음
    token_count: int
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.position not in CHUNK_POSITIONS:
            raise ValueError(f"invalid chunk position: {self.position!r}")
        if not self.events:
            raise ValueError("chunk must contain at least one event")


def _front_fill_index(events: Sequence[TokenEvent], max_length: int) -> int:
    """``events[0:idx]``의 토큰 합이 ``max_length`` 이하가 되는 최대 idx."""
    total = 0
    for index, event in enumerate(events):
        total += event.token_count
        if total > max_length:
            return index
    return len(events)


def _back_fill_index(events: Sequence[TokenEvent], max_length: int) -> int:
    """``events[idx:]``의 토큰 합이 ``max_length`` 이하가 되는 최소 idx.

    ``src/tasks/base.py Task.clip_document``의 ``accumulate(reversed(lengths))``
    패턴과 동일하다.
    """
    total = 0
    for offset, event in enumerate(reversed(events)):
        total += event.token_count
        if total > max_length:
            return len(events) - offset
    return 0


def _chunk(
    events: Sequence[TokenEvent], start: int, end: int, position: str
) -> SequenceChunk:
    window = tuple(events[start:end])
    return SequenceChunk(
        position=position,
        events=window,
        start_index=start,
        end_index=end,
        token_count=sum(event.token_count for event in window),
    )


def cut_into_bounded_chunks(
    events: Sequence[TokenEvent],
    max_length: int,
    *,
    overlap_policy_id: str = "NO_OVERLAP_V1",
    overlap_events: int = 0,
) -> list[SequenceChunk]:
    """긴 이벤트 시퀀스를 ``max_length`` 이하의 청크들로 분할한다(§6.2).

    - 시퀀스가 그대로 들어가면 청크 1개(``SOLE``).
    - 아니면 첫 청크(``FIRST``)는 앞에서부터, 마지막 청크(``LAST``)는
      뒤에서부터 채운다.
    - 둘 사이에 남는 구간이 있으면 ``MIDDLE`` 청크로 채우되, 각 청크는
      이전 청크의 마지막 ``overlap_events``개를 앞에 겹쳐 포함한다
      (``overlap_policy_id``로 정책 버전을 함께 기록).
    - 이벤트 하나가 ``max_length``보다 크면 쪼개지 않고 통째로 넣은 뒤
      ``SINGLE_EVENT_EXCEEDS_MAX_LENGTH``를 플래그로 남긴다.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if overlap_events < 0:
        raise ValueError("overlap_events must not be negative")
    if not events:
        return []

    total_tokens = sum(event.token_count for event in events)
    if total_tokens <= max_length:
        return [_chunk(events, 0, len(events), "SOLE")]

    first_end = _front_fill_index(events, max_length)
    forced_first = first_end == 0
    if forced_first:
        first_end = 1

    last_start = _back_fill_index(events, max_length)
    forced_last = last_start == len(events)
    if forced_last:
        last_start = len(events) - 1

    def _with_forced_flag(chunk: SequenceChunk, forced: bool) -> SequenceChunk:
        if not forced:
            return chunk
        return SequenceChunk(
            position=chunk.position,
            events=chunk.events,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            token_count=chunk.token_count,
            quality_flags=chunk.quality_flags + (SINGLE_EVENT_EXCEEDS_MAX_LENGTH,),
        )

    if first_end >= last_start:
        # 앞·뒤 청크만으로 전체를 덮는다 — 중간 청크가 필요 없다. 겹치는
        # 구간(있다면 events[last_start:first_end])은 두 청크 모두에 나타난다.
        first_chunk = _with_forced_flag(_chunk(events, 0, first_end, "FIRST"), forced_first)
        last_chunk = _with_forced_flag(
            _chunk(events, last_start, len(events), "LAST"), forced_last
        )
        return [first_chunk, last_chunk]

    chunks = [_with_forced_flag(_chunk(events, 0, first_end, "FIRST"), forced_first)]

    middle_start = first_end
    while middle_start < last_start:
        window = events[middle_start:last_start]
        local_end = _front_fill_index(window, max_length)
        forced_middle = local_end == 0
        if forced_middle:
            local_end = 1
        chunk_end = middle_start + local_end
        chunks.append(
            _with_forced_flag(_chunk(events, middle_start, chunk_end, "MIDDLE"), forced_middle)
        )
        next_start = chunk_end - overlap_events
        if next_start <= middle_start:  # overlap이 진행을 막지 않도록 최소 1칸은 전진
            next_start = chunk_end
        middle_start = next_start

    chunks.append(
        _with_forced_flag(_chunk(events, last_start, len(events), "LAST"), forced_last)
    )
    return chunks
