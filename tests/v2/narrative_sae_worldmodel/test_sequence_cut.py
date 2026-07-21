"""§6.2 이벤트 경계 유지 시퀀스 절단 회귀 테스트."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sequence_curation.sequence_cut import (
    SINGLE_EVENT_EXCEEDS_MAX_LENGTH,
    SequenceChunk,
    TokenEvent,
    cut_into_bounded_chunks,
)


def _events(*sizes: int) -> list[TokenEvent]:
    return [TokenEvent(event_id=f"e{i}", tokens=tuple(f"t{i}_{j}" for j in range(n))) for i, n in enumerate(sizes)]


def _all_tokens(chunks: list[SequenceChunk]) -> list[str]:
    result = []
    for chunk in chunks:
        for event in chunk.events:
            result.extend(event.tokens)
    return result


def test_empty_events_returns_empty() -> None:
    assert cut_into_bounded_chunks([], max_length=10) == []


def test_fits_in_one_chunk_returns_sole() -> None:
    events = _events(3, 4, 2)
    chunks = cut_into_bounded_chunks(events, max_length=20)
    assert len(chunks) == 1
    assert chunks[0].position == "SOLE"
    assert chunks[0].token_count == 9
    assert chunks[0].start_index == 0
    assert chunks[0].end_index == 3


def test_exact_fit_boundary_is_sole() -> None:
    events = _events(5, 5)
    chunks = cut_into_bounded_chunks(events, max_length=10)
    assert len(chunks) == 1
    assert chunks[0].position == "SOLE"


def test_two_chunks_no_middle_needed() -> None:
    # 4개 이벤트, 각 3토큰(총 12). max_length=6 -> 앞 2개, 뒤 2개로 정확히 나뉨.
    events = _events(3, 3, 3, 3)
    chunks = cut_into_bounded_chunks(events, max_length=6)
    assert [c.position for c in chunks] == ["FIRST", "LAST"]
    assert chunks[0].start_index == 0 and chunks[0].end_index == 2
    assert chunks[1].start_index == 2 and chunks[1].end_index == 4
    assert all(c.token_count <= 6 for c in chunks)


def test_never_splits_inside_an_event() -> None:
    events = _events(7, 3, 5, 2, 9, 4, 6)
    chunks = cut_into_bounded_chunks(events, max_length=10)
    # 모든 이벤트의 토큰이 원본 순서 그대로, 쪼개짐 없이 전부 등장해야 한다
    # (겹치는 이벤트는 여러 청크에 통째로 반복되는 것만 허용).
    seen_ids_per_position = [set(e.event_id for e in c.events) for c in chunks]
    all_ids = set(e.event_id for e in events)
    covered = set().union(*seen_ids_per_position)
    assert covered == all_ids
    for chunk in chunks:
        for event in chunk.events:
            original = next(e for e in events if e.event_id == event.event_id)
            assert event.tokens == original.tokens  # 토큰 시퀀스가 그대로 보존됨


def test_middle_chunks_needed_without_overlap() -> None:
    events = _events(*([2] * 10))  # 10개 이벤트, 각 2토큰 = 총 20
    chunks = cut_into_bounded_chunks(events, max_length=6, overlap_events=0)
    assert chunks[0].position == "FIRST"
    assert chunks[-1].position == "LAST"
    assert all(c.position == "MIDDLE" for c in chunks[1:-1])
    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.token_count <= 6
    # overlap 없이 인접 청크는 이어 붙어야 한다(겹침도 빈틈도 없음).
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_index == b.start_index


def test_middle_chunks_with_overlap_repeat_boundary_events() -> None:
    events = _events(*([2] * 10))
    chunks = cut_into_bounded_chunks(events, max_length=6, overlap_events=1)
    for a, b in zip(chunks, chunks[1:]):
        # overlap_events=1이면 다음 청크가 이전 청크의 마지막 1개 이벤트를 다시 포함한다.
        assert b.start_index <= a.end_index
        assert b.start_index >= a.end_index - 1


def test_single_event_exceeding_max_length_is_kept_whole_and_flagged() -> None:
    events = _events(3, 50, 3)
    chunks = cut_into_bounded_chunks(events, max_length=10)
    huge = next(c for c in chunks if any(e.event_id == "e1" for e in c.events))
    assert SINGLE_EVENT_EXCEEDS_MAX_LENGTH in huge.quality_flags
    assert huge.token_count == 50  # 잘리지 않고 그대로 들어감


def test_all_tokens_preserved_across_chunks_when_no_overlap() -> None:
    events = _events(4, 4, 4, 4, 4)
    chunks = cut_into_bounded_chunks(events, max_length=8, overlap_events=0)
    # overlap이 없으면 전체 토큰 수가 원본과 정확히 같아야 한다(중복도 누락도 없음).
    assert sum(c.token_count for c in chunks) == sum(e.token_count for e in events)


def test_rejects_non_positive_max_length() -> None:
    with pytest.raises(ValueError):
        cut_into_bounded_chunks(_events(1, 1), max_length=0)


def test_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError):
        cut_into_bounded_chunks(_events(1, 1), max_length=5, overlap_events=-1)


def test_chunk_rejects_invalid_position() -> None:
    with pytest.raises(ValueError):
        SequenceChunk(
            position="MAYBE",
            events=(TokenEvent("e0", ("t0",)),),
            start_index=0,
            end_index=1,
            token_count=1,
        )


def test_chunk_rejects_empty_events() -> None:
    with pytest.raises(ValueError):
        SequenceChunk(position="SOLE", events=(), start_index=0, end_index=0, token_count=0)
