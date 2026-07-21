"""§6.3 학습 전 편집 액션(curation transaction) 회귀 테스트."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sequence_curation.curation_actions import (
    CurationTransaction,
    build_curation_transaction,
)


def test_include_and_exclude_need_no_parameters() -> None:
    tx = build_curation_transaction(
        sequence_id="seq_1",
        action="INCLUDE",
        reason="사람 검토 승인",
        applied_rule_ref="decisions.jsonl:seq_1",
        before_tokens=["a", "b"],
        after_tokens=["a", "b"],
        affected_split="TRAIN",
    )
    assert tx.action == "INCLUDE"
    assert tx.parameters == {}


def test_same_tokens_produce_same_hash_when_replayed() -> None:
    # before_sha/after_sha는 namespace가 달라("curation_before" vs "curation_after")
    # 내용이 같아도 값 자체는 다르다 - 이건 의도된 설계(네임스페이스 분리로 해시
    # 충돌 방지)다. 대신 "같은 입력을 다시 넣으면 같은 해시가 나온다"(재현성)를
    # 검증한다 - test_transaction_id_is_deterministic이 transaction_id 수준에서
    # 이미 이걸 확인하지만 여기서는 before_sequence_sha 자체를 직접 확인한다.
    tx1 = build_curation_transaction(
        sequence_id="seq_1", action="INCLUDE", reason="r", applied_rule_ref="ref",
        before_tokens=["a", "b"], after_tokens=["a", "b"], affected_split="TRAIN",
    )
    tx2 = build_curation_transaction(
        sequence_id="seq_1", action="INCLUDE", reason="r2", applied_rule_ref="ref2",
        before_tokens=["a", "b"], after_tokens=["a", "b"], affected_split="TRAIN",
    )
    assert tx1.before_sequence_sha == tx2.before_sequence_sha
    assert tx1.after_sequence_sha == tx2.after_sequence_sha


def test_exclude_produces_different_before_after_hash() -> None:
    tx = build_curation_transaction(
        sequence_id="seq_1",
        action="EXCLUDE",
        reason="사람 검토 반려",
        applied_rule_ref="decisions.jsonl:seq_1",
        before_tokens=["a", "b"],
        after_tokens=[],
        affected_split="TRAIN",
    )
    assert tx.before_sequence_sha != tx.after_sequence_sha


def test_transaction_id_is_deterministic() -> None:
    kwargs = dict(
        sequence_id="seq_1",
        action="EXCLUDE",
        reason="r",
        applied_rule_ref="ref",
        before_tokens=["a"],
        after_tokens=[],
        affected_split="TRAIN",
    )
    tx1 = build_curation_transaction(**kwargs)
    tx2 = build_curation_transaction(**kwargs)
    assert tx1.transaction_id == tx2.transaction_id  # 같은 입력 -> 같은 ID (멱등)


def test_different_reason_changes_transaction_id_but_not_sequence_hashes() -> None:
    base = dict(
        sequence_id="seq_1",
        action="EXCLUDE",
        applied_rule_ref="ref",
        before_tokens=["a"],
        after_tokens=[],
        affected_split="TRAIN",
    )
    tx1 = build_curation_transaction(reason="사유 A", **base)
    tx2 = build_curation_transaction(reason="사유 B", **base)
    assert tx1.transaction_id != tx2.transaction_id
    assert tx1.before_sequence_sha == tx2.before_sequence_sha


@pytest.mark.parametrize(
    "action,parameters",
    [
        ("MASK", {"masked_event_ids": ["e1"]}),
        ("REPLACE", {"replaced_event_ids": ["e1"], "replacement_tokens": ["x"]}),
        ("SPLIT", {"split_at_event_index": 2}),
        ("MERGE", {"merged_sequence_ids": ["seq_2"]}),
        ("REORDER", {"new_event_order": [1, 0]}),
        ("REWINDOW", {"new_start_timestamp": "t0", "new_end_timestamp": "t1"}),
    ],
)
def test_actions_with_required_parameters_succeed(action: str, parameters: dict) -> None:
    tx = build_curation_transaction(
        sequence_id="seq_1",
        action=action,
        reason="r",
        applied_rule_ref="ref",
        before_tokens=["a", "b"],
        after_tokens=["a"],
        affected_split="DEVELOPMENT",
        parameters=parameters,
    )
    assert tx.action == action
    assert tx.parameters == parameters


@pytest.mark.parametrize(
    "action",
    ["MASK", "REPLACE", "SPLIT", "MERGE", "REORDER", "REWINDOW"],
)
def test_actions_missing_required_parameters_raise(action: str) -> None:
    with pytest.raises(ValueError):
        build_curation_transaction(
            sequence_id="seq_1",
            action=action,
            reason="r",
            applied_rule_ref="ref",
            before_tokens=["a"],
            after_tokens=["a"],
            affected_split="TRAIN",
        )


def test_invalid_action_rejected() -> None:
    with pytest.raises(ValueError):
        build_curation_transaction(
            sequence_id="seq_1",
            action="DELETE",
            reason="r",
            applied_rule_ref="ref",
            before_tokens=["a"],
            after_tokens=[],
            affected_split="TRAIN",
        )


def test_invalid_split_rejected() -> None:
    with pytest.raises(ValueError):
        build_curation_transaction(
            sequence_id="seq_1",
            action="EXCLUDE",
            reason="r",
            applied_rule_ref="ref",
            before_tokens=["a"],
            after_tokens=[],
            affected_split="TEST",
        )


def test_direct_construction_also_validates() -> None:
    with pytest.raises(ValueError):
        CurationTransaction(
            transaction_id="x",
            sequence_id="seq_1",
            action="MASK",
            reason="r",
            applied_rule_ref="ref",
            before_sequence_sha="a",
            after_sequence_sha="b",
            affected_split="TRAIN",
            parameters={},  # masked_event_ids 없음
        )
