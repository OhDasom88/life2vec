"""§5.3 결정 -> §6.3 curation transaction 연결부 회귀 테스트."""

from __future__ import annotations

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sequence_curation.review_to_action import (
    transaction_from_decision,
    transactions_from_decisions,
)


def _decision(instance_id: str, decision: str, reason_codes: list[str] | None = None) -> dict:
    return {
        "narrative_instance_id": instance_id,
        "narrative_template_id": "A01",
        "decision": decision,
        "reviewer": "tester",
        "reasons_at_review_time": [{"code": c, "detail": "d"} for c in (reason_codes or [])],
        "priority_score": 1.0,
        "grounding_search_decision": None,
        "reviewed_at_utc": "2026-07-21T00:00:00Z",
    }


def test_accept_becomes_include_with_unchanged_tokens() -> None:
    tx = transaction_from_decision(
        _decision("seq_1", "ACCEPT"), before_tokens=["a", "b"], affected_split="TRAIN"
    )
    assert tx is not None
    assert tx.action == "INCLUDE"
    assert tx.sequence_id == "seq_1"


def test_reject_becomes_exclude_with_emptied_tokens() -> None:
    tx = transaction_from_decision(
        _decision("seq_1", "REJECT"), before_tokens=["a", "b"], affected_split="TRAIN"
    )
    assert tx is not None
    assert tx.action == "EXCLUDE"
    assert tx.before_sequence_sha != tx.after_sequence_sha


def test_skip_produces_no_transaction() -> None:
    tx = transaction_from_decision(
        _decision("seq_1", "SKIP"), before_tokens=["a"], affected_split="TRAIN"
    )
    assert tx is None


def test_reason_summary_included_in_reason_text() -> None:
    tx = transaction_from_decision(
        _decision("seq_1", "REJECT", ["MODEL_EXPERT_DISAGREEMENT"]),
        before_tokens=["a"],
        affected_split="TRAIN",
    )
    assert tx is not None
    assert "MODEL_EXPERT_DISAGREEMENT" in tx.reason


def test_transactions_from_decisions_skips_missing_tokens_and_skip_decisions() -> None:
    decisions = {
        "a": _decision("a", "ACCEPT"),
        "b": _decision("b", "REJECT"),
        "c": _decision("c", "SKIP"),  # 결정 자체가 SKIP -> 제외
        "d": _decision("d", "ACCEPT"),  # 토큰이 없어서 -> 제외
    }
    tokens_by_id = {"a": ["t1"], "b": ["t2"]}
    transactions = transactions_from_decisions(
        decisions, tokens_by_sequence_id=tokens_by_id, affected_split="DEVELOPMENT"
    )
    ids = {tx.sequence_id for tx in transactions}
    assert ids == {"a", "b"}
    assert all(tx.affected_split == "DEVELOPMENT" for tx in transactions)
