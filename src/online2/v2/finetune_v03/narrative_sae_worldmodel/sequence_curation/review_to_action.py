"""§5.3 사람 검토 결정 -> §6.3 curation transaction 연결부.

``narrative_grounding``의 ``decisions.jsonl``(§5.3 UI가 쌓는 사람 결정 로그)을
그대로 학습 데이터 curation 결정으로 승격한다: ACCEPT는 시퀀스를 사전학습
입력에 포함(``INCLUDE``), REJECT는 제외(``EXCLUDE``). SKIP은 아직 결정이
아니므로 트랜잭션을 만들지 않는다.

decisions.jsonl은 ``narrative_instance_id``만 갖고 있고 실제 토큰 시퀀스는
갖고 있지 않다(§5.3 UI가 다루는 건 narrative 객체이지 원시 토큰이 아니다).
그래서 이 모듈은 토큰을 직접 조회하지 않고 호출자가 ``tokens_by_sequence_id``로
주도록 한다 — online2 ``sequences.parquet``/``event_tokens.parquet``에서
가져오는 책임은 호출자(예: 배치 스크립트, UI)에게 남긴다.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .curation_actions import CurationTransaction, build_curation_transaction

_ACTION_BY_DECISION = {"ACCEPT": "INCLUDE", "REJECT": "EXCLUDE"}


def _reason_summary(decision_record: Mapping[str, Any]) -> str:
    reasons = decision_record.get("reasons_at_review_time") or []
    if not reasons:
        return "검토 사유 없음"
    return "; ".join(f"{reason['code']}({reason['detail']})" for reason in reasons)


def transaction_from_decision(
    decision_record: Mapping[str, Any],
    *,
    before_tokens: Sequence[str],
    affected_split: str,
) -> CurationTransaction | None:
    """decisions.jsonl 레코드 하나 -> curation transaction.

    SKIP이면 ``None``을 반환한다 — "아직 결정 안 됨"을 빈 트랜잭션으로
    위장하지 않는다.
    """
    decision = decision_record.get("decision")
    action = _ACTION_BY_DECISION.get(decision)
    if action is None:
        return None

    after_tokens = list(before_tokens) if action == "INCLUDE" else []
    reviewer = decision_record.get("reviewer", "anonymous")
    return build_curation_transaction(
        sequence_id=decision_record["narrative_instance_id"],
        action=action,
        reason=f"§5.3 사람 결정={decision} (reviewer={reviewer}): {_reason_summary(decision_record)}",
        applied_rule_ref=(
            f"decisions.jsonl:{decision_record['narrative_instance_id']}:"
            f"{decision_record.get('reviewed_at_utc', '')}"
        ),
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        affected_split=affected_split,
    )


def transactions_from_decisions(
    decisions: Mapping[str, Mapping[str, Any]],
    *,
    tokens_by_sequence_id: Mapping[str, Sequence[str]],
    affected_split: str,
) -> list[CurationTransaction]:
    """여러 결정을 한 번에 변환한다.

    ``tokens_by_sequence_id``에 없는 항목은 조용히 건너뛴다(에러를 던지지
    않는다) — 호출자가 토큰 조회 범위를 의도적으로 좁혔을 수 있어서다(예:
    특정 farm 배치만 처리 중). SKIP 항목도 결과에서 빠진다.
    """
    transactions: list[CurationTransaction] = []
    for record in decisions.values():
        tokens = tokens_by_sequence_id.get(record["narrative_instance_id"])
        if tokens is None:
            continue
        transaction = transaction_from_decision(
            record, before_tokens=tokens, affected_split=affected_split
        )
        if transaction is not None:
            transactions.append(transaction)
    return transactions
