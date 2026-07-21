"""학습 전 편집 액션 (계획서 §6.3) — immutable curation transaction.

원래 계획(``README.md``)은 CF1S(``counterfactual/cf1s/core_raw_transaction.py``)의
``ValidatedRawTransaction``에 ``edit_purpose`` 필드를 추가해 하나의 스키마로
통합하는 것이었다. 착수 전 확인한 결과 이건 위험한 선택이다: CF1S의
트랜잭션 해시(``canonical_validated_transaction_sha`` 등)는 이미 검증을
통과한 55건(Dev3 3 + Primary32 32 + Validation20 20) stable lock의 무결성
근거이고, frozen dataclass에 필드를 하나만 추가해도 재직렬화 시 해시가
달라져 55건 전체 재인증을 유발할 수 있다 — `concept_governance/README.md`가
이미 vocab 변경에 대해 문서화한 것과 같은 종류의 리스크를, 이번엔 CF1S
코드 자체를 건드려서 만들게 된다.

그래서 CF1S 프로덕션 파일은 건드리지 않는다. 대신 같은 설계 원칙(불변
트랜잭션, 해시 체인, fail-closed 검증)을 공유하는 **별도** 스키마를 여기
만든다. 해시 유틸은 새로 만들지 않고 ``src/online2/canonical.py``의
``stable_id``/``canonical_json``을 그대로 재사용한다(online2 코퍼스와 동일한
네임스페이스 해싱 방식).

CF1S 편집(``core_raw_transaction.py``)과 이 모듈의 차이:
CF1S는 미세조정 이후 **인과검증** 목적의 최소 변경 편집이고, 여기는
**사전학습 입력 curation** 목적의 편집이다(저품질/오류 서사를 학습셋에서
빼거나 마스킹). 대상도 다르다 — CF1S는 case/event 단위, 여기는 online2
sequence 단위(``narrative_grounding.DataWindow.window_id``와 동일한 ID 재사용).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.online2.canonical import stable_id

CURATION_ACTIONS = frozenset(
    {"INCLUDE", "EXCLUDE", "MASK", "REPLACE", "SPLIT", "MERGE", "REORDER", "REWINDOW"}
)
DATA_SPLITS = frozenset({"TRAIN", "DEVELOPMENT", "HOLDOUT"})

# 액션별로 parameters에 반드시 있어야 하는 키. 구체적인 값 형태까지 강타입으로
# 만들지 않는다 — 액션 8종 전부를 위한 전용 클래스를 만들면 이 시점에 과설계다.
_REQUIRED_PARAMETER_KEYS: dict[str, frozenset[str]] = {
    "INCLUDE": frozenset(),
    "EXCLUDE": frozenset(),
    "MASK": frozenset({"masked_event_ids"}),
    "REPLACE": frozenset({"replaced_event_ids", "replacement_tokens"}),
    "SPLIT": frozenset({"split_at_event_index"}),
    "MERGE": frozenset({"merged_sequence_ids"}),
    "REORDER": frozenset({"new_event_order"}),
    "REWINDOW": frozenset({"new_start_timestamp", "new_end_timestamp"}),
}


@dataclass(frozen=True)
class CurationTransaction:
    """§6.3 편집 하나를 나타내는 immutable 레코드. 원본을 덮어쓰지 않는다."""

    transaction_id: str
    sequence_id: str
    action: str
    reason: str
    applied_rule_ref: str
    before_sequence_sha: str
    after_sequence_sha: str
    affected_split: str
    vocab_impact: bool = False
    retrain_required: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in CURATION_ACTIONS:
            raise ValueError(f"invalid curation action: {self.action!r}")
        if self.affected_split not in DATA_SPLITS:
            raise ValueError(f"invalid affected_split: {self.affected_split!r}")
        missing = _REQUIRED_PARAMETER_KEYS[self.action] - set(self.parameters.keys())
        if missing:
            raise ValueError(
                f"action {self.action} missing required parameters: {sorted(missing)}"
            )


def build_curation_transaction(
    *,
    sequence_id: str,
    action: str,
    reason: str,
    applied_rule_ref: str,
    before_tokens: Sequence[str],
    after_tokens: Sequence[str],
    affected_split: str,
    parameters: Mapping[str, Any] | None = None,
    vocab_impact: bool = False,
    retrain_required: bool = False,
) -> CurationTransaction:
    """before/after 토큰 시퀀스에서 해시를 계산해 ``CurationTransaction``을 만든다.

    ``transaction_id``는 (sequence_id, action, before_sha, after_sha, reason,
    applied_rule_ref)의 해시다 — 같은 편집을 두 번 만들면 같은 ID가 나온다
    (멱등, 중복 append 방지에 쓸 수 있다).
    """
    before_sha = stable_id(
        "curation_before", {"sequence_id": sequence_id, "tokens": list(before_tokens)}
    )
    after_sha = stable_id(
        "curation_after", {"sequence_id": sequence_id, "tokens": list(after_tokens)}
    )
    transaction_id = stable_id(
        "curation_tx",
        {
            "sequence_id": sequence_id,
            "action": action,
            "before_sha": before_sha,
            "after_sha": after_sha,
            "reason": reason,
            "applied_rule_ref": applied_rule_ref,
        },
    )
    return CurationTransaction(
        transaction_id=transaction_id,
        sequence_id=sequence_id,
        action=action,
        reason=reason,
        applied_rule_ref=applied_rule_ref,
        before_sequence_sha=before_sha,
        after_sequence_sha=after_sha,
        affected_split=affected_split,
        vocab_impact=vocab_impact,
        retrain_required=retrain_required,
        parameters=dict(parameters or {}),
    )
