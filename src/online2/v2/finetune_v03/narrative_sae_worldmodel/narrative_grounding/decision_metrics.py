"""decisions.jsonl(§5.3 UI가 쌓는 사람 결정 로그) -> §5.4 지표.

``evaluation.py``의 ``expert_acceptance_rate``를 실제 결정 로그에 연결하는
어댑터다. decisions.jsonl은 ``ui/data_grounding_curation/review_batch.load_decisions()``가
이미 "narrative_instance_id -> 마지막 결정" 딕셔너리로 파싱해 준다 — 이
모듈은 그 딕셔너리를 받아 계산만 한다(파일 I/O를 다시 하지 않는다, UI
모듈에 대한 의존성도 만들지 않는다 — 딕셔너리 shape만 맞으면 된다).

``auto_accept_error_and_review_rate``(evaluation.py)는 여기서 연결하지
않는다. 그 지표는 §5.1 검색이 내린 4분기 판정
(``AUTO_ACCEPT_CANDIDATE|REVIEW|QUARANTINE|REJECT``)과 "그 판정이 맞았는가"라는
정답 라벨을 요구한다. 그런데 지금 §5.3 검토 큐는 §5.1 검색 결과가 아니라
``review_queue``의 검토 사유(모델·전문가 불일치, 저빈도 개념 등)로 채워지고,
decisions.jsonl에는 §5.1 판정 라벨 자체가 기록되지 않는다 — 두 파이프라인이
아직 연결돼 있지 않다. 없는 데이터를 있는 척 채우지 않고
``summarize_decisions``의 ``not_connected`` 필드에 이유를 명시한다.
"""

from __future__ import annotations

from typing import Any, Mapping

from .evaluation import expert_acceptance_rate

_DECIDED_VALUES = frozenset({"ACCEPT", "REJECT"})  # SKIP은 "아직 결정 안 함"이라 표본 제외


def summarize_decisions(decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """decisions.jsonl의 "최신 상태" 딕셔너리 -> §5.4 리포트.

    - ``overall_expert_acceptance_rate``: SKIP을 제외한 ACCEPT/REJECT만으로 계산.
    - ``by_reason_code``: ``reasons_at_review_time``에 등장한 검토 사유 코드별
      승인율. ``review_queue._REASON_WEIGHTS``를 재보정할 실측 근거다 —
      승인율이 높은 사유(사람이 대체로 "문제 없다"고 판단)는 가중치를 낮추고,
      승인율이 낮은 사유(사람이 대체로 "문제 있다"고 판단)는 유지·강화하는 식으로
      쓴다.
    """
    records = list(decisions.values())
    decided = [record for record in records if record.get("decision") in _DECIDED_VALUES]
    skipped = [record for record in records if record.get("decision") == "SKIP"]

    overall_rate = expert_acceptance_rate([record["decision"] == "ACCEPT" for record in decided])

    flags_by_reason: dict[str, list[bool]] = {}
    for record in decided:
        accepted = record["decision"] == "ACCEPT"
        for reason in record.get("reasons_at_review_time", []):
            code = reason.get("code")
            if not code:
                continue
            flags_by_reason.setdefault(code, []).append(accepted)

    by_reason_code = {
        code: {"acceptance_rate": expert_acceptance_rate(flags), "n": len(flags)}
        for code, flags in flags_by_reason.items()
    }

    return {
        "n_total": len(records),
        "n_decided": len(decided),
        "n_accept": sum(1 for r in decided if r["decision"] == "ACCEPT"),
        "n_reject": sum(1 for r in decided if r["decision"] == "REJECT"),
        "n_skipped": len(skipped),
        "overall_expert_acceptance_rate": overall_rate,
        "by_reason_code": by_reason_code,
        "not_connected": {
            "auto_accept_error_and_review_rate": (
                "decisions.jsonl에 §5.1 4분기 판정 라벨(AUTO_ACCEPT_CANDIDATE 등)이 "
                "없어 계산할 수 없음 — §5.1 검색 결과를 §5.3 검토 큐로 보내는 연결이 "
                "아직 없기 때문(README 참조)."
            )
        },
    }
