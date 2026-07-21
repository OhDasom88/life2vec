"""decisions.jsonl(§5.3 UI가 쌓는 사람 결정 로그) -> §5.4 지표.

``evaluation.py``의 ``expert_acceptance_rate``를 실제 결정 로그에 연결하는
어댑터다. decisions.jsonl은 ``ui/data_grounding_curation/review_batch.load_decisions()``가
이미 "narrative_instance_id -> 마지막 결정" 딕셔너리로 파싱해 준다 — 이
모듈은 그 딕셔너리를 받아 계산만 한다(파일 I/O를 다시 하지 않는다, UI
모듈에 대한 의존성도 만들지 않는다 — 딕셔너리 shape만 맞으면 된다).

§5.1 검색 결과가 ``search_to_review.py``를 통해 §5.3 큐로 들어오게 된 뒤로는
``grounding_search_decision``(REVIEW|QUARANTINE) 필드가 실린 레코드가
생긴다 — 이 모듈이 그 서브셋의 "사람 확인율"을 ``grounding_search_confirmation``으로
계산한다. 다만 ``evaluation.auto_accept_error_and_review_rate``는 **여전히
연결하지 않는다**: 그 지표는 AUTO_ACCEPT_CANDIDATE·REJECT까지 포함한 전체
후보 모집단 위에서 계산해야 ``human_review_rate``가 의미를 가지는데,
``search_to_review.py``는 애초에 REVIEW/QUARANTINE만 큐에 올리도록 설계돼
있어(AUTO_ACCEPT는 이미 확신, REJECT는 이미 배제) 그 두 등급의 라벨만
생긴다. 전체 모집단을 로깅하는 별도 파이프라인(모든 §5.1 검색 결과 기록)
없이는 이 지표를 정직하게 채울 수 없다 — ``not_connected``에 이유를 명시한다.
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
    - ``grounding_search_confirmation``: ``grounding_search_decision``이 채워진
      레코드(§5.1에서 넘어온 것)만 골라 REVIEW/QUARANTINE 등급별 사람 확인율.
      "§5.1이 애매하다고 판단한 후보 중 사람이 실제로 몇 %를 승인했는가"이며,
      §5.1 임계값(text_to_window._REVIEW_THRESHOLD 등) 재보정의 실측 근거가 된다.
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

    flags_by_search_decision: dict[str, list[bool]] = {}
    for record in decided:
        search_decision = record.get("grounding_search_decision")
        if not search_decision:
            continue
        flags_by_search_decision.setdefault(search_decision, []).append(
            record["decision"] == "ACCEPT"
        )
    grounding_search_confirmation = {
        search_decision: {"confirmation_rate": expert_acceptance_rate(flags), "n": len(flags)}
        for search_decision, flags in flags_by_search_decision.items()
    }

    return {
        "n_total": len(records),
        "n_decided": len(decided),
        "n_accept": sum(1 for r in decided if r["decision"] == "ACCEPT"),
        "n_reject": sum(1 for r in decided if r["decision"] == "REJECT"),
        "n_skipped": len(skipped),
        "overall_expert_acceptance_rate": overall_rate,
        "by_reason_code": by_reason_code,
        "grounding_search_confirmation": grounding_search_confirmation,
        "not_connected": {
            "auto_accept_error_and_review_rate": (
                "REVIEW/QUARANTINE 서브셋의 사람 확인율은 grounding_search_confirmation으로 "
                "계산되지만, 이 지표는 AUTO_ACCEPT_CANDIDATE·REJECT까지 포함한 전체 후보 "
                "모집단이 있어야 human_review_rate가 의미를 가진다 — search_to_review.py는 "
                "그 두 등급을 애초에 큐에 올리지 않으므로(설계상 의도) 전체 모집단 로깅 "
                "없이는 여기서 계산할 수 없다."
            )
        },
    }
