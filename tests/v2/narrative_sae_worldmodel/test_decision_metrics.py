"""decisions.jsonl -> §5.4 지표 어댑터 회귀 테스트."""

from __future__ import annotations

import pytest

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.decision_metrics import (
    summarize_decisions,
)


def _record(instance_id: str, decision: str, reason_codes: list[str]) -> dict:
    return {
        "narrative_instance_id": instance_id,
        "narrative_template_id": "A01",
        "decision": decision,
        "reviewer": "tester",
        "reasons_at_review_time": [{"code": code, "detail": ""} for code in reason_codes],
        "priority_score": float(len(reason_codes)),
        "reviewed_at_utc": "2026-07-21T00:00:00Z",
    }


def test_empty_decisions_returns_zeroed_report() -> None:
    report = summarize_decisions({})
    assert report["n_total"] == 0
    assert report["n_decided"] == 0
    assert report["overall_expert_acceptance_rate"] == 0.0
    assert report["by_reason_code"] == {}


def test_skip_excluded_from_overall_rate() -> None:
    decisions = {
        "a": _record("a", "ACCEPT", ["LOW_FREQUENCY_CONCEPT"]),
        "b": _record("b", "SKIP", ["LOW_FREQUENCY_CONCEPT"]),
    }
    report = summarize_decisions(decisions)
    assert report["n_total"] == 2
    assert report["n_decided"] == 1
    assert report["n_skipped"] == 1
    assert report["overall_expert_acceptance_rate"] == 1.0  # SKIP은 분모에서 빠짐


def test_overall_rate_mixes_accept_and_reject() -> None:
    decisions = {
        "a": _record("a", "ACCEPT", []),
        "b": _record("b", "ACCEPT", []),
        "c": _record("c", "REJECT", []),
        "d": _record("d", "REJECT", []),
    }
    report = summarize_decisions(decisions)
    assert report["n_accept"] == 2
    assert report["n_reject"] == 2
    assert report["overall_expert_acceptance_rate"] == 0.5


def test_by_reason_code_breaks_down_acceptance() -> None:
    decisions = {
        # MODEL_EXPERT_DISAGREEMENT로 걸린 3건 중 1건만 승인 -> 대체로 진짜 문제였다는 신호.
        "a": _record("a", "REJECT", ["MODEL_EXPERT_DISAGREEMENT"]),
        "b": _record("b", "REJECT", ["MODEL_EXPERT_DISAGREEMENT"]),
        "c": _record("c", "ACCEPT", ["MODEL_EXPERT_DISAGREEMENT"]),
        # LOW_FREQUENCY_CONCEPT로 걸린 2건은 전부 승인 -> 대체로 오탐(과잉 플래그)이라는 신호.
        "d": _record("d", "ACCEPT", ["LOW_FREQUENCY_CONCEPT"]),
        "e": _record("e", "ACCEPT", ["LOW_FREQUENCY_CONCEPT"]),
    }
    report = summarize_decisions(decisions)
    by_reason = report["by_reason_code"]
    assert by_reason["MODEL_EXPERT_DISAGREEMENT"]["n"] == 3
    assert by_reason["MODEL_EXPERT_DISAGREEMENT"]["acceptance_rate"] == pytest.approx(1 / 3)
    assert by_reason["LOW_FREQUENCY_CONCEPT"]["n"] == 2
    assert by_reason["LOW_FREQUENCY_CONCEPT"]["acceptance_rate"] == 1.0


def test_multi_reason_record_counts_toward_each_code() -> None:
    decisions = {
        "a": _record("a", "ACCEPT", ["LOW_FREQUENCY_CONCEPT", "MODEL_EXPERT_DISAGREEMENT"]),
    }
    report = summarize_decisions(decisions)
    assert report["by_reason_code"]["LOW_FREQUENCY_CONCEPT"]["n"] == 1
    assert report["by_reason_code"]["MODEL_EXPERT_DISAGREEMENT"]["n"] == 1


def test_auto_accept_error_rate_explicitly_marked_not_connected() -> None:
    report = summarize_decisions({})
    assert "auto_accept_error_and_review_rate" in report["not_connected"]
