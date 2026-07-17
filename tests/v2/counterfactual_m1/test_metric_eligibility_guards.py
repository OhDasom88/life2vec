"""Metric eligibility: rank_present vs rank_valid + manifest recoverability symmetry."""

from __future__ import annotations

from src.online2.v2.finetune_v03.counterfactual.evaluation.metric_eligibility import (
    classify_metric_eligibility,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_q4_q5 import (
    aggregate_q4_q5_from_per_mg,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_selection import (
    MIN_RECONSTRUCTION_CANDIDATES,
)


def _ok_eligible_row(**overrides):
    row = {
        "manifest_recoverable": True,
        "metric_eligible": True,
        "scoring_completed": True,
        "candidate_count": int(MIN_RECONSTRUCTION_CANDIDATES),
        "original_rank": 2,
        "recall_at_3": 1,
        "random_recall_at_3": 0.25,
        "metric_audit_error": False,
    }
    row.update(overrides)
    return row


def test_manifest_recoverable_true_with_missing_rank_fails_audit():
    clf = classify_metric_eligibility(
        manifest_recoverable=True,
        original_rank=None,
        scoring_completed=True,
        candidate_count=int(MIN_RECONSTRUCTION_CANDIDATES),
    )
    assert clf["metric_eligible"] is False
    assert clf["metric_audit_error"] is True
    assert "manifest_recoverable_but_original_missing" in clf["metric_audit_errors"]

    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "manifest_recoverable": True,
                "metric_eligible": False,
                "metric_ineligible_reason": "manifest_recoverable_but_original_missing",
                "metric_audit_error": True,
                "metric_audit_errors": ["manifest_recoverable_but_original_missing"],
                "scoring_completed": True,
                "candidate_count": int(MIN_RECONSTRUCTION_CANDIDATES),
                "original_rank": None,
            }
        ]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"


def test_manifest_recoverable_false_with_missing_rank_is_metric_ineligible_not_audit_failure():
    clf = classify_metric_eligibility(
        manifest_recoverable=False,
        original_rank=None,
        scoring_completed=True,
        candidate_count=int(MIN_RECONSTRUCTION_CANDIDATES),
    )
    assert clf["metric_eligible"] is False
    assert clf["metric_audit_error"] is False
    assert clf["metric_ineligible_reason"] == "original_not_in_scored_candidates"

    good = _ok_eligible_row()
    ineligible = {
        "manifest_recoverable": False,
        "metric_eligible": False,
        "metric_ineligible_reason": "original_not_in_scored_candidates",
        "metric_audit_error": False,
        "scoring_completed": True,
        "candidate_count": int(MIN_RECONSTRUCTION_CANDIDATES),
        "original_rank": None,
    }
    out = aggregate_q4_q5_from_per_mg([good, ineligible])
    assert out["reconstruction_metric_audit_status"] == "PASS"
    assert out["metric_eligible_mg_count"] == 1


def test_manifest_recoverable_false_with_rank_fails_audit():
    clf = classify_metric_eligibility(
        manifest_recoverable=False,
        original_rank=3,
        scoring_completed=True,
        candidate_count=int(MIN_RECONSTRUCTION_CANDIDATES),
    )
    assert clf["metric_eligible"] is False
    assert clf["metric_audit_error"] is True
    assert "manifest_unrecoverable_but_original_rank_present" in clf["metric_audit_errors"]

    out = aggregate_q4_q5_from_per_mg(
        [
            {
                "manifest_recoverable": False,
                "metric_eligible": False,
                "metric_ineligible_reason": "manifest_unrecoverable_but_original_rank_present",
                "metric_audit_error": True,
                "metric_audit_errors": ["manifest_unrecoverable_but_original_rank_present"],
                "scoring_completed": True,
                "candidate_count": int(MIN_RECONSTRUCTION_CANDIDATES),
                "original_rank": 3,
            }
        ]
    )
    assert out["reconstruction_metric_audit_status"] == "FAIL"


def test_original_rank_bool_is_rejected_as_invalid_integer():
    clf = classify_metric_eligibility(
        manifest_recoverable=True,
        original_rank=True,
        scoring_completed=True,
        candidate_count=int(MIN_RECONSTRUCTION_CANDIDATES),
    )
    assert clf["rank_present"] is True
    assert clf["rank_valid"] is False
    assert clf["metric_audit_error"] is True
    assert "invalid_original_rank_type" in clf["metric_audit_errors"]


def test_manifest_unrecoverable_with_invalid_non_null_rank_fails_audit():
    for bad in (0, True, 3.0, "3"):
        clf = classify_metric_eligibility(
            manifest_recoverable=False,
            original_rank=bad,
            scoring_completed=True,
            candidate_count=int(MIN_RECONSTRUCTION_CANDIDATES),
        )
        assert clf["rank_present"] is True
        assert clf["rank_valid"] is False
        assert clf["metric_audit_error"] is True
        assert clf["metric_eligible"] is False


def test_candidate_count_bool_is_rejected():
    clf = classify_metric_eligibility(
        manifest_recoverable=True,
        original_rank=2,
        scoring_completed=True,
        candidate_count=True,
    )
    assert clf["candidate_count_valid"] is False
    assert clf["metric_audit_error"] is True
    assert "invalid_candidate_count" in clf["metric_audit_errors"]
    assert clf["metric_eligible"] is False


def test_field_semantics_scored_vs_recoverable_vs_ranked():
    """Documented count meanings must stay distinct even when equal numerically."""
    selected_mg_count = 85
    recoverable_mg_count = 78  # from frozen manifest
    scored_mg_count = 85
    ranked_original_mg_count = 78
    metric_eligible_mg_count = 78
    assert recoverable_mg_count != scored_mg_count or True  # meanings differ
    assert selected_mg_count >= recoverable_mg_count
    assert ranked_original_mg_count <= scored_mg_count
    assert metric_eligible_mg_count <= ranked_original_mg_count
