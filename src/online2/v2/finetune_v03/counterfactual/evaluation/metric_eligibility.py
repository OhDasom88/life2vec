"""Per-MG metric eligibility: frozen-manifest recoverability + rank_present/rank_valid."""

from __future__ import annotations

import numbers
from typing import Any, Dict, List, Optional

from .reconstruction_selection import MIN_RECONSTRUCTION_CANDIDATES


def is_strict_nonneg_int(value: object) -> bool:
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and int(value) >= 0
    )


def is_strict_rank_int(value: object) -> bool:
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and int(value) >= 1
    )


def classify_metric_eligibility(
    *,
    manifest_recoverable: bool,
    original_rank: Any,
    scoring_completed: bool,
    candidate_count: Any,
    min_candidates: int = MIN_RECONSTRUCTION_CANDIDATES,
    early_ineligible_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify a per-MG row. Never mutates manifest_recoverable.

    rank_present = original_rank is not None (any non-null, including invalid types).
    rank_valid = strict Integral (not bool) and >= 1.
    """
    reasons: List[str] = []
    audit_errors: List[str] = []

    candidate_count_valid = is_strict_nonneg_int(candidate_count)
    if not candidate_count_valid:
        reasons.append("invalid_candidate_count")
        audit_errors.append("invalid_candidate_count")

    rank_present = original_rank is not None
    rank_valid = is_strict_rank_int(original_rank)
    if rank_present and not rank_valid:
        reasons.append("invalid_original_rank_type")
        audit_errors.append("invalid_original_rank_type")

    recoverability_rank_mismatch = bool(manifest_recoverable) != bool(rank_present)
    if recoverability_rank_mismatch:
        if manifest_recoverable and not rank_present:
            tag = "manifest_recoverable_but_original_missing"
        else:
            tag = "manifest_unrecoverable_but_original_rank_present"
        reasons.append(tag)
        audit_errors.append(tag)

    consistency_error = recoverability_rank_mismatch or (rank_present and not rank_valid)

    cand_n = int(candidate_count) if candidate_count_valid else -1
    cand_ok = candidate_count_valid and cand_n >= int(min_candidates)

    metric_eligible = (
        bool(manifest_recoverable)
        and bool(scoring_completed) is True
        and candidate_count_valid
        and cand_ok
        and rank_valid
        and not consistency_error
    )

    if not metric_eligible and not audit_errors:
        if early_ineligible_reason:
            reasons.append(str(early_ineligible_reason))
        elif not cand_ok and candidate_count_valid:
            reasons.append("insufficient_candidates")
        elif not scoring_completed:
            reasons.append("scoring_incomplete")
        elif not manifest_recoverable and not rank_present:
            reasons.append("original_not_in_scored_candidates")

    # Deduplicate while preserving order
    seen: set = set()
    uniq_reasons: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq_reasons.append(r)
    seen_a: set = set()
    uniq_audit: List[str] = []
    for r in audit_errors:
        if r not in seen_a:
            seen_a.add(r)
            uniq_audit.append(r)

    primary = uniq_reasons[0] if uniq_reasons else (None if metric_eligible else "metric_ineligible")
    return {
        "manifest_recoverable": bool(manifest_recoverable),
        "metric_eligible": bool(metric_eligible),
        "metric_ineligible_reason": primary,
        "metric_ineligible_reasons": uniq_reasons,
        "metric_audit_error": bool(uniq_audit),
        "metric_audit_errors": uniq_audit,
        "rank_present": rank_present,
        "rank_valid": rank_valid,
        "candidate_count_valid": candidate_count_valid,
        "consistency_error": consistency_error,
    }
