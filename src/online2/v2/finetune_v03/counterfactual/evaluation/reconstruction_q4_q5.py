"""Strict Q4/Q5 aggregation from per-MG rows (metric_eligible contract)."""

from __future__ import annotations

import numbers
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .reconstruction_selection import MIN_RECONSTRUCTION_CANDIDATES


def is_strict_int(value: object) -> bool:
    """True for int / numpy·pandas Integral; False for bool and floats."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def aggregate_q4_q5_from_per_mg(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate Q4/Q5 only from explicitly metric_eligible rows.

    Contract for metric_eligible=True:
    - candidate_count is strict int and >= MIN_RECONSTRUCTION_CANDIDATES
    - scoring_completed is True
    - original_rank is strict int and >= 1
    - recall_at_3 is strict int in {0, 1}
    - 0 < random_recall_at_3 <= 1 (float allowed; bool rejected)
    """
    selected = list(rows)
    missing_eligibility = 0
    eligible: List[Mapping[str, Any]] = []
    ineligible: List[Mapping[str, Any]] = []
    ineligible_reasons: Counter = Counter()
    audit_errors: List[str] = []

    for i, row in enumerate(selected):
        if "metric_eligible" not in row:
            missing_eligibility += 1
            audit_errors.append(f"row[{i}].missing_metric_eligible")
            continue
        if row.get("metric_audit_error") is True:
            detail = row.get("metric_audit_errors") or row.get("metric_ineligible_reasons") or []
            audit_errors.append(
                f"row[{i}].metric_audit_error:{detail!r}"
            )
            # still classify for counts, but audit already failed
            flag = row.get("metric_eligible")
            if flag is False:
                reason = row.get("metric_ineligible_reason")
                reasons_list = row.get("metric_ineligible_reasons")
                if not reason and isinstance(reasons_list, (list, tuple)) and reasons_list:
                    reason = reasons_list[0]
                if reason:
                    ineligible.append(row)
                    ineligible_reasons[str(reason)] += 1
            continue
        flag = row.get("metric_eligible")
        if flag is True:
            cand = row.get("candidate_count")
            if not is_strict_int(cand):
                audit_errors.append(
                    f"row[{i}].metric_eligible_invalid_candidate_count_type:{cand!r}"
                )
                continue
            cand_n = int(cand)  # safe: already strict Integral, not bool
            if cand_n < int(MIN_RECONSTRUCTION_CANDIDATES):
                audit_errors.append(
                    f"row[{i}].metric_eligible_insufficient_candidate_count:{cand!r}"
                )
                continue
            if row.get("scoring_completed") is not True:
                audit_errors.append(f"row[{i}].metric_eligible_scoring_not_completed")
                continue
            # Eligible rows must also be manifest_recoverable when the field is present
            if "manifest_recoverable" in row and row.get("manifest_recoverable") is not True:
                audit_errors.append(
                    f"row[{i}].metric_eligible_without_manifest_recoverable"
                )
                continue
            rank = row.get("original_rank")
            if not is_strict_int(rank):
                audit_errors.append(
                    f"row[{i}].metric_eligible_invalid_original_rank_type:{rank!r}"
                )
                continue
            rank_n = int(rank)
            if rank_n < 1:
                audit_errors.append(
                    f"row[{i}].metric_eligible_invalid_original_rank:{rank!r}"
                )
                continue
            r3 = row.get("recall_at_3")
            if not is_strict_int(r3) or int(r3) not in (0, 1):
                audit_errors.append(
                    f"row[{i}].metric_eligible_invalid_recall_at_3:{r3!r}"
                )
                continue
            rand = row.get("random_recall_at_3")
            if isinstance(rand, bool) or rand is None:
                audit_errors.append(
                    f"row[{i}].metric_eligible_invalid_random_recall_at_3:{rand!r}"
                )
                continue
            try:
                rand_f = float(rand)
            except (TypeError, ValueError):
                audit_errors.append(
                    f"row[{i}].metric_eligible_invalid_random_recall_at_3:{rand!r}"
                )
                continue
            if not (0.0 < rand_f <= 1.0):
                audit_errors.append(
                    f"row[{i}].metric_eligible_invalid_random_recall_at_3:{rand!r}"
                )
                continue
            eligible.append(row)
        elif flag is False:
            reason = row.get("metric_ineligible_reason")
            reasons_list = row.get("metric_ineligible_reasons")
            if not reason and isinstance(reasons_list, (list, tuple)) and reasons_list:
                reason = reasons_list[0]
            if not reason:
                audit_errors.append(f"row[{i}].metric_ineligible_missing_reason")
                continue
            ineligible.append(row)
            ineligible_reasons[str(reason)] += 1
        else:
            audit_errors.append(f"row[{i}].metric_eligible_invalid:{flag!r}")

    base = {
        "selected_mg_count": len(selected),
        "metric_eligible_mg_count": len(eligible),
        "metric_ineligible_mg_count": len(ineligible),
        "missing_metric_eligibility_count": missing_eligibility,
        "metric_ineligible_reason_counts": dict(sorted(ineligible_reasons.items())),
        "audit_errors": audit_errors,
        "min_reconstruction_candidates": int(MIN_RECONSTRUCTION_CANDIDATES),
    }

    if audit_errors or missing_eligibility > 0:
        return {
            **base,
            "q4_q5_evaluable": False,
            "reconstruction_metric_audit_status": "FAIL",
            "q4_q5_status": "NOT_EVALUATED",
            "mean_recall_at_3": None,
            "mean_random_recall_at_3": None,
            "lift": None,
            "q4": None,
            "q5": None,
        }

    if not eligible:
        return {
            **base,
            "q4_q5_evaluable": False,
            "reconstruction_metric_audit_status": "PASS",
            "q4_q5_status": "NOT_EVALUATED",
            "mean_recall_at_3": None,
            "mean_random_recall_at_3": None,
            "lift": None,
            "q4": None,
            "q5": None,
        }

    n = len(eligible)
    mean_r3 = sum(float(r.get("recall_at_3") or 0) for r in eligible) / n
    mean_rand = sum(float(r.get("random_recall_at_3") or 0) for r in eligible) / n
    lift: Optional[float] = (mean_r3 / mean_rand) if mean_rand > 0 else 0.0
    q4 = mean_r3 > mean_rand
    q5 = bool(lift is not None and lift > 1.0)
    return {
        **base,
        "q4_q5_evaluable": True,
        "reconstruction_metric_audit_status": "PASS",
        "q4_q5_status": "EVALUATED",
        "mean_recall_at_3": mean_r3,
        "mean_random_recall_at_3": mean_rand,
        "lift": lift,
        "q4": q4,
        "q5": q5,
    }
