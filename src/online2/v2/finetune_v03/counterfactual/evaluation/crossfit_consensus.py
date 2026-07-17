"""Crossfit consensus: identity agreement + holdout-valid split rate."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence


def summarize_crossfit_splits(
    split_results: Sequence[Mapping[str, Any]],
    *,
    min_agreement_splits: int = 2,
    min_holdout_valid_splits: int = 2,
) -> Dict[str, Any]:
    """Aggregate independent split-level CF results.

    Does NOT average delta_r across different candidates.
    Consensus requires:
      canonical locus/feature/direction agreement ≥ min_agreement_splits
      AND holdout_effect_valid ≥ min_holdout_valid_splits
      AND Gate4 pass + non-OPERATIONAL on agreed splits
    """
    total = len(split_results)
    if total == 0:
        return {
            "crossfit_consensus_status": "UNSTABLE_OR_UNVALIDATED",
            "final_cf_candidate": None,
            "candidate_agreement_splits": 0,
            "holdout_valid_splits": 0,
            "total_splits": 0,
            "split_results": [],
        }

    keys = []
    holdout_valid = 0
    for row in split_results:
        key = str(
            row.get("canonical_locus_key")
            or "|".join(
                [
                    str(row.get("feature") or ""),
                    str(row.get("locus_id") or ""),
                    str(row.get("edit_direction") or ""),
                ]
            )
        )
        keys.append(key)
        if row.get("holdout_effect_valid") is True or row.get("material_improvement_holdout") is True:
            # only count if this split's selected candidate matches later consensus key
            pass
        if bool(row.get("holdout_effect_valid") or row.get("material_improvement_holdout")):
            holdout_valid += 0  # counted after consensus key chosen

    counts = Counter(keys)
    best_key, agree_n = counts.most_common(1)[0]
    agreed_rows = [r for r, k in zip(split_results, keys) if k == best_key]

    holdout_ok_n = sum(
        1
        for r in agreed_rows
        if bool(r.get("holdout_effect_valid") or r.get("material_improvement_holdout"))
    )
    gate4_ok = all(
        str(r.get("gate4_status") or "PASSED") == "PASSED" for r in agreed_rows
    )
    operational_ok = all(
        str(r.get("operational_eligibility") or "OK") != "OPERATIONAL" for r in agreed_rows
    )

    stable = (
        agree_n >= int(min_agreement_splits)
        and holdout_ok_n >= int(min_holdout_valid_splits)
        and gate4_ok
        and operational_ok
        and best_key
        and best_key != "|"
        and not any(bool(r.get("is_noop")) for r in agreed_rows)
    )

    final = None
    if stable:
        # pick representative from agreed rows with best (lowest) holdout delta if present
        def _hkey(r: Mapping[str, Any]):
            dr = r.get("delta_r_holdout")
            return (float(dr) if dr is not None else 0.0, str(r.get("candidate_id") or ""))

        final = dict(sorted(agreed_rows, key=_hkey)[0])
        final["crossfit_consensus_member"] = True

    return {
        "crossfit_consensus_status": "STABLE_VALID_CF" if stable else "UNSTABLE_OR_UNVALIDATED",
        "final_cf_candidate": final,
        "candidate_agreement_splits": int(agree_n),
        "holdout_valid_splits": int(holdout_ok_n),
        "total_splits": int(total),
        "canonical_locus_key": best_key if agree_n else None,
        "split_results": [dict(r) for r in split_results],
        # Explicitly do not report mean_delta_r across heterogeneous candidates
        "mean_delta_r_holdout": None,
        "note": "split-level results preserved; heterogeneous candidate ΔR not averaged",
    }
