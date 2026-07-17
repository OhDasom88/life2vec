"""CF validity schema — cohort-aware (Problem crossfit vs Example OOF-null)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .fold_execution_context import EVAL_CROSSFIT, EVAL_OOF_REF, FoldExecutionContext


STATUS_INDEPENDENT = "INDEPENDENTLY_EVALUATED"
STATUS_NOT_INDEPENDENT = "NOT_INDEPENDENTLY_EVALUABLE"


def _truthy_all(flags: Sequence[Optional[bool]]) -> bool:
    return all(bool(x) for x in flags)


def build_cf_validity(
    *,
    ctx: FoldExecutionContext,
    structurally_valid: bool,
    raw_edit_valid: bool,
    retokenization_valid: bool,
    stage_a_valid: bool,
    search_effect_valid: Optional[bool],
    holdout_effect_valid: Optional[bool],
    reason_codes: Optional[Sequence[str]] = None,
    is_noop: bool = False,
) -> Dict[str, Any]:
    """Build validity record.

    Example 35 (oof_reference_only): cf_valid / search / holdout effects are null.
    Problem 20 (crossfit): cf_valid = all six booleans true.
    """
    codes = list(reason_codes or [])
    if ctx.evaluation_mode == EVAL_OOF_REF or not ctx.independently_evaluable:
        return {
            "evaluation_mode": EVAL_OOF_REF,
            "structurally_valid": bool(structurally_valid),
            "raw_edit_valid": bool(raw_edit_valid),
            "retokenization_valid": bool(retokenization_valid),
            "stage_a_valid": bool(stage_a_valid),
            "search_effect_valid": None,
            "holdout_effect_valid": None,
            "cf_valid": None,
            "cf_evaluation_status": STATUS_NOT_INDEPENDENT,
            "is_noop": bool(is_noop),
            "reason_codes": codes,
            "management_intervention_allowed": False,
        }

    search_ok = bool(search_effect_valid) if search_effect_valid is not None else False
    holdout_ok = bool(holdout_effect_valid) if holdout_effect_valid is not None else False
    structural = [
        bool(structurally_valid),
        bool(raw_edit_valid),
        bool(retokenization_valid),
        bool(stage_a_valid),
    ]
    cf_valid = _truthy_all(structural + [search_ok, holdout_ok]) and not is_noop
    if is_noop:
        codes = codes or ["CANONICAL_NO_OP"]
    elif not cf_valid and "NO_VALID_CF" not in codes:
        if not search_ok:
            codes.append("SEARCH_EFFECT_INVALID")
        if not holdout_ok:
            codes.append("HOLDOUT_EFFECT_INVALID")
    return {
        "evaluation_mode": EVAL_CROSSFIT,
        "structurally_valid": bool(structurally_valid),
        "raw_edit_valid": bool(raw_edit_valid),
        "retokenization_valid": bool(retokenization_valid),
        "stage_a_valid": bool(stage_a_valid),
        "search_effect_valid": search_ok,
        "holdout_effect_valid": holdout_ok,
        "cf_valid": bool(cf_valid),
        "cf_evaluation_status": STATUS_INDEPENDENT,
        "is_noop": bool(is_noop),
        "reason_codes": codes,
        "management_intervention_allowed": bool(cf_valid),
    }


def llm_case_payload(
    *,
    selected: Optional[Mapping[str, Any]],
    validity: Mapping[str, Any],
    invalid_candidates_excluded: int = 0,
) -> Dict[str, Any]:
    """Case-level LLM payload: never include invalid intervention details."""
    is_noop = bool(
        (selected or {}).get("is_noop")
        or str((selected or {}).get("source") or "").startswith("noop")
        or validity.get("cf_valid") is not True
    )
    if validity.get("cf_evaluation_status") == STATUS_NOT_INDEPENDENT:
        status = "NOT_INDEPENDENTLY_EVALUABLE"
        allowed = False
    elif validity.get("cf_valid") is True and not is_noop:
        status = "VALID_CF"
        allowed = True
    else:
        status = "NO_VALID_CF"
        allowed = False
    return {
        "cf_status": status,
        "selected_candidate": "NO_OP" if is_noop or not allowed else str(
            (selected or {}).get("candidate_id") or (selected or {}).get("source") or "UNKNOWN"
        ),
        "invalid_candidates_excluded": int(invalid_candidates_excluded),
        "management_intervention_allowed": bool(allowed),
        "cf_evaluation_status": validity.get("cf_evaluation_status"),
        "reason_codes": list(validity.get("reason_codes") or []),
    }


def strip_invalid_candidate_details(
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop intervention details from non-valid candidates for LLM-facing exports."""
    out = []
    for c in candidates:
        row = dict(c)
        valid = row.get("cf_valid")
        if valid is not True:
            for k in (
                "target_raw",
                "proposed_token_bundle",
                "actual_retokenized_bundle",
                "locus",
                "mlm_proposed_bundle",
                "_after_batch",
            ):
                row.pop(k, None)
            row["intervention_detail_excluded"] = True
        out.append(row)
    return out
