"""Top-k MLM funnel conservation and A5 contract checks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


TERMINAL_FAILURE_REASONS = frozenset(
    {
        "ROLE_SIGNATURE_MISMATCH",
        "NON_INVERTIBLE_BUNDLE",
        "EMPTY_INTERVAL_INTERSECTION",
        "TARGET_RAW_COLLAPSED_TO_NOOP",
        "RETOKENIZATION_MISMATCH",
        "STAGE_A_REENCODE_FAILURE",
        "STAGE_A_PARITY_FAILURE",
        "CRITIC_EXECUTION_FAILURE",
        "INVALID_CRITIC_SCORE",
        "NON_MATERIAL_SEARCH_DELTA",
    }
)

INVERSION_FAIL_REASONS = frozenset(
    {
        "ROLE_SIGNATURE_MISMATCH",
        "NON_INVERTIBLE_BUNDLE",
        "EMPTY_INTERVAL_INTERSECTION",
        "TARGET_RAW_COLLAPSED_TO_NOOP",
    }
)
GATE4_FAIL_REASONS = frozenset({"RETOKENIZATION_MISMATCH"})
STAGE_A_FAIL_REASONS = frozenset({"STAGE_A_REENCODE_FAILURE", "STAGE_A_PARITY_FAILURE"})
CRITIC_EXEC_FAIL_REASONS = frozenset({"CRITIC_EXECUTION_FAILURE", "INVALID_CRITIC_SCORE"})
NON_MATERIAL_FAIL_REASONS = frozenset({"NON_MATERIAL_SEARCH_DELTA"})

SELECTION_TOPK = "TOPK_SELECTED"
SELECTION_PRUNED = "NOT_SELECTED_TOPK"
TERMINAL_VALID = "VALID"
TERMINAL_FAILED = "FAILED"


def empty_funnel_counts() -> Dict[str, int]:
    return {
        "n_bank_unique": 0,
        "n_scored_unique": 0,
        "n_topk": 0,
        "n_not_selected_topk": 0,
        "n_inversion_ok": 0,
        "n_gate4_pass": 0,
        "n_stage_a_ok": 0,
        "n_critic_scored": 0,
        "n_material_search_valid": 0,
        "n_inversion_terminal_fail": 0,
        "n_gate4_terminal_fail": 0,
        "n_stage_a_terminal_fail": 0,
        "n_critic_execution_terminal_fail": 0,
        "n_non_material_terminal_fail": 0,
        "n_other_postcritic_terminal_fail": 0,
    }


def _i(funnel: Mapping[str, Any], key: str) -> int:
    return int(funnel.get(key) or 0)


def validate_funnel_monotonicity(funnel: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    chain = (
        "n_bank_unique",
        "n_scored_unique",
        "n_topk",
        "n_inversion_ok",
        "n_gate4_pass",
        "n_stage_a_ok",
        "n_critic_scored",
        "n_material_search_valid",
    )
    vals = [_i(funnel, k) for k in chain]
    for a, b, ka, kb in zip(vals, vals[1:], chain, chain[1:]):
        if a < b:
            errs.append(f"monotonicity_violated:{ka}<{kb}")
    return len(errs) == 0, errs


def validate_funnel_conservation(funnel: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """Enforce Gate4→StageA→critic-execution partitioned conservation equations."""
    errs: List[str] = []
    n_topk = _i(funnel, "n_topk")
    n_inversion_ok = _i(funnel, "n_inversion_ok")
    n_inversion_fail = _i(funnel, "n_inversion_terminal_fail")
    n_gate4_pass = _i(funnel, "n_gate4_pass")
    n_gate4_fail = _i(funnel, "n_gate4_terminal_fail")
    n_stage_a_ok = _i(funnel, "n_stage_a_ok")
    n_stage_a_fail = _i(funnel, "n_stage_a_terminal_fail")
    n_critic_scored = _i(funnel, "n_critic_scored")
    n_critic_exec_fail = _i(funnel, "n_critic_execution_terminal_fail")
    n_material = _i(funnel, "n_material_search_valid")
    n_non_material = _i(funnel, "n_non_material_terminal_fail")
    n_other_post = _i(funnel, "n_other_postcritic_terminal_fail")

    if n_topk != n_inversion_ok + n_inversion_fail:
        errs.append("n_topk != n_inversion_ok + n_inversion_terminal_fail")
    if n_inversion_ok != n_gate4_pass + n_gate4_fail:
        errs.append("n_inversion_ok != n_gate4_pass + n_gate4_terminal_fail")
    if n_gate4_pass != n_stage_a_ok + n_stage_a_fail:
        errs.append("n_gate4_pass != n_stage_a_ok + n_stage_a_terminal_fail")
    if n_stage_a_ok != n_critic_scored + n_critic_exec_fail:
        errs.append("n_stage_a_ok != n_critic_scored + n_critic_execution_terminal_fail")
    if n_critic_scored != n_material + n_non_material + n_other_post:
        errs.append(
            "n_critic_scored != n_material_search_valid "
            "+ n_non_material_terminal_fail + n_other_postcritic_terminal_fail"
        )
    total_fail = (
        n_inversion_fail
        + n_gate4_fail
        + n_stage_a_fail
        + n_critic_exec_fail
        + n_non_material
        + n_other_post
    )
    if n_topk != n_material + total_fail:
        errs.append("n_topk != n_material_search_valid + Σ(terminal_fail)")
    return len(errs) == 0, errs


def validate_topk_terminal_outcomes(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[bool, List[str]]:
    """Each TOPK_SELECTED candidate has exactly one terminal outcome."""
    errs: List[str] = []
    for i, rec in enumerate(records):
        if str(rec.get("selection_status")) != SELECTION_TOPK:
            continue
        status = str(rec.get("terminal_status") or "")
        reason = rec.get("terminal_failure_reason")
        if status == TERMINAL_VALID:
            if reason not in (None, "", "null"):
                errs.append(f"record[{i}]: VALID must have null terminal_failure_reason")
        elif status == TERMINAL_FAILED:
            if reason is None or str(reason) not in TERMINAL_FAILURE_REASONS:
                errs.append(f"record[{i}]: FAILED needs known terminal_failure_reason")
        else:
            errs.append(f"record[{i}]: missing terminal_status")
    return len(errs) == 0, errs


def validate_non_topk_are_pruned(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    for i, rec in enumerate(records):
        if str(rec.get("selection_status")) == SELECTION_PRUNED:
            if rec.get("terminal_failure_reason") is not None:
                errs.append(f"record[{i}]: pruned must not carry terminal_failure_reason")
            if str(rec.get("terminal_status") or "") == TERMINAL_FAILED:
                errs.append(f"record[{i}]: pruned must not be FAILED")
    return len(errs) == 0, errs


def aggregate_topk_funnel(
    *,
    n_bank_unique: int,
    n_scored_unique: int,
    topk_records: Sequence[Mapping[str, Any]],
    n_not_selected_topk: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate TOPK_SELECTED records into conservation counts."""
    counts = empty_funnel_counts()
    counts["n_bank_unique"] = int(n_bank_unique)
    counts["n_scored_unique"] = int(n_scored_unique)
    topk = [r for r in topk_records if str(r.get("selection_status")) == SELECTION_TOPK]
    counts["n_topk"] = len(topk)
    if n_not_selected_topk is not None:
        counts["n_not_selected_topk"] = int(n_not_selected_topk)
    else:
        counts["n_not_selected_topk"] = max(0, counts["n_scored_unique"] - counts["n_topk"])

    for rec in topk:
        status = str(rec.get("terminal_status") or "")
        reason = str(rec.get("terminal_failure_reason") or "")
        if status == TERMINAL_VALID:
            counts["n_inversion_ok"] += 1
            counts["n_gate4_pass"] += 1
            counts["n_stage_a_ok"] += 1
            counts["n_critic_scored"] += 1
            counts["n_material_search_valid"] += 1
            continue
        if reason in INVERSION_FAIL_REASONS:
            counts["n_inversion_terminal_fail"] += 1
        elif reason in GATE4_FAIL_REASONS:
            counts["n_inversion_ok"] += 1
            counts["n_gate4_terminal_fail"] += 1
        elif reason in STAGE_A_FAIL_REASONS:
            counts["n_inversion_ok"] += 1
            counts["n_gate4_pass"] += 1
            counts["n_stage_a_terminal_fail"] += 1
        elif reason in CRITIC_EXEC_FAIL_REASONS:
            counts["n_inversion_ok"] += 1
            counts["n_gate4_pass"] += 1
            counts["n_stage_a_ok"] += 1
            counts["n_critic_execution_terminal_fail"] += 1
        elif reason in NON_MATERIAL_FAIL_REASONS:
            counts["n_inversion_ok"] += 1
            counts["n_gate4_pass"] += 1
            counts["n_stage_a_ok"] += 1
            counts["n_critic_scored"] += 1
            counts["n_non_material_terminal_fail"] += 1
        else:
            # reached critic scoring then other post-critic terminal
            counts["n_inversion_ok"] += 1
            counts["n_gate4_pass"] += 1
            counts["n_stage_a_ok"] += 1
            counts["n_critic_scored"] += 1
            counts["n_other_postcritic_terminal_fail"] += 1

    candidate_failure_counts: Dict[str, int] = {}
    for rec in topk:
        if str(rec.get("terminal_status")) == TERMINAL_FAILED:
            r = str(rec.get("terminal_failure_reason") or "UNKNOWN")
            candidate_failure_counts[r] = candidate_failure_counts.get(r, 0) + 1

    out = dict(counts)
    out["topk_records"] = [dict(r) for r in topk]
    out["candidate_failure_counts"] = candidate_failure_counts
    return out


def evaluate_a5_contracts(
    *,
    lift_meta: Optional[Mapping[str, Any]] = None,
    mlm_score_used_in_critic_ranking: bool = False,
    rng_or_grouped_masker_used: bool = False,
    funnel: Optional[Mapping[str, Any]] = None,
    topk_records: Optional[Sequence[Mapping[str, Any]]] = None,
    all_candidate_records: Optional[Sequence[Mapping[str, Any]]] = None,
    require_full_conservation: Optional[bool] = None,
) -> Dict[str, Any]:
    """A5 PASS requires mask/Gate4 invariants + funnel conservation + pruning rules."""
    lift = dict(lift_meta or {})
    checks: Dict[str, bool] = {}
    errs: List[str] = []

    if lift.get("modified_channels") is None:
        checks["token_id_only_mask"] = True
    else:
        checks["token_id_only_mask"] = list(lift.get("modified_channels") or []) == ["token_id"]
        if not checks["token_id_only_mask"]:
            errs.append("token_id_only_mask")

    checks["window_offset_ok"] = bool(lift.get("unchanged_non_target_positions", True))
    checks["auxiliary_attention_preserved"] = bool(
        lift.get("unchanged_auxiliary_channels", True)
    ) and bool(lift.get("unchanged_attention_mask", True))
    checks["mlm_score_not_in_critic_ranking"] = not bool(mlm_score_used_in_critic_ranking)
    checks["no_rng_grouped_masker"] = not bool(rng_or_grouped_masker_used)
    checks["position_preserving_gate4"] = True

    funnel_d = dict(funnel or {})
    records = list(topk_records or funnel_d.get("topk_records") or [])
    pending = bool(funnel_d.get("pending_downstream_stages"))
    if require_full_conservation is None:
        require_full_conservation = not pending

    if require_full_conservation and funnel_d:
        mono_ok, mono_errs = validate_funnel_monotonicity(funnel_d)
        cons_ok, cons_errs = validate_funnel_conservation(funnel_d)
    else:
        mono_ok, mono_errs = (True, [])
        cons_ok, cons_errs = (True, [])
        # still require inversion partition when topk present
        if funnel_d and "n_topk" in funnel_d:
            if _i(funnel_d, "n_topk") != _i(funnel_d, "n_inversion_ok") + _i(
                funnel_d, "n_inversion_terminal_fail"
            ):
                cons_ok = False
                cons_errs = ["n_topk != n_inversion_ok + n_inversion_terminal_fail"]

    # Only validate terminal outcomes for finalized records
    finalized = [
        r
        for r in records
        if str(r.get("terminal_status")) in {TERMINAL_VALID, TERMINAL_FAILED}
    ]
    term_ok, term_errs = validate_topk_terminal_outcomes(finalized) if finalized else (True, [])
    prune_src = list(all_candidate_records or records)
    prune_ok, prune_errs = (
        validate_non_topk_are_pruned(prune_src) if prune_src else (True, [])
    )

    checks["funnel_monotonicity"] = mono_ok
    checks["funnel_conservation"] = cons_ok
    checks["topk_single_terminal_outcome"] = term_ok
    checks["non_topk_pruning_only"] = prune_ok
    errs.extend(mono_errs + cons_errs + term_errs + prune_errs)

    if not funnel_d and not lift:
        passed = (
            checks["mlm_score_not_in_critic_ranking"] and checks["no_rng_grouped_masker"]
        )
    else:
        passed = all(
            checks[k]
            for k in (
                "token_id_only_mask",
                "window_offset_ok",
                "auxiliary_attention_preserved",
                "position_preserving_gate4",
                "mlm_score_not_in_critic_ranking",
                "no_rng_grouped_masker",
                "funnel_monotonicity",
                "funnel_conservation",
                "topk_single_terminal_outcome",
                "non_topk_pruning_only",
            )
        )

    return {
        "a5_pass": bool(passed),
        "checks": checks,
        "errors": errs,
        "pending_downstream_stages": pending,
    }


# ---------------------------------------------------------------------------
# CF-0 candidate-level funnel (cf0_v1)
# ---------------------------------------------------------------------------

CF0_SCHEMA_VERSION = "cf0_v1"
PLAUSIBILITY_DEFINITION_CF0 = "BANK_SUPPORT_PROXY"

TERMINAL_REJECTED = "REJECTED"
TERMINAL_EXECUTION_ERROR = "EXECUTION_ERROR"

CASE_EXECUTION_ERROR_STAGES = frozenset(
    {
        "FEATURE_SCHEMA_DISPATCH",
        "CANDIDATE_GENERATION",
        "FEATURE_BIN_LOOKUP",
        "RETOKENIZATION",
        "STAGE_A_FORWARD",
        "CRITIC_FORWARD",
        "ARTIFACT_LOAD",
        "UNHANDLED_EXCEPTION",
    }
)

EXECUTION_ERROR_REASONS = frozenset(
    {
        "STAGE_A_REENCODE_FAILURE",
        "STAGE_A_PARITY_FAILURE",
        "STAGE_A_FORWARD_ERROR",
        "CRITIC_EXECUTION_FAILURE",
        "INVALID_CRITIC_SCORE",
        "ARTIFACT_MISSING",
        "UNHANDLED_EXCEPTION",
    }
)

CANDIDATE_REJECTION_REASONS = frozenset(
    {
        "ROLE_SIGNATURE_MISMATCH",
        "NON_INVERTIBLE_BUNDLE",
        "EMPTY_INTERVAL_INTERSECTION",
        "TARGET_RAW_COLLAPSED_TO_NOOP",
        "PHYSICAL_BOUND_VIOLATION",
        "EMPTY_SAFE_INTERIOR",
        "GATE0_BASELINE_FAIL",
        "RETOKENIZATION_MISMATCH",
        "NON_MATERIAL_SEARCH_DELTA",
        "LOW_FOLD_AGREEMENT",
        "HOLDOUT_EFFECT_INVALID",
        "NO_ELIGIBLE_LOCUS",
    }
)

CF0_FUNNEL_STAGES = (
    "editable_loci_count",
    "raw_generated_candidate_count",
    "deduplicated_candidate_count",
    "bank_supported_candidate_count",
    "decoder_scored_candidate_count",
    "hard_constraint_pass_count",
    "plausibility_pass_count",
    "effect_pass_count",
    "stability_pass_count",
    "valid_cf_count",
    "selected_non_noop_count",
)


def _sources_of(row: Mapping[str, Any]) -> set:
    srcs = row.get("candidate_sources")
    if srcs:
        return {str(s) for s in srcs}
    src = row.get("source")
    return {str(src)} if src is not None else set()


def is_mlm_origin(row: Mapping[str, Any]) -> bool:
    return "constrained_mlm" in _sources_of(row)


def count_mlm_provenance(
    *,
    raw_proposals: Sequence[Mapping[str, Any]],
    dedup_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """Separate MLM raw proposals, MLM-origin unique candidates, and valid MLM CF."""
    n_mlm_raw = sum(
        1
        for r in raw_proposals
        if str(r.get("source")) == "constrained_mlm" and not bool(r.get("is_noop"))
    )
    n_mlm_origin = sum(
        1 for r in dedup_rows if (not bool(r.get("is_noop"))) and is_mlm_origin(r)
    )
    n_valid_mlm = sum(
        1
        for r in dedup_rows
        if (not bool(r.get("is_noop")))
        and is_mlm_origin(r)
        and r.get("cf_valid") is True
    )
    return {
        "n_mlm_raw_proposals": int(n_mlm_raw),
        "n_mlm_origin_candidates": int(n_mlm_origin),
        "n_valid_mlm_cf": int(n_valid_mlm),
    }


def assign_primary_terminal(
    row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assign exactly one primary terminal outcome for a non-NO_OP candidate."""
    all_reasons: List[str] = []
    if row.get("cf_valid") is True:
        return {
            "terminal_status": TERMINAL_VALID,
            "terminal_failure_reason": None,
            "primary_terminal_failure_reason": None,
            "all_failure_reasons": [],
            "is_execution_error": False,
        }

    inv = row.get("inversion_pass")
    gate0 = row.get("gate0_pass")
    gate4_raw = row.get("gate4_raw_pass")
    if gate4_raw is None:
        gate4_raw = row.get("gate4_pass")
    if inv is False:
        reason = str(row.get("inversion_reason_code") or "NON_INVERTIBLE_BUNDLE")
        all_reasons.append(reason)
    elif gate0 is False:
        all_reasons.append("GATE0_BASELINE_FAIL")
    elif gate4_raw is False:
        all_reasons.append("RETOKENIZATION_MISMATCH")
    elif row.get("stage_a_pass") is False:
        sa_reason = str(row.get("stage_a_failure_reason") or "STAGE_A_REENCODE_FAILURE")
        all_reasons.append(sa_reason)
    elif row.get("critic_scored") is False:
        cr = str(row.get("critic_failure_reason") or "CRITIC_EXECUTION_FAILURE")
        all_reasons.append(cr)
    elif row.get("hard_constraint_pass") and row.get("critic_scored"):
        folds = row.get("delta_r_folds_search") or row.get("delta_r_folds")
        all_neg = True
        if folds is not None:
            try:
                all_neg = all(float(x) < 0.0 for x in list(folds))
            except Exception:
                all_neg = False
        if not all_neg:
            all_reasons.append("LOW_FOLD_AGREEMENT")
        if not bool(row.get("effect_pass") or row.get("search_material")):
            if "LOW_FOLD_AGREEMENT" not in all_reasons:
                all_reasons.append("NON_MATERIAL_SEARCH_DELTA")
        if (
            row.get("effect_pass") or row.get("search_material")
        ) and row.get("holdout_effect_valid") is False:
            all_reasons.append("HOLDOUT_EFFECT_INVALID")

    # Preserve any explicit reasons already attached
    for r in row.get("all_failure_reasons") or []:
        if str(r) not in all_reasons:
            all_reasons.append(str(r))
    for r in row.get("reason_codes") or []:
        rs = str(r)
        if rs in EXECUTION_ERROR_REASONS or rs in CANDIDATE_REJECTION_REASONS:
            if rs not in all_reasons:
                all_reasons.append(rs)

    if not all_reasons:
        all_reasons.append("NON_MATERIAL_SEARCH_DELTA")

    # Pipeline-order primary: first applicable failure
    priority = [
        "ROLE_SIGNATURE_MISMATCH",
        "NON_INVERTIBLE_BUNDLE",
        "EMPTY_INTERVAL_INTERSECTION",
        "PHYSICAL_BOUND_VIOLATION",
        "EMPTY_SAFE_INTERIOR",
        "TARGET_RAW_COLLAPSED_TO_NOOP",
        "GATE0_BASELINE_FAIL",
        "RETOKENIZATION_MISMATCH",
        "STAGE_A_REENCODE_FAILURE",
        "STAGE_A_PARITY_FAILURE",
        "STAGE_A_FORWARD_ERROR",
        "CRITIC_EXECUTION_FAILURE",
        "INVALID_CRITIC_SCORE",
        "ARTIFACT_MISSING",
        "UNHANDLED_EXCEPTION",
        "LOW_FOLD_AGREEMENT",
        "NON_MATERIAL_SEARCH_DELTA",
        "HOLDOUT_EFFECT_INVALID",
        "NO_ELIGIBLE_LOCUS",
    ]
    primary = all_reasons[0]
    for p in priority:
        if p in all_reasons:
            primary = p
            break

    is_exec = primary in EXECUTION_ERROR_REASONS
    return {
        "terminal_status": TERMINAL_EXECUTION_ERROR if is_exec else TERMINAL_REJECTED,
        "terminal_failure_reason": primary,
        "primary_terminal_failure_reason": primary,
        "all_failure_reasons": all_reasons,
        "is_execution_error": is_exec,
    }


def first_zero_stage(counts: Mapping[str, Any]) -> Optional[str]:
    """Return the first CF-0 funnel stage whose count is 0."""
    for stage in CF0_FUNNEL_STAGES:
        if int(counts.get(stage) or 0) == 0:
            return stage
    return None


def validate_cf0_monotonicity(counts: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    chain = (
        "deduplicated_candidate_count",
        "bank_supported_candidate_count",
        "decoder_scored_candidate_count",
        "hard_constraint_pass_count",
        "plausibility_pass_count",
        "effect_pass_count",
        "stability_pass_count",
        "valid_cf_count",
        "selected_non_noop_count",
    )
    errs: List[str] = []
    vals = [int(counts.get(k) or 0) for k in chain]
    for a, b, ka, kb in zip(vals, vals[1:], chain, chain[1:]):
        if a < b:
            errs.append(f"monotonicity_violated:{ka}<{kb}")
    raw_n = int(counts.get("raw_generated_candidate_count") or 0)
    dedup_n = int(counts.get("deduplicated_candidate_count") or 0)
    if raw_n < dedup_n:
        errs.append("raw_generated_candidate_count < deduplicated_candidate_count")
    return len(errs) == 0, errs


def validate_cf0_conservation(counts: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    dedup = int(counts.get("deduplicated_candidate_count") or 0)
    valid = int(counts.get("valid_cf_count") or 0)
    rej = int(counts.get("candidate_rejection_count") or 0)
    exe = int(counts.get("candidate_execution_error_count") or 0)
    if dedup != valid + rej + exe:
        errs.append(
            "deduplicated_candidate_count != valid_cf_count "
            "+ candidate_rejection_count + candidate_execution_error_count"
        )
    g0 = int(counts.get("gate0_pass_count") or 0)
    inv = int(counts.get("inversion_pass_count") or 0)
    g4 = int(counts.get("gate4_pass_count") or 0)
    hard = int(counts.get("hard_constraint_pass_count") or 0)
    if hard > g0 or hard > inv or hard > g4:
        errs.append("hard_constraint_pass_count exceeds gate0/inversion/gate4 pass counts")
    if counts.get("hard_equals_gate4_invariant_expected") and hard != g4:
        errs.append("hard_constraint_pass_count != gate4_pass_count under sequential invariant")
    term_sum = sum(int(v) for v in (counts.get("terminal_reason_counts") or {}).values())
    if term_sum != rej:
        errs.append("sum(terminal_reason_counts) != candidate_rejection_count")
    exec_sum = sum(int(v) for v in (counts.get("execution_error_reason_counts") or {}).values())
    if exec_sum != exe:
        errs.append("sum(execution_error_reason_counts) != candidate_execution_error_count")
    return len(errs) == 0, errs


def classify_case_execution_error_stage(exc: BaseException | str) -> str:
    """Map a case-level exception to a dynamic first_execution_error_stage."""
    msg = str(exc)
    low = msg.lower()
    if "no bin edges" in low or "bin edges" in low:
        return "FEATURE_BIN_LOOKUP"
    if "feature schema" in low or "unsupported_continuous_bin" in low:
        return "FEATURE_SCHEMA_DISPATCH"
    if "retoken" in low or "gate4" in low or "gate0" in low:
        return "RETOKENIZATION"
    if "stage a" in low or "stage_a" in low or "reencode" in low:
        return "STAGE_A_FORWARD"
    if "critic" in low:
        return "CRITIC_FORWARD"
    if "artifact" in low or "file not found" in low or "no such file" in low:
        return "ARTIFACT_LOAD"
    if "candidate" in low and "generat" in low:
        return "CANDIDATE_GENERATION"
    return "UNHANDLED_EXCEPTION"


def is_observational_sensitivity_row(row: Mapping[str, Any]) -> bool:
    if str(row.get("operational_eligibility") or "") == "OBSERVATIONAL_SENSITIVITY_ONLY":
        return True
    labels = {str(x) for x in (row.get("validity_labels") or [])}
    if "MODEL_SENSITIVITY_ONLY" in labels or "NOT_RECOMMENDATION" in labels:
        return True
    if str(row.get("source") or "") == "schema_categorical_adjacent":
        return True
    if str(row.get("candidate_source") or "") == "schema_categorical_adjacent":
        return True
    return False


def is_recommendation_eligible_row(row: Mapping[str, Any]) -> bool:
    if row.get("cf_valid") is not True:
        return False
    if is_observational_sensitivity_row(row):
        return False
    if row.get("actionability") is not True:
        return False
    if str(row.get("operational_eligibility") or "") != "RECOMMENDATION_ELIGIBLE":
        return False
    if row.get("recommendation_eligible") is False:
        return False
    return True


def build_cf0_case_funnel(
    *,
    case_id: str,
    cohort: str,
    editable_loci_count: int,
    raw_proposals: Sequence[Mapping[str, Any]],
    dedup_rows: Sequence[Mapping[str, Any]],
    selected: Optional[Mapping[str, Any]] = None,
    bank_unique_bundle_count: int = 0,
    decoder_scored_unique_bundle_count: int = 0,
    warnings: Optional[Sequence[Mapping[str, Any]]] = None,
    evaluation_mode: Optional[str] = None,
    unsupported_locus_count: int = 0,
    candidate_generation_skip_reason_counts: Optional[Mapping[str, int]] = None,
    edit_strategy_kind: Optional[str] = None,
    case_execution_error: bool = False,
    first_execution_error_stage: Optional[str] = None,
    execution_error_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build per-case cf0_v1 funnel from non-NO_OP deduplicated candidate ledger."""
    non_noop = [dict(r) for r in dedup_rows if not bool(r.get("is_noop"))]
    noops = [r for r in dedup_rows if bool(r.get("is_noop"))]
    raw_non_noop = [r for r in raw_proposals if not bool(r.get("is_noop"))]

    finalized: List[Dict[str, Any]] = []
    for row in non_noop:
        term = assign_primary_terminal(row)
        out = dict(row)
        out.update(term)
        # Stage flags with defaults from pass bits
        out["bank_supported"] = bool(row.get("bank_supported", True))
        out["decoder_scored"] = bool(row.get("decoder_scored", True))
        out["inversion_pass"] = bool(row.get("inversion_pass", True))
        out["gate0_pass"] = bool(row.get("gate0_pass", False))
        out["gate4_pass"] = bool(row.get("gate4_pass", False))
        out["hard_constraint_pass"] = bool(
            row.get("hard_constraint_pass")
            if row.get("hard_constraint_pass") is not None
            else (
                out["inversion_pass"]
                and out["gate0_pass"]
                and out["gate4_pass"]
            )
        )
        out["plausibility_pass"] = bool(
            row.get("plausibility_pass")
            if row.get("plausibility_pass") is not None
            else (out["hard_constraint_pass"] and out["bank_supported"])
        )
        out["effect_pass"] = bool(row.get("effect_pass") or row.get("search_material"))
        out["stability_pass"] = bool(
            row.get("stability_pass")
            if row.get("stability_pass") is not None
            else out["effect_pass"]
        )
        out["cf_valid"] = row.get("cf_valid") is True
        finalized.append(out)

    selected = dict(selected or {})
    selected_non_noop = 0
    selected_actionable_non_noop = 0
    noop_selected = False
    if selected:
        if (
            bool(selected.get("is_noop"))
            or str(selected.get("source")) == "noop"
            or str(selected.get("selection_reason") or "")
            == "no_material_candidate_canonical_noop"
        ):
            noop_selected = True
        else:
            selected_non_noop = 1
            if (
                selected.get("actionability") is True
                and str(selected.get("operational_eligibility") or "")
                == "RECOMMENDATION_ELIGIBLE"
                and not is_observational_sensitivity_row(selected)
            ):
                selected_actionable_non_noop = 1

    gate0_pass_count = sum(1 for f in finalized if f.get("gate0_pass"))
    inversion_pass_count = sum(1 for f in finalized if f.get("inversion_pass"))
    gate4_pass_count = sum(1 for f in finalized if f.get("gate4_pass"))
    hard_constraint_pass_count = sum(1 for f in finalized if f.get("hard_constraint_pass"))

    model_valid = [f for f in finalized if f.get("cf_valid") is True]
    observational_valid = [f for f in model_valid if is_observational_sensitivity_row(f)]
    recommendation_valid = [f for f in model_valid if is_recommendation_eligible_row(f)]

    counts: Dict[str, Any] = {
        "editable_loci_count": int(editable_loci_count),
        "unsupported_locus_count": int(unsupported_locus_count),
        "candidate_generation_skip_reason_counts": {
            str(k): int(v)
            for k, v in dict(candidate_generation_skip_reason_counts or {}).items()
        },
        "raw_generated_candidate_count": int(len(raw_non_noop)),
        "generated_candidate_count": int(len(finalized)),
        "deduplicated_candidate_count": int(len(finalized)),
        "bank_supported_candidate_count": sum(1 for f in finalized if f.get("bank_supported")),
        "decoder_scored_candidate_count": sum(1 for f in finalized if f.get("decoder_scored")),
        "gate0_pass_count": int(gate0_pass_count),
        "inversion_pass_count": int(inversion_pass_count),
        "gate4_pass_count": int(gate4_pass_count),
        "hard_constraint_pass_count": int(hard_constraint_pass_count),
        "plausibility_pass_count": sum(1 for f in finalized if f.get("plausibility_pass")),
        "effect_pass_count": sum(1 for f in finalized if f.get("effect_pass")),
        "stability_pass_count": sum(1 for f in finalized if f.get("stability_pass")),
        # model_valid_cf: schema/effect model conditions; conservation uses this.
        "valid_cf_count": int(len(model_valid)),
        "model_valid_cf_count": int(len(model_valid)),
        "observational_sensitivity_valid_count": int(len(observational_valid)),
        "recommendation_valid_cf_count": int(len(recommendation_valid)),
        "selected_non_noop_count": int(selected_non_noop),
        "selected_actionable_non_noop_count": int(selected_actionable_non_noop),
        "valid_cf_definition": (
            "model_valid_cf=cf_valid; recommendation_valid_cf requires "
            "actionability and RECOMMENDATION_ELIGIBLE; observational "
            "sensitivity never counts as Natural Valid Action CF"
        ),
    }

    term_counts: Dict[str, int] = {}
    exec_counts: Dict[str, int] = {}
    rej_n = 0
    exec_n = 0
    pending = False
    for f in finalized:
        status = str(f.get("terminal_status") or "")
        if status == TERMINAL_VALID:
            continue
        if status not in {TERMINAL_REJECTED, TERMINAL_EXECUTION_ERROR, TERMINAL_FAILED}:
            pending = True
            continue
        reason = str(
            f.get("primary_terminal_failure_reason")
            or f.get("terminal_failure_reason")
            or "UNKNOWN"
        )
        if f.get("is_execution_error") or status == TERMINAL_EXECUTION_ERROR:
            exec_n += 1
            exec_counts[reason] = exec_counts.get(reason, 0) + 1
        else:
            rej_n += 1
            term_counts[reason] = term_counts.get(reason, 0) + 1

    counts["candidate_rejection_count"] = int(rej_n)
    counts["candidate_execution_error_count"] = int(exec_n)
    counts["terminal_reason_counts"] = term_counts
    counts["execution_error_reason_counts"] = exec_counts
    # When hard == gate4 because gate4 is only evaluated after gate0+inversion
    counts["hard_equals_gate4_invariant_expected"] = True

    mlm = count_mlm_provenance(raw_proposals=raw_proposals, dedup_rows=finalized)
    # Recompute n_valid_mlm_cf against finalized with cf_valid
    mlm["n_valid_mlm_cf"] = sum(
        1 for f in finalized if is_mlm_origin(f) and f.get("cf_valid") is True
    )
    mlm["n_mlm_origin_candidates"] = sum(1 for f in finalized if is_mlm_origin(f))

    mono_ok, mono_errs = validate_cf0_monotonicity(counts)
    cons_ok, cons_errs = validate_cf0_conservation(counts)

    if case_execution_error:
        audit_status = "FAIL_EXECUTION_ERROR"
        pending_downstream = False
        fz = None
    elif pending:
        audit_status = "PENDING_DOWNSTREAM"
        pending_downstream = True
        fz = first_zero_stage(counts)
    elif exec_n > 0:
        audit_status = "FAIL_EXECUTION_ERROR"
        pending_downstream = False
        fz = first_zero_stage(counts)
    elif not mono_ok or not cons_ok:
        audit_status = "FAIL_ACCOUNTING_INVARIANT"
        pending_downstream = False
        fz = first_zero_stage(counts)
    else:
        audit_status = "PASS"
        pending_downstream = False
        fz = first_zero_stage(counts)

    # pending_downstream_stages=false only when every candidate has terminal or exec error
    if (
        not case_execution_error
        and not pending
        and all(
            str(f.get("terminal_status"))
            in {TERMINAL_VALID, TERMINAL_REJECTED, TERMINAL_EXECUTION_ERROR, TERMINAL_FAILED}
            for f in finalized
        )
    ):
        pending_downstream = False

    payload = {
        "schema_version": CF0_SCHEMA_VERSION,
        "case_id": str(case_id),
        "cohort": str(cohort),
        "evaluation_mode": evaluation_mode,
        "edit_strategy_kind": edit_strategy_kind,
        **counts,
        **mlm,
        "bank_unique_bundle_count": int(bank_unique_bundle_count),
        "decoder_scored_unique_bundle_count": int(decoder_scored_unique_bundle_count),
        "noop_control_count": int(len(noops)),
        "noop_selected": bool(noop_selected),
        "plausibility_definition": PLAUSIBILITY_DEFINITION_CF0,
        "plausibility_definition_version": CF0_SCHEMA_VERSION,
        "full_plausibility_evaluation": False,
        "first_zero_stage": fz,
        "case_execution_error": bool(case_execution_error),
        "case_execution_error_count": 1 if case_execution_error else 0,
        "first_execution_error_stage": (
            first_execution_error_stage if case_execution_error else None
        ),
        "execution_error_reason": execution_error_reason if case_execution_error else None,
        "pending_downstream_stages": pending_downstream,
        "monotonicity_pass": bool(mono_ok) if not case_execution_error else False,
        "conservation_pass": bool(cons_ok) if not case_execution_error else False,
        "monotonicity_errors": mono_errs if not case_execution_error else [],
        "conservation_errors": cons_errs if not case_execution_error else [],
        "funnel_audit_status": audit_status,
        "candidates": [
            {
                "dedup_candidate_id": f.get("dedup_candidate_id") or f.get("candidate_id"),
                "candidate_id": f.get("candidate_id"),
                "candidate_sources": list(_sources_of(f)),
                "source": f.get("source"),
                "terminal_status": f.get("terminal_status"),
                "terminal_failure_reason": f.get("terminal_failure_reason"),
                "all_failure_reasons": list(f.get("all_failure_reasons") or []),
                "inversion_pass": f.get("inversion_pass"),
                "gate0_pass": f.get("gate0_pass"),
                "gate4_pass": f.get("gate4_pass"),
                "hard_constraint_pass": f.get("hard_constraint_pass"),
                "bank_supported": f.get("bank_supported"),
                "decoder_scored": f.get("decoder_scored"),
                "plausibility_pass": f.get("plausibility_pass"),
                "effect_pass": f.get("effect_pass"),
                "stability_pass": f.get("stability_pass"),
                "cf_valid": f.get("cf_valid"),
                "actionability": f.get("actionability"),
                "operational_eligibility": f.get("operational_eligibility"),
                "recommendation_eligible": f.get("recommendation_eligible"),
                "validity_labels": list(f.get("validity_labels") or []),
                "delta_r_search": f.get("delta_r_search"),
                "search_material": f.get("search_material"),
            }
            for f in finalized
        ],
        "warnings": [dict(w) for w in (warnings or [])],
    }
    return payload
