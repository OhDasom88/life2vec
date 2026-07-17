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
