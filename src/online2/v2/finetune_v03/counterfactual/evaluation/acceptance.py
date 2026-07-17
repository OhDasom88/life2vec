"""Acceptance report: implementation PASS | INCOMPLETE | FAIL from artifacts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


_PASS_LIKE = {"PASS"}
_NOT_RUN = {"NOT_RUN", "NOT_EVALUATED", "SKIPPED", "DEFERRED"}


def build_acceptance_report(
    *,
    structural_smoke: str = "PASS",
    behavioral_cf_smoke: str = "NO_VALID_CF",
    normal_over_edit_smoke: str = "PASS",
    constrained_mlm_execution: str = "NOT_RUN",
    mlm_reconstruction_quality: str = "NOT_EVALUATED",
    mlm_cf_candidate_outcome: str = "NOT_EVALUATED",
    scenario_outcome: str = "NO_VALID_CF",
    leakage_free: Optional[bool] = None,
    gate0_ok: Optional[bool] = None,
    retokenize_ok: Optional[bool] = None,
    noop_policy_ok: Optional[bool] = None,
    cohort_integrity: Optional[str] = None,
    full_retokenization: Optional[str] = None,
    mg_retokenization_smoke: Optional[str] = None,
    dependency_closure_retokenization: Optional[str] = None,
    calibration_ok: Optional[bool] = None,
    gt_normal_verified: Optional[bool] = None,
    source_lock_status: Optional[str] = None,
    tokenizer_integrity: Optional[str] = None,
    sign_agreement_hard_gate: Optional[str] = None,
    mlm_required_for_pass: bool = False,
    acceptance_auto: bool = True,
    over_edit_gt_normal: Optional[Mapping[str, Any]] = None,
    over_edit_route_normal: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute implementation_acceptance from real flags — never hardcode True callers.

    PASS: required stages executed and passed (MLM optional unless mlm_required_for_pass).
    INCOMPLETE: required stage not run / not evaluated.
    FAIL: executed but contract violation.
    """
    fail_reasons: List[str] = []
    incomplete_reasons: List[str] = []

    def _check_bool(name: str, val: Optional[bool], *, required: bool = True) -> None:
        if val is None:
            if required:
                incomplete_reasons.append(f"{name}_unevaluated")
            return
        if val is False:
            fail_reasons.append(name)

    def _check_status(name: str, status: str, *, allow_not_run: bool = False) -> None:
        s = str(status)
        if s in _PASS_LIKE:
            return
        if allow_not_run and s in _NOT_RUN:
            return
        if s in _NOT_RUN:
            incomplete_reasons.append(f"{name}_not_run")
            return
        if s in {"PARTIAL", "PARTIAL_PASS"}:
            incomplete_reasons.append(name)
            return
        fail_reasons.append(name)

    if structural_smoke not in _PASS_LIKE | {"PARTIAL_PASS"}:
        fail_reasons.append("structural_smoke")
    elif structural_smoke == "PARTIAL_PASS":
        incomplete_reasons.append("structural_smoke_partial")

    _check_bool("leakage_free", leakage_free)
    _check_bool("gate0_ok", gate0_ok)
    _check_bool("retokenize_ok", retokenize_ok)
    _check_bool("noop_policy_ok", noop_policy_ok)
    _check_bool("calibration_ok", calibration_ok, required=True)

    if cohort_integrity is not None:
        _check_status("cohort_integrity", cohort_integrity)
    else:
        incomplete_reasons.append("cohort_integrity_unevaluated")

    # Prefer precise MG smoke label; keep full_retokenization as legacy alias
    if mg_retokenization_smoke is None and full_retokenization is not None:
        mg_retokenization_smoke = full_retokenization
    if mg_retokenization_smoke is not None:
        _check_status("mg_retokenization_smoke", mg_retokenization_smoke)
    else:
        incomplete_reasons.append("mg_retokenization_smoke_unevaluated")

    if dependency_closure_retokenization is None:
        dependency_closure_retokenization = "NOT_EVALUATED"
    _check_status(
        "dependency_closure_retokenization",
        dependency_closure_retokenization,
        allow_not_run=True,
    )
    # Legacy field mirrors MG smoke only (not full dependency closure)
    if full_retokenization is None:
        full_retokenization = mg_retokenization_smoke

    if source_lock_status is not None:
        _check_status("source_lock_status", source_lock_status, allow_not_run=False)
    if tokenizer_integrity is not None:
        _check_status("tokenizer_integrity", tokenizer_integrity)
    if sign_agreement_hard_gate is not None:
        _check_status("sign_agreement_hard_gate", sign_agreement_hard_gate)

    _check_status(
        "constrained_mlm_execution",
        constrained_mlm_execution,
        allow_not_run=not mlm_required_for_pass,
    )
    if constrained_mlm_execution in _PASS_LIKE:
        if mlm_reconstruction_quality not in _PASS_LIKE | _NOT_RUN:
            fail_reasons.append("mlm_reconstruction_quality")
    if constrained_mlm_execution in _NOT_RUN and mlm_required_for_pass:
        incomplete_reasons.append("mlm_required")

    if normal_over_edit_smoke not in _PASS_LIKE | _NOT_RUN:
        fail_reasons.append("normal_over_edit")
    elif normal_over_edit_smoke in _NOT_RUN:
        incomplete_reasons.append("normal_over_edit_not_run")
    if gt_normal_verified is False:
        fail_reasons.append("gt_normal_unverified")
    elif gt_normal_verified is None and normal_over_edit_smoke in _PASS_LIKE:
        incomplete_reasons.append("gt_normal_unverified")

    if not acceptance_auto:
        fail_reasons.append("acceptance_not_auto")

    if fail_reasons:
        implementation = "FAIL"
    elif incomplete_reasons:
        implementation = "INCOMPLETE"
    else:
        implementation = "PASS"

    report = {
        "implementation_acceptance": implementation,
        "implementation_fail_reasons": fail_reasons,
        "implementation_incomplete_reasons": incomplete_reasons,
        "scenario_outcome": scenario_outcome,
        "structural_smoke": structural_smoke,
        "structural_policy_smoke": structural_smoke,
        "behavioral_cf_smoke": behavioral_cf_smoke,
        "cohort_integrity": cohort_integrity,
        "full_retokenization": full_retokenization,
        "mg_retokenization_smoke": mg_retokenization_smoke,
        "dependency_closure_retokenization": dependency_closure_retokenization,
        "constrained_mlm_execution": constrained_mlm_execution,
        "mlm_reconstruction_quality": mlm_reconstruction_quality,
        "mlm_cf_candidate_outcome": mlm_cf_candidate_outcome,
        "normal_over_edit_smoke": normal_over_edit_smoke,
        "gt_normal_verified": gt_normal_verified,
        "calibration_ok": calibration_ok,
        "leakage_free": leakage_free,
        "gate0_ok": gate0_ok,
        "retokenize_ok": retokenize_ok,
        "noop_policy_ok": noop_policy_ok,
        "source_lock_status": source_lock_status,
        "tokenizer_integrity": tokenizer_integrity,
        "sign_agreement_hard_gate": sign_agreement_hard_gate,
        "over_edit_gt_normal": dict(over_edit_gt_normal or {}),
        "over_edit_route_normal": dict(over_edit_route_normal or {}),
        "all_pass": implementation == "PASS",
    }
    if extra:
        report["extra"] = dict(extra)
    return report


def mlm_outcome_from_candidates(
    *,
    mlm_executed: bool,
    n_eligible_loci: int = 0,
    n_valid_mlm_cf: int = 0,
    inversion_failures: int = 0,
    reconstruction_pass: Optional[bool] = None,
    deferred: bool = False,
) -> Dict[str, str]:
    """Honest MLM status. Deferred/not-run must never become PASS."""
    if deferred or not mlm_executed:
        return {
            "constrained_mlm_execution": "NOT_RUN",
            "mlm_reconstruction_quality": "NOT_EVALUATED",
            "mlm_cf_candidate_outcome": "NOT_EVALUATED",
        }
    execution = "PASS"
    if reconstruction_pass is None:
        quality = "NOT_EVALUATED"
    else:
        quality = "PASS" if reconstruction_pass else "FAIL"
    if n_eligible_loci <= 0:
        cand = "NO_ELIGIBLE_LOCUS"
    elif inversion_failures > 0 and n_valid_mlm_cf <= 0:
        cand = "INVERSION_FAILED"
    elif n_valid_mlm_cf > 0:
        cand = "VALID_MLM_CF_FOUND"
    else:
        cand = "NO_VALID_MLM_CF"
    return {
        "constrained_mlm_execution": execution,
        "mlm_reconstruction_quality": quality,
        "mlm_cf_candidate_outcome": cand,
    }


def evaluate_p1_acceptance_conditions(
    *,
    a1_preflight_pass: bool,
    a2_functional: bool,
    a3_curated_non_original_critic: bool,
    a4_outcome_equals_natural: bool,
    a5_contracts: bool,
    a6_final_code_lock: bool,
    a7_rerun_after_lock: bool,
    a8_artifact_lock: bool,
    a8_1_bank_hashes: bool = True,
    a8_2_selection_exact_match: bool = True,
    q1: bool = False,
    q2: bool = False,
    q3: bool = False,
    q4: bool = False,
    q5: bool = False,
    mlm_cf_candidate_outcome: str = "NOT_EVALUATED",
    natural_integration_outcome: str = "NOT_EVALUATED",
    selection_manifest_chain_hash: Optional[str] = None,
    reconstruction_metric_audit_status: Optional[str] = None,
    quality_override: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """P1 acceptance with numbered conditions A1–A8 / Q1–Q5 / R1–R3.

    Official quality enum: PASS | FAIL | NOT_EVALUATED.
    PROVISIONAL_PASS is forbidden on official quality field.
    A8 = A8-1 ∧ A8-2.
    """
    a8_combined = (
        bool(a8_artifact_lock)
        and bool(selection_manifest_chain_hash)
        and bool(a8_1_bank_hashes)
        and bool(a8_2_selection_exact_match)
    )
    conditions = {
        "A1": bool(a1_preflight_pass),
        "A2": bool(a2_functional),
        "A3": bool(a3_curated_non_original_critic),
        "A4": bool(a4_outcome_equals_natural),
        "A5": bool(a5_contracts),
        "A6": bool(a6_final_code_lock),
        "A7": bool(a7_rerun_after_lock),
        "A8": a8_combined,
        "A8_1": bool(a8_artifact_lock)
        and bool(selection_manifest_chain_hash)
        and bool(a8_1_bank_hashes),
        "A8_2": bool(a8_2_selection_exact_match),
        "Q1": bool(q1),
        "Q2": bool(q2),
        "Q3": bool(q3),
        "Q4": bool(q4),
        "Q5": bool(q5),
    }
    impl_pass = all(conditions[k] for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"))

    # Quality / audit separation
    # A8-2 OK → audit can be PASS even if Q fails; Q alone never forces audit FAIL
    if not a8_2_selection_exact_match:
        quality = "NOT_EVALUATED"
        audit = "FAIL"
        impl_pass = False
        conditions["A8"] = False
        conditions["A8_2"] = False
    elif quality_override is not None:
        quality = str(quality_override)
        if quality == "PROVISIONAL_PASS":
            quality = "NOT_EVALUATED"
            audit = "FAIL"
        elif quality == "NOT_EVALUATED":
            audit = str(reconstruction_metric_audit_status or "FAIL")
        else:
            # Prefer explicit audit from caller; default PASS when A8-2 ok
            audit = str(reconstruction_metric_audit_status or "PASS")
    else:
        quality_pass = all(conditions[k] for k in ("Q1", "Q2", "Q3", "Q4", "Q5"))
        quality = "PASS" if quality_pass else "FAIL"
        audit = str(reconstruction_metric_audit_status or "PASS")

    if quality not in {"PASS", "FAIL", "NOT_EVALUATED"}:
        quality = "NOT_EVALUATED"
        audit = "FAIL"

    # Authoritative contract: audit FAIL → quality NOT_EVALUATED → impl FAIL → R3
    if str(audit) == "FAIL":
        quality = "NOT_EVALUATED"
        impl_pass = False

    quality_pass_flag = quality == "PASS"
    if impl_pass and quality_pass_flag:
        readiness = "PASS"
        r_id = "R1"
    elif impl_pass:
        readiness = "CONDITIONAL_PASS"
        r_id = "R2"
    else:
        readiness = "FAIL"
        r_id = "R3"

    # Invariant: CONDITIONAL_PASS requires audit PASS
    if readiness == "CONDITIONAL_PASS" and str(audit) != "PASS":
        readiness = "FAIL"
        r_id = "R3"
        impl_pass = False
        quality = "NOT_EVALUATED" if str(audit) == "FAIL" else quality


    out = {
        "mlm_implementation": "PASS" if impl_pass else "FAIL",
        "mlm_reconstruction_quality": quality,
        "reconstruction_metric_audit_status": audit,
        "mlm_cf_candidate_outcome": mlm_cf_candidate_outcome,
        "natural_integration_outcome": natural_integration_outcome,
        "p1_readiness": readiness,
        "acceptance_conditions": {
            "A1_A8": "PASS" if impl_pass else "FAIL",
            "Q1_Q5": "PASS" if quality_pass_flag else ("NOT_EVALUATED" if quality == "NOT_EVALUATED" else "FAIL"),
            "R": readiness,
            "R_id": r_id,
            **{
                k: (
                    "PASS"
                    if v
                    else (
                        "NOT_EVALUATED"
                        if k.startswith("Q") and quality == "NOT_EVALUATED"
                        else "FAIL"
                    )
                )
                for k, v in conditions.items()
            },
        },
        "selection_manifest_chain_hash": selection_manifest_chain_hash,
        "rerun_after_final_lock": bool(a7_rerun_after_lock),
        "final_code_source_lock": "PASS" if a6_final_code_lock else "FAIL",
        "artifact_lock": "PASS" if a8_artifact_lock else "FAIL",
        "a8_1_bank_hashes": "PASS" if a8_1_bank_hashes else "FAIL",
        "a8_2_selection_exact_match": "PASS" if a8_2_selection_exact_match else "FAIL",
    }
    if extra:
        out["extra"] = dict(extra)
    return out
