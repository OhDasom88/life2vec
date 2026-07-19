"""Deterministic inference contract and pure/forced identity controls."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def verify_deterministic_contract(flags: Mapping[str, Any]) -> Dict[str, Any]:
    requested = bool(flags.get("deterministic_algorithms_requested", False))
    enabled = bool(flags.get("deterministic_algorithms_enabled", False))
    violations = int(flags.get("deterministic_violation_count") or 0)
    warnings = int(flags.get("deterministic_warning_count") or 0)
    verified = (
        bool(flags.get("model_eval_mode", False))
        and bool(flags.get("inference_mode", False))
        and bool(flags.get("dropout_disabled", False))
        and bool(flags.get("fixed_rng_seeds", False))
        and requested
        and enabled
        and violations == 0
        and warnings == 0
        and bool(flags.get("batch_order_fixed", False))
        and str(flags.get("mixed_precision_policy") or "") == "LOCKED"
        and bool(flags.get("checkpoint_hash_verified", False))
    )
    return {
        "deterministic_algorithms_requested": requested,
        "deterministic_algorithms_enabled": enabled,
        "deterministic_violation_count": violations,
        "deterministic_warning_count": warnings,
        "deterministic_contract_verified": verified,
    }


def pure_repeat_noise_floors(
    risks: Sequence[float],
    logits: Sequence[float],
) -> Dict[str, float]:
    def _max_pairwise(vals: Sequence[float]) -> float:
        arr = [float(v) for v in vals]
        if len(arr) < 2:
            return 0.0
        best = 0.0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                best = max(best, abs(arr[i] - arr[j]))
        return best

    risk_noise = _max_pairwise(risks)
    logit_noise = _max_pairwise(logits)
    return {
        "pure_repeat_risk_noise_floor": risk_noise,
        "pure_repeat_logit_noise_floor": logit_noise,
        "effective_risk_identity_tolerance": max(1e-6, 2.0 * risk_noise),
        "effective_logit_identity_tolerance": max(1e-6, 2.0 * logit_noise),
    }


def forced_identity_pass(
    *,
    abs_risk_delta: float,
    abs_logit_delta: float,
    risk_tol: float,
    logit_tol: float,
    forced_path_executed: bool,
) -> Dict[str, Any]:
    ok = (
        bool(forced_path_executed)
        and abs(float(abs_risk_delta)) <= float(risk_tol)
        and abs(float(abs_logit_delta)) <= float(logit_tol)
    )
    return {
        "forced_path_executed": bool(forced_path_executed),
        "forced_identity_abs_risk_delta": float(abs_risk_delta),
        "forced_identity_abs_logit_delta": float(abs_logit_delta),
        "effective_risk_identity_tolerance": float(risk_tol),
        "effective_logit_identity_tolerance": float(logit_tol),
        "forced_identity_pass": bool(ok),
        "within_noise_floor": bool(ok),
    }
