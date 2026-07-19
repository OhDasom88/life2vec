"""Baseline pure-repeat and forced-identity contracts for CF-1S Core."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .core_contract import CoreContractError, fold_median


ForwardFn = Callable[[int], Mapping[str, Any]]
# forward(fold_id) -> {"risk": float, "logit": Sequence[float]}


def pure_repeat_risk_noise(risks: Sequence[float]) -> float:
    vals = [float(x) for x in risks]
    if len(vals) < 2:
        return 0.0
    return float(max(vals) - min(vals))


def pure_repeat_logit_noise(logits_by_repeat: Sequence[Sequence[float]]) -> float:
    if len(logits_by_repeat) < 2:
        return 0.0
    mats = [[float(x) for x in row] for row in logits_by_repeat]
    n_class = len(mats[0])
    best = 0.0
    for c in range(n_class):
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                best = max(best, abs(mats[i][c] - mats[j][c]))
    return float(best)


def run_pure_repeat_baseline(
    *,
    requested_fold_ids: Sequence[int],
    forward_fn: ForwardFn,
    repeat_count: int = 3,
) -> Dict[str, Any]:
    folds = [int(f) for f in requested_fold_ids]
    by_fold: Dict[str, Any] = {}
    case_risk_noise = 0.0
    case_logit_noise = 0.0
    for fold_id in folds:
        repeats = []
        for _ in range(int(repeat_count)):
            out = forward_fn(fold_id)
            repeats.append(
                {
                    "risk": float(out["risk"]),
                    "logit": [float(x) for x in out["logit"]],
                }
            )
        risks = [r["risk"] for r in repeats]
        logits = [r["logit"] for r in repeats]
        risk_noise = pure_repeat_risk_noise(risks)
        logit_noise = pure_repeat_logit_noise(logits)
        # element-wise median for logits
        n_class = len(logits[0])
        ref_logit = [
            fold_median([logits[r][c] for r in range(len(logits))]) for c in range(n_class)
        ]
        by_fold[str(fold_id)] = {
            "repeats": repeats,
            "baseline_reference_risk": fold_median(risks),
            "baseline_reference_logit": ref_logit,
            "pure_repeat_risk_noise": risk_noise,
            "pure_repeat_logit_noise": logit_noise,
        }
        case_risk_noise = max(case_risk_noise, risk_noise)
        case_logit_noise = max(case_logit_noise, logit_noise)
    return {
        "baseline_reference_policy": "MEDIAN_OF_3_PURE_REPEATS",
        "requested_fold_ids": folds,
        "by_fold": by_fold,
        "case_pure_repeat_risk_noise": float(case_risk_noise),
        "case_pure_repeat_logit_noise": float(case_logit_noise),
        "candidate_specific_baseline_recompute_count": 0,
    }


def check_noise_ceilings(
    *,
    case_pure_repeat_risk_noise: float,
    case_pure_repeat_logit_noise: float,
    max_risk: float = 0.001,
    max_logit: float = 0.001,
) -> Dict[str, Any]:
    risk_ok = float(case_pure_repeat_risk_noise) <= float(max_risk)
    logit_ok = float(case_pure_repeat_logit_noise) <= float(max_logit)
    return {
        "noise_ceiling_pass": risk_ok and logit_ok,
        "risk_noise_ok": risk_ok,
        "logit_noise_ok": logit_ok,
        "case_pure_repeat_risk_noise": float(case_pure_repeat_risk_noise),
        "case_pure_repeat_logit_noise": float(case_pure_repeat_logit_noise),
        "max_allowed_pure_repeat_risk_noise": float(max_risk),
        "max_allowed_pure_repeat_logit_noise": float(max_logit),
    }


def forced_identity_tolerances(
    *,
    case_pure_repeat_risk_noise: float,
    case_pure_repeat_logit_noise: float,
    risk_ceiling: float = 0.003,
    logit_ceiling: float = 0.003,
) -> Dict[str, float]:
    risk_tol = min(max(1e-6, 3.0 * float(case_pure_repeat_risk_noise)), float(risk_ceiling))
    logit_tol = min(max(1e-6, 3.0 * float(case_pure_repeat_logit_noise)), float(logit_ceiling))
    return {
        "case_forced_identity_risk_tolerance": float(risk_tol),
        "case_forced_identity_logit_tolerance": float(logit_tol),
    }


def evaluate_forced_identity(
    *,
    baseline: Mapping[str, Any],
    identity_by_fold: Mapping[str, Mapping[str, Any]],
    risk_tolerance: float,
    logit_tolerance: float,
) -> Dict[str, Any]:
    failed: List[int] = []
    details = {}
    for fold_s, base in baseline["by_fold"].items():
        fold_id = int(fold_s)
        obs = identity_by_fold.get(fold_s) or identity_by_fold.get(fold_id)  # type: ignore[arg-type]
        if obs is None:
            failed.append(fold_id)
            details[fold_s] = {"pass": False, "reason": "MISSING_IDENTITY_OUTPUT"}
            continue
        risk_err = abs(float(obs["risk"]) - float(base["baseline_reference_risk"]))
        logit_base = list(base["baseline_reference_logit"])
        logit_obs = [float(x) for x in obs["logit"]]
        if len(logit_base) != len(logit_obs):
            failed.append(fold_id)
            details[fold_s] = {"pass": False, "reason": "LOGIT_LEN_MISMATCH"}
            continue
        logit_err = max(abs(a - b) for a, b in zip(logit_obs, logit_base)) if logit_base else 0.0
        ok = risk_err <= float(risk_tolerance) and logit_err <= float(logit_tolerance)
        if not ok:
            failed.append(fold_id)
        details[fold_s] = {
            "pass": ok,
            "risk_abs_error": risk_err,
            "logit_max_abs_error": logit_err,
        }
    return {
        "transaction_mode": "FORCED_IDENTITY",
        "forced_identity_all_requested_folds_pass": len(failed) == 0,
        "forced_identity_failed_fold_ids": sorted(failed),
        "forced_identity_counted_as_scientific_candidate": False,
        "details": details,
        "case_execution_status": "PASS" if not failed else "NOT_EVALUABLE",
    }


def deltas_from_baseline(
    *,
    baseline: Mapping[str, Any],
    edited_by_fold: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for fold_s, base in baseline["by_fold"].items():
        obs = edited_by_fold.get(fold_s) or edited_by_fold.get(int(fold_s))  # type: ignore[arg-type]
        if obs is None:
            raise CoreContractError(f"missing edited output for fold {fold_s}")
        out[fold_s] = float(obs["risk"]) - float(base["baseline_reference_risk"])
    return out
