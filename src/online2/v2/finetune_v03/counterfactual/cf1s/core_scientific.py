"""Scientific numeric oracles for Development3 multi-event evidence."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from .core_contract import CoreContractError


def identity_error_fold(identity_risk: float, baseline_risk: float) -> float:
    return abs(float(identity_risk) - float(baseline_risk))


def max_search_identity_reconstruction_error(
    identity_by_fold: Mapping[str, float],
    baseline_by_fold: Mapping[str, float],
) -> float:
    """Fold-scoped only — never cross-compare fold0 baseline with fold1 identity."""
    errs = []
    for fold_id, base in baseline_by_fold.items():
        if fold_id not in identity_by_fold:
            raise CoreContractError(f"identity missing fold {fold_id}")
        errs.append(identity_error_fold(identity_by_fold[fold_id], base))
    if not errs:
        raise CoreContractError("no fold identity errors")
    return float(max(errs))


def runtime_repeat_noise(risks: Sequence[float]) -> float:
    vals = [float(x) for x in risks]
    if len(vals) < 2:
        return 0.0
    m = 0.0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            m = max(m, abs(vals[i] - vals[j]))
    return float(m)


def delta_risk(candidate_risk: float, baseline_risk: float) -> float:
    return float(candidate_risk) - float(baseline_risk)


def direction_sign(expected_effect_direction: str) -> int:
    d = str(expected_effect_direction).upper()
    if d in ("RISK_INCREASE", "+1", "INCREASE"):
        return 1
    if d in ("RISK_DECREASE", "-1", "DECREASE"):
        return -1
    raise CoreContractError(f"unknown expected_effect_direction: {expected_effect_direction}")


def directed_delta(delta: float, *, expected_effect_direction: str) -> float:
    return float(direction_sign(expected_effect_direction)) * float(delta)


def material(delta_risk_value: float, *, locked_threshold: float) -> bool:
    return abs(float(delta_risk_value)) >= float(locked_threshold)


def direction_consistent(delta: float, *, expected_effect_direction: str, locked_threshold: float) -> bool:
    return directed_delta(delta, expected_effect_direction=expected_effect_direction) >= float(
        locked_threshold
    )


def incremental_directed(
    bundle_delta: float,
    parent_deltas: Sequence[float],
    *,
    expected_effect_direction: str,
) -> float:
    if not parent_deltas:
        raise CoreContractError("NOT_EVALUABLE: parent deltas missing")
    db = directed_delta(bundle_delta, expected_effect_direction=expected_effect_direction)
    parents = [
        directed_delta(p, expected_effect_direction=expected_effect_direction) for p in parent_deltas
    ]
    return float(db - max(parents))


def residual_risk(bundle_delta: float, atomic_deltas: Sequence[float]) -> float:
    return float(bundle_delta) - float(sum(atomic_deltas))


def stable_effect_scope(
    fold_deltas: Mapping[str, float],
    *,
    required_fold_ids: Sequence[str],
    expected_effect_direction: str,
    locked_threshold: float,
) -> bool:
    for fid in required_fold_ids:
        if fid not in fold_deltas:
            return False
        if not direction_consistent(
            fold_deltas[fid],
            expected_effect_direction=expected_effect_direction,
            locked_threshold=locked_threshold,
        ):
            return False
    return True


def evaluate_claim_levels(
    *,
    model_input_delivery_verified: bool,
    stable_effect_replicated: bool,
    multievent_increment_supported: bool,
) -> Dict[str, Any]:
    levels = []
    if model_input_delivery_verified:
        levels.append("MODEL_INPUT_DELIVERY_VERIFIED")
    if stable_effect_replicated:
        levels.append("BUNDLE_MODEL_EFFECT_REPLICATED")
    if multievent_increment_supported:
        levels.append("MULTIEVENT_ADDITIONAL_MODEL_EFFECT_SUPPORTED")
    highest = levels[-1] if levels else "NONE"
    return {
        "claim_levels": levels,
        "highest_claim_level": highest,
        "multievent_additional_effect_claimed": bool(multievent_increment_supported),
    }


def scientific_status_from_effects(
    *,
    stable_effect_replicated: bool,
    multievent_increment_supported: bool,
    inconclusive: bool = False,
    not_evaluable: bool = False,
    not_evaluated: bool = False,
    no_material_control: bool = False,
) -> Dict[str, Any]:
    if not_evaluated:
        return {
            "development_scientific_result": "NOT_EVALUATED",
            "stable_effect_status": "NOT_EVALUABLE",
            "multievent_increment_status": "NOT_EVALUABLE",
            "reason": "NOT_EVALUATED",
        }
    if not_evaluable:
        return {
            "development_scientific_result": "NOT_EVALUABLE",
            "stable_effect_status": "NOT_EVALUABLE",
            "multievent_increment_status": "NOT_EVALUABLE",
            "reason": "EVIDENCE_INVALID",
        }
    if inconclusive:
        return {
            "development_scientific_result": "INCONCLUSIVE",
            "stable_effect_status": "NOT_REPLICATED",
            "multievent_increment_status": "MULTIEVENT_INCREMENT_NOT_SUPPORTED",
            "reason": "SCOPE_DISAGREEMENT",
        }
    if no_material_control and not stable_effect_replicated:
        return {
            "development_scientific_result": "NOT_EVALUATED",
            "stable_effect_status": "NOT_EVALUATED",
            "multievent_increment_status": "NOT_EVALUATED",
            "reason": "CONTROL_ONLY_NO_MATERIAL_EFFECT_EVALUATION",
        }
    if multievent_increment_supported:
        return {
            "development_scientific_result": "SUPPORTED",
            "stable_effect_status": "STABLE_EFFECT_REPLICATED",
            "multievent_increment_status": "MULTIEVENT_INCREMENT_SUPPORTED",
            "reason": "MULTIEVENT_INCREMENT_SUPPORTED",
        }
    if stable_effect_replicated:
        return {
            "development_scientific_result": "NOT_SUPPORTED",
            "stable_effect_status": "STABLE_EFFECT_REPLICATED",
            "multievent_increment_status": "MULTIEVENT_INCREMENT_NOT_SUPPORTED",
            "reason": "BUNDLE_EFFECT_REPLICATED_INCREMENT_NOT_SUPPORTED",
        }
    return {
        "development_scientific_result": "NOT_SUPPORTED",
        "stable_effect_status": "NOT_REPLICATED",
        "multievent_increment_status": "MULTIEVENT_INCREMENT_NOT_SUPPORTED",
        "reason": "BUNDLE_EFFECT_NOT_REPLICATED",
    }


def check_risk_logit_contract(risk: float, abnormal_logit: float, *, tolerance: float = 1e-6) -> None:
    if not math.isfinite(risk) or not math.isfinite(abnormal_logit):
        raise CoreContractError("RISK_LOGIT_CONTRACT_VIOLATION: non-finite")
    expected = 1.0 / (1.0 + math.exp(-float(abnormal_logit)))
    if abs(float(risk) - expected) > float(tolerance):
        raise CoreContractError("RISK_LOGIT_CONTRACT_VIOLATION")


def select_observed_median(
    observations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Select one actual observation by median-risk rank; never mix risk/logit rows."""
    if not observations:
        raise CoreContractError("BASELINE_OBSERVATIONS_MISSING")
    normalized = []
    required = ("risk", "logit", "trace_ref", "critic_input_sha")
    for index, observation in enumerate(observations):
        row = dict(observation)
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise CoreContractError(
                "BASELINE_OBSERVATION_INCOMPLETE:" + ",".join(sorted(missing))
            )
        check_risk_logit_contract(float(row["risk"]), float(row["logit"]))
        normalized.append((float(row["risk"]), str(row.get("observation_id", "")), index, row))
    normalized.sort(key=lambda item: (item[0], item[1], item[2]))
    # For even repeats choose the lower ranked actual observation. No averaging.
    selected = normalized[(len(normalized) - 1) // 2][3]
    selected["median_selection_rule"] = "LOWER_MEDIAN_ACTUAL_OBSERVATION"
    return selected


def control_result_from_delta(
    *,
    delta_risk_value: Optional[float],
    locked_threshold: float,
    evaluable: bool = True,
) -> str:
    if not evaluable or delta_risk_value is None or not math.isfinite(float(delta_risk_value)):
        return "CONTROL_NOT_EVALUABLE"
    if abs(float(delta_risk_value)) <= float(locked_threshold):
        return "CONTROL_WITHIN_LOCKED_THRESHOLD"
    return "CONTROL_EXCEEDED_LOCKED_THRESHOLD"


def derive_coverage_axes(case_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep execution, material-effect, and control coverage distinct."""
    total = len(case_rows)
    execution = sum(1 for row in case_rows if bool(row.get("execution_complete")))
    material = sum(
        1
        for row in case_rows
        if row.get("selection_disposition") == "SELECTED_MATERIAL"
        and bool(row.get("scientifically_evaluable"))
    )
    control = sum(
        1
        for row in case_rows
        if row.get("selection_disposition") == "SELECTED_CONTROL_NO_MATERIAL"
        and row.get("control_result")
        in ("CONTROL_WITHIN_LOCKED_THRESHOLD", "CONTROL_EXCEEDED_LOCKED_THRESHOLD")
    )

    def envelope(name: str, count: int) -> Dict[str, Any]:
        status = "COMPLETE" if total > 0 and count == total else ("NONE" if count == 0 else "PARTIAL")
        return {"coverage_axis": name, "status": status, "count": count, "total": total}

    return {
        "execution_coverage": envelope("execution_coverage", execution),
        "material_effect_evaluation_coverage": envelope(
            "material_effect_evaluation_coverage", material
        ),
        "control_evaluation_coverage": envelope("control_evaluation_coverage", control),
    }
