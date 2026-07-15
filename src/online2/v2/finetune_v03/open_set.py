"""Reject / Known / UNKNOWN_ABNORMAL routing for v0.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.online2.v2.finetune_v03.config import RoutingThresholds


@dataclass
class RouteResult:
    decision: str  # NORMAL | KNOWN | UNKNOWN_ABNORMAL | REJECT
    reasons: list
    scores: Dict[str, float]


def route_case(
    *,
    p_abnormal: float,
    fine_probs: np.ndarray,
    normal_class_id: int,
    prototype_distance: Optional[float] = None,
    prototype_tau: Optional[float] = None,
    energy: Optional[float] = None,
    energy_tau: Optional[float] = None,
    fold_agreement: Optional[float] = None,
    head_contradiction: bool = False,
    use_reject: bool = True,
    use_open_set: bool = True,
    thr: Optional[RoutingThresholds] = None,
) -> RouteResult:
    thr = thr or RoutingThresholds()
    reasons = []
    scores = {
        "p_abnormal": float(p_abnormal),
        "max_fine_prob": float(np.max(fine_probs)),
        "p_fine_normal": float(fine_probs[int(normal_class_id)]),
    }
    if prototype_distance is not None:
        scores["prototype_distance"] = float(prototype_distance)
    if energy is not None:
        scores["energy"] = float(energy)
    if fold_agreement is not None:
        scores["fold_agreement"] = float(fold_agreement)

    # Reject: low confidence / contradiction / border binary
    if use_reject:
        margin = abs(float(p_abnormal) - float(thr.tau_binary))
        if margin < 0.05:
            reasons.append("binary_near_threshold")
        if float(np.max(fine_probs)) < float(thr.tau_fine) and float(thr.tau_fine) > 0:
            reasons.append("low_fine_confidence")
        if head_contradiction:
            reasons.append("fine_binary_contradiction")
        if fold_agreement is not None and fold_agreement < float(thr.min_fold_agreement):
            reasons.append("low_fold_agreement")
        if reasons:
            return RouteResult("REJECT", reasons, scores)

    # Normal
    fine_ok = float(fine_probs[int(normal_class_id)]) >= float(thr.tau_fine) or float(thr.tau_fine) <= 0
    if float(p_abnormal) < float(thr.tau_binary) and fine_ok and not head_contradiction:
        return RouteResult("NORMAL", ["p_abnormal_below_tau"], scores)

    # Known
    max_p = float(np.max(fine_probs))
    proto_ok = True
    if prototype_distance is not None and prototype_tau is not None:
        proto_ok = float(prototype_distance) <= float(prototype_tau)
        scores["prototype_tau"] = float(prototype_tau)
    energy_ok = True
    if energy is not None and (energy_tau is not None or thr.energy_tau is not None):
        et = float(energy_tau if energy_tau is not None else thr.energy_tau)  # type: ignore
        energy_ok = float(energy) <= et
        scores["energy_tau"] = et
    fold_ok = fold_agreement is None or float(fold_agreement) >= float(thr.min_fold_agreement)

    known = (
        float(p_abnormal) >= float(thr.tau_binary)
        and max_p >= float(thr.tau_fine)
        and proto_ok
        and energy_ok
        and fold_ok
    )
    if known:
        return RouteResult("KNOWN", ["passed_known_gates"], scores)

    if use_open_set and float(p_abnormal) >= float(thr.tau_binary):
        return RouteResult("UNKNOWN_ABNORMAL", ["abnormal_not_known"], scores)

    if use_reject:
        return RouteResult("REJECT", ["fallback_reject"], scores)
    # Forced closed-set fallback
    return RouteResult("KNOWN", ["open_set_disabled_force_known"], scores)


def energy_from_logits(logits: np.ndarray) -> float:
    """E = logsumexp(logits) — higher → more in-distribution typically for soft max models."""
    m = float(np.max(logits))
    return float(m + np.log(np.sum(np.exp(logits - m))))
