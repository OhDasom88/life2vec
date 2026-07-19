"""Mutually exclusive signed interaction classification for CF-1S."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .fold_effect import required_strict_majority_count, sign_of


EMERGENT = "EMERGENT_BUNDLE_MODEL_EFFECT"
ADDITIVE = "ADDITIVE_CUMULATIVE_MODEL_SENSITIVITY"
AMPLIFICATION = "NONLINEAR_MULTI_EVENT_MODEL_AMPLIFICATION"
CANCELLATION = "MODEL_OUTPUT_CANCELLATION"
INDETERMINATE = "INTERACTION_INDETERMINATE"


def classify_interaction(
    *,
    bundle_delta: float,
    atomic_deltas: Sequence[float],
    atomic_sum_noise_floor: float = 0.001,
    additive_tolerance_abs: float = 0.001,
    amplification_threshold_abs: float = 0.001,
    cancellation_threshold_abs: float = 0.001,
    effect_epsilon: float = 0.01,
) -> Dict[str, Any]:
    atomics = [float(x) for x in atomic_deltas]
    atomic_sum = float(sum(atomics))
    interaction_delta = float(bundle_delta) - atomic_sum
    abs_atomic_sum = abs(atomic_sum)
    reference_direction = None
    aligned_interaction = None
    if abs_atomic_sum >= float(atomic_sum_noise_floor):
        reference_direction = sign_of(atomic_sum, zero_tol=0.0)
        aligned_interaction = float(reference_direction) * float(interaction_delta)

    # Mutually exclusive order with ADDITIVE_FIRST boundary precedence.
    # Float guard keeps exact policy boundary (e.g. 0.001) inside ADDITIVE.
    eps = 1e-12
    if abs_atomic_sum < float(atomic_sum_noise_floor) - eps and abs(
        float(bundle_delta)
    ) >= float(effect_epsilon):
        cls = EMERGENT
    elif abs(interaction_delta) <= float(additive_tolerance_abs) + eps:
        cls = ADDITIVE
    elif (
        reference_direction is not None
        and sign_of(bundle_delta) == reference_direction
        and aligned_interaction is not None
        and aligned_interaction > float(amplification_threshold_abs) + eps
    ):
        cls = AMPLIFICATION
    elif (
        reference_direction is not None
        and (
            sign_of(bundle_delta) == -reference_direction
            or (
                aligned_interaction is not None
                and aligned_interaction < -float(cancellation_threshold_abs) - eps
            )
        )
    ):
        cls = CANCELLATION
    else:
        cls = INDETERMINATE

    return {
        "bundle_delta": float(bundle_delta),
        "atomic_sum_delta": atomic_sum,
        "interaction_delta": interaction_delta,
        "reference_direction": reference_direction,
        "aligned_interaction": aligned_interaction,
        "interaction_class": cls,
        "boundary_precedence": "ADDITIVE_FIRST",
    }


def classify_interaction_by_fold(
    *,
    bundle_deltas_by_fold: Sequence[float],
    atomic_deltas_by_fold: Sequence[Sequence[float]],
    **kwargs: Any,
) -> Dict[str, Any]:
    classes: List[str] = []
    deltas: List[float] = []
    directions: List[Optional[int]] = []
    for bundle_delta, atomics in zip(bundle_deltas_by_fold, atomic_deltas_by_fold):
        row = classify_interaction(
            bundle_delta=float(bundle_delta),
            atomic_deltas=atomics,
            **kwargs,
        )
        classes.append(row["interaction_class"])
        deltas.append(float(row["interaction_delta"]))
        aligned = row.get("aligned_interaction")
        if aligned is None:
            directions.append(None)
        else:
            directions.append(sign_of(float(aligned)))
    return {
        "interaction_class_by_fold": classes,
        "interaction_delta_by_fold": deltas,
        "aligned_interaction_direction_by_fold": directions,
    }


def stable_interaction_fields(
    *,
    search_class: str,
    holdout_class: str,
    holdout_classes_by_fold: Sequence[str],
    holdout_aligned_directions_by_fold: Sequence[Optional[int]],
    holdout_interaction_magnitude: float,
    magnitude_min: float = 0.001,
) -> Dict[str, Any]:
    evaluable_class_folds = [c for c in holdout_classes_by_fold if c]
    if not evaluable_class_folds:
        return {
            "interaction_class_stable": False,
            "stable_interaction_class": None,
            "holdout_interaction_class_consensus_fraction": 0.0,
            "holdout_interaction_direction_consensus_fraction": 0.0,
            "interaction_stability_status": "NOT_EVALUABLE",
        }

    required = required_strict_majority_count(len(evaluable_class_folds))
    # Class consensus
    from collections import Counter

    class_counts = Counter(evaluable_class_folds)
    top_class, top_count = class_counts.most_common(1)[0]
    class_frac = top_count / float(len(evaluable_class_folds))
    class_majority = top_count >= required

    dir_folds = [d for d in holdout_aligned_directions_by_fold if d in (-1, 1)]
    if not dir_folds:
        dir_majority = False
        dir_frac = 0.0
    else:
        dir_required = required_strict_majority_count(len(dir_folds))
        pos = sum(1 for d in dir_folds if d > 0)
        neg = sum(1 for d in dir_folds if d < 0)
        dir_top = max(pos, neg)
        dir_frac = dir_top / float(len(dir_folds))
        dir_majority = dir_top >= dir_required

    stable = (
        search_class == holdout_class
        and class_majority
        and top_class == holdout_class
        and dir_majority
        and abs(float(holdout_interaction_magnitude)) >= float(magnitude_min)
    )
    return {
        "interaction_class_stable": bool(stable),
        "stable_interaction_class": holdout_class if stable else None,
        "holdout_interaction_class_consensus_fraction": float(class_frac),
        "holdout_interaction_direction_consensus_fraction": float(dir_frac),
        "interaction_stability_status": "STABLE" if stable else "NOT_STABLE",
        "required_agreement_count": required,
    }


def three_event_residuals(
    *,
    delta_a: float,
    delta_b: float,
    delta_c: float,
    delta_ab: float,
    delta_ac: float,
    delta_bc: float,
    delta_abc: float,
) -> Dict[str, float]:
    total = float(delta_abc) - float(delta_a) - float(delta_b) - float(delta_c)
    third = (
        float(delta_abc)
        - float(delta_ab)
        - float(delta_ac)
        - float(delta_bc)
        + float(delta_a)
        + float(delta_b)
        + float(delta_c)
    )
    return {
        "total_nonadditivity_abc": total,
        "third_order_residual_abc": third,
    }
