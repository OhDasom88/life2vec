"""Feature-schema Path A candidate dispatch (continuous / circular / boolean).

Observational circular/boolean edits are model-sensitivity probes only — never
recommendation-eligible actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.online2.v2.tokenizer import wind_compass

# Production tokenizer compass8 order (must not invent a parallel mapping).
COMPASS8_CATEGORIES: Tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
COMPASS8_CANONICAL_DEG: Dict[str, float] = {
    "N": 0.0,
    "NE": 45.0,
    "E": 90.0,
    "SE": 135.0,
    "S": 180.0,
    "SW": 225.0,
    "W": 270.0,
    "NW": 315.0,
}

OBSERVATIONAL_LABELS = (
    "MODEL_SENSITIVITY_ONLY",
    "NON_CAUSAL_OBSERVATION_PERTURBATION",
    "NOT_AN_ACTION",
    "NOT_RECOMMENDATION",
)

KIND_LINEAR = "linear_bins"
KIND_CIRCULAR = "circular_categorical"
KIND_BOOLEAN = "boolean_observation"
KIND_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class PathAEditStrategy:
    kind: str
    feature: str
    feature_type: str
    skip_reason: Optional[str] = None


def resolve_path_a_edit_strategy(
    *,
    feature: str,
    feature_type: Optional[str] = None,
    circular_encoding: Optional[str] = None,
    absolute_encoding: bool = False,
    global_relative_encoding: bool = False,
    farm_relative_encoding: bool = False,
) -> PathAEditStrategy:
    feat = str(feature).strip().lower()
    ftype = str(feature_type or "").strip().lower()
    circ = str(circular_encoding or "none").strip().lower()

    if circ == "compass8" or ftype == "circular" or feat == "wind_direction_deg":
        return PathAEditStrategy(
            kind=KIND_CIRCULAR,
            feature=feat,
            feature_type="CIRCULAR_CATEGORICAL",
        )
    if ftype == "boolean" or feat == "rain_detected":
        if absolute_encoding or global_relative_encoding or farm_relative_encoding:
            return PathAEditStrategy(
                kind=KIND_UNSUPPORTED,
                feature=feat,
                feature_type="BOOLEAN_WITH_BINNING",
                skip_reason="NO_SCHEMA_SUPPORTED_CANDIDATE",
            )
        return PathAEditStrategy(
            kind=KIND_BOOLEAN,
            feature=feat,
            feature_type="BOOLEAN_OBSERVATION",
        )
    if absolute_encoding or ftype in {"continuous", "numeric", "temperature", "humidity", "ratio"}:
        return PathAEditStrategy(
            kind=KIND_LINEAR,
            feature=feat,
            feature_type="CONTINUOUS_LINEAR",
        )
    # Default: treat measurable abs-binned features as linear when type unknown;
    # otherwise unsupported locus skip (not a candidate rejection).
    if ftype in {"", "unknown"} and absolute_encoding:
        return PathAEditStrategy(
            kind=KIND_LINEAR,
            feature=feat,
            feature_type="CONTINUOUS_LINEAR",
        )
    if absolute_encoding:
        return PathAEditStrategy(
            kind=KIND_LINEAR,
            feature=feat,
            feature_type="CONTINUOUS_LINEAR",
        )
    return PathAEditStrategy(
        kind=KIND_UNSUPPORTED,
        feature=feat,
        feature_type=str(feature_type or "UNSUPPORTED").upper(),
        skip_reason="NO_SCHEMA_SUPPORTED_CANDIDATE",
    )


def resolve_strategy_from_spec(feature: str, spec: Any) -> PathAEditStrategy:
    if spec is None:
        return PathAEditStrategy(
            kind=KIND_UNSUPPORTED,
            feature=str(feature).strip().lower(),
            feature_type="MISSING_SPEC",
            skip_reason="NO_SCHEMA_SUPPORTED_CANDIDATE",
        )
    return resolve_path_a_edit_strategy(
        feature=feature,
        feature_type=getattr(spec, "type", None),
        circular_encoding=getattr(spec, "circular_encoding", None),
        absolute_encoding=bool(getattr(spec, "absolute_encoding", False)),
        global_relative_encoding=bool(getattr(spec, "global_relative_encoding", False)),
        farm_relative_encoding=bool(getattr(spec, "farm_relative_encoding", False)),
    )


def _observational_meta(**extra: Any) -> Dict[str, Any]:
    return {
        "actionability": False,
        "operational_eligibility": "OBSERVATIONAL_SENSITIVITY_ONLY",
        "validity_labels": list(OBSERVATIONAL_LABELS),
        "recommendation_eligible": False,
        **extra,
    }


def compass8_adjacent_categories(current: str) -> List[str]:
    cats = list(COMPASS8_CATEGORIES)
    if current not in cats:
        return []
    i = cats.index(current)
    return [cats[(i - 1) % 8], cats[(i + 1) % 8]]


def compass8_edit_direction(current: str, target: str) -> str:
    cats = list(COMPASS8_CATEGORIES)
    if current not in cats or target not in cats:
        return "unknown"
    i = cats.index(current)
    j = cats.index(target)
    cw = (j - i) % 8
    ccw = (i - j) % 8
    if cw == 0:
        return "none"
    if cw <= ccw:
        return "clockwise"
    return "counterclockwise"


def build_compass8_candidates(
    *,
    feature: str,
    observed_raw: Optional[float],
    wind_speed_mps: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if observed_raw is None:
        return []
    try:
        obs = float(observed_raw)
    except (TypeError, ValueError):
        return []
    if obs != obs:  # NaN
        return []
    current = wind_compass(obs)
    if current not in COMPASS8_CATEGORIES:
        return []
    neighbors = compass8_adjacent_categories(current)
    out: List[Dict[str, Any]] = []
    conf = "NOMINAL"
    if wind_speed_mps is not None:
        try:
            if float(wind_speed_mps) <= 0.5:
                conf = "LOW_WHEN_WIND_SPEED_NEAR_ZERO"
        except (TypeError, ValueError):
            pass
    for neigh in neighbors:
        target = float(COMPASS8_CANONICAL_DEG[neigh])
        direction = compass8_edit_direction(current, neigh)
        row = {
            "feature": feature,
            "observed_raw": float(obs),
            "target_raw": target,
            "policy": "compass8_adjacent_canonical",
            "from_category": current,
            "to_category": neigh,
            "candidate_categories": list(neighbors),
            "current_category": current,
            "probe_side": direction,
            "edit_direction": direction,
            "source": "schema_categorical_adjacent",
            "candidate_source": "schema_categorical_adjacent",
            "is_noop": False,
            "feature_type": "CIRCULAR_CATEGORICAL",
            "wind_direction_semantic_confidence": conf,
            **_observational_meta(),
        }
        out.append(row)
    return out


def _as_binary_state(raw: float, *, eps: float = 1e-9) -> Optional[int]:
    if abs(float(raw) - 0.0) <= eps:
        return 0
    if abs(float(raw) - 1.0) <= eps:
        return 1
    return None


def build_boolean_toggle_candidates(
    *,
    feature: str,
    observed_raw: Optional[float],
) -> List[Dict[str, Any]]:
    if observed_raw is None:
        return []
    try:
        obs = float(observed_raw)
    except (TypeError, ValueError):
        return []
    if obs != obs:
        return []
    state = _as_binary_state(obs)
    if state is None:
        return []
    target = 1.0 if state == 0 else 0.0
    direction = "zero_to_positive" if state == 0 else "positive_to_zero"
    return [
        {
            "feature": feature,
            "observed_raw": float(state),
            "target_raw": float(target),
            "policy": "boolean_toggle",
            "from_state": int(state),
            "to_state": int(target),
            "probe_side": direction,
            "edit_direction": direction,
            "source": "schema_categorical_adjacent",
            "candidate_source": "schema_categorical_adjacent",
            "is_noop": False,
            "feature_type": "BOOLEAN_OBSERVATION",
            **_observational_meta(),
        }
    ]


def build_schema_dispatched_candidates(
    *,
    strategy: PathAEditStrategy,
    observed_raw: Optional[float],
    edges: Optional[Sequence[float]] = None,
    wind_speed_mps: Optional[float] = None,
    build_linear_fn=None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return (candidates, locus_skip_reason). Skip reason is locus-level only."""
    if strategy.kind == KIND_UNSUPPORTED:
        return [], strategy.skip_reason or "NO_SCHEMA_SUPPORTED_CANDIDATE"
    if strategy.kind == KIND_CIRCULAR:
        return (
            build_compass8_candidates(
                feature=strategy.feature,
                observed_raw=observed_raw,
                wind_speed_mps=wind_speed_mps,
            ),
            None,
        )
    if strategy.kind == KIND_BOOLEAN:
        return (
            build_boolean_toggle_candidates(
                feature=strategy.feature,
                observed_raw=observed_raw,
            ),
            None,
        )
    if strategy.kind == KIND_LINEAR:
        if build_linear_fn is None:
            from .path_a_generator import build_path_a_candidates

            build_linear_fn = build_path_a_candidates
        if edges is None or len(edges) < 2:
            raise RuntimeError(f"no bin edges for feature={strategy.feature}")
        return (
            build_linear_fn(
                feature=strategy.feature,
                observed_raw=float(observed_raw),
                edges=edges,
            ),
            None,
        )
    return [], "NO_SCHEMA_SUPPORTED_CANDIDATE"
