"""Production schema-dispatch atomic targets; fixture adjacency is separate."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from ..candidates.path_a_schema_dispatch import (
    build_schema_dispatched_candidates,
    resolve_strategy_from_spec,
)
from .execution_context import ProductionContractViolation


def fixture_adjacency_targets(observed_raw: Any, schema_kind: str) -> List[Any]:
    try:
        val = float(observed_raw)
    except (TypeError, ValueError):
        return []
    kind = str(schema_kind).lower()
    if kind in {"boolean"}:
        return [0.0 if val >= 0.5 else 1.0]
    if kind in {"ordinal", "continuous", "circular_categorical"}:
        return [val - 1.0, val + 1.0]
    return [val - 1.0, val + 1.0]


def production_schema_targets(
    locus: Mapping[str, Any],
    feature_record: Any,
    *,
    feature_spec: Any = None,
    bin_edges: Optional[Sequence[float]] = None,
    forbid_default_adjacency: bool = True,
) -> List[Any]:
    """Resolve adjacency targets via path_a schema dispatch."""
    if forbid_default_adjacency:
        # Callers must not fall back to default_adjacency_targets in production.
        pass
    feature = str(locus.get("feature_id") or locus.get("feature") or "")
    strategy = resolve_strategy_from_spec(feature, feature_spec)
    # If no rich spec, synthesize a minimal type from edit-policy schema_kind.
    if feature_spec is None and feature_record is not None:
        kind = str(getattr(feature_record, "schema_kind", "") or "").lower()
        type_map = {
            "boolean": "BOOLEAN_OBSERVATION",
            "circular_categorical": "CIRCULAR_CATEGORICAL",
            "ordinal": "ORDINAL",
            "continuous": "CONTINUOUS",
        }
        feature_type = type_map.get(kind, kind.upper() or "UNSUPPORTED")

        class _Spec:
            type = feature_type
            circular_encoding = "compass8" if "circular" in kind else None
            absolute_encoding = True
            global_relative_encoding = False
            farm_relative_encoding = False

        strategy = resolve_strategy_from_spec(feature, _Spec())

    candidates, _skip = build_schema_dispatched_candidates(
        strategy=strategy,
        observed_raw=locus.get("observed_raw"),
        edges=bin_edges,
    )
    targets = []
    for c in candidates:
        if "target_raw" in c:
            targets.append(c["target_raw"])
    return targets


def assert_not_default_adjacency_in_production(
    *,
    provenance_mode: str,
    default_adjacency_use_count: int,
) -> None:
    if str(provenance_mode).upper() == "PRODUCTION" and default_adjacency_use_count > 0:
        raise ProductionContractViolation(
            "default_adjacency_targets used in production"
        )
