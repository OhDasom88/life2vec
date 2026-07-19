"""Development3 cohort phase state machine and dependency closure."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError
from .core_scientific import (
    evaluate_claim_levels,
    incremental_directed,
    max_search_identity_reconstruction_error,
    runtime_repeat_noise,
    scientific_status_from_effects,
    stable_effect_scope,
)


class DevelopmentPhase(str, Enum):
    INPUT_LOCK = "INPUT_LOCK"
    ATTRIBUTION = "ATTRIBUTION"
    CANDIDATE_UNIVERSE = "CANDIDATE_UNIVERSE"
    SEARCH_BASELINE_IDENTITY = "SEARCH_BASELINE_IDENTITY"
    THRESHOLD_LOCK = "THRESHOLD_LOCK"
    SEARCH_EFFECTS = "SEARCH_EFFECTS"
    SELECTION = "SELECTION"
    CLOSURE_FREEZE = "CLOSURE_FREEZE"
    REEVAL_BASELINE_IDENTITY = "REEVAL_BASELINE_IDENTITY"
    REEVAL_EFFECTS = "REEVAL_EFFECTS"
    RESULTS = "RESULTS"


ALLOWED_TRANSITIONS = {
    DevelopmentPhase.INPUT_LOCK: {DevelopmentPhase.ATTRIBUTION},
    DevelopmentPhase.ATTRIBUTION: {DevelopmentPhase.CANDIDATE_UNIVERSE},
    DevelopmentPhase.CANDIDATE_UNIVERSE: {DevelopmentPhase.SEARCH_BASELINE_IDENTITY},
    DevelopmentPhase.SEARCH_BASELINE_IDENTITY: {DevelopmentPhase.THRESHOLD_LOCK},
    DevelopmentPhase.THRESHOLD_LOCK: {DevelopmentPhase.SEARCH_EFFECTS},
    DevelopmentPhase.SEARCH_EFFECTS: {DevelopmentPhase.SELECTION},
    DevelopmentPhase.SELECTION: {DevelopmentPhase.CLOSURE_FREEZE},
    DevelopmentPhase.CLOSURE_FREEZE: {DevelopmentPhase.REEVAL_BASELINE_IDENTITY},
    DevelopmentPhase.REEVAL_BASELINE_IDENTITY: {DevelopmentPhase.REEVAL_EFFECTS},
    DevelopmentPhase.REEVAL_EFFECTS: {DevelopmentPhase.RESULTS},
    DevelopmentPhase.RESULTS: set(),
}

CANDIDATE_BUDGET_2EVENT = 10
CANDIDATE_BUDGET_3EVENT = 5
DEPENDENCY_CLOSURE_BUDGET = 50


def required_parents_for_bundle(bundle_event_ids: Sequence[str], *, for_incremental: bool = True) -> List[Tuple[str, ...]]:
    """Return required parent bundles as sorted event-id tuples."""
    ev = tuple(sorted(str(x) for x in bundle_event_ids))
    n = len(ev)
    if n == 2:
        return [(ev[0],), (ev[1],)]
    if n == 3:
        a, b, c = ev
        if for_incremental:
            # ABC incremental needs AB, AC, BC; residual/atomic needs A,B,C separately
            return [(a, b), (a, c), (b, c)]
        return [(a,), (b,), (c,)]
    if n == 1:
        return []
    raise CoreContractError(f"unsupported bundle size {n}")


def full_abc_dependency_closure(bundle_event_ids: Sequence[str]) -> List[Tuple[str, ...]]:
    ev = tuple(sorted(str(x) for x in bundle_event_ids))
    if len(ev) != 3:
        raise CoreContractError("full_abc_dependency_closure requires 3 events")
    a, b, c = ev
    return [(a,), (b,), (c,), (a, b), (a, c), (b, c)]


def expand_dependency_closure(
    selected_bundles: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Auto-expand parents for selected bundles before freeze.
    selected_bundles: [{candidate_id, event_ids, kind}]
    """
    deps: Dict[Tuple[str, ...], str] = {}
    reasons: Dict[str, List[str]] = {}

    def _add(key: Tuple[str, ...], reason: str) -> None:
        deps.setdefault(key, reason)
        reasons.setdefault(reason, []).append("|".join(key))

    for bundle in selected_bundles:
        eids = tuple(sorted(str(x) for x in bundle["event_ids"]))
        cid = str(bundle["candidate_id"])
        kind = str(bundle.get("kind") or ("PAIR" if len(eids) == 2 else "TRIPLE"))
        _add(eids, f"selected:{cid}")
        if len(eids) == 1:
            pass  # atomic: self only
        elif len(eids) == 2:
            for p in required_parents_for_bundle(eids):
                _add(p, f"parent_of:{cid}")
        elif len(eids) == 3:
            for p in full_abc_dependency_closure(eids):
                _add(p, f"parent_of:{cid}")
        else:
            raise CoreContractError(f"unsupported selected bundle {kind}")

    unique = sorted(deps.keys(), key=lambda t: (len(t), t))
    return {
        "dependencies": [{"event_ids": list(k), "reason": deps[k]} for k in unique],
        "unique_dependency_count": len(unique),
        "dependency_closure_budget": DEPENDENCY_CLOSURE_BUDGET,
    }


def apply_dependency_budget(
    selected_bundles: List[Mapping[str, Any]],
    *,
    budget: int = DEPENDENCY_CLOSURE_BUDGET,
) -> Dict[str, Any]:
    """Deterministically drop trailing bundles until dependency count ≤ budget."""
    remaining = list(selected_bundles)
    excluded: List[Dict[str, Any]] = []
    while True:
        closure = expand_dependency_closure(remaining)
        if closure["unique_dependency_count"] <= budget:
            return {
                "selected_bundles": remaining,
                "excluded_bundles": excluded,
                "closure": closure,
            }
        if not remaining:
            raise CoreContractError("dependency budget unsatisfiable")
        dropped = remaining.pop()  # drop from canonical rear
        excluded.append(
            {
                **dict(dropped),
                "exclusion_reason": "EXCLUDED_DEPENDENCY_BUDGET",
            }
        )


@dataclass
class PhaseStateMachine:
    phase: DevelopmentPhase = DevelopmentPhase.INPUT_LOCK
    closure_frozen: bool = False
    closure_bytes: Optional[bytes] = None
    history: List[str] = field(default_factory=list)

    def advance(self, next_phase: DevelopmentPhase) -> None:
        allowed = ALLOWED_TRANSITIONS[self.phase]
        if next_phase not in allowed:
            raise CoreContractError(f"illegal phase transition {self.phase} -> {next_phase}")
        self.phase = next_phase
        self.history.append(next_phase.value)

    def freeze_closure(self, closure: Mapping[str, Any]) -> Dict[str, Any]:
        if self.phase == DevelopmentPhase.SELECTION:
            self.advance(DevelopmentPhase.CLOSURE_FREEZE)
        elif self.phase != DevelopmentPhase.CLOSURE_FREEZE:
            raise CoreContractError(f"cannot freeze closure in phase {self.phase}")
        body = {k: v for k, v in dict(closure).items() if k != "evaluation_closure_hash"}
        body["closure_kind"] = "SELECTION_BLIND_REEVALUATION_CLOSURE"
        digest = canonical_json_sha256(body)
        self.closure_frozen = True
        self.closure_bytes = digest.encode("utf-8")
        return {**body, "evaluation_closure_hash": digest}

    def assert_closure_immutable(self, closure: Mapping[str, Any]) -> None:
        if not self.closure_frozen or self.closure_bytes is None:
            raise CoreContractError("closure not frozen")
        body = {k: v for k, v in closure.items() if k != "evaluation_closure_hash"}
        body["closure_kind"] = "SELECTION_BLIND_REEVALUATION_CLOSURE"
        expected = canonical_json_sha256(body).encode("utf-8")
        if expected != self.closure_bytes:
            raise CoreContractError("closure mutated after freeze")

def build_selection_blind_closure(
    *,
    case_id: str,
    candidate_identity_hashes: Sequence[str],
    search_result_hash: str,
    selection_manifest_sha256: str,
    candidate_budget_policy_sha256: str,
    threshold_policy_sha256: str,
    risk_definition_sha256: str,
    candidate_universe_hash: str,
    parent_graph_sha256: str,
    expected_effect_direction_by_candidate: Mapping[str, str],
    required_parent_transaction_shas: Sequence[str] = (),
) -> Dict[str, Any]:
    body = {
        "closure_kind": "SELECTION_BLIND_REEVALUATION_CLOSURE",
        "case_id": case_id,
        "candidate_identity_hashes": sorted(candidate_identity_hashes),
        "search_result_hash": search_result_hash,
        "selection_manifest_sha256": selection_manifest_sha256,
        "candidate_budget_policy_sha256": candidate_budget_policy_sha256,
        "threshold_policy_sha256": threshold_policy_sha256,
        "risk_definition_sha256": risk_definition_sha256,
        "candidate_universe_hash": candidate_universe_hash,
        "parent_graph_sha256": parent_graph_sha256,
        "required_parent_transaction_shas": sorted(
            set(str(value) for value in required_parent_transaction_shas)
        ),
        "expected_effect_direction_by_candidate": dict(
            sorted(expected_effect_direction_by_candidate.items())
        ),
        "holdout_effect_included": False,
        "scientifically_independent_holdout": False,
    }
    evaluation_closure_hash = canonical_json_sha256(body)
    return {
        **body,
        "evaluation_closure_hash": evaluation_closure_hash,
        "chain": {
            "candidate_universe_hash": candidate_universe_hash,
            "search_result_hash": search_result_hash,
            "selection_manifest_hash": selection_manifest_sha256,
            "evaluation_closure_hash": evaluation_closure_hash,
        },
    }


def migrate_legacy_holdout_closure(legacy: Mapping[str, Any]) -> Dict[str, Any]:
    """Read-only adapter for old artifacts."""
    if legacy.get("closure_kind") != "HOLDOUT_BLIND_EVALUATION_CLOSURE":
        raise CoreContractError("not a legacy holdout closure")
    out = dict(legacy)
    out["legacy_enum"] = "HOLDOUT_BLIND_EVALUATION_CLOSURE"
    out["semantic_role"] = "SELECTION_BLIND_REEVALUATION_CLOSURE"
    out["closure_kind"] = "SELECTION_BLIND_REEVALUATION_CLOSURE"
    out["scientifically_independent_holdout"] = False
    return out


def evaluate_bundle_effects(
    *,
    search_fold_deltas: Mapping[str, float],
    reeval_fold_deltas: Mapping[str, float],
    search_parent_deltas_by_fold: Mapping[str, Sequence[float]],
    reeval_parent_deltas_by_fold: Mapping[str, Sequence[float]],
    expected_effect_direction: str,
    locked_threshold: float,
) -> Dict[str, Any]:
    search_stable = stable_effect_scope(
        search_fold_deltas,
        required_fold_ids=["0", "1"],
        expected_effect_direction=expected_effect_direction,
        locked_threshold=locked_threshold,
    )
    reeval_stable = stable_effect_scope(
        reeval_fold_deltas,
        required_fold_ids=["2"],
        expected_effect_direction=expected_effect_direction,
        locked_threshold=locked_threshold,
    )
    stable_replicated = bool(search_stable and reeval_stable)

    def _inc_scope(fold_deltas, parent_by_fold, required):
        for fid in required:
            if fid not in fold_deltas or fid not in parent_by_fold:
                return False
            inc = incremental_directed(
                fold_deltas[fid],
                parent_by_fold[fid],
                expected_effect_direction=expected_effect_direction,
            )
            if inc < float(locked_threshold):
                return False
        return True

    search_inc = _inc_scope(search_fold_deltas, search_parent_deltas_by_fold, ["0", "1"])
    reeval_inc = _inc_scope(reeval_fold_deltas, reeval_parent_deltas_by_fold, ["2"])
    multi = bool(stable_replicated and search_inc and reeval_inc)

    status = scientific_status_from_effects(
        stable_effect_replicated=stable_replicated,
        multievent_increment_supported=multi,
    )
    claims = evaluate_claim_levels(
        model_input_delivery_verified=True,
        stable_effect_replicated=stable_replicated,
        multievent_increment_supported=multi,
    )
    return {**status, **claims, "interaction_status": "INTERACTION_DIAGNOSTIC_ONLY"}


def development_readiness_axes(
    *,
    execution_ready: bool,
    evaluable_cases: int,
    locked_cohort_size: int = 3,
    scientific_counts: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    if evaluable_cases <= 0:
        coverage = "NONE"
    elif evaluable_cases < locked_cohort_size:
        coverage = "PARTIAL"
    else:
        coverage = "COMPLETE"
    return {
        "development_execution_readiness": "READY" if execution_ready else "NOT_READY",
        "development_evaluation_coverage": coverage,
        "development_scientific_result": dict(scientific_counts or {}),
        "cohort_completion_claim_allowed": coverage == "COMPLETE",
    }
