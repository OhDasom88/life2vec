"""Case-level disposition classification and expected-trace-profile verification (P0-6)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .core_contract import CoreContractError
from .core_selection import SELECTED_CONTROL_NO_MATERIAL, SELECTED_MATERIAL

CONSTRUCTIBLE_SELECTED = "CONSTRUCTIBLE_SELECTED"
CONSTRUCTIBLE_ALL_GATE_FAILED = "CONSTRUCTIBLE_ALL_GATE_FAILED"
CONSTRUCTIBLE_ALL_EXCLUDED = "CONSTRUCTIBLE_ALL_EXCLUDED"
NOT_CONSTRUCTIBLE = "NOT_CONSTRUCTIBLE"
NOT_EVALUATED = "NOT_EVALUATED"
NOT_EVALUABLE = "NOT_EVALUABLE"

CASE_DISPOSITIONS = (
    CONSTRUCTIBLE_SELECTED,
    CONSTRUCTIBLE_ALL_GATE_FAILED,
    CONSTRUCTIBLE_ALL_EXCLUDED,
    NOT_CONSTRUCTIBLE,
    NOT_EVALUATED,
    NOT_EVALUABLE,
)

_SELECTED_CANDIDATE_DISPOSITIONS = (SELECTED_MATERIAL, SELECTED_CONTROL_NO_MATERIAL)
_TERMINAL_STATES = ("COMPLETED", "FAILED", "DENIED")


class ForwardCountRule(str, Enum):
    NONZERO_REQUIRED = "NONZERO_REQUIRED"
    ZERO_IF_BLOCKED = "ZERO_IF_BLOCKED"
    ZERO_NORMAL = "ZERO_NORMAL"
    PARTIAL_ALLOWED = "PARTIAL_ALLOWED"


@dataclass(frozen=True)
class ExpectedTraceProfile:
    required_kind_scope_pairs: FrozenSet[Tuple[str, str]]
    forward_count_rule: str
    requires_fold2_closure: bool
    required_terminal: FrozenSet[str]
    forbid_post_terminal_side_effects: bool


PROFILES: Dict[str, ExpectedTraceProfile] = {
    CONSTRUCTIBLE_SELECTED: ExpectedTraceProfile(
        required_kind_scope_pairs=frozenset(
            {
                ("ATTRIBUTION_MODEL_LOAD", "search"),
                ("ATTRIBUTION_FORWARD", "search"),
                ("PIPELINE_INVOCATION", "search"),
                ("CHECKPOINT_FORWARD", "search"),
            }
        ),
        forward_count_rule=ForwardCountRule.NONZERO_REQUIRED.value,
        requires_fold2_closure=True,
        required_terminal=frozenset({"COMPLETED"}),
        forbid_post_terminal_side_effects=False,
    ),
    CONSTRUCTIBLE_ALL_GATE_FAILED: ExpectedTraceProfile(
        required_kind_scope_pairs=frozenset(
            {
                ("ATTRIBUTION_MODEL_LOAD", "search"),
                ("ATTRIBUTION_FORWARD", "search"),
            }
        ),
        forward_count_rule=ForwardCountRule.ZERO_IF_BLOCKED.value,
        requires_fold2_closure=False,
        required_terminal=frozenset({"DENIED", "FAILED"}),
        forbid_post_terminal_side_effects=False,
    ),
    # Construction + search executed to completion (no authorization/scope
    # gate denial anywhere in trace) but every candidate was scientifically
    # excluded at selection (e.g. mixed/partial-material across search
    # folds, below-threshold). Distinct from CONSTRUCTIBLE_ALL_GATE_FAILED,
    # which models a candidate forward being denied/blocked before it could
    # run — here every constructed candidate is unavoidably forward-scored
    # first (search has no pre-selection by budget), so nonzero forward
    # count is the normal, expected state.
    CONSTRUCTIBLE_ALL_EXCLUDED: ExpectedTraceProfile(
        required_kind_scope_pairs=frozenset(
            {
                ("ATTRIBUTION_MODEL_LOAD", "search"),
                ("ATTRIBUTION_FORWARD", "search"),
                ("PIPELINE_INVOCATION", "search"),
                ("CHECKPOINT_FORWARD", "search"),
            }
        ),
        forward_count_rule=ForwardCountRule.NONZERO_REQUIRED.value,
        requires_fold2_closure=False,
        required_terminal=frozenset({"COMPLETED"}),
        forbid_post_terminal_side_effects=False,
    ),
    NOT_CONSTRUCTIBLE: ExpectedTraceProfile(
        required_kind_scope_pairs=frozenset(),
        forward_count_rule=ForwardCountRule.ZERO_NORMAL.value,
        requires_fold2_closure=False,
        required_terminal=frozenset({"FAILED"}),
        forbid_post_terminal_side_effects=False,
    ),
    NOT_EVALUATED: ExpectedTraceProfile(
        required_kind_scope_pairs=frozenset(),
        forward_count_rule=ForwardCountRule.ZERO_NORMAL.value,
        requires_fold2_closure=False,
        required_terminal=frozenset({"DENIED"}),
        forbid_post_terminal_side_effects=True,
    ),
    NOT_EVALUABLE: ExpectedTraceProfile(
        required_kind_scope_pairs=frozenset(),
        forward_count_rule=ForwardCountRule.PARTIAL_ALLOWED.value,
        requires_fold2_closure=False,
        required_terminal=frozenset({"FAILED"}),
        forbid_post_terminal_side_effects=False,
    ),
}


def classify_case_disposition(proposal: Mapping[str, Any]) -> str:
    """Recompute the case-level disposition from already-verified proposal fields.

    Never trusts a self-declared case-level disposition — that is exactly the
    spoofable shortcut P0-6 closes. Only derives from fields G1/G4/G6/G9 already
    cross-check (construction_status, execution_evidence, candidate_results).
    """
    construction_status = str(proposal.get("construction_status") or "")
    scientific_status = str(proposal.get("scientific_status") or "")
    evidence = proposal.get("execution_evidence") or {}
    candidate_results = proposal.get("candidate_results") or []
    has_selected = any(
        row.get("disposition") in _SELECTED_CANDIDATE_DISPOSITIONS
        for row in candidate_results
    )

    if construction_status == "NOT_CONSTRUCTIBLE":
        return NOT_CONSTRUCTIBLE
    if scientific_status == "NOT_EVALUABLE":
        return NOT_EVALUABLE
    if not evidence.get("production_input_verified") or not evidence.get(
        "pipeline_integrity_verified"
    ):
        return NOT_EVALUATED
    if has_selected:
        return CONSTRUCTIBLE_SELECTED
    # scientific_status is "NOT_EVALUATED" whenever nothing was selected —
    # including cases where construction/search fully executed but every
    # candidate was excluded (e.g. mixed/partial-material across search
    # folds). The real execution-blocked NOT_EVALUATED case is already
    # caught above via the evidence-flag check. A non-empty candidate
    # ledger means search actually ran and scored every constructed
    # candidate (no pre-selection by budget) — that is CONSTRUCTIBLE_
    # ALL_EXCLUDED, not a gate denial (empty ledger, forward never ran).
    if candidate_results:
        return CONSTRUCTIBLE_ALL_EXCLUDED
    return CONSTRUCTIBLE_ALL_GATE_FAILED


def case_trace_events(
    all_events: Sequence[Mapping[str, Any]], case_id: str
) -> List[Dict[str, Any]]:
    """Derive a case's own trace by filtering the global chain on case_id.

    No independent per-case trace-root is written today; filtering the shared
    chain (rather than trusting package-aggregate counts) is what prevents one
    case's trace from satisfying another case's requirement — every downstream
    check in this module only ever sees rows returned from here.
    """
    rows = [dict(event) for event in all_events if event.get("case_id") == case_id]
    if not rows:
        raise CoreContractError(f"no trace events for case {case_id}")
    return rows


def evaluate_case_trace_profile(
    *,
    case_id: str,
    disposition: str,
    case_events: Sequence[Mapping[str, Any]],
    closure_sha: Optional[str] = None,
) -> None:
    if disposition not in PROFILES:
        raise CoreContractError(f"unknown case disposition: {disposition}")
    profile = PROFILES[disposition]
    completed = [e for e in case_events if e.get("state") == "COMPLETED"]

    for kind, scope in profile.required_kind_scope_pairs:
        if not any(e.get("kind") == kind and e.get("scope") == scope for e in completed):
            raise CoreContractError(
                f"case {case_id} disposition {disposition} missing required trace {kind}/{scope}"
            )

    candidate_forward_count = sum(
        1
        for e in completed
        if e.get("kind") == "PIPELINE_INVOCATION"
        and (e.get("extra") or {}).get("forward_kind") == "CANDIDATE_EFFECT_FORWARD"
    )
    rule = profile.forward_count_rule
    if rule == ForwardCountRule.NONZERO_REQUIRED.value and candidate_forward_count <= 0:
        raise CoreContractError(
            f"case {case_id} disposition {disposition} requires nonzero candidate forward"
        )
    if (
        rule in (ForwardCountRule.ZERO_IF_BLOCKED.value, ForwardCountRule.ZERO_NORMAL.value)
        and candidate_forward_count != 0
    ):
        raise CoreContractError(
            f"case {case_id} disposition {disposition} forbids candidate forward"
        )

    if profile.requires_fold2_closure:
        fold2_replay = [
            e
            for e in completed
            if e.get("kind") in ("CHECKPOINT_FORWARD", "CRITIC_LOAD")
            and e.get("scope") in ("selection_blind_reevaluation", "holdout", "fold2")
            and e.get("fold_id") == 2
            and e.get("closure_frozen") is True
        ]
        if not fold2_replay:
            raise CoreContractError(
                f"case {case_id} disposition {disposition} missing Fold2 closure replay trace"
            )
        if closure_sha is not None and any(
            e.get("closure_sha") != closure_sha for e in fold2_replay
        ):
            raise CoreContractError(f"case {case_id} Fold2 replay not bound to frozen closure")

    if profile.required_terminal:
        terminal_events = [e for e in case_events if e.get("state") in _TERMINAL_STATES]
        observed_terminal = {e.get("state") for e in terminal_events}
        if not observed_terminal & profile.required_terminal:
            raise CoreContractError(
                f"case {case_id} disposition {disposition} missing required terminal state "
                f"{sorted(profile.required_terminal)}"
            )
        if "FAILED" in profile.required_terminal:
            failed_events = [e for e in terminal_events if e.get("state") == "FAILED"]
            if failed_events and not any(
                (e.get("extra") or {}).get("terminal_failure") is True
                for e in failed_events
            ):
                raise CoreContractError(
                    f"case {case_id} disposition {disposition} FAILED terminal "
                    "missing terminal_failure marker"
                )

    if profile.forbid_post_terminal_side_effects:
        denied_sequences = [
            e.get("sequence") for e in case_events if e.get("state") == "DENIED"
        ]
        if denied_sequences:
            last_denied_seq = max(denied_sequences)
            for e in case_events:
                seq = e.get("sequence")
                if (
                    seq is not None
                    and seq > last_denied_seq
                    and e.get("state") in ("STARTED", "COMPLETED")
                ):
                    raise CoreContractError(
                        f"case {case_id} disposition {disposition} has side effect after DENIED"
                    )
