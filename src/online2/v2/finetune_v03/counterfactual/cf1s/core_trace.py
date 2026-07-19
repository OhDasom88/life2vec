"""Global hash-chained load/forward trace with REQUESTED→ALLOWED|DENIED→STARTED→COMPLETED|FAILED."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .core_canonical import canonical_json_sha256
from .core_contract import CoreContractError


class TraceKind(str, Enum):
    PIPELINE_INVOCATION = "PIPELINE_INVOCATION"
    CHECKPOINT_FORWARD = "CHECKPOINT_FORWARD"
    STAGE_A_LOAD = "STAGE_A_LOAD"
    CRITIC_LOAD = "CRITIC_LOAD"
    ATTRIBUTION_MODEL_LOAD = "ATTRIBUTION_MODEL_LOAD"
    ATTRIBUTION_FORWARD = "ATTRIBUTION_FORWARD"


class TraceState(str, Enum):
    REQUESTED = "REQUESTED"
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


_TERMINAL = {TraceState.DENIED, TraceState.COMPLETED, TraceState.FAILED}


@dataclass
class TraceInvocation:
    invocation_id: str
    kind: TraceKind
    states: List[TraceState] = field(default_factory=list)
    terminal: bool = False


class GlobalForwardTrace:
    """Single deterministic writer; parallel worker merge is out of scope."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._prev_sha: Optional[str] = None
        self._seq = 0
        self._invocations: Dict[str, TraceInvocation] = {}
        self._closure_frozen = False
        self._closure_sha: Optional[str] = None

    def mark_closure_frozen(self, *, closure_sha: str) -> None:
        self._closure_frozen = True
        self._closure_sha = str(closure_sha)

    def _append(
        self,
        *,
        kind: TraceKind,
        state: TraceState,
        invocation_id: str,
        phase: str,
        scope: str,
        case_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        transaction_sha: Optional[str] = None,
        fold_id: Optional[int] = None,
        checkpoint_sha: Optional[str] = None,
        stage_a_pairing_sha: Optional[str] = None,
        failure_code: Optional[str] = None,
        parent_invocation_id: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        inv = self._invocations.get(invocation_id)
        if inv is None:
            if state != TraceState.REQUESTED:
                raise CoreContractError(
                    f"trace invocation {invocation_id} must start with REQUESTED"
                )
            inv = TraceInvocation(invocation_id=invocation_id, kind=kind)
            self._invocations[invocation_id] = inv
        else:
            if inv.kind != kind:
                raise CoreContractError("trace invocation kind mismatch")
            if inv.terminal:
                raise CoreContractError(
                    f"terminal state already reached for invocation {invocation_id}"
                )
            self._assert_transition(inv, state)

        if (
            scope in ("selection_blind_reevaluation", "holdout", "fold2")
            and kind
            in (
                TraceKind.ATTRIBUTION_MODEL_LOAD,
                TraceKind.ATTRIBUTION_FORWARD,
                TraceKind.CRITIC_LOAD,
                TraceKind.CHECKPOINT_FORWARD,
            )
            and not self._closure_frozen
            and state in (TraceState.ALLOWED, TraceState.STARTED, TraceState.COMPLETED)
        ):
            raise CoreContractError(
                "selection-blind reevaluation load/forward before closure freeze"
            )

        self._seq += 1
        body = {
            "trace_event_id": f"tev_{self._seq:08d}",
            "sequence": self._seq,
            "previous_event_sha": self._prev_sha,
            "kind": kind.value,
            "state": state.value,
            "invocation_id": invocation_id,
            "parent_invocation_id": parent_invocation_id,
            "phase": phase,
            "scope": scope,
            "case_id": case_id,
            "candidate_id": candidate_id,
            "transaction_sha": transaction_sha,
            "fold_id": fold_id,
            "checkpoint_sha": checkpoint_sha,
            "stage_a_pairing_sha": stage_a_pairing_sha,
            "closure_sha": self._closure_sha,
            "closure_frozen": self._closure_frozen,
            "failure_code": failure_code,
            "extra": dict(extra or {}),
        }
        event_sha = canonical_json_sha256(body)
        body["event_sha"] = event_sha
        self.events.append(body)
        self._prev_sha = event_sha

        inv.states.append(state)
        if state in _TERMINAL:
            inv.terminal = True
        return body

    @staticmethod
    def _assert_transition(inv: TraceInvocation, state: TraceState) -> None:
        states = inv.states
        if not states:
            raise CoreContractError("internal: empty invocation states")
        last = states[-1]
        if last == TraceState.REQUESTED:
            if state not in (TraceState.ALLOWED, TraceState.DENIED):
                raise CoreContractError("REQUESTED must be followed by ALLOWED or DENIED")
            return
        if last == TraceState.ALLOWED:
            if state != TraceState.STARTED:
                raise CoreContractError("ALLOWED must be followed by STARTED")
            return
        if last == TraceState.STARTED:
            if state not in (TraceState.COMPLETED, TraceState.FAILED):
                raise CoreContractError("STARTED must be followed by COMPLETED or FAILED")
            return
        if last == TraceState.DENIED:
            raise CoreContractError("DENIED forbids further events")
        raise CoreContractError(f"illegal transition from {last} to {state}")

    def begin_operation(
        self,
        *,
        kind: TraceKind,
        invocation_id: str,
        phase: str,
        scope: str,
        allow: bool = True,
        denial_code: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Emit REQUESTED then ALLOWED or DENIED. Returns True iff allowed."""
        self._append(
            kind=kind,
            state=TraceState.REQUESTED,
            invocation_id=invocation_id,
            phase=phase,
            scope=scope,
            **kwargs,
        )
        if allow:
            self._append(
                kind=kind,
                state=TraceState.ALLOWED,
                invocation_id=invocation_id,
                phase=phase,
                scope=scope,
                **kwargs,
            )
            return True
        self._append(
            kind=kind,
            state=TraceState.DENIED,
            invocation_id=invocation_id,
            phase=phase,
            scope=scope,
            failure_code=denial_code or "DENIED",
            **kwargs,
        )
        return False

    def start(self, *, kind: TraceKind, invocation_id: str, phase: str, scope: str, **kwargs: Any) -> None:
        self._append(
            kind=kind,
            state=TraceState.STARTED,
            invocation_id=invocation_id,
            phase=phase,
            scope=scope,
            **kwargs,
        )

    def complete(self, *, kind: TraceKind, invocation_id: str, phase: str, scope: str, **kwargs: Any) -> None:
        self._append(
            kind=kind,
            state=TraceState.COMPLETED,
            invocation_id=invocation_id,
            phase=phase,
            scope=scope,
            **kwargs,
        )

    def fail(
        self,
        *,
        kind: TraceKind,
        invocation_id: str,
        phase: str,
        scope: str,
        failure_code: str,
        **kwargs: Any,
    ) -> None:
        self._append(
            kind=kind,
            state=TraceState.FAILED,
            invocation_id=invocation_id,
            phase=phase,
            scope=scope,
            failure_code=failure_code,
            **kwargs,
        )

    def summarize(self) -> Dict[str, Any]:
        """All summary counts are derived only from this global hash chain."""
        requested = denied = failed = 0
        checkpoint_forward_count = 0
        pipeline_invocation_count = 0
        stage_a_load_count = 0
        critic_load_count = 0
        attribution_load_count = 0
        attribution_forward_count = 0
        fold2_critic_load_before_closure = 0
        fold2_attribution_load = 0
        fold2_attribution_forward = 0
        fold2_candidate_forward_before_closure = 0
        nonrequested_fold_forward = 0

        for ev in self.events:
            st = ev["state"]
            kind = ev["kind"]
            if st == TraceState.REQUESTED.value:
                requested += 1
            elif st == TraceState.DENIED.value:
                denied += 1
            elif st == TraceState.FAILED.value:
                failed += 1
            if st == TraceState.COMPLETED.value:
                if kind == TraceKind.CHECKPOINT_FORWARD.value:
                    checkpoint_forward_count += 1
                elif kind == TraceKind.PIPELINE_INVOCATION.value:
                    pipeline_invocation_count += 1
                elif kind == TraceKind.STAGE_A_LOAD.value:
                    stage_a_load_count += 1
                elif kind == TraceKind.CRITIC_LOAD.value:
                    critic_load_count += 1
                    if not ev.get("closure_frozen") and ev.get("scope") in (
                        "selection_blind_reevaluation",
                        "holdout",
                        "fold2",
                    ):
                        fold2_critic_load_before_closure += 1
                elif kind == TraceKind.ATTRIBUTION_MODEL_LOAD.value:
                    attribution_load_count += 1
                    if ev.get("scope") in ("selection_blind_reevaluation", "holdout", "fold2"):
                        fold2_attribution_load += 1
                elif kind == TraceKind.ATTRIBUTION_FORWARD.value:
                    attribution_forward_count += 1
                    if ev.get("scope") in ("selection_blind_reevaluation", "holdout", "fold2"):
                        fold2_attribution_forward += 1

                if (
                    kind in (
                        TraceKind.ATTRIBUTION_MODEL_LOAD.value,
                        TraceKind.ATTRIBUTION_FORWARD.value,
                        TraceKind.CRITIC_LOAD.value,
                        TraceKind.CHECKPOINT_FORWARD.value,
                    )
                    and (ev.get("extra") or {}).get("requested_fold") is False
                ):
                    nonrequested_fold_forward += 1
                if (
                    kind == TraceKind.CHECKPOINT_FORWARD.value
                    and not ev.get("closure_frozen")
                    and ev.get("scope")
                    in ("selection_blind_reevaluation", "holdout", "fold2")
                    and (ev.get("extra") or {}).get("forward_kind")
                    == "CANDIDATE_EFFECT_FORWARD"
                ):
                    fold2_candidate_forward_before_closure += 1

        # Validate per-invocation invariants
        for inv in self._invocations.values():
            if not inv.states or inv.states[0] != TraceState.REQUESTED:
                raise CoreContractError(f"invocation {inv.invocation_id} missing REQUESTED")
            if TraceState.ALLOWED in inv.states and TraceState.DENIED in inv.states:
                raise CoreContractError(f"invocation {inv.invocation_id} ALLOWED and DENIED")
            if TraceState.DENIED in inv.states:
                if any(
                    s in inv.states
                    for s in (TraceState.STARTED, TraceState.COMPLETED, TraceState.FAILED)
                ):
                    raise CoreContractError(
                        f"DENIED invocation {inv.invocation_id} has side-effect states"
                    )

        chain_root = self._prev_sha or canonical_json_sha256({"events": []})
        return {
            "trace_event_count": len(self.events),
            "requested_count": requested,
            "denied_count": denied,
            "failed_count": failed,
            "pipeline_invocation_count": pipeline_invocation_count,
            "checkpoint_forward_count": checkpoint_forward_count,
            "stage_a_load_count": stage_a_load_count,
            "critic_load_count": critic_load_count,
            "attribution_load_count": attribution_load_count,
            "attribution_forward_count": attribution_forward_count,
            "fold2_critic_load_before_closure_count": fold2_critic_load_before_closure,
            "fold2_attribution_load_count": fold2_attribution_load,
            "fold2_attribution_forward_count": fold2_attribution_forward,
            "fold2_candidate_forward_before_closure_count": fold2_candidate_forward_before_closure,
            "nonrequested_fold_forward_count": nonrequested_fold_forward,
            "global_trace_chain_root_sha": chain_root,
            "events": list(self.events),
        }

    def to_forward_counts(self) -> Dict[str, int]:
        s = self.summarize()
        # Map to legacy envelope keys; only COMPLETED checkpoint/pipeline counts.
        baseline = sum(
            1
            for e in self.events
            if e["kind"] == TraceKind.PIPELINE_INVOCATION.value
            and e["state"] == TraceState.COMPLETED.value
            and (e.get("extra") or {}).get("forward_kind") == "BASELINE_FORWARD"
        )
        identity = sum(
            1
            for e in self.events
            if e["kind"] == TraceKind.PIPELINE_INVOCATION.value
            and e["state"] == TraceState.COMPLETED.value
            and (e.get("extra") or {}).get("forward_kind") == "FORCED_IDENTITY_FORWARD"
        )
        candidate = sum(
            1
            for e in self.events
            if e["kind"] == TraceKind.PIPELINE_INVOCATION.value
            and e["state"] == TraceState.COMPLETED.value
            and (e.get("extra") or {}).get("forward_kind") == "CANDIDATE_EFFECT_FORWARD"
        )
        return {
            "attribution_forward_count": s["attribution_forward_count"],
            "baseline_forward_count": baseline,
            "identity_forward_count": identity,
            "candidate_effect_forward_count": candidate,
            "nonrequested_fold_forward_count": s["nonrequested_fold_forward_count"],
            "holdout_attribution_forward_count": s["fold2_attribution_forward_count"],
            "holdout_candidate_forward_before_closure_count": s[
                "fold2_candidate_forward_before_closure_count"
            ],
            "fold2_attribution_load_count": s["fold2_attribution_load_count"],
            "fold2_attribution_forward_count": s["fold2_attribution_forward_count"],
            "fold2_critic_load_before_closure_count": s[
                "fold2_critic_load_before_closure_count"
            ],
            "stage_a_load_count": s["stage_a_load_count"],
            "critic_load_count": s["critic_load_count"],
            "attribution_load_count": s["attribution_load_count"],
            "pipeline_invocation_count": s["pipeline_invocation_count"],
            "checkpoint_forward_count": s["checkpoint_forward_count"],
            "trace_event_count": s["trace_event_count"],
            "requested_count": s["requested_count"],
            "denied_count": s["denied_count"],
            "failed_count": s["failed_count"],
        }


def validate_persisted_trace(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Rebuild and validate a persisted trace without trusting in-memory state."""
    rebuilt = GlobalForwardTrace()
    expected_previous: Optional[str] = None
    expected_sequence = 1
    seen_event_ids = set()

    for raw in events:
        event = dict(raw)
        if event.get("sequence") != expected_sequence:
            raise CoreContractError("trace sequence gap or duplicate")
        expected_event_id = f"tev_{expected_sequence:08d}"
        if event.get("trace_event_id") != expected_event_id:
            raise CoreContractError("trace_event_id mismatch")
        if event["trace_event_id"] in seen_event_ids:
            raise CoreContractError("duplicate trace_event_id")
        seen_event_ids.add(event["trace_event_id"])
        if event.get("previous_event_sha") != expected_previous:
            raise CoreContractError("trace previous_event_sha mismatch")
        supplied_sha = event.pop("event_sha", None)
        computed_sha = canonical_json_sha256(event)
        if supplied_sha != computed_sha:
            raise CoreContractError("trace event_sha mismatch")

        kind = TraceKind(str(event["kind"]))
        state = TraceState(str(event["state"]))
        if event.get("closure_frozen") and (
            not rebuilt._closure_frozen
            or str(event.get("closure_sha")) != str(rebuilt._closure_sha)
        ):
            closure_sha = event.get("closure_sha")
            if not closure_sha:
                raise CoreContractError("frozen trace event missing closure_sha")
            rebuilt.mark_closure_frozen(closure_sha=str(closure_sha))
        if bool(event.get("closure_frozen")) != rebuilt._closure_frozen:
            raise CoreContractError("trace closure state regression")
        rebuilt._append(
            kind=kind,
            state=state,
            invocation_id=str(event["invocation_id"]),
            phase=str(event["phase"]),
            scope=str(event["scope"]),
            case_id=event.get("case_id"),
            candidate_id=event.get("candidate_id"),
            transaction_sha=event.get("transaction_sha"),
            fold_id=event.get("fold_id"),
            checkpoint_sha=event.get("checkpoint_sha"),
            stage_a_pairing_sha=event.get("stage_a_pairing_sha"),
            failure_code=event.get("failure_code"),
            parent_invocation_id=event.get("parent_invocation_id"),
            extra=event.get("extra") or {},
        )
        if rebuilt.events[-1]["event_sha"] != supplied_sha:
            raise CoreContractError("trace deterministic rebuild mismatch")
        expected_previous = supplied_sha
        expected_sequence += 1

    for invocation_id, invocation in rebuilt._invocations.items():
        if not invocation.terminal:
            raise CoreContractError(f"trace invocation missing terminal: {invocation_id}")

    # A failed logical operation is acceptable only when an explicit retry completed.
    failed_logical_ids = set()
    completed_logical_ids = set()
    for event in rebuilt.events:
        logical_id = str(
            (event.get("extra") or {}).get("logical_operation_id")
            or event.get("invocation_id")
        )
        if event["state"] == TraceState.FAILED.value:
            failed_logical_ids.add(logical_id)
        elif event["state"] == TraceState.COMPLETED.value:
            completed_logical_ids.add(logical_id)
    unresolved = sorted(failed_logical_ids - completed_logical_ids)
    if unresolved:
        raise CoreContractError(
            "trace failed logical operation without successful retry: "
            + ",".join(unresolved)
        )
    return rebuilt.summarize()
