"""Atomic multi-event retokenization + Stage-A reencode for CF-1S."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


@dataclass
class MultiEventRetokenizeResult:
    ok: bool
    length_preserving: bool
    edited_event_ids: List[str] = field(default_factory=list)
    affected_target_event_ids: List[str] = field(default_factory=list)
    grounding_failure_count: int = 0
    inversion_failure_count: int = 0
    gate0_failure_count: int = 0
    gate4_failure_count: int = 0
    failure_reason: Optional[str] = None
    edits_applied: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def _as_token_len(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def apply_multi_event_edits_atomically(
    *,
    edits: Sequence[Mapping[str, Any]],
    ground_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    invert_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    gate0_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    gate4_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    retokenize_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    reencode_fn: Optional[Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]] = None,
    length_preserving_edit_only: bool = True,
) -> MultiEventRetokenizeResult:
    """Apply all edits atomically; any atomic failure rejects the composite."""
    prepared: List[Dict[str, Any]] = []
    grounding_fail = inversion_fail = gate0_fail = gate4_fail = 0

    for edit in edits:
        g = dict(ground_fn(edit))
        if not bool(g.get("ok", g.get("raw_grounding_status") == "EXACT")):
            grounding_fail += 1
            return MultiEventRetokenizeResult(
                ok=False,
                length_preserving=False,
                grounding_failure_count=grounding_fail,
                failure_reason="GROUNDING_FAILED",
                details={"failed_edit": dict(edit), "grounding": g},
            )
        inv = dict(invert_fn({**edit, **g}))
        if not bool(inv.get("ok", True)):
            inversion_fail += 1
            return MultiEventRetokenizeResult(
                ok=False,
                length_preserving=False,
                grounding_failure_count=grounding_fail,
                inversion_failure_count=inversion_fail,
                failure_reason="INVERSION_FAILED",
                details={"failed_edit": dict(edit), "inversion": inv},
            )
        tok = dict(retokenize_fn({**edit, **g, **inv}))
        orig_len = _as_token_len(tok.get("original_token_length") or edit.get("original_token_length"))
        new_len = _as_token_len(tok.get("retokenized_token_length") or tok.get("tokens"))
        length_preserving = (
            orig_len is not None and new_len is not None and orig_len == new_len
        )
        if length_preserving_edit_only and not length_preserving:
            return MultiEventRetokenizeResult(
                ok=False,
                length_preserving=False,
                grounding_failure_count=grounding_fail,
                inversion_failure_count=inversion_fail,
                failure_reason="LENGTH_CHANGE_REJECTED",
                details={"failed_edit": dict(edit), "original_len": orig_len, "new_len": new_len},
            )
        g0 = dict(gate0_fn({**edit, **tok}))
        if not bool(g0.get("ok", g0.get("gate0_pass", False))):
            gate0_fail += 1
            return MultiEventRetokenizeResult(
                ok=False,
                length_preserving=length_preserving,
                grounding_failure_count=grounding_fail,
                inversion_failure_count=inversion_fail,
                gate0_failure_count=gate0_fail,
                failure_reason="GATE0_FAILED",
                details={"failed_edit": dict(edit), "gate0": g0},
            )
        g4 = dict(gate4_fn({**edit, **tok}))
        if not bool(g4.get("ok", g4.get("gate4_pass", False))):
            gate4_fail += 1
            return MultiEventRetokenizeResult(
                ok=False,
                length_preserving=length_preserving,
                grounding_failure_count=grounding_fail,
                inversion_failure_count=inversion_fail,
                gate0_failure_count=gate0_fail,
                gate4_failure_count=gate4_fail,
                failure_reason="GATE4_FAILED",
                details={"failed_edit": dict(edit), "gate4": g4},
            )
        prepared.append({**edit, **g, **inv, **tok, "length_preserving": length_preserving})

    affected: List[str] = []
    if reencode_fn is not None:
        re = dict(reencode_fn(prepared))
        if not bool(re.get("ok", True)):
            return MultiEventRetokenizeResult(
                ok=False,
                length_preserving=all(p.get("length_preserving") for p in prepared),
                grounding_failure_count=grounding_fail,
                inversion_failure_count=inversion_fail,
                gate0_failure_count=gate0_fail,
                gate4_failure_count=gate4_fail,
                edited_event_ids=[str(p.get("event_id")) for p in prepared],
                failure_reason="STAGE_A_REENCODE_FAILED",
                details={"reencode": re},
            )
        affected = [str(x) for x in (re.get("affected_target_event_ids") or [])]

    return MultiEventRetokenizeResult(
        ok=True,
        length_preserving=all(p.get("length_preserving") for p in prepared),
        edited_event_ids=[str(p.get("event_id")) for p in prepared],
        affected_target_event_ids=affected,
        grounding_failure_count=grounding_fail,
        inversion_failure_count=inversion_fail,
        gate0_failure_count=gate0_fail,
        gate4_failure_count=gate4_fail,
        edits_applied=prepared,
    )


def parity_sample_rule(candidate_id: str, *, mod: int = 10) -> bool:
    digest = __import__("hashlib").sha256(str(candidate_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % int(mod) == 0


def classify_parity_status(
    *,
    failure_count: int,
    fallback_failure_count: int,
    minimum_coverage_met: bool,
) -> str:
    if not minimum_coverage_met:
        return "NOT_EVALUABLE"
    if failure_count == 0:
        return "PASS"
    if fallback_failure_count == 0:
        return "PASS_WITH_FALLBACK"
    return "FAILED"
