"""Gate 4 — retokenize and compare requested token direction."""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def compare_token_bundles(
    proposed_to_tokens: Sequence[str],
    retokenized_tokens: Sequence[str],
    *,
    require_subset: bool = True,
) -> Dict[str, Any]:
    prop = [str(t) for t in proposed_to_tokens]
    got = [str(t) for t in retokenized_tokens]
    if require_subset:
        ok = all(t in got for t in prop) if prop else True
    else:
        ok = prop == got
    return {
        "gate": "gate4_local_rules",
        "status": "PASSED" if ok else "REJECTED",
        "reason_codes": [] if ok else ["ERR_TOKEN_BUNDLE_CONFLICT"],
        "details": {"proposed": prop, "retokenized": got},
    }


def build_token_edits(
    *,
    event_id: str,
    measurement_group_id: str,
    feature: str,
    from_tokens: Sequence[str],
    to_tokens: Sequence[str],
    gate4_passed: bool,
    derived_from_edit_scope: bool = False,
) -> Optional[Dict[str, Any]]:
    if not gate4_passed:
        return None
    return {
        "event_id": event_id,
        "measurement_group_id": measurement_group_id,
        "feature": feature,
        "from_tokens": list(from_tokens),
        "to_tokens": list(to_tokens),
        "derived_from_edit_scope": derived_from_edit_scope,
    }
