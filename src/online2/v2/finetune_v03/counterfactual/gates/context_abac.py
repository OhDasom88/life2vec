"""Gate 1 — Context ABAC / editability prefilter."""

from __future__ import annotations
from typing import Any, Dict, Mapping, Optional

EDITABLE_PATH_A_TYPES = {"OBSERVED_STATE"}
EDITABLE_PATH_B_TYPES = {"ACTUATOR_STATE", "ACTUATOR_COMMAND"}
BLOCKED_TYPES = {"DERIVED_STATE", "IMMUTABLE_CONTEXT", "NON_INVERTIBLE", "UNKNOWN_SEMANTICS"}


def check_path_a(feature: str, semantics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    spec = semantics.get(feature) or {}
    ftype = str(spec.get("feature_type", "UNKNOWN_SEMANTICS"))
    reasons = []
    ok = True
    if ftype in BLOCKED_TYPES or ftype == "DERIVED_STATE":
        ok = False
        reasons.append("ERR_DERIVED_OR_IMMUTABLE")
    if not bool(spec.get("editable_path_a", False)):
        ok = False
        reasons.append("ERR_NOT_EDITABLE_PATH_A")
    if ftype not in EDITABLE_PATH_A_TYPES and ok:
        ok = False
        reasons.append("ERR_FEATURE_TYPE")
    return {
        "gate": "gate1_context_abac",
        "status": "PASSED" if ok else "REJECTED",
        "reason_codes": reasons,
        "details": {"feature": feature, "feature_type": ftype},
    }


def check_path_b(
    feature: str,
    semantics: Mapping[str, Mapping[str, Any]],
    *,
    whitelist: Optional[set] = None,
) -> Dict[str, Any]:
    spec = semantics.get(feature) or {}
    ftype = str(spec.get("feature_type", "UNKNOWN_SEMANTICS"))
    reasons = []
    ok = True
    if ftype in BLOCKED_TYPES:
        ok = False
        reasons.append("ERR_BLOCKED_TYPE")
    if not bool(spec.get("editable_path_b", False)):
        ok = False
        reasons.append("ERR_NOT_EDITABLE_PATH_B")
    if ftype not in EDITABLE_PATH_B_TYPES:
        ok = False
        reasons.append("ERR_FEATURE_TYPE")
    if whitelist is not None and feature not in whitelist:
        ok = False
        reasons.append("ERR_NOT_IN_WHITELIST")
    return {
        "gate": "gate1_context_abac",
        "status": "PASSED" if ok else "REJECTED",
        "reason_codes": reasons,
        "details": {"feature": feature, "feature_type": ftype},
    }
