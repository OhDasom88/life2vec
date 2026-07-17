from __future__ import annotations
from typing import Any, Dict, List, Mapping, Sequence

def eligibility_for_path(path: str, *, gate2_status: str = "PASSED") -> str:
    if path == "A":
        return "EXPERT_REVIEW_REQUIRED" if gate2_status == "PARTIAL" else "EXPLANATORY_ONLY"
    # Path B B0
    if gate2_status == "REJECTED":
        return "BLOCKED"
    return "EXPERT_REVIEW_REQUIRED"


def assert_no_operational(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for r in rows if r.get("operational_eligibility") == "OPERATIONAL_CANDIDATE")
