from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict

class GateResult(TypedDict, total=False):
    gate: str
    status: str  # PASSED | PARTIAL | REJECTED
    reason_codes: List[str]
    details: Dict[str, Any]
