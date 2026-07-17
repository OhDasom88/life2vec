from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict

class InterventionLocus(TypedDict, total=False):
    case_id: str
    level: str  # event | mg | span
    path: str  # A | B
    event_ids: List[str]
    span_id: Optional[str]
    measurement_group_id: Optional[str]
    feature: str
    attribution: Dict[str, Any]
    content_hint: Dict[str, Any]
    raw_join_ok: bool
    gate1: Dict[str, Any]
