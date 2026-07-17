from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict

class PathACandidate(TypedDict, total=False):
    locus_id: str
    feature: str
    observed_raw: float
    target_raw: float
    policy: str
    direction_hint: Dict[str, Any]
    from_bin: Dict[str, Any]
    to_bin: Dict[str, Any]

class PathBCandidate(TypedDict, total=False):
    span_id: str
    feature: str
    operation: str
    cut_events: int
    from_hours: float
    to_hours: float
    intensity_policy: str
    capacity_status: str
