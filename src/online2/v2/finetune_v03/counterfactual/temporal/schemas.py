from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict

class ActuatorSpan(TypedDict, total=False):
    span_id: str
    case_id: str
    farm_id: str
    zone_id: str
    feature: str
    start_time: str
    end_time: str
    estimate_duration_min: float
    lower_bound_min: float
    upper_bound_min: float
    sampling_interval_min: float
    missing_gap_count: int
    span_confidence: str
    source_event_ids: List[str]
    assumption: str
    state: str
