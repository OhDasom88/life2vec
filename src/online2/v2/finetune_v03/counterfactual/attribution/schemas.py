from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class TokenAttributionRow(TypedDict, total=False):
    case_id: str
    fold_id: int
    event_id: str
    measurement_group_id: str
    same_time_group_id: str
    token_index: int
    token_string: str
    token_role: str
    feature: str
    signed_attribution: float
    absolute_attribution: float
    normalized_signed_attribution: float
    normalized_absolute_attribution: float
    normalization_method: str
    attribution_method: str
