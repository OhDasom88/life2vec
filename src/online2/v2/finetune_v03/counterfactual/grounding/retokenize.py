"""Full measurement bundle retokenization helpers (ABS channel)."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

from .raw_target import value_to_bin_index


def retokenize_abs_value(feature: str, value: float, edges: Sequence[float]) -> List[str]:
    k = value_to_bin_index(value, edges)
    return [f"FEATURE|{feature}", f"VALUE_ABS|ABS_B{k:02d}"]


def retokenize_actuator_literal(feature: str, on: bool) -> List[str]:
    lit = "POSITIVE" if on else "ZERO"
    return [f"FEATURE|{feature}", f"OBSERVED_VALUE|{lit}"]
