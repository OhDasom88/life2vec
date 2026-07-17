"""Path A bin decode and nearest feasible interior target."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def decode_interval(edges: Sequence[float], bin_index: int) -> Tuple[float, float]:
    edges = list(edges)
    if len(edges) < 2:
        raise ValueError("edges need >= 2")
    k = int(bin_index)
    k = max(0, min(k, len(edges) - 2))
    return float(edges[k]), float(edges[k + 1])


def value_to_bin_index(value: float, edges: Sequence[float]) -> int:
    edges = list(edges)
    if len(edges) < 2:
        return 0
    # same convention as BinRuleV2: bisect_left on edges[1:-1]
    import bisect
    return int(bisect.bisect_left(edges[1:-1], float(value)))


def nearest_feasible_interior(
    observed: float,
    target_interval: Tuple[float, float],
    *,
    safe_interval: Optional[Tuple[float, float]] = None,
    eps: float = 1e-6,
) -> float:
    lo, hi = float(target_interval[0]), float(target_interval[1])
    if safe_interval is not None:
        lo = max(lo, float(safe_interval[0]))
        hi = min(hi, float(safe_interval[1]))
    if lo > hi:
        # empty after safety — return clamp to safe mid if possible
        if safe_interval is not None:
            return float(0.5 * (safe_interval[0] + safe_interval[1]))
        return float(observed)
    if lo <= observed <= hi:
        return float(observed)
    if observed < lo:
        return float(lo + eps * max(1.0, abs(lo)))
    return float(hi - eps * max(1.0, abs(hi)))


def adjacent_bin_targets(observed: float, edges: Sequence[float]) -> List[Dict[str, Any]]:
    edges = list(edges)
    k = value_to_bin_index(observed, edges)
    out = []
    for side, kk in (("lower", k - 1), ("upper", k + 1)):
        if kk < 0 or kk > len(edges) - 2:
            continue
        lo, hi = decode_interval(edges, kk)
        out.append(
            {
                "side": side,
                "from_bin": {"channel": "ABS", "index": k, "lo": edges[k], "hi": edges[k + 1]},
                "to_bin": {"channel": "ABS", "index": kk, "lo": lo, "hi": hi},
                "interval": (lo, hi),
            }
        )
    return out
