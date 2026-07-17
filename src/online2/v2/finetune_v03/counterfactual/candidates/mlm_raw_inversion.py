"""MLM token-bundle → raw inversion with interval intersection + interior margin."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..grounding.raw_target import decode_interval, value_to_bin_index
from ..grounding.retokenize import retokenize_abs_value
from ..gates.local_rules import compare_token_bundles


def _parse_abs_bin(token: str) -> Optional[int]:
    s = str(token)
    if "ABS_B" not in s:
        return None
    try:
        return int(s.split("ABS_B")[-1])
    except ValueError:
        return None


def _parse_rel_bin(token: str, prefix: str) -> Optional[int]:
    s = str(token)
    marker = f"{prefix}_B"
    if marker not in s:
        return None
    try:
        return int(s.split(marker)[-1])
    except ValueError:
        return None


def interval_intersection(
    intervals: Sequence[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    if not intervals:
        return None
    lo = max(float(a) for a, _ in intervals)
    hi = min(float(b) for _, b in intervals)
    if lo > hi:
        return None
    return (lo, hi)


def interior_target(
    current_raw: float,
    interval: Tuple[float, float],
    *,
    margin_ratio: float = 0.001,
    safe_interval: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[float], Optional[str]]:
    lo, hi = float(interval[0]), float(interval[1])
    if safe_interval is not None:
        lo = max(lo, float(safe_interval[0]))
        hi = min(hi, float(safe_interval[1]))
    width = hi - lo
    if width <= 0:
        return None, "EMPTY_INTERVAL_INTERSECTION"
    margin = abs(width) * float(margin_ratio)
    ilo, ihi = lo + margin, hi - margin
    if ilo > ihi:
        return None, "EMPTY_SAFE_INTERIOR"
    v = float(current_raw)
    if v < ilo:
        return float(ilo), None
    if v > ihi:
        return float(ihi), None
    return v, None


def invert_mlm_bundle_to_raw(
    *,
    feature: str,
    observed_raw: float,
    mlm_proposed_bundle: Sequence[str],
    edges_abs: Sequence[float],
    edges_global: Optional[Sequence[float]] = None,
    edges_farm: Optional[Sequence[float]] = None,
    global_ref_value: Optional[float] = None,
    farm_ref_value: Optional[float] = None,
    margin_ratio: float = 0.001,
    safe_interval: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Map MLM ABS/GLOBAL/FARM token bins to a feasible raw via intersection + interior.

    Fails with NON_INVERTIBLE_BUNDLE / EMPTY_INTERVAL_INTERSECTION /
    EMPTY_SAFE_INTERIOR / PHYSICAL_BOUND_VIOLATION / RETOKENIZATION_MISMATCH.
    """
    bundle = [str(t) for t in mlm_proposed_bundle]
    intervals: List[Tuple[float, float]] = []
    abs_tok = next((t for t in bundle if t.startswith("VALUE_ABS|")), None)
    if abs_tok is None:
        return {
            "ok": False,
            "reason_code": "NON_INVERTIBLE_BUNDLE",
            "mlm_proposed_bundle": bundle,
            "selected_target_raw": None,
        }
    abs_bin = _parse_abs_bin(abs_tok)
    if abs_bin is None:
        return {
            "ok": False,
            "reason_code": "NON_INVERTIBLE_BUNDLE",
            "mlm_proposed_bundle": bundle,
        }
    intervals.append(decode_interval(edges_abs, abs_bin))

    # GLOBAL / FARM residuals map back to absolute via frozen refs when available
    g_tok = next((t for t in bundle if "GLOBAL" in t and "_B" in t), None)
    if g_tok is not None and edges_global is not None and global_ref_value is not None:
        gb = _parse_rel_bin(g_tok, "GLOBAL")
        if gb is not None:
            glo, ghi = decode_interval(edges_global, gb)
            intervals.append((float(global_ref_value) + glo, float(global_ref_value) + ghi))
    f_tok = next((t for t in bundle if "FARM" in t and "_B" in t), None)
    if f_tok is not None and edges_farm is not None and farm_ref_value is not None:
        fb = _parse_rel_bin(f_tok, "FARM")
        if fb is not None:
            flo, fhi = decode_interval(edges_farm, fb)
            intervals.append((float(farm_ref_value) + flo, float(farm_ref_value) + fhi))

    inter = interval_intersection(intervals)
    if inter is None:
        return {
            "ok": False,
            "reason_code": "EMPTY_INTERVAL_INTERSECTION",
            "mlm_proposed_bundle": bundle,
            "raw_interval_intersection": None,
        }
    target, err = interior_target(
        observed_raw, inter, margin_ratio=margin_ratio, safe_interval=safe_interval
    )
    if err:
        return {
            "ok": False,
            "reason_code": err,
            "mlm_proposed_bundle": bundle,
            "raw_interval_intersection": list(inter),
        }
    if safe_interval is not None:
        if not (safe_interval[0] <= float(target) <= safe_interval[1]):
            return {
                "ok": False,
                "reason_code": "PHYSICAL_BOUND_VIOLATION",
                "mlm_proposed_bundle": bundle,
                "raw_interval_intersection": list(inter),
                "selected_target_raw": target,
            }

    # Round-trip ABS (and full bundle when refs provided) must match proposed ABS at minimum
    actual_abs = retokenize_abs_value(feature, float(target), edges_abs)
    gate = compare_token_bundles(
        [t for t in bundle if t.startswith("VALUE_ABS|") or t.startswith("FEATURE|")],
        [t for t in actual_abs if t.startswith("VALUE_ABS|") or t.startswith("FEATURE|")],
        require_subset=False,
    )
    if gate["status"] != "PASSED":
        return {
            "ok": False,
            "reason_code": "RETOKENIZATION_MISMATCH",
            "mlm_proposed_bundle": bundle,
            "raw_interval_intersection": list(inter),
            "selected_target_raw": target,
            "actual_retokenized_bundle": actual_abs,
            "bundle_match_status": "MISMATCH",
        }
    return {
        "ok": True,
        "reason_code": None,
        "mlm_proposed_bundle": bundle,
        "raw_interval_intersection": list(inter),
        "selected_target_raw": float(target),
        "actual_retokenized_bundle": actual_abs,
        "bundle_match_status": "EXACT",
    }
