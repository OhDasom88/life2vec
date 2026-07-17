"""Path A content candidates + adjacent perturbation direction probe helpers."""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..grounding.raw_target import decode_interval, nearest_feasible_interior, adjacent_bin_targets
from ..evaluation.bank_integrity import as_sequence_list


def build_path_a_candidates(
    *,
    feature: str,
    observed_raw: float,
    edges: Sequence[float],
    direction_hint: Optional[Dict[str, Any]] = None,
    safe_interval: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    targets = adjacent_bin_targets(observed_raw, edges)
    out = []
    for t in targets:
        raw = nearest_feasible_interior(
            observed_raw,
            t["interval"],
            safe_interval=safe_interval,
        )
        out.append(
            {
                "feature": feature,
                "observed_raw": float(observed_raw),
                "target_raw": float(raw),
                "policy": "nearest_feasible_interior",
                "direction_hint": direction_hint or {"status": "proposal_only"},
                "from_bin": t.get("from_bin"),
                "to_bin": t.get("to_bin"),
                "probe_side": t.get("side"),
                "source": "adjacent_bin",
                "is_noop": False,
            }
        )
    return out


def candidates_from_mlm_bundles(
    *,
    feature: str,
    observed_raw: float,
    edges: Sequence[float],
    bundles: Sequence[Mapping[str, Any]],
    safe_interval: Optional[Tuple[float, float]] = None,
    margin_ratio: float = 0.001,
    edges_global: Optional[Sequence[float]] = None,
    edges_farm: Optional[Sequence[float]] = None,
    global_ref_value: Optional[float] = None,
    farm_ref_value: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Map MLM bundles → raw via interval intersection + interior margin; reject non-invertible."""
    from .mlm_raw_inversion import invert_mlm_bundle_to_raw

    out = []
    for b in bundles:
        if isinstance(b, Mapping):
            toks = as_sequence_list(b.get("tokens"))
            score = b.get("score")
            is_orig = bool(
                b.get("is_original")
                or str(b.get("source") or "").startswith("noop")
            )
        else:
            toks = list(getattr(b, "tokens", []) or [])
            score = getattr(b, "score", None)
            is_orig = bool(
                getattr(b, "is_original", False)
                or str(getattr(b, "source", "") or "").startswith("noop")
            )
        if is_orig:
            out.append(
                {
                    "feature": feature,
                    "observed_raw": float(observed_raw),
                    "target_raw": float(observed_raw),
                    "policy": "mlm_original_noop",
                    "proposed_token_bundle": toks,
                    "source": "noop",
                    "is_noop": True,
                    "mlm_score": score,
                }
            )
            continue
        inv = invert_mlm_bundle_to_raw(
            feature=feature,
            observed_raw=observed_raw,
            mlm_proposed_bundle=toks,
            edges_abs=edges,
            edges_global=edges_global,
            edges_farm=edges_farm,
            global_ref_value=global_ref_value,
            farm_ref_value=farm_ref_value,
            margin_ratio=margin_ratio,
            safe_interval=safe_interval,
        )
        if not inv.get("ok"):
            out.append(
                {
                    "feature": feature,
                    "observed_raw": float(observed_raw),
                    "target_raw": None,
                    "policy": "mlm_inversion_rejected",
                    "proposed_token_bundle": toks,
                    "source": "constrained_mlm",
                    "is_noop": False,
                    "mlm_score": score,
                    "inversion_ok": False,
                    "inversion_reason_code": inv.get("reason_code"),
                    "rejected": True,
                }
            )
            continue
        out.append(
            {
                "feature": feature,
                "observed_raw": float(observed_raw),
                "target_raw": float(inv["selected_target_raw"]),
                "policy": "mlm_interval_intersection_interior",
                "proposed_token_bundle": toks,
                "actual_retokenized_bundle": inv.get("actual_retokenized_bundle"),
                "raw_interval_intersection": inv.get("raw_interval_intersection"),
                "bundle_match_status": inv.get("bundle_match_status"),
                "source": "constrained_mlm",
                "is_noop": False,
                "mlm_score": score,
                "inversion_ok": True,
            }
        )
    return out


def choose_direction_hint(delta_r_lower: float, delta_r_upper: float) -> Dict[str, Any]:
    if delta_r_lower < delta_r_upper:
        value = "decrease"
    elif delta_r_upper < delta_r_lower:
        value = "increase"
    else:
        value = "tie"
    return {
        "value": value,
        "source": "adjacent_raw_perturbation",
        "status": "proposal_only",
        "delta_r_lower": float(delta_r_lower),
        "delta_r_upper": float(delta_r_upper),
    }
