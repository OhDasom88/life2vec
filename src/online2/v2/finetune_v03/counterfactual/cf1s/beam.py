"""Single-event screening and beam search for CF-1S multi-event composition."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..candidates.cf1s_composer import multi_event_candidate_id


EffectFn = Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]


def _gap_seconds(a: Mapping[str, Any], b: Mapping[str, Any]) -> Optional[float]:
    ta = a.get("event_time_epoch")
    tb = b.get("event_time_epoch")
    if ta is None or tb is None:
        return None
    return abs(float(ta) - float(tb))


def pair_temporal_ok(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    mode: str,
    min_sparse_seconds: float,
    max_contiguous_gap_seconds: float,
) -> bool:
    if str(a.get("event_id")) == str(b.get("event_id")):
        return False
    if str(a.get("same_time_group_id") or a.get("event_time")) == str(
        b.get("same_time_group_id") or b.get("event_time")
    ):
        return False
    if str(a.get("event_time")) == str(b.get("event_time")):
        return False
    gap = _gap_seconds(a, b)
    if gap is None:
        # Epoch missing: reject for temporal modes (NOT silent allow).
        return False
    if mode == "CONTIGUOUS":
        return gap <= float(max_contiguous_gap_seconds)
    if mode == "SPARSE":
        return gap >= float(min_sparse_seconds)
    return True


def screen_single_event_atomics(
    atomics: Sequence[Mapping[str, Any]],
    *,
    effect_fn: EffectFn,
    single_budget: int = 20,
) -> Dict[str, Any]:
    """Evaluate all atomic directions, keep top-|Δrisk| within budget."""
    scored: List[Dict[str, Any]] = []
    for atomic in atomics:
        eff = dict(effect_fn([atomic]))
        delta = float(eff.get("delta_risk_search_aggregate", eff.get("delta_r_search", 0.0)))
        row = dict(atomic)
        row["delta_risk_search_aggregate"] = delta
        row["abs_delta_risk_search"] = abs(delta)
        row["effect_search_by_fold"] = list(
            eff.get("delta_risk_search_by_fold") or [delta]
        )
        row["single_screened"] = True
        scored.append(row)
    scored.sort(
        key=lambda r: (
            -float(r["abs_delta_risk_search"]),
            str(r.get("atomic_candidate_id") or ""),
        )
    )
    kept = scored[: int(single_budget)]
    return {
        "n_single_screened": len(scored),
        "n_single_kept": len(kept),
        "singles": kept,
        "all_scored_singles": scored,
    }


def _bundle_direction_pattern(edits: Sequence[Mapping[str, Any]]) -> str:
    deltas = []
    for e in edits:
        try:
            deltas.append(float(e["target_raw"]) - float(e["observed_raw"]))
        except (TypeError, ValueError, KeyError):
            deltas.append(0.0)
    if all(d > 0 for d in deltas):
        return "COHERENT_UP"
    if all(d < 0 for d in deltas):
        return "COHERENT_DOWN"
    return "MIXED_DIRECTION"


def compose_with_beam(
    *,
    case_id: str,
    screened_singles: Sequence[Mapping[str, Any]],
    effect_fn: EffectFn,
    beam_width: int = 8,
    two_event_max: int = 32,
    three_event_max: int = 24,
    total_max: int = 100,
    min_sparse_seconds: float = 3600.0,
    max_contiguous_gap_seconds: float = 1800.0,
) -> Dict[str, Any]:
    """Beam: top singles → 2-event → top-B → 3-event with global budget caps."""
    singles = list(screened_singles)
    # Count singles against total realization budget.
    selected: List[Dict[str, Any]] = []
    for s in singles:
        if len(selected) >= int(total_max):
            break
        cid = str(s.get("atomic_candidate_id") or multi_event_candidate_id(case_id=case_id, edits=[s]))
        selected.append(
            {
                "multi_event_candidate_id": cid,
                "sequence_edit_candidate_id": cid,
                "case_id": case_id,
                "feature_id": s.get("feature_id"),
                "n_events": 1,
                "bundle_mode": "SINGLE",
                "direction_pattern": _bundle_direction_pattern([s]),
                "edits": [dict(s)],
                "atomic_parent_candidate_ids": [cid],
                "delta_risk_search_aggregate": float(s.get("delta_risk_search_aggregate") or 0.0),
                "abs_delta_risk_search": float(s.get("abs_delta_risk_search") or 0.0),
                "delta_risk_search_by_fold": list(s.get("effect_search_by_fold") or []),
            }
        )

    # Group screened singles by feature for same-feature bundles.
    by_feature: Dict[str, List[Dict[str, Any]]] = {}
    for s in singles:
        by_feature.setdefault(str(s.get("feature_id")), []).append(dict(s))

    two_event: List[Dict[str, Any]] = []
    for feature_id, members in by_feature.items():
        profiles = {str(m.get("feature_edit_profile_id")) for m in members}
        if len(profiles) != 1:
            continue
        for mode in ("CONTIGUOUS", "SPARSE"):
            for a, b in combinations(members, 2):
                if not pair_temporal_ok(
                    a,
                    b,
                    mode=mode,
                    min_sparse_seconds=min_sparse_seconds,
                    max_contiguous_gap_seconds=max_contiguous_gap_seconds,
                ):
                    continue
                edits = [dict(a), dict(b)]
                eff = dict(effect_fn(edits))
                delta = float(
                    eff.get("delta_risk_search_aggregate", eff.get("delta_r_search", 0.0))
                )
                cid = multi_event_candidate_id(case_id=case_id, edits=edits)
                two_event.append(
                    {
                        "multi_event_candidate_id": cid,
                        "sequence_edit_candidate_id": cid,
                        "case_id": case_id,
                        "feature_id": feature_id,
                        "n_events": 2,
                        "bundle_mode": mode,
                        "direction_pattern": _bundle_direction_pattern(edits),
                        "edits": edits,
                        "atomic_parent_candidate_ids": [
                            str(a.get("atomic_candidate_id") or ""),
                            str(b.get("atomic_candidate_id") or ""),
                        ],
                        "delta_risk_search_aggregate": delta,
                        "abs_delta_risk_search": abs(delta),
                        "delta_risk_search_by_fold": list(
                            eff.get("delta_risk_search_by_fold") or [delta]
                        ),
                    }
                )
    two_event.sort(
        key=lambda r: (-float(r["abs_delta_risk_search"]), r["multi_event_candidate_id"])
    )
    # Global two-event cap (not per mode).
    two_kept = two_event[: int(two_event_max)]
    beam = two_kept[: int(beam_width)]

    remaining = int(total_max) - len(selected)
    for b in two_kept[: max(0, remaining)]:
        selected.append(b)

    three_event: List[Dict[str, Any]] = []
    # Expand from beam pairs + one more screened single of same feature.
    for pair in beam:
        feature_id = str(pair.get("feature_id"))
        members = by_feature.get(feature_id) or []
        pair_ids = {str(e.get("event_id")) for e in pair.get("edits") or []}
        for mode in ("CONTIGUOUS", "SPARSE"):
            for c in members:
                if str(c.get("event_id")) in pair_ids:
                    continue
                edits = list(pair.get("edits") or []) + [dict(c)]
                ok = True
                for x, y in combinations(edits, 2):
                    if not pair_temporal_ok(
                        x,
                        y,
                        mode=mode,
                        min_sparse_seconds=min_sparse_seconds,
                        max_contiguous_gap_seconds=max_contiguous_gap_seconds,
                    ):
                        ok = False
                        break
                if not ok:
                    continue
                eff = dict(effect_fn(edits))
                delta = float(
                    eff.get("delta_risk_search_aggregate", eff.get("delta_r_search", 0.0))
                )
                cid = multi_event_candidate_id(case_id=case_id, edits=edits)
                three_event.append(
                    {
                        "multi_event_candidate_id": cid,
                        "sequence_edit_candidate_id": cid,
                        "case_id": case_id,
                        "feature_id": feature_id,
                        "n_events": 3,
                        "bundle_mode": mode,
                        "direction_pattern": _bundle_direction_pattern(edits),
                        "edits": edits,
                        "atomic_parent_candidate_ids": [
                            str(e.get("atomic_candidate_id") or "") for e in edits
                        ],
                        "delta_risk_search_aggregate": delta,
                        "abs_delta_risk_search": abs(delta),
                        "delta_risk_search_by_fold": list(
                            eff.get("delta_risk_search_by_fold") or [delta]
                        ),
                        "pairwise_parent_candidate_ids": [
                            pair["multi_event_candidate_id"]
                        ],
                    }
                )
    three_event.sort(
        key=lambda r: (-float(r["abs_delta_risk_search"]), r["multi_event_candidate_id"])
    )
    three_kept = three_event[: int(three_event_max)]
    remaining = int(total_max) - len(selected)
    for b in three_kept[: max(0, remaining)]:
        selected.append(b)

    # Dedup by candidate id preserving order.
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in selected:
        dedup.setdefault(row["multi_event_candidate_id"], row)
    final = list(dedup.values())[: int(total_max)]

    return {
        "bundles": final,
        "n_single_screened": len(singles),
        "n_two_event_raw": len(two_event),
        "n_two_event_kept": len(two_kept),
        "n_three_event_raw": len(three_event),
        "n_three_event_kept": len(three_kept),
        "n_total_selected": len(final),
        "beam_width": int(beam_width),
        "two_event_max": int(two_event_max),
        "three_event_max": int(three_event_max),
        "total_max": int(total_max),
        "budget_ok": len(final) <= int(total_max)
        and len(two_kept) <= int(two_event_max)
        and len(three_kept) <= int(three_event_max),
        "cross_profile_bundle_count": 0,
    }
