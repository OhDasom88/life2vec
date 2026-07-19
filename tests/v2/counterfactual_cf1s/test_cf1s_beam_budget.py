from __future__ import annotations

from src.online2.v2.finetune_v03.counterfactual.cf1s.beam import (
    compose_with_beam,
    pair_temporal_ok,
    screen_single_event_atomics,
)


def _effect(edits):
    # Larger |delta| for more edits to make beam prefer multi-event.
    mag = 0.01 * len(edits)
    return {
        "delta_risk_search_aggregate": mag,
        "delta_r_search": mag,
        "delta_risk_search_by_fold": [mag, mag],
        "model_effects_evaluated": True,
    }


def test_pair_temporal_requires_epoch():
    a = {"event_id": "e1", "event_time": "t1", "same_time_group_id": "g1"}
    b = {"event_id": "e2", "event_time": "t2", "same_time_group_id": "g2"}
    assert pair_temporal_ok(
        a, b, mode="SPARSE", min_sparse_seconds=3600, max_contiguous_gap_seconds=1800
    ) is False
    a["event_time_epoch"] = 0
    b["event_time_epoch"] = 7200
    assert pair_temporal_ok(
        a, b, mode="SPARSE", min_sparse_seconds=3600, max_contiguous_gap_seconds=1800
    )
    assert pair_temporal_ok(
        a, b, mode="CONTIGUOUS", min_sparse_seconds=3600, max_contiguous_gap_seconds=1800
    ) is False


def test_beam_enforces_global_two_three_and_total_caps():
    singles = []
    for i in range(12):
        singles.append(
            {
                "atomic_candidate_id": f"a{i}",
                "event_id": f"e{i}",
                "event_time": f"t{i}",
                "event_time_epoch": float(i * 3600),
                "same_time_group_id": f"t{i}",
                "feature_id": "circulation_fan",
                "feature_edit_profile_id": "p1",
                "mg_id": f"m{i}",
                "observed_raw": 0.0,
                "target_raw": 1.0,
                "delta_risk_search_aggregate": 0.02 - 0.001 * i,
                "abs_delta_risk_search": abs(0.02 - 0.001 * i),
                "effect_search_by_fold": [0.02],
            }
        )
    screened = screen_single_event_atomics(singles, effect_fn=_effect, single_budget=20)
    assert screened["n_single_screened"] == 12
    out = compose_with_beam(
        case_id="C",
        screened_singles=screened["singles"],
        effect_fn=_effect,
        beam_width=8,
        two_event_max=32,
        three_event_max=24,
        total_max=100,
        min_sparse_seconds=3600,
        max_contiguous_gap_seconds=1800,
    )
    assert out["n_single_screened"] > 0
    assert out["n_two_event_kept"] <= 32
    assert out["n_three_event_kept"] <= 24
    assert out["n_total_selected"] <= 100
    assert out["budget_ok"] is True
    # Both directions of atomics are not collapsed here; beam consumes screened singles.
    n_two = sum(1 for b in out["bundles"] if b["n_events"] == 2)
    n_three = sum(1 for b in out["bundles"] if b["n_events"] == 3)
    assert n_two <= 32
    assert n_three <= 24
