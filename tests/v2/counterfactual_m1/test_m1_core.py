
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.online2.v2.finetune_v03.counterfactual.attribution.aggregation import (
    aggregate_events,
    aggregate_measurement_groups,
    aggregate_spans,
    infer_token_role,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.path_b_b0_generator import (
    generate_b0_operations,
    select_best_operation,
)
from src.online2.v2.finetune_v03.counterfactual.gates.causal_consistency import (
    calculate_vpd,
    recompute_derived,
)
from src.online2.v2.finetune_v03.counterfactual.gates.context_abac import check_path_a, check_path_b
from src.online2.v2.finetune_v03.counterfactual.gates.local_rules import compare_token_bundles
from src.online2.v2.finetune_v03.counterfactual.gates.safety_bounds import project_to_safe
from src.online2.v2.finetune_v03.counterfactual.grounding.raw_target import (
    adjacent_bin_targets,
    nearest_feasible_interior,
    value_to_bin_index,
)
from src.online2.v2.finetune_v03.counterfactual.temporal.span_builder import build_zoh_spans


def test_mg_aggregation_avoids_token_count_bias():
    rows = []
    # MG1: 1 strong abs token
    rows.append(dict(case_id="c", fold_id=0, event_id="e", measurement_group_id="m1", feature="t",
                     token_string="VALUE_ABS|ABS_B01", token_role="abs_value",
                     normalized_signed_attribution=1.0, normalized_absolute_attribution=1.0))
    # MG2: many weak meta tokens
    for i in range(10):
        rows.append(dict(case_id="c", fold_id=0, event_id="e", measurement_group_id="m2", feature="h",
                         token_string="UNIT|C", token_role="unit",
                         normalized_signed_attribution=1.0, normalized_absolute_attribution=1.0))
    mg = aggregate_measurement_groups(pd.DataFrame(rows))
    s1 = float(mg.loc[mg.measurement_group_id == "m1", "signed_attribution"].iloc[0])
    s2 = float(mg.loc[mg.measurement_group_id == "m2", "signed_attribution"].iloc[0])
    assert s1 == pytest.approx(1.0)
    assert s2 == pytest.approx(0.0)  # unit weight 0


def test_event_aggregation():
    mg = pd.DataFrame([
        {"case_id":"c","fold_id":0,"event_id":"e","measurement_group_id":"m1","feature":"t","signed_attribution":1.0,"value_token_count":1},
        {"case_id":"c","fold_id":0,"event_id":"e","measurement_group_id":"m2","feature":"h","signed_attribution":0.0,"value_token_count":1},
    ])
    ev = aggregate_events(mg)
    assert float(ev.iloc[0]["signed_attribution"]) == pytest.approx(0.5)


def test_span_duration_positive_ratio():
    ev = pd.DataFrame([
        {"case_id":"c","fold_id":0,"event_id":"e1","signed_attribution":1.0},
        {"case_id":"c","fold_id":0,"event_id":"e2","signed_attribution":-1.0},
        {"case_id":"c","fold_id":0,"event_id":"e3","signed_attribution":1.0},
    ])
    spans = pd.DataFrame([{
        "case_id":"c","fold_id":0,"span_id":"s1","feature":"fan",
        "source_event_ids":["e1","e2","e3"],
        "sampling_interval_min":60,
        "event_durations_min":[60,120,60],  # middle negative lasts longer
        "estimate_duration_min":240,
        "missing_gap_count":0,
    }])
    out = aggregate_spans(ev, spans)
    # positive duration = 60+60=120 / 240 = 0.5
    assert float(out.iloc[0]["duration_weighted_positive_ratio"]) == pytest.approx(0.5)


def test_zoh_gap_splits_span():
    df = pd.DataFrame([
        {"timestamp":"2025-01-01T10:00:00Z","raw_value":1,"event_id":"a"},
        {"timestamp":"2025-01-01T11:00:00Z","raw_value":1,"event_id":"b"},
        # gap > 1.5h
        {"timestamp":"2025-01-01T14:00:00Z","raw_value":1,"event_id":"c"},
    ])
    spans = build_zoh_spans(df, feature="circulation_fan", farm_id="F", zone_id="1", nominal_interval_min=60)
    assert len(spans) == 2


def test_gate1_blocks_derived():
    sem = {"vpd": {"feature_type":"DERIVED_STATE","editable_path_a":False,"editable_path_b":False}}
    assert check_path_a("vpd", sem)["status"] == "REJECTED"


def test_gate2_partial_unknown_operational():
    r = project_to_safe(25.0, hard=(0,40), agronomic=(10,35), operational=None, operational_known=False)
    assert r["status"] == "PARTIAL"


def test_gate3_vpd_recompute():
    state = {}
    out = recompute_derived(state, {"inside_temp_c": 25.0, "inside_humidity_pct": 60.0})
    assert out["status"] == "PASSED"
    assert "vpd" in state
    assert state["vpd"] == pytest.approx(calculate_vpd(25.0, 60.0))


def test_gate4_conflict():
    r = compare_token_bundles(["VALUE_ABS|ABS_B01"], ["VALUE_ABS|ABS_B02"])
    assert r["status"] == "REJECTED"


def test_nearest_feasible_interior():
    v = nearest_feasible_interior(30.0, (20.0, 22.0))
    assert 20.0 <= v <= 22.0


def test_b0_includes_noop_and_limits_ops():
    span = {"span_id":"s","feature":"circulation_fan","source_event_ids":["e1","e2","e3"],
            "estimate_duration_min":180,"sampling_interval_min":60,"n_events":3}
    attr = {"duration_weighted_positive_ratio":0.95,"start_mass_ratio":0.1,"end_mass_ratio":0.8,"signed":1.0}
    ops = generate_b0_operations(span, attr, clear_whitelist={"circulation_fan"})
    names = {o["operation"] for o in ops}
    assert "NO_OP" in names
    assert names <= {"NO_OP","truncate_start","truncate_end","clear_span"}


def test_tie_prefers_noop():
    best = select_best_operation([
        {"operation":"truncate_end","model_space_delta_r":-0.1},
        {"operation":"NO_OP","model_space_delta_r":-0.1},
    ])
    assert best["operation"] == "NO_OP"


def test_no_operational_candidate_eligibility():
    from src.online2.v2.finetune_v03.counterfactual.evaluation.metrics import eligibility_for_path
    assert eligibility_for_path("B", gate2_status="PARTIAL") == "EXPERT_REVIEW_REQUIRED"
    assert eligibility_for_path("A") == "EXPLANATORY_ONLY"


def test_span_event_alignment_window():
    from src.online2.v2.finetune_v03.counterfactual.temporal.span_event_align import (
        align_spans_to_events,
        alignment_summary,
    )

    emb = pd.DataFrame(
        [
            {"event_id": "a1", "event_index": 0, "timestamp": "2025-03-20T03:00:00Z", "view": "ACTUATOR", "zone": "1"},
            {"event_id": "a2", "event_index": 1, "timestamp": "2025-03-20T04:00:00Z", "view": "ACTUATOR", "zone": "1"},
            {"event_id": "a3", "event_index": 2, "timestamp": "2025-03-20T05:00:00Z", "view": "ACTUATOR", "zone": "1"},
        ]
    )
    emb["timestamp"] = pd.to_datetime(emb["timestamp"], utc=True)
    spans = pd.DataFrame(
        [
            {
                "span_id": "s1",
                "zone_id": "1",
                "start_time": "2025-03-20T03:00:00+00:00",
                "end_time": "2025-03-20T05:00:00+00:00",
                "n_events": 2,
                "source_event_ids": ["x"],
            }
        ]
    )
    out = align_spans_to_events(spans, emb)
    assert out.iloc[0]["alignment_status"] == "ALIGNED"
    assert set(out.iloc[0]["source_event_ids"]) == {"a1", "a2"}
    assert alignment_summary(out)["aligned_rate"] == 1.0


def test_perturbation_noop_clone():
    import torch
    from src.online2.v2.finetune_v03.counterfactual.evaluation.perturbation import (
        interpolate_event_indices,
        noop_batch,
        zero_event_indices,
    )

    b = {
        "event_mean": torch.ones(1, 3, 4),
        "event_max": torch.ones(1, 3, 4),
        "padding_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    n = noop_batch(b)
    assert torch.allclose(n["event_mean"], b["event_mean"])
    z = zero_event_indices(b, [1])
    assert float(z["event_mean"][0, 1].sum()) == 0.0
    assert float(z["event_mean"][0, 0].sum()) == 4.0
    amp = interpolate_event_indices(b, [0], alpha=0.5, toward="amplify")
    assert float(amp["event_mean"][0, 0, 0]) == pytest.approx(1.5)
