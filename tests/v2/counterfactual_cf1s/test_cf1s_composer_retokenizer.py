from __future__ import annotations

from src.online2.v2.finetune_v03.counterfactual.candidates.cf1s_composer import (
    compose_atomic_edits_for_locus,
    compose_multi_event_bundles,
    multi_event_candidate_id,
)
from src.online2.v2.finetune_v03.counterfactual.retokenization.multi_event_retokenizer import (
    apply_multi_event_edits_atomically,
    classify_parity_status,
    parity_sample_rule,
)


def _ok(_):
    return {"ok": True, "gate0_pass": True, "gate4_pass": True, "raw_grounding_status": "EXACT"}


def test_candidate_id_order_independent():
    edits_a = [
        {"event_time": "t1", "event_id": "e1", "feature_id": "f", "target_raw": 1, "mg_id": "m1"},
        {"event_time": "t2", "event_id": "e2", "feature_id": "f", "target_raw": 0, "mg_id": "m2"},
    ]
    edits_b = list(reversed(edits_a))
    assert multi_event_candidate_id(case_id="C", edits=edits_a) == multi_event_candidate_id(
        case_id="C", edits=edits_b
    )


def test_multi_event_requires_distinct_timestamp():
    loci = [
        {
            "locus_key": "k1",
            "event_id": "e1",
            "event_time": "t0",
            "event_time_epoch": 0,
            "feature_id": "circulation_fan",
            "feature_edit_profile_id": "p1",
            "mg_id": "m1",
            "observed_raw": 0.0,
        },
        {
            "locus_key": "k2",
            "event_id": "e2",
            "event_time": "t0",  # same timestamp
            "event_time_epoch": 0,
            "feature_id": "circulation_fan",
            "feature_edit_profile_id": "p1",
            "mg_id": "m2",
            "observed_raw": 1.0,
        },
        {
            "locus_key": "k3",
            "event_id": "e3",
            "event_time": "t1",
            "event_time_epoch": 7200,
            "feature_id": "circulation_fan",
            "feature_edit_profile_id": "p1",
            "mg_id": "m3",
            "observed_raw": 0.0,
        },
    ]
    atomic = {
        "k1": [{"event_id": "e1", "event_time": "t0", "event_time_epoch": 0, "feature_id": "circulation_fan", "mg_id": "m1", "observed_raw": 0.0, "target_raw": 1.0, "atomic_candidate_id": "a1", "feature_edit_profile_id": "p1"}],
        "k2": [{"event_id": "e2", "event_time": "t0", "event_time_epoch": 0, "feature_id": "circulation_fan", "mg_id": "m2", "observed_raw": 1.0, "target_raw": 0.0, "atomic_candidate_id": "a2", "feature_edit_profile_id": "p1"}],
        "k3": [{"event_id": "e3", "event_time": "t1", "event_time_epoch": 7200, "feature_id": "circulation_fan", "mg_id": "m3", "observed_raw": 0.0, "target_raw": 1.0, "atomic_candidate_id": "a3", "feature_edit_profile_id": "p1"}],
    }
    out = compose_multi_event_bundles(
        case_id="C",
        selected_loci=loci,
        atomic_by_locus_key=atomic,
        min_sparse_seconds=1,
        max_contiguous_gap_seconds=10_000,
    )
    for b in out["bundles"]:
        times = [e["event_time"] for e in b["edits"]]
        assert len(times) == len(set(times))


def test_atomic_failure_rejects_composite():
    edits = [
        {"event_id": "e1", "original_token_length": 2, "target_raw": 1},
        {"event_id": "e2", "original_token_length": 2, "target_raw": 0},
    ]

    def ground(edit):
        if edit["event_id"] == "e2":
            return {"ok": False}
        return {"ok": True, "raw_grounding_status": "EXACT"}

    def retok(edit):
        return {
            "original_token_length": 2,
            "retokenized_token_length": 2,
            "tokens": ["a", "b"],
        }

    res = apply_multi_event_edits_atomically(
        edits=edits,
        ground_fn=ground,
        invert_fn=_ok,
        gate0_fn=_ok,
        gate4_fn=_ok,
        retokenize_fn=retok,
    )
    assert res.ok is False
    assert res.grounding_failure_count == 1
    assert res.failure_reason == "GROUNDING_FAILED"


def test_length_change_rejected():
    edits = [{"event_id": "e1", "original_token_length": 2, "target_raw": 1}]

    def retok(_):
        return {
            "original_token_length": 2,
            "retokenized_token_length": 3,
            "tokens": ["a", "b", "c"],
        }

    res = apply_multi_event_edits_atomically(
        edits=edits,
        ground_fn=_ok,
        invert_fn=_ok,
        gate0_fn=_ok,
        gate4_fn=_ok,
        retokenize_fn=retok,
        length_preserving_edit_only=True,
    )
    assert res.ok is False
    assert res.failure_reason == "LENGTH_CHANGE_REJECTED"


def test_parity_status_and_sample_rule():
    assert classify_parity_status(
        failure_count=0, fallback_failure_count=0, minimum_coverage_met=True
    ) == "PASS"
    assert classify_parity_status(
        failure_count=1, fallback_failure_count=0, minimum_coverage_met=True
    ) == "PASS_WITH_FALLBACK"
    assert classify_parity_status(
        failure_count=1, fallback_failure_count=1, minimum_coverage_met=True
    ) == "FAILED"
    assert classify_parity_status(
        failure_count=0, fallback_failure_count=0, minimum_coverage_met=False
    ) == "NOT_EVALUABLE"
    # Deterministic boolean
    assert isinstance(parity_sample_rule("abc123"), bool)


def test_atomic_edits_not_from_saliency_sign():
    locus = {
        "event_id": "e",
        "event_time": "t",
        "feature_id": "f",
        "mg_id": "m",
        "observed_raw": 1.0,
        "edit_class": "EDITABLE_CONTROL_LIKE",
    }
    atomics = compose_atomic_edits_for_locus(locus, adjacency_targets=[0.0])
    assert atomics[0]["direction_inferred_from_saliency"] is False
