"""Family projection, temporal arms, identity, and legacy isolation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.core_candidate_family import (
    build_complete_families,
    project_parent_subset,
    validate_family_membership,
    verify_parent_matches_bundle_projection,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_contract import (
    assert_core_import_graph_clean,
    temporal_arm_for_gaps,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_identity import (
    check_noise_ceilings,
    evaluate_forced_identity,
    pure_repeat_logit_noise,
    pure_repeat_risk_noise,
    run_pure_repeat_baseline,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_manifest import FoldForwardTrace
from src.online2.v2.finetune_v03.counterfactual.cf1s.core_preflight import CORE_MODULE_RELPATHS

ROOT = Path(__file__).resolve().parents[3]


def _atomic(eid, t, pos, direction="DOWN", feature="temp"):
    return {
        "event_id": eid,
        "feature_id": feature,
        "edit_direction": direction,
        "target_raw": 1.0,
        "schema_target": "bin1",
        "sequence_position": pos,
        "event_time_epoch": float(t),
        "to_tokens": ["TOK"],
        "sentence_tokens": ["TOK"],
    }


def test_mixed_direction_rejected():
    mem = validate_family_membership(
        [
            _atomic("A", 0, 0, direction="DOWN"),
            _atomic("B", 1000, 1, direction="UP"),
        ]
    )
    assert mem["ok"] is False
    assert mem["reason"] == "MIXED_DIRECTION"


def test_temporal_arms_and_dead_zone():
    assert temporal_arm_for_gaps([1000])[0] == "CONTIGUOUS"
    assert temporal_arm_for_gaps([4000])[0] == "SPARSE"
    arm, reason = temporal_arm_for_gaps([2000])
    assert arm is None and reason == "REJECT_TEMPORAL_GAP_DEAD_ZONE"
    arm, reason = temporal_arm_for_gaps([1000, 4000])
    assert reason == "MIXED_TEMPORAL_ARM"
    # 3-event contiguous
    mem = validate_family_membership(
        [_atomic("A", 0, 0), _atomic("B", 1000, 1), _atomic("C", 2000, 2)]
    )
    assert mem["ok"] and mem["temporal_arm"] == "CONTIGUOUS"


def test_parent_exact_subset_projection_hashes():
    bundle = [_atomic("A", 0, 0), _atomic("B", 1000, 1)]
    parent = project_parent_subset(bundle, ["A"])
    check = verify_parent_matches_bundle_projection(
        bundle_atomics=bundle,
        parent_atomics=parent["atomics"],
        keep_event_ids=["A"],
    )
    assert check["parent_atomic_target_hash_matches_bundle_projection"]
    assert check["parent_raw_transaction_hash_matches_bundle_projection"]


def test_complete_two_and_three_event_families():
    atomics = [
        _atomic("A", 0, 0),
        _atomic("B", 1000, 1),
        _atomic("C", 2000, 2),
        _atomic("D", 8000, 3),  # sparse vs A
    ]
    fam = build_complete_families(atomics)
    assert fam["two_event_family_evaluable"]
    assert fam["three_event_family_evaluable"]
    complete3 = [f for f in fam["three_event_families"] if f["complete"]]
    assert complete3
    assert set(complete3[0]["candidates"]) >= {"A", "B", "C", "AB", "AC", "BC", "ABC"}


def test_pure_repeat_noise_and_forced_identity_all_folds():
    assert pure_repeat_risk_noise([0.2, 0.21, 0.205]) == pytest.approx(0.01)
    assert pure_repeat_logit_noise([[0.1, 0.2], [0.1, 0.25], [0.1, 0.22]]) == pytest.approx(0.05)

    def fwd(fold_id: int):
        return {"risk": 0.2, "logit": [0.0, 0.2]}

    base = run_pure_repeat_baseline(requested_fold_ids=[0, 1], forward_fn=fwd)
    assert base["baseline_reference_policy"] == "MEDIAN_OF_3_PURE_REPEATS"
    noise = check_noise_ceilings(
        case_pure_repeat_risk_noise=base["case_pure_repeat_risk_noise"],
        case_pure_repeat_logit_noise=base["case_pure_repeat_logit_noise"],
    )
    assert noise["noise_ceiling_pass"]
    identity = evaluate_forced_identity(
        baseline=base,
        identity_by_fold={
            "0": {"risk": 0.2, "logit": [0.0, 0.2]},
            "1": {"risk": 0.2, "logit": [0.0, 0.2]},
        },
        risk_tolerance=1e-6,
        logit_tolerance=1e-6,
    )
    assert identity["forced_identity_all_requested_folds_pass"]
    bad = evaluate_forced_identity(
        baseline=base,
        identity_by_fold={
            "0": {"risk": 0.2, "logit": [0.0, 0.2]},
            "1": {"risk": 0.5, "logit": [0.0, 0.5]},
        },
        risk_tolerance=1e-6,
        logit_tolerance=1e-6,
    )
    assert bad["forced_identity_failed_fold_ids"] == [1]
    assert bad["case_execution_status"] == "NOT_EVALUABLE"


def test_holdout_forward_before_freeze_forbidden():
    trace = FoldForwardTrace()
    with pytest.raises(Exception):
        trace.record(scope="holdout", fold_ids=[2])
    trace.record(scope="search", fold_ids=[0, 1])
    trace.mark_closure_frozen()
    trace.record(scope="holdout", fold_ids=[2])
    summary = trace.summarize(search_fold_ids=[0, 1], holdout_fold_ids=[2])
    assert summary["holdout_forward_before_freeze_count"] == 0
    assert summary["nonrequested_fold_forward_count"] == 0


def test_core_import_graph_excludes_legacy():
    paths = [ROOT / rel for rel in CORE_MODULE_RELPATHS if (ROOT / rel).exists()]
    # Also include core_execution/readiness which are in CORE_MODULE_RELPATHS
    out = assert_core_import_graph_clean(paths)
    assert out["core_imports_legacy_runner_count"] == 0
    assert out["core_imports_production_bridge_count"] == 0
    assert out["core_uses_legacy_readiness_count"] == 0
