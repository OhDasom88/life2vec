"""Unit tests for CF M2 prereq blockers (no GPU required for most)."""

from __future__ import annotations

import numpy as np
import pytest

from src.online2.v2.finetune_v03.counterfactual.candidates.inference_masker import (
    assert_only_group_changed,
    mask_measurement_group,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.constrained_mlm import (
    build_observed_bundle_bank,
    filter_bundles,
    mlm_reconstruction_metrics,
    select_mlm_candidates,
    BundleCandidate,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.crossfit_critic import (
    rank_candidates_on_search_folds,
    resolve_fold_split,
    select_best_on_search,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.over_edit import (
    build_over_edit_report,
    summarize_over_edit,
)
from src.online2.v2.finetune_v03.counterfactual.gates.local_rules import compare_token_bundles
from src.online2.v2.finetune_v03.counterfactual.grounding.locus_raw import is_exact
from src.online2.v2.finetune_v03.counterfactual.m2_preflight import assert_strict_no_fallback
from src.online2.v2.finetune_v03.counterfactual.stage_a_reencoder import (
    affected_targets_for_edits,
    build_reverse_dependency_index,
)


def test_inference_mask_is_deterministic():
    ids = [1, 2, 3, 4, 5]
    groups = ["a", "mg1", "mg1", "b", "mg1"]
    roles = ["feature_identity", "abs_value", "global_value", "feature_identity", "abs_value"]
    tokens = ["FEATURE|t", "VALUE_ABS|ABS_B01", "VALUE_GLOBAL|G01", "FEATURE|h", "VALUE_ABS|ABS_B02"]
    r1 = mask_measurement_group(ids, groups, roles, target_group_id="mg1", mask_id=99, tokens=tokens)
    r2 = mask_measurement_group(ids, groups, roles, target_group_id="mg1", mask_id=99, tokens=tokens)
    assert list(r1.masked_indices) == list(r2.masked_indices)
    assert list(r1.masked_ids) == list(r2.masked_ids)


def test_inference_mask_only_value_roles():
    ids = [1, 2, 3]
    groups = ["mg1", "mg1", "mg1"]
    roles = ["feature_identity", "abs_value", "unit"]
    tokens = ["FEATURE|t", "VALUE_ABS|ABS_B01", "UNIT|C"]
    r = mask_measurement_group(ids, groups, roles, target_group_id="mg1", mask_id=99, tokens=tokens)
    assert r.masked_ids[0] == 1  # feature preserved
    assert r.masked_ids[1] == 99
    assert r.masked_ids[2] == 3  # unit preserved
    assert_only_group_changed(ids, r.masked_ids, groups, "mg1")


def test_holdout_not_overlapping_search():
    split = resolve_fold_split(
        checkpoint_paths=["/a/f0.pt", "/a/f1.pt", "/a/f2.pt"],
        fold_ids=[0, 1, 2],
        search_fold_ids=[0, 1],
        holdout_fold_ids=[2],
    )
    assert split.search_fold_ids == [0, 1]
    assert split.holdout_fold_ids == [2]
    with pytest.raises(ValueError):
        resolve_fold_split(
            checkpoint_paths=["/a/f0.pt", "/a/f1.pt"],
            fold_ids=[0, 1],
            search_fold_ids=[0, 1],
            holdout_fold_ids=[1],
        )


def test_rank_prefers_lower_delta_then_noop():
    cands = [
        {"source": "adjacent_bin", "delta_r_search": -0.02, "folds_improved_search": 2, "is_noop": False, "structurally_valid": True, "delta_r_folds_search": [-0.02, -0.02]},
        {"source": "noop", "delta_r_search": -0.02, "folds_improved_search": 2, "is_noop": True, "structurally_valid": True},
        {"source": "constrained_mlm", "delta_r_search": -0.05, "folds_improved_search": 2, "is_noop": False, "structurally_valid": True, "delta_r_folds_search": [-0.04, -0.06]},
    ]
    ranked = rank_candidates_on_search_folds(cands, min_search_folds_improved=1)
    assert ranked[0]["source"] == "constrained_mlm"
    best = select_best_on_search(ranked, require_material=True)
    assert best["source"] == "constrained_mlm"


def test_mlm_target_event_excluded_from_bundle_bank():
    rows = [
        {"feature": "t", "event_id": "e1", "tokens": ["A"], "token_ids": [1], "roles": ["abs_value"]},
        {"feature": "t", "event_id": "e2", "tokens": ["B"], "token_ids": [2], "roles": ["abs_value"]},
    ]
    bank = build_observed_bundle_bank(rows, exclude_event_id="e1")
    assert all(b["event_id"] != "e1" for b in bank["t"])


def test_mlm_original_bundle_becomes_noop():
    scored = [
        BundleCandidate("constrained_mlm", ["A"], [1], -1.0, False, "t", "mg"),
        BundleCandidate("constrained_mlm", ["B"], [2], -0.5, True, "t", "mg"),
    ]
    out = select_mlm_candidates(scored, top_k=3, exclude_original_from_edits=True)
    assert any(c.source == "noop" and c.is_original for c in out)
    assert all(not c.is_original for c in out if c.source == "constrained_mlm")


def test_mlm_reconstruction_metrics_lift():
    scored = [
        BundleCandidate("constrained_mlm", ["ORIG"], [1], 0.0, True, "t", "mg"),
        BundleCandidate("constrained_mlm", ["X"], [2], -1.0, False, "t", "mg"),
    ]
    m = mlm_reconstruction_metrics(scored, original_tokens=["ORIG"])
    assert m["recall@1"] == 1.0
    assert m["MRR"] == 1.0


def test_gate4_rejects_inconsistent_mlm_bundle():
    r = compare_token_bundles(["VALUE_ABS|ABS_B01"], ["VALUE_ABS|ABS_B02"])
    assert r["status"] == "REJECTED"


def test_gate4_compares_proposed_to_actual_not_identity_noop():
    proposed = ["FEATURE|t", "VALUE_ABS|ABS_B03"]
    actual = ["FEATURE|t", "VALUE_ABS|ABS_B03"]
    r = compare_token_bundles(proposed, actual)
    assert r["status"] == "PASSED"


def test_normal_nonnoop_counted_as_over_edit():
    rows = [
        {
            "is_normal": True,
            "candidates": [{}],
            "selected": {"is_noop": False, "source": "adjacent_bin", "delta_r_holdout": -0.001},
        }
    ]
    s = summarize_over_edit(rows)
    assert s["n_edit_selected"] == 1
    assert s["normal_over_edit_rate"] == 1.0


def test_strict_mode_rejects_synthetic_fallback():
    cfg = {"cf_m2_prereq": {"enabled": True, "fallback_policy": "error"}}
    with pytest.raises(RuntimeError):
        assert_strict_no_fallback(cfg, attribution_mode="synthetic_fallback")


def test_affected_targets_include_context_dependents():
    class W:
        def __init__(self, idxs):
            self.event_indices = idxs

    class Mod:
        def construct_target_window(self, events, t, max_length=1024):
            # window for target t includes t-1 and t
            idxs = [x for x in (t - 1, t, t + 1) if 0 <= x < len(events)]
            return W(idxs)

    events = [object(), object(), object()]
    dep = build_reverse_dependency_index(events, Mod(), max_length=1024)
    # editing event 1 affects targets that include 1
    aff = affected_targets_for_edits(dep, [1])
    assert 0 in aff or 1 in aff
    assert 1 in aff


def test_is_exact_helper():
    assert is_exact({"raw_grounding_status": "EXACT"})
    assert not is_exact({"raw_grounding_status": "NEAREST"})
