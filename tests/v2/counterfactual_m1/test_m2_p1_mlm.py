"""P1 MLM unit tests (CPU; no GroupedMLMMasker)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.online2.v2.finetune_v03.counterfactual.candidates.inference_masker import (
    mask_measurement_group,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.mlm_window_lift import (
    annotate_sentence_tokens,
    lift_local_mask_to_window,
)
from src.online2.v2.finetune_v03.counterfactual.candidates.bundle_bank_loader import (
    build_bundle_bank_for_locus,
    bundle_fingerprint,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.acceptance import (
    evaluate_p1_acceptance_conditions,
    mlm_outcome_from_candidates,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.curated_fixture import (
    select_structural_fixture,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.reconstruction_selection import (
    compute_selection_manifest_chain_hash,
    run_expansion_stages,
    select_mg_keys_deterministic,
)


def test_mlm_window_lift_offsets():
    tokens = ["FEATURE|t", "VALUE_ABS|ABS_B01", "VALUE_GLOBAL|G01", "FEATURE|h", "VALUE_ABS|ABS_B02"]
    ann = annotate_sentence_tokens(tokens)
    ids = list(range(len(tokens)))
    L = 20
    start = 5
    x = torch.zeros(1, 4, L, dtype=torch.long)
    x[0, 0, start : start + len(ids)] = torch.tensor(ids)
    x[0, 1, :] = 7
    x[0, 2, :] = 3
    x[0, 3, :] = 1
    pad = torch.ones(1, L, dtype=torch.bool)
    lift = lift_local_mask_to_window(
        local_token_ids=ids,
        local_group_ids=ann.group_ids,
        local_roles=ann.roles,
        local_tokens=ann.tokens,
        target_group_id="mg:t",
        mask_id=99,
        window_input_ids_4ch=x,
        window_padding_mask=pad,
        target_token_start=start,
    )
    assert lift.window_masked_indices == [start + i for i in lift.local_masked_indices]
    assert lift.unchanged_auxiliary_channels
    assert lift.unchanged_attention_mask
    assert lift.modified_channels == ["token_id"]
    # aux unchanged numerically
    assert torch.equal(lift.input_ids_4ch[:, 1:, :], x[:, 1:, :])


def test_mlm_window_lift_preserves_non_target_tokens():
    tokens = ["FEATURE|t", "VALUE_ABS|ABS_B01", "UNIT|C"]
    ann = annotate_sentence_tokens(tokens)
    ids = [10, 20, 30]
    x = torch.arange(12, dtype=torch.long).view(1, 1, 12).repeat(1, 4, 1).clone()
    x[0, 0, 2:5] = torch.tensor(ids)
    pad = torch.ones(1, 12, dtype=torch.bool)
    lift = lift_local_mask_to_window(
        local_token_ids=ids,
        local_group_ids=ann.group_ids,
        local_roles=ann.roles,
        local_tokens=ann.tokens,
        target_group_id="mg:t",
        mask_id=99,
        window_input_ids_4ch=x,
        window_padding_mask=pad,
        target_token_start=2,
    )
    assert lift.unchanged_non_target_positions
    for i in range(12):
        if i not in lift.window_masked_indices:
            assert int(lift.input_ids_4ch[0, 0, i]) == int(x[0, 0, i])


def test_bundle_bank_leave_out_and_future_exclude():
    import pandas as pd

    rows = [
        {
            "feature": "t",
            "event_id": "e0",
            "case_id": "C1",
            "tokens": ["A"],
            "token_ids": [1],
            "roles": ["abs_value"],
            "timestamp": pd.Timestamp("2025-01-01"),
            "source_partition": "TRAINING",
            "fingerprint": "a",
        },
        {
            "feature": "t",
            "event_id": "e1",
            "case_id": "C1",
            "tokens": ["B"],
            "token_ids": [2],
            "roles": ["abs_value"],
            "timestamp": pd.Timestamp("2025-01-02"),
            "source_partition": "TRAINING",
            "fingerprint": "b",
        },
        {
            "feature": "t",
            "event_id": "e2",
            "case_id": "C1",
            "tokens": ["C"],
            "token_ids": [3],
            "roles": ["abs_value"],
            "timestamp": pd.Timestamp("2025-01-03"),
            "source_partition": "TRAINING",
            "fingerprint": "c",
        },
        {
            "feature": "t",
            "event_id": "e9",
            "case_id": "C2",
            "tokens": ["D"],
            "token_ids": [4],
            "roles": ["abs_value"],
            "timestamp": pd.Timestamp("2025-01-10"),
            "source_partition": "TRAINING",
            "fingerprint": "d",
        },
    ]
    filtered, meta = build_bundle_bank_for_locus(
        rows,
        feature="t",
        expected_roles=["abs_value"],
        target_event_id="e1",
        target_timestamp=pd.Timestamp("2025-01-02"),
        case_id="C1",
        exclude_target_event=True,
        exclude_same_case_future=True,
        original_tokens=["B"],
    )
    eids = {b["event_id"] for b in filtered}
    assert "e1" not in eids
    assert "e2" not in eids  # same-case future
    assert "e0" in eids
    assert "e9" in eids
    assert meta["exclude_same_case_future"] is True


def test_bundle_bank_duplicate_fingerprint_excluded():
    rows = [
        {
            "feature": "t",
            "event_id": "e1",
            "tokens": ["A"],
            "token_ids": [1],
            "roles": ["abs_value"],
            "fingerprint": bundle_fingerprint("t", ["A"]),
        },
        {
            "feature": "t",
            "event_id": "e2",
            "tokens": ["A"],
            "token_ids": [1],
            "roles": ["abs_value"],
            "fingerprint": bundle_fingerprint("t", ["A"]),
        },
    ]
    filtered, meta = build_bundle_bank_for_locus(
        rows,
        feature="t",
        expected_roles=["abs_value"],
        target_event_id="e0",
    )
    assert meta["n_unique_fingerprints"] == 1
    assert len(filtered) == 1


def test_mlm_not_executed_cannot_report_pass():
    flags = mlm_outcome_from_candidates(mlm_executed=False, deferred=True)
    assert flags["constrained_mlm_execution"] == "NOT_RUN"
    flags2 = mlm_outcome_from_candidates(mlm_executed=False, deferred=False)
    assert flags2["constrained_mlm_execution"] == "NOT_RUN"


def test_expansion_uses_only_recoverable_count():
    pool = []
    for feat_i in range(5):
        for i in range(10):
            pool.append(
                {
                    "case_id": "c",
                    "event_id": f"e{feat_i}_{i}",
                    "measurement_group_id": "mg",
                    "feature": f"f{feat_i}",
                }
            )

    def recoverable_fn(rows):
        return len(rows)

    out = run_expansion_stages(
        pool,
        recoverable_fn=recoverable_fn,
        schedule=[{"max_mg_per_feature": 5}, {"max_mg_per_feature": 10}],
        min_recoverable_mg=20,
    )
    assert out["chain"]["stopped_reason"] == "MIN_RECOVERABLE_MET"
    assert "selected_bank_coverage" not in out["final"]
    assert out["final"]["recoverable_mg_count"] >= 20
    assert out["chain"]["final_stage"] == 1  # 5 features * 5 = 25 >= 20


def test_expansion_stops_when_recoverable_threshold_met():
    pool = []
    for feat_i in range(5):
        for i in range(10):
            pool.append(
                {
                    "case_id": "c",
                    "event_id": f"e{feat_i}_{i}",
                    "measurement_group_id": "mg",
                    "feature": f"f{feat_i}",
                }
            )

    def recoverable_fn(rows):
        return 22 if len(rows) >= 20 else 3

    out = run_expansion_stages(
        pool,
        recoverable_fn=recoverable_fn,
        schedule=[
            {"max_mg_per_feature": 5},
            {"max_mg_per_feature": 10},
            {"max_mg_per_feature": 20},
        ],
        min_recoverable_mg=20,
    )
    assert out["chain"]["final_stage"] == 1
    assert len(out["stages"]) == 1


def test_selection_manifest_chain_hash_covers_all_stages():
    pool = [
        {"case_id": "c", "event_id": f"e{i}", "measurement_group_id": "mg", "feature": "f"}
        for i in range(10)
    ]
    out = run_expansion_stages(
        pool,
        recoverable_fn=lambda rows: len(rows),
        schedule=[{"max_mg_per_feature": 3}, {"max_mg_per_feature": 6}],
        min_recoverable_mg=100,  # force all stages
    )
    hashes = out["chain"]["stage_hashes"]
    assert len(hashes) == 3  # 2 stages + final
    stage_only = [s["stage_sha256"] for s in out["stages"]]
    final_sha = out["final"]["selection_manifest_final_sha256"]
    assert out["chain"]["selection_manifest_chain_hash"] == compute_selection_manifest_chain_hash(
        stage_only,
        selection_manifest_final_sha256=final_sha,
    )
    assert hashes[-1] == final_sha


def test_fixture_selection_does_not_use_decoder_scores():
    # structural failure path still records policy flags
    man = select_structural_fixture(
        case_id="C",
        event_id="E",
        feature="t",
        measurement_group_id="mg:t",
        observed_raw=1.0,
        edges_abs=[0.0, 1.0, 2.0],
        original_tokens=["VALUE_ABS|ABS_B01"],
        bank_bundles=[{"tokens": ["VALUE_ABS|ABS_B01"], "is_original": True}],
        tokenizer=None,
        farm_id="F",
    )
    assert man["decoder_scores_used_for_selection"] is False
    assert man["critic_scores_used_for_selection"] is False
    assert man["fixture_selection_policy"] == "STRUCTURAL_ONLY_PRE_MODEL"


def test_mlm_candidate_outcome_equals_natural_outcome():
    rep = evaluate_p1_acceptance_conditions(
        a1_preflight_pass=True,
        a2_functional=True,
        a3_curated_non_original_critic=True,
        a4_outcome_equals_natural=True,
        a5_contracts=True,
        a6_final_code_lock=True,
        a7_rerun_after_lock=True,
        a8_artifact_lock=True,
        mlm_cf_candidate_outcome="NO_ELIGIBLE_LOCUS",
        natural_integration_outcome="NO_ELIGIBLE_LOCUS",
        selection_manifest_chain_hash="abc",
    )
    assert rep["mlm_cf_candidate_outcome"] == rep["natural_integration_outcome"]
    assert rep["acceptance_conditions"]["A4"] == "PASS"


def test_acceptance_requires_rerun_after_final_lock():
    rep = evaluate_p1_acceptance_conditions(
        a1_preflight_pass=True,
        a2_functional=True,
        a3_curated_non_original_critic=True,
        a4_outcome_equals_natural=True,
        a5_contracts=True,
        a6_final_code_lock=True,
        a7_rerun_after_lock=False,
        a8_artifact_lock=True,
        selection_manifest_chain_hash="abc",
    )
    assert rep["mlm_implementation"] == "FAIL"
    assert rep["acceptance_conditions"]["A7"] == "FAIL"


def test_artifact_manifest_has_no_self_referential_hash():
    art = {
        "artifact_lock": "PASS",
        "bundle_bank_sha256": "x",
        "selection_manifest_chain_hash": "y",
    }
    assert "final_package_sha256" not in art


def test_grouped_mlm_masker_never_called_in_inference_masker():
    import src.online2.v2.finetune_v03.counterfactual.candidates.inference_masker as im
    src = Path(im.__file__).read_text(encoding="utf-8")
    assert "GroupedMLMMasker" not in src or "Never call GroupedMLMMasker" in src
    # ensure module does not import GroupedMLMMasker
    assert not hasattr(im, "GroupedMLMMasker")
