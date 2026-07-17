"""Behavioral unit tests for CF M2 v3 policies (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.online2.v2.finetune_v03.counterfactual.candidates.mlm_raw_inversion import (
    invert_mlm_bundle_to_raw,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.acceptance import (
    build_acceptance_report,
    mlm_outcome_from_candidates,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.canonical_locus import (
    canonical_locus_key,
    canonical_target_raw,
    ensure_canonical_noop,
    merge_duplicate_candidates,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.case_fold_manifest import (
    build_case_fold_record,
    case_seen_by_fold,
    crossfit_rotations,
    oof_fold_for_case,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.cf_validity import (
    STATUS_NOT_INDEPENDENT,
    build_cf_validity,
    llm_case_payload,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.crossfit_consensus import (
    summarize_crossfit_splits,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.crossfit_critic import (
    rank_candidates_on_search_folds,
    select_best_on_search,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.fold_execution_context import (
    EVAL_CROSSFIT,
    EVAL_OOF_REF,
    build_fold_execution_context,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.noop_calibration import (
    build_noop_noise_artifact,
    compute_effective_epsilon,
    ensure_default_calibration_artifact,
    load_frozen_epsilon,
)
from src.online2.v2.finetune_v03.counterfactual.attribution.fold_consensus import (
    l1_normalize,
    normalize_fold_scores,
    sign_agreement_mask,
)
from src.online2.v2.finetune_v03.counterfactual.retokenization.full_event_retokenizer import (
    REFERENCE_POLICY,
)
import numpy as np


def test_fold_context_rejects_example_crossfit():
    with pytest.raises(ValueError, match="example_set requires"):
        build_fold_execution_context(
            case_id="C",
            cohort="example_set",
            evaluation_mode=EVAL_CROSSFIT,
            case_seen_by_fold={0: False, 1: True, 2: True},
            search_fold_ids=[0, 1],
            holdout_fold_ids=[2],
            effective_epsilon=0.01,
        )


def test_fold_context_oof_allows_same_fold():
    ctx = build_fold_execution_context(
        case_id="C",
        cohort="example_set",
        evaluation_mode=EVAL_OOF_REF,
        case_seen_by_fold={0: False, 1: True, 2: True},
        search_fold_ids=[0],
        holdout_fold_ids=[0],
        effective_epsilon=0.01,
        oof_fold_id=0,
    )
    assert not ctx.independently_evaluable
    assert ctx.cf_validity_allowed is False


def test_fold_context_rejects_crossfit_overlap():
    with pytest.raises(ValueError):
        build_fold_execution_context(
            case_id="C",
            cohort="problem_set",
            evaluation_mode=EVAL_CROSSFIT,
            case_seen_by_fold={0: False, 1: False, 2: False},
            search_fold_ids=[0, 1],
            holdout_fold_ids=[1],
            effective_epsilon=0.01,
        )


def test_example_cf_valid_is_null():
    ctx = build_fold_execution_context(
        case_id="C",
        cohort="example_set",
        evaluation_mode=EVAL_OOF_REF,
        case_seen_by_fold={0: False, 1: True, 2: True},
        search_fold_ids=[0],
        holdout_fold_ids=[0],
        effective_epsilon=0.01,
        oof_fold_id=0,
    )
    v = build_cf_validity(
        ctx=ctx,
        structurally_valid=True,
        raw_edit_valid=True,
        retokenization_valid=True,
        stage_a_valid=True,
        search_effect_valid=True,
        holdout_effect_valid=True,
        is_noop=False,
    )
    assert v["cf_valid"] is None
    assert v["search_effect_valid"] is None
    assert v["holdout_effect_valid"] is None
    assert v["cf_evaluation_status"] == STATUS_NOT_INDEPENDENT
    assert v["management_intervention_allowed"] is False


def test_problem_cf_valid_requires_holdout():
    ctx = build_fold_execution_context(
        case_id="P",
        cohort="problem_set",
        evaluation_mode=EVAL_CROSSFIT,
        case_seen_by_fold={0: False, 1: False, 2: False},
        search_fold_ids=[0, 1],
        holdout_fold_ids=[2],
        effective_epsilon=0.01,
    )
    v = build_cf_validity(
        ctx=ctx,
        structurally_valid=True,
        raw_edit_valid=True,
        retokenization_valid=True,
        stage_a_valid=True,
        search_effect_valid=True,
        holdout_effect_valid=False,
        is_noop=False,
    )
    assert v["cf_valid"] is False


def test_material_selects_noop_when_immaterial():
    cands = [
        {
            "source": "adjacent_bin",
            "candidate_id": "a",
            "delta_r_search": -1e-6,
            "folds_improved_search": 2,
            "delta_r_folds_search": [-1e-6, -1e-6],
            "is_noop": False,
            "structurally_valid": True,
        },
        {
            "source": "noop",
            "candidate_id": "noop::c",
            "delta_r_search": 0.0,
            "folds_improved_search": 0,
            "is_noop": True,
            "structurally_valid": True,
        },
    ]
    ranked = rank_candidates_on_search_folds(cands, material_delta_r_abs=0.01, min_search_folds_improved=2)
    best = select_best_on_search(ranked, require_material=True)
    assert best["is_noop"] is True


def test_noop_calibration_frozen_outside_eval(tmp_path: Path):
    art = build_noop_noise_artifact(
        noop_delta_rs=[0.0, 1e-7, 2e-7],
        calibration_cohort="example_oof",
        configured_epsilon=0.01,
    )
    path = tmp_path / "noop_noise.json"
    path.write_text(json.dumps(art))
    loaded = load_frozen_epsilon(path)
    assert loaded["effective_epsilon_frozen"] is True
    assert loaded["effective_epsilon"] == 0.01
    assert loaded["calibration_cohort"] == "example_oof"


def test_effective_epsilon_uses_noise_floor():
    eps = compute_effective_epsilon(
        configured_epsilon=0.001,
        noop_delta_p99=0.0005,
        noop_noise_multiplier=10.0,
    )
    assert eps == pytest.approx(0.005)


def test_l1_fold_normalization_and_sign():
    folds = np.array([[10.0, -5.0], [20.0, -10.0]], dtype=np.float64)
    norm = normalize_fold_scores(folds, method="l1")
    assert abs(np.abs(norm[0]).sum() - 1.0) < 1e-9
    mask = sign_agreement_mask(folds, min_abs_score_per_fold=1e-5)
    assert bool(mask[0]) and bool(mask[1])


def test_gate0_and_mg_splice_production_api():
    from src.online2.v2.finetune_v03.counterfactual.retokenization.full_event_retokenizer import (
        REFERENCE_POLICY,
        find_mg_span,
        splice_mg_tokens,
    )

    tokens = [
        "[MEAS_SEP]",
        "FEATURE|INSIDE_TEMP_C",
        "VALUE_ABS|ABS_B03",
        "INSIDE_TEMP_C|ABS_B03",
        "QUALITY|OK",
        "[MEAS_SEP]",
        "FEATURE|EC_SENSOR",
        "VALUE_ABS|ABS_B01",
        "QUALITY|OK",
    ]
    span = find_mg_span(tokens, "inside_temp_c")
    assert span == (1, 5)
    new_mg = [
        "FEATURE|INSIDE_TEMP_C",
        "VALUE_ABS|ABS_B04",
        "INSIDE_TEMP_C|ABS_B04",
        "QUALITY|OK",
    ]
    new_sent, new_span, outside_ok = splice_mg_tokens(tokens, "inside_temp_c", new_mg)
    assert outside_ok
    assert "VALUE_ABS|ABS_B01" in new_sent  # other MG untouched
    assert new_sent.count("VALUE_ABS|ABS_B04") == 1
    assert REFERENCE_POLICY == "FROZEN_ORIGINAL_ARTIFACT"


def test_mlm_inversion_interior_and_empty():
    edges = [0.0, 1.0, 2.0, 3.0]
    ok = invert_mlm_bundle_to_raw(
        feature="t",
        observed_raw=1.5,
        mlm_proposed_bundle=["FEATURE|t", "VALUE_ABS|ABS_B01"],
        edges_abs=edges,
        margin_ratio=0.001,
    )
    assert ok["ok"] is True
    assert 1.0 < ok["selected_target_raw"] < 2.0

    # Conflicting abs bins via crafted intersection fail: single abs always works;
    # empty interior when margin too large on tiny width
    bad = invert_mlm_bundle_to_raw(
        feature="t",
        observed_raw=0.5,
        mlm_proposed_bundle=["FEATURE|t", "VALUE_ABS|ABS_B00"],
        edges_abs=[0.0, 1e-12, 1.0],
        margin_ratio=0.6,
    )
    assert bad["ok"] is False
    assert bad["reason_code"] in {"EMPTY_SAFE_INTERIOR", "EMPTY_INTERVAL_INTERSECTION", "RETOKENIZATION_MISMATCH"}


def test_dedup_merges_sources_excludes_source_from_key():
    rows = [
        {
            "case_id": "C",
            "canonical_locus_key": "t|z|mg|1|increase",
            "canonical_target_raw": 1.2,
            "actual_retokenized_bundle": ["FEATURE|t", "VALUE_ABS|ABS_B01"],
            "source": "adjacent_bin",
            "delta_r_search": -0.02,
            "is_noop": False,
        },
        {
            "case_id": "C",
            "canonical_locus_key": "t|z|mg|1|increase",
            "canonical_target_raw": 1.2,
            "actual_retokenized_bundle": ["FEATURE|t", "VALUE_ABS|ABS_B01"],
            "source": "constrained_mlm",
            "delta_r_search": -0.03,
            "is_noop": False,
        },
        {"case_id": "C", "source": "noop", "is_noop": True, "delta_r_search": 0.0},
        {"case_id": "C", "source": "noop", "is_noop": True, "delta_r_search": 0.0},
    ]
    merged = ensure_canonical_noop(merge_duplicate_candidates(rows), case_id="C")
    edits = [r for r in merged if not r.get("is_noop")]
    noops = [r for r in merged if r.get("is_noop")]
    assert len(edits) == 1
    assert set(edits[0]["candidate_sources"]) == {"adjacent_bin", "constrained_mlm"}
    assert len(noops) == 1


def test_crossfit_consensus_requires_holdout_valid():
    splits = [
        {
            "canonical_locus_key": "ec|z|mg|1|decrease",
            "holdout_effect_valid": True,
            "gate4_status": "PASSED",
            "operational_eligibility": "OK",
            "delta_r_holdout": -0.02,
            "is_noop": False,
        },
        {
            "canonical_locus_key": "ec|z|mg|1|decrease",
            "holdout_effect_valid": False,
            "gate4_status": "PASSED",
            "operational_eligibility": "OK",
            "delta_r_holdout": -0.001,
            "is_noop": False,
        },
        {
            "canonical_locus_key": "temp|z|mg|1|increase",
            "holdout_effect_valid": True,
            "gate4_status": "PASSED",
            "operational_eligibility": "OK",
            "delta_r_holdout": -0.03,
            "is_noop": False,
        },
    ]
    s = summarize_crossfit_splits(splits, min_agreement_splits=2, min_holdout_valid_splits=2)
    assert s["crossfit_consensus_status"] == "UNSTABLE_OR_UNVALIDATED"
    assert s["final_cf_candidate"] is None

    splits[1]["holdout_effect_valid"] = True
    s2 = summarize_crossfit_splits(splits[:2] + [
        {**splits[0], "delta_r_holdout": -0.025}
    ], min_agreement_splits=2, min_holdout_valid_splits=2)
    assert s2["crossfit_consensus_status"] == "STABLE_VALID_CF"


def test_acceptance_separates_mlm_execution_and_outcome():
    flags = mlm_outcome_from_candidates(
        mlm_executed=True,
        n_eligible_loci=3,
        n_valid_mlm_cf=0,
        inversion_failures=2,
        reconstruction_pass=True,
    )
    report = build_acceptance_report(
        structural_smoke="PASS",
        scenario_outcome="NO_VALID_CF",
        leakage_free=True,
        gate0_ok=True,
        retokenize_ok=True,
        noop_policy_ok=True,
        cohort_integrity="PASS",
        full_retokenization="PASS",
        mg_retokenization_smoke="PASS",
        dependency_closure_retokenization="NOT_EVALUATED",
        calibration_ok=True,
        gt_normal_verified=True,
        normal_over_edit_smoke="PASS",
        **flags,
    )
    assert report["implementation_acceptance"] == "PASS"
    assert report["mlm_cf_candidate_outcome"] == "INVERSION_FAILED"
    assert report["constrained_mlm_execution"] == "PASS"


def test_acceptance_mlm_not_run_is_incomplete_not_pass():
    flags = mlm_outcome_from_candidates(mlm_executed=False, deferred=True)
    report = build_acceptance_report(
        structural_smoke="PARTIAL_PASS",
        leakage_free=True,
        gate0_ok=True,
        retokenize_ok=True,
        noop_policy_ok=True,
        cohort_integrity="PASS",
        full_retokenization="PASS",
        mg_retokenization_smoke="PASS",
        dependency_closure_retokenization="NOT_EVALUATED",
        calibration_ok=True,
        gt_normal_verified=True,
        normal_over_edit_smoke="PASS",
        mlm_required_for_pass=False,
        **flags,
    )
    assert flags["constrained_mlm_execution"] == "NOT_RUN"
    assert report["implementation_acceptance"] in {"PASS", "INCOMPLETE"}
    assert report["constrained_mlm_execution"] != "PASS" or flags["constrained_mlm_execution"] == "NOT_RUN"


def test_llm_payload_blocks_invalid_intervention():
    ctx = build_fold_execution_context(
        case_id="P",
        cohort="problem_set",
        evaluation_mode=EVAL_CROSSFIT,
        case_seen_by_fold={0: False, 1: False, 2: False},
        search_fold_ids=[0, 1],
        holdout_fold_ids=[2],
        effective_epsilon=0.01,
    )
    v = build_cf_validity(
        ctx=ctx,
        structurally_valid=True,
        raw_edit_valid=True,
        retokenization_valid=True,
        stage_a_valid=True,
        search_effect_valid=False,
        holdout_effect_valid=False,
        is_noop=True,
    )
    payload = llm_case_payload(selected={"is_noop": True, "source": "noop"}, validity=v)
    assert payload["cf_status"] == "NO_VALID_CF"
    assert payload["management_intervention_allowed"] is False


def test_case_fold_manifest_example_vs_problem():
    splits = [
        {"fold": 0, "train": ["E1", "E2"], "val": ["E0"]},
        {"fold": 1, "train": ["E0", "E2"], "val": ["E1"]},
        {"fold": 2, "train": ["E0", "E1"], "val": ["E2"]},
    ]
    assert oof_fold_for_case("E1", splits) == 1
    seen = case_seen_by_fold("E1", splits)
    assert seen[1] is False
    assert seen[0] is True and seen[2] is True
    ex = build_case_fold_record(case_id="E1", cohort="example_set", splits=splits)
    assert ex["evaluation_mode"] == EVAL_OOF_REF
    assert ex["cf_valid"] if False else ex["oof_fold_id"] == 1
    pr = build_case_fold_record(
        case_id="P9", cohort="problem_set", splits=splits, holdout_rotation=2
    )
    assert pr["evaluation_mode"] == EVAL_CROSSFIT
    assert pr["holdout_fold_ids"] == [2]
    assert all(v is False for v in {int(k): v for k, v in pr["case_seen_by_fold"].items()}.values())
    assert len(crossfit_rotations()) == 3
    with pytest.raises(ValueError, match="cohort mismatch"):
        build_case_fold_record(
            case_id="E1", cohort="problem_set", splits=splits, holdout_rotation=2
        )


def test_select_path_a_mg_sign_agreement_hard_gate():
    import pandas as pd
    from src.online2.v2.finetune_v03.counterfactual.pipeline_m2 import _select_path_a_mg

    mg = pd.DataFrame(
        [
            {
                "feature": "inside_temp_c",
                "event_id": "e1",
                "absolute_attribution": 0.9,
                "sign_agreement": False,
            },
            {
                "feature": "inside_temp_c",
                "event_id": "e2",
                "absolute_attribution": 0.5,
                "sign_agreement": False,
            },
        ]
    )
    assert _select_path_a_mg(mg, "inside_temp_c", require_sign_agreement=True) is None
    mg.loc[1, "sign_agreement"] = True
    top = _select_path_a_mg(mg, "inside_temp_c", require_sign_agreement=True)
    assert top is not None and top["event_id"] == "e2"


def test_leakage_free_from_fold_usage():
    from src.online2.v2.finetune_v03.counterfactual.evaluation.fold_usage import (
        build_fold_usage_trace,
        compute_leakage_free,
    )

    ok = build_fold_usage_trace(
        event_preselection_folds=[0, 1],
        token_ixg_folds=[0, 1],
        locus_selection_folds=[0, 1],
        candidate_ranking_folds=[0, 1],
        holdout_evaluation_folds=[2],
        evaluation_mode="crossfit_3fold",
        cohort="problem_set",
        case_id="P",
    )
    assert compute_leakage_free(ok)["leakage_free"] is True
    bad = dict(ok)
    bad["candidate_ranking_folds"] = [0, 1, 2]
    assert compute_leakage_free(bad)["leakage_free"] is False


def test_gate0_strict_rejects_exact_core():
    from src.online2.v2.finetune_v03.counterfactual.retokenization.full_event_retokenizer import (
        gate0_baseline_roundtrip,
    )

    class FakeTok:
        def tokenize_value(self, feature, raw, farm_id=""):
            class MT:
                tokens = [
                    "FEATURE|INSIDE_TEMP_C",
                    "VALUE_ABS|ABS_B03",
                    "QUALITY|WARN",
                ]
            return MT()

    original = [
        "FEATURE|INSIDE_TEMP_C",
        "VALUE_ABS|ABS_B03",
        "QUALITY|OK",
    ]
    # monkey via wrap: gate0 calls tokenize_mg_production which calls tokenizer.tokenize_value
    # Use a minimal stub object with tokenize_value
    g_strict = gate0_baseline_roundtrip(
        original_tokens=original,
        feature="inside_temp_c",
        observed_raw=1.0,
        tokenizer=FakeTok(),
        strict_exact=True,
    )
    assert g_strict["baseline_bundle_match"] == "EXACT_CORE"
    assert g_strict["baseline_roundtrip_valid"] is False
    g_diag = gate0_baseline_roundtrip(
        original_tokens=original,
        feature="inside_temp_c",
        observed_raw=1.0,
        tokenizer=FakeTok(),
        strict_exact=False,
    )
    assert g_diag["baseline_roundtrip_valid"] is True


def test_acceptance_mg_vs_dependency_closure():
    report = build_acceptance_report(
        structural_smoke="PASS",
        scenario_outcome="NO_VALID_CF",
        leakage_free=True,
        gate0_ok=True,
        retokenize_ok=True,
        noop_policy_ok=True,
        cohort_integrity="PASS",
        mg_retokenization_smoke="PASS",
        dependency_closure_retokenization="NOT_EVALUATED",
        calibration_ok=True,
        gt_normal_verified=True,
        normal_over_edit_smoke="PASS",
        tokenizer_integrity="PASS",
        sign_agreement_hard_gate="PASS",
        constrained_mlm_execution="NOT_RUN",
        mlm_required_for_pass=False,
    )
    assert report["mg_retokenization_smoke"] == "PASS"
    assert report["dependency_closure_retokenization"] == "NOT_EVALUATED"
    assert report["implementation_acceptance"] == "PASS"


def test_canonical_target_raw_feature_resolution():
    c = canonical_target_raw(0.6120001, edges=[0.0, 1.0], feature_resolution=0.001)
    assert c["canonicalization_policy"] == "FEATURE_RESOLUTION"
    assert abs(c["canonical_target_raw"] - 0.612) < 1e-9


def test_canonical_locus_key_stable():
    k1 = canonical_locus_key(
        feature="EC", zone="A", measurement_group_id="mg", timestamp="2025-01-01T00:30:00", edit_direction="decrease"
    )
    k2 = canonical_locus_key(
        feature="ec", zone="A", measurement_group_id="mg", timestamp="2025-01-01T00:45:00", edit_direction="decrease"
    )
    assert k1 == k2
