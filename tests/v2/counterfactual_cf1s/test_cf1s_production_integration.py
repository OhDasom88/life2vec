"""Production integration mock tests for CF-1S (no GPU / no real smoke)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pytest

from src.online2.v2.finetune_v03.counterfactual.cf1s.acceptance import (
    evaluate_selector_acceptance,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.evaluation_closure import (
    build_evaluation_closure,
    candidate_identity_projection,
    holdout_set_matches_closure,
    identity_projection_sha256,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.execution_context import (
    CF1SCallbackBundle,
    CF1SExecutionContext,
    ProductionContractViolation,
    fixture_identity_runner,
    fixture_parity_runner,
    fixture_pass_through,
    fixture_stage_a_noop,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.fold_effect import (
    incremental_abs_gain_by_fold,
    multi_event_only_fold_pass,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.production_preflight import (
    ProductionPreflightError,
    assert_production_preflight,
    production_preflight,
    resolve_prediction_cutoff_time,
    validate_case_scoped_cutoff,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.readiness import (
    build_production_readiness,
    require_readiness_for_production_command,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.status import (
    LIMITED,
    NOT_SUPPORTED,
    SUPPORTED,
    derive_selector_feasibility_status,
)
from src.online2.v2.finetune_v03.counterfactual.pipeline_cf1s import (
    load_yaml_policy,
    run_cf1s_case_core,
    run_cf1s_case_fixture,
    run_cf1s_case_production,
)


ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = ROOT / "conf/m1/cf1s_policies"


def _policies():
    return (
        load_yaml_policy(POLICY_DIR / "CF1S_EDIT_POLICY_V1.yaml"),
        load_yaml_policy(POLICY_DIR / "CF1S_SEARCH_POLICY_V1.yaml"),
        load_yaml_policy(POLICY_DIR / "CF1S_SALIENCY_POLICY_V1.yaml"),
        load_yaml_policy(POLICY_DIR / "CF1S_ACCEPTANCE_POLICY_V1.yaml"),
    )


def _mock_effect(delta_search=0.02, delta_holdout=0.02):
    def search_fn(edits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        n = max(len(edits), 1)
        scale = 1.0 if n == 1 else (1.5 if n == 2 else 2.0)
        d = float(delta_search) * scale
        return {
            "delta_risk_search_by_fold": [d, d],
            "delta_risk_search_aggregate": d,
            "model_effects_evaluated": True,
            "simulated": False,
        }

    def holdout_fn(edits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        n = max(len(edits), 1)
        scale = 1.0 if n == 1 else (1.5 if n == 2 else 2.0)
        d = float(delta_holdout) * scale
        return {
            "delta_risk_holdout_by_fold": [d],
            "delta_risk_holdout_aggregate": d,
            "model_effects_evaluated": True,
            "simulated": False,
        }

    return search_fn, holdout_fn


def _production_bundle(
    *,
    search_fn=None,
    holdout_fn=None,
    atomic_targets=None,
):
    sfn, hfn = _mock_effect()
    search_fn = search_fn or sfn
    holdout_fn = holdout_fn or hfn

    def atomic_target_fn(locus, record):
        if atomic_targets is not None:
            return list(atomic_targets)
        obs = locus.get("observed_raw")
        try:
            v = float(obs)
        except (TypeError, ValueError):
            return [1.0]
        return [0.0 if v >= 0.5 else 1.0]

    def parity_runner(edits, effect):
        return {
            "parity_audited": True,
            "parity_status": "PASS",
            "full_reencode_fallback_used": False,
            "effect_source": "INCREMENTAL_VERIFIED",
            "finalized_effect_values": True,
            "finalized_effect": dict(effect),
            "simulated": False,
        }

    return CF1SCallbackBundle(
        atomic_target_fn=atomic_target_fn,
        ground_fn=lambda e: {"ok": True, "simulated": False},
        invert_fn=lambda e: {"ok": True, "simulated": False},
        gate0_fn=lambda e: {"ok": True, "gate0_pass": True, "simulated": False},
        gate4_fn=lambda e: {"ok": True, "gate4_pass": True, "simulated": False},
        retokenize_fn=lambda edit: {
            "ok": True,
            "original_token_length": int(edit.get("original_token_length") or 1),
            "retokenized_token_length": int(edit.get("original_token_length") or 1),
            "tokens": ["P"] * int(edit.get("original_token_length") or 1),
            "retokenized_mg_bundle": ["P"] * int(edit.get("original_token_length") or 1),
            "simulated": False,
        },
        stage_a_reencode_fn=fixture_stage_a_noop,
        search_effect_fn=search_fn,
        holdout_effect_fn=holdout_fn,
        identity_runner=lambda: {
            "risks": [0.1, 0.1, 0.1],
            "logits": [0.0, 0.0, 0.0],
            "forced_risk_delta": 0.0,
            "forced_logit_delta": 0.0,
            "simulated": False,
        },
        parity_runner=parity_runner,
    )


def test_production_wrapper_does_not_call_fixture(monkeypatch, sample_events):
    edit, search, saliency, acceptance = _policies()
    called = {"fixture": 0}

    def boom(*_a, **_k):
        called["fixture"] += 1
        raise AssertionError("fixture wrapper must not be called")

    monkeypatch.setattr(
        "src.online2.v2.finetune_v03.counterfactual.pipeline_cf1s.run_cf1s_case_fixture",
        boom,
    )
    out = run_cf1s_case_production(
        case_id="F1_2025-01-01_2025-01-14",
        events=sample_events,
        edit_policy=edit,
        search_policy=search,
        saliency_policy=saliency,
        acceptance_policy=acceptance,
        callbacks=_production_bundle(),
        cutoff_time="2025-01-14T00:00:00+00:00",
    )
    assert called["fixture"] == 0
    assert out["execution_mode"] == "PRODUCTION"
    assert out["evaluation_closure"]["search_parent_completion_before_holdout"] is True


def test_missing_callback_fails_before_candidates(sample_events):
    edit, search, saliency, acceptance = _policies()
    bundle = _production_bundle()
    # Break a required callback
    broken = CF1SCallbackBundle(
        atomic_target_fn=bundle.atomic_target_fn,
        ground_fn=bundle.ground_fn,
        invert_fn=bundle.invert_fn,
        gate0_fn=bundle.gate0_fn,
        gate4_fn=bundle.gate4_fn,
        retokenize_fn=bundle.retokenize_fn,
        stage_a_reencode_fn=bundle.stage_a_reencode_fn,
        search_effect_fn=bundle.search_effect_fn,
        holdout_effect_fn=None,  # type: ignore[arg-type]
        identity_runner=bundle.identity_runner,
        parity_runner=bundle.parity_runner,
    )
    ctx = CF1SExecutionContext(
        case_id="F1_2025-01-01_2025-01-14",
        events=sample_events,
        edit_policy=edit,
        search_policy=search,
        saliency_policy=saliency,
        acceptance_policy=acceptance,
        callbacks=broken,
        provenance_mode="PRODUCTION",
        uses_fixture_defaults=False,
        model_effects_evaluated=True,
    )
    with pytest.raises(ProductionContractViolation):
        run_cf1s_case_core(ctx)


def test_simulated_result_fatal_in_production(sample_events):
    edit, search, saliency, acceptance = _policies()

    def sim_search(edits):
        return {
            "delta_risk_search_by_fold": [0.02, 0.02],
            "delta_risk_search_aggregate": 0.02,
            "simulated": True,
        }

    sfn, hfn = _mock_effect()
    bundle = _production_bundle(search_fn=sim_search, holdout_fn=hfn)
    with pytest.raises(ProductionContractViolation):
        run_cf1s_case_production(
            case_id="F1_2025-01-01_2025-01-14",
            events=sample_events,
            edit_policy=edit,
            search_policy=search,
            saliency_policy=saliency,
            acceptance_policy=acceptance,
            callbacks=bundle,
        )


def test_single_only_stable_not_supported_primary():
    status = derive_selector_feasibility_status(
        model_effects_evaluated=True,
        n_evaluable_composites=3,
        n_stable_model_sensitivity=2,
        identity_and_determinism_ok=True,
        n_stable_incremental_multi_event=0,
        n_stable_multi_event_only=0,
        n_stable_multi_event=0,
        n_stable_single_event=2,
    )
    assert status == NOT_SUPPORTED


def test_stable_multi_event_without_incremental_is_limited():
    status = derive_selector_feasibility_status(
        model_effects_evaluated=True,
        n_evaluable_composites=3,
        n_stable_model_sensitivity=1,
        identity_and_determinism_ok=True,
        n_stable_incremental_multi_event=0,
        n_stable_multi_event_only=0,
        n_stable_multi_event=1,
        n_stable_single_event=0,
    )
    assert status == LIMITED


def test_incremental_and_multi_event_only_fold_helpers():
    gains = incremental_abs_gain_by_fold(
        bundle_deltas_by_fold=[0.05, 0.04],
        atomic_parent_deltas_by_fold=[[0.01, 0.01], [0.02, 0.01]],
    )
    assert gains[0] == pytest.approx(0.03)
    me = multi_event_only_fold_pass(
        bundle_deltas_by_fold=[0.02, 0.03],
        atomic_parent_deltas_by_fold=[[0.001, 0.001], [0.0, 0.0]],
        epsilon=0.01,
    )
    assert me["multi_event_only_pass"] is True


def test_holdout_before_search_closure_fails(sample_events):
    edit, search, saliency, acceptance = _policies()
    sfn, _ = _mock_effect()
    calls = {"n": 0}

    def early_holdout(edits):
        calls["n"] += 1
        # Will be wrapped; should raise before this if order wrong.
        return {
            "delta_risk_holdout_by_fold": [0.02],
            "delta_risk_holdout_aggregate": 0.02,
            "simulated": False,
        }

    # Inject a search_fn that illegally calls holdout via shared mutable — instead
    # verify wrapper rejects direct holdout before freeze.
    ctx = CF1SExecutionContext(
        case_id="F1_2025-01-01_2025-01-14",
        events=sample_events,
        edit_policy=edit,
        search_policy=search,
        saliency_policy=saliency,
        acceptance_policy=acceptance,
        callbacks=_production_bundle(search_fn=sfn, holdout_fn=early_holdout),
        provenance_mode="PRODUCTION",
        uses_fixture_defaults=False,
        model_effects_evaluated=True,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.execution_context import (
        wrap_holdout_effect_fn,
    )

    wrapped = wrap_holdout_effect_fn(ctx, early_holdout)
    with pytest.raises(ProductionContractViolation):
        wrapped([])
    assert ctx.counters.holdout_call_before_candidate_freeze_count == 1


def test_evaluation_closure_identity_excludes_effects():
    proj = candidate_identity_projection(
        candidate_id="c1",
        case_id="F1",
        selector="SALIENCY_TOP_K",
        edits=[{"event_id": "e", "feature_id": "f", "observed_raw": 0, "target_raw": 1}],
        parent_candidate_ids=[],
    )
    sha1 = identity_projection_sha256(proj)
    # Effects must not be part of projection.
    assert "delta" not in proj
    assert holdout_set_matches_closure(sha1, sha1)


def test_closure_includes_parents_for_three_event():
    bundle = {
        "multi_event_candidate_id": "abc",
        "n_events": 3,
        "edits": [
            {
                "event_id": "e0",
                "feature_id": "f",
                "mg_id": "m0",
                "observed_raw": 0,
                "target_raw": 1,
                "atomic_candidate_id": "a0",
            },
            {
                "event_id": "e1",
                "feature_id": "f",
                "mg_id": "m1",
                "observed_raw": 0,
                "target_raw": 1,
                "atomic_candidate_id": "a1",
            },
            {
                "event_id": "e2",
                "feature_id": "f",
                "mg_id": "m2",
                "observed_raw": 0,
                "target_raw": 1,
                "atomic_candidate_id": "a2",
            },
        ],
        "atomic_parent_candidate_ids": ["a0", "a1", "a2"],
    }
    closure = build_evaluation_closure(
        [bundle], case_id="F1", selector="SALIENCY_TOP_K", already_scored={}
    )
    # 3 atomics + 3 pairs + 1 triple
    assert closure["n_candidates"] == 7
    assert closure["n_closure_only"] == 7


def test_preflight_fold_partition_and_paths():
    cfg = {
        "run_dir": str(ROOT / "outputs/online2/v2_finetune_v03/runs/cv_repeated_20260715_161945"),
        "embeddings_dir": str(ROOT / "outputs/online2/v2_finetune_v02/event_embeddings"),
        "labels_path": str(ROOT / "outputs/online2/v2_finetune_v02/labels_example35_problem20.csv"),
        "label_map_path": str(ROOT / "outputs/online2/v2_finetune/label_map.json"),
        "stage_a_ckpt": str(ROOT / "outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt"),
        "abspos_reference_path": str(ROOT / "outputs/online2/v2_build/abspos_reference.json"),
        "tokenizer_path": str(ROOT / "outputs/online2/build-v8-active80-r3/tokenization_registry.json"),
        "vocabulary_path": str(ROOT / "outputs/online2/v2_build/life2vec_token_registry_v2.json"),
        "feature_schema_path": str(ROOT / "outputs/online2/v2_build/feature_schema_v2.yaml"),
        "binning_registry_path": str(
            ROOT / "outputs/online2/v2_build/binning_registry_v2_transductive.json"
        ),
        "events_path": str(ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet"),
        "cells_path": str(ROOT / "outputs/online2/build-v8-active80-r3/cell_occurrences.parquet"),
        "repeat": 0,
        "critic": {"search_fold_ids": [0, 1], "holdout_fold_ids": [2]},
    }
    saliency = load_yaml_policy(POLICY_DIR / "CF1S_SALIENCY_POLICY_V1.yaml")
    report = production_preflight(cfg, saliency_policy=saliency)
    # Paths may or may not exist in CI; fold partition should be ok.
    assert report["fold_partition"]["ok"] is True
    bad = dict(cfg)
    bad["critic"] = {"search_fold_ids": [0, 1], "holdout_fold_ids": [1]}
    bad_report = production_preflight(bad, saliency_policy=saliency)
    assert bad_report["ok"] is False
    with pytest.raises(ProductionPreflightError):
        assert_production_preflight(bad, saliency_policy=saliency)


def test_case_scoped_cutoff_ignores_other_cases():
    cut = resolve_prediction_cutoff_time("F1_2025-01-01_2025-01-14")
    assert cut["ok"] is True
    ok = validate_case_scoped_cutoff(
        case_id="F1_2025-01-01_2025-01-14",
        case_scoped_event_times=["2025-01-10T00:00:00Z"],
        prediction_cutoff_time=cut["prediction_cutoff_time"],
    )
    assert ok["case_evaluable"] is True
    bad = validate_case_scoped_cutoff(
        case_id="F1_2025-01-01_2025-01-14",
        case_scoped_event_times=["2025-01-20T00:00:00Z"],
        prediction_cutoff_time=cut["prediction_cutoff_time"],
    )
    assert bad["case_evaluable"] is False
    assert bad["future_case_input_event_policy"] == "CASE_NOT_EVALUABLE"


def test_readiness_blocks_without_gates():
    ready = build_production_readiness(
        fixture_regression_tests_pass=True,
        production_mock_integration_tests_pass=True,
        production_preflight_pass=True,
        production_callback_bundle_complete=True,
        invariants={
            "production_contract_checked_before_execution": True,
            "fixture_default_callback_use_count": 0,
            "simulated_production_result_count": 0,
            "schema_dispatch_is_atomic_target_authority": True,
            "search_holdout_scorers_isolated": True,
            "holdout_call_before_candidate_freeze_count": 0,
            "selected_candidate_manifest_locked": True,
            "evaluation_closure_manifest_locked": True,
            "parent_candidate_added_after_freeze_count": 0,
            "evaluation_closure_locked_before_parent_completion": True,
            "search_parent_completion_before_holdout": True,
            "parent_completion_search_complete_before_holdout": True,
            "holdout_call_before_search_closure_complete_count": 0,
            "holdout_set_matches_evaluation_closure": True,
            "candidate_identity_projection_locked": True,
            "parity_resolved_before_effect_finalization": True,
            "parent_metrics_use_finalized_effects_only": True,
            "pre_fallback_effect_used_for_metric_count": 0,
            "incremental_gain_by_fold_complete": True,
            "search_multi_event_only_strict_majority_applied": True,
            "holdout_multi_event_only_strict_majority_applied": True,
            "acceptance_policy_evaluator_active": True,
            "random_matched_subset_statistics_active": True,
            "production_dependency_closure_locked": True,
        },
    )
    assert ready["readiness_pass"] is True
    assert ready["primary_32_allowed"] is False
    require_readiness_for_production_command(ready)

    blocked = build_production_readiness(
        fixture_regression_tests_pass=False,
        production_mock_integration_tests_pass=True,
        production_preflight_pass=True,
        invariants={},
    )
    assert blocked["production_command_execution_allowed"] is False
    with pytest.raises(RuntimeError):
        require_readiness_for_production_command(blocked)


def test_acceptance_evaluator_supported_vs_limited():
    acceptance = load_yaml_policy(POLICY_DIR / "CF1S_ACCEPTANCE_POLICY_V1.yaml")
    supported_bundle = {
        "n_events": 2,
        "stable_model_sensitivity_valid": True,
        "stable_incremental_multi_event_pass": True,
        "stable_multi_event_only_effect_pass": False,
        "parent_family_complete": True,
        "finalized_effect_values": True,
        "model_effects_evaluated": True,
        "multi_event_candidate_id": "x",
    }
    limited_bundle = {
        "n_events": 2,
        "stable_model_sensitivity_valid": True,
        "stable_incremental_multi_event_pass": False,
        "stable_multi_event_only_effect_pass": False,
        "parent_family_complete": True,
        "finalized_effect_values": True,
        "model_effects_evaluated": True,
        "multi_event_candidate_id": "y",
    }
    acc = evaluate_selector_acceptance(
        [supported_bundle],
        acceptance_policy=acceptance,
        model_effects_evaluated=True,
        identity_and_determinism_ok=True,
        parity_ok=True,
    )
    assert acc["feasibility_status"] == SUPPORTED
    acc2 = evaluate_selector_acceptance(
        [limited_bundle],
        acceptance_policy=acceptance,
        model_effects_evaluated=True,
        identity_and_determinism_ok=True,
        parity_ok=True,
    )
    assert acc2["feasibility_status"] == LIMITED


def test_fixture_regression_still_not_evaluable(sample_events):
    edit, search, saliency, acceptance = _policies()
    out = run_cf1s_case_fixture(
        case_id="F_TEST",
        events=sample_events,
        edit_policy=edit,
        search_policy=search,
        saliency_policy=saliency,
        acceptance_policy=acceptance,
        cutoff_time="2025-01-01T05:00:00",
        model_effects_evaluated=False,
    )
    assert out["execution_mode"] == "FIXTURE_CONTRACT"
    assert out["primary_multi_event_edit_feasibility_status"] == "NOT_EVALUABLE"
    assert out["evaluation_closure"]["locked"] is True
