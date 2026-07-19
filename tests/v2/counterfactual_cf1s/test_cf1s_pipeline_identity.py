from __future__ import annotations

from pathlib import Path

from src.online2.v2.finetune_v03.counterfactual.cf1s.identity import (
    forced_identity_pass,
    pure_repeat_noise_floors,
    verify_deterministic_contract,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.cf1s_funnel_accounting import (
    empty_cf1s_funnel_counts,
    funnel_accounting_status,
    validate_cf1s_funnel_monotonicity,
)
from src.online2.v2.finetune_v03.counterfactual.pipeline_cf1s import (
    load_yaml_policy,
    lock_manifest,
    run_cf1s_case_fixture,
)


ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = ROOT / "conf/m1/cf1s_policies"


def test_deterministic_contract_requested_vs_verified():
    bad = verify_deterministic_contract(
        {
            "model_eval_mode": True,
            "inference_mode": True,
            "dropout_disabled": True,
            "fixed_rng_seeds": True,
            "deterministic_algorithms_requested": True,
            "deterministic_algorithms_enabled": False,
            "deterministic_violation_count": 0,
            "deterministic_warning_count": 0,
            "batch_order_fixed": True,
            "mixed_precision_policy": "LOCKED",
            "checkpoint_hash_verified": True,
        }
    )
    assert bad["deterministic_algorithms_requested"] is True
    assert bad["deterministic_contract_verified"] is False

    good = verify_deterministic_contract(
        {
            "model_eval_mode": True,
            "inference_mode": True,
            "dropout_disabled": True,
            "fixed_rng_seeds": True,
            "deterministic_algorithms_requested": True,
            "deterministic_algorithms_enabled": True,
            "deterministic_violation_count": 0,
            "deterministic_warning_count": 0,
            "batch_order_fixed": True,
            "mixed_precision_policy": "LOCKED",
            "checkpoint_hash_verified": True,
        }
    )
    assert good["deterministic_contract_verified"] is True


def test_risk_logit_noise_floors_separated():
    noise = pure_repeat_noise_floors([0.1, 0.100001, 0.1], [1.0, 1.0, 1.0])
    assert noise["pure_repeat_risk_noise_floor"] > 0
    assert noise["pure_repeat_logit_noise_floor"] == 0.0
    forced = forced_identity_pass(
        abs_risk_delta=0.0,
        abs_logit_delta=0.0,
        risk_tol=noise["effective_risk_identity_tolerance"],
        logit_tol=noise["effective_logit_identity_tolerance"],
        forced_path_executed=True,
    )
    assert forced["forced_identity_pass"] is True


def test_funnel_failure_counts_separated():
    funnel = empty_cf1s_funnel_counts()
    funnel.update(
        {
            "n_common_eligible_loci": 10,
            "n_selector_ranked": 10,
            "n_round_robin_selected": 8,
            "n_atomic_candidates": 8,
            "n_grounding_valid": 7,
            "n_inversion_valid": 7,
            "n_gate0_pass": 6,
            "n_gate4_pass": 6,
            "grounding_failure_count": 1,
            "inversion_failure_count": 0,
            "gate0_failure_count": 1,
            "gate4_failure_count": 0,
        }
    )
    ok, errs = validate_cf1s_funnel_monotonicity(funnel)
    assert ok, errs
    assert funnel_accounting_status(funnel) == "PASS"


def test_pipeline_fixture_end_to_end(sample_events):
    edit = load_yaml_policy(POLICY_DIR / "CF1S_EDIT_POLICY_V1.yaml")
    search = load_yaml_policy(POLICY_DIR / "CF1S_SEARCH_POLICY_V1.yaml")
    saliency = load_yaml_policy(POLICY_DIR / "CF1S_SALIENCY_POLICY_V1.yaml")
    acceptance = load_yaml_policy(POLICY_DIR / "CF1S_ACCEPTANCE_POLICY_V1.yaml")
    out = run_cf1s_case_fixture(
        case_id="F420458_2025-02-16_2025-03-01",
        events=sample_events,
        edit_policy=edit,
        search_policy=search,
        saliency_policy=saliency,
        acceptance_policy=acceptance,
        cutoff_time="2025-01-01T05:00:00",
    )
    assert out["action_authorization"] is False
    assert out["recommendation_authorization"] is False
    assert out["cf1b_authorization"] is False
    assert out["execution_mode"] == "FIXTURE_CONTRACT"
    assert out["model_effects_evaluated"] is False
    assert out["universe"]["raw_mg_duplicate_detected_count"] > 0
    assert out["universe"]["raw_mg_duplicate_remaining_count"] == 0
    assert out["invariants"]["saliency_stability_application"] == "RANKING_TIER"
    assert out["invariants"]["primary_feasibility_selector"] == "SALIENCY_TOP_K"
    assert out["invariants"]["beam_search_enabled"] is True
    assert out["funnel"]["n_single_screened"] > 0
    assert out["funnel"]["n_multi_event_composites"] <= 100
    assert out["funnel"]["n_two_event_composites"] <= 32
    assert out["funnel"]["n_three_event_composites"] <= 24
    assert out["funnel_accounting_status"] == "PASS"
    # Fixture without model effects must be NOT_EVALUABLE, not LIMITED.
    assert out["primary_multi_event_edit_feasibility_status"] == "NOT_EVALUABLE"
    assert out["feasibility_status_by_selector"]["SALIENCY_TOP_K"] == "NOT_EVALUABLE"
    assert out["feasibility_status_by_selector"]["RANDOM_EDITABLE_LOCUS"] == "NOT_EVALUABLE"
    assert out["feasibility_status_by_selector"]["RECENCY_TOP_K"] == "NOT_EVALUABLE"
    assert out["saliency_guidance_efficiency_status"] == "NOT_EVALUABLE"
    assert out["parity"]["reencode_parity_audit_status"] == "NOT_EVALUABLE"
    assert out["lock"]["final_primary_lock"] is False
    assert out["lock"]["fixture_contract_snapshot"] is True
    assert out["baseline_success_can_rescue_primary_feasibility"] is False
    assert "realizations" in out["selector_results"]["RANDOM_EDITABLE_LOCUS"]


def test_lock_manifest_fixture_not_primary_lock():
    man = lock_manifest(
        root=ROOT,
        parent_cf0r_correction_sha256="c",
        fixture_contract_snapshot=True,
        final_primary_lock=False,
    )
    assert man["fixture_contract_snapshot"] is True
    assert man["final_primary_lock"] is False
    assert man["policy_locked_before_primary_32"] is False
    assert man["code_locked_before_primary_32"] is False
    assert "CODE_TREE_SHA256" in man
    assert man["development_case_count"] == 3
    assert man["primary_internal_validation_case_count"] == 32
