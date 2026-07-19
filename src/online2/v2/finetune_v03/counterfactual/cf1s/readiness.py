"""CF1S_PRODUCTION_READINESS artifact derivation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def build_production_readiness(
    *,
    fixture_regression_tests_pass: bool,
    production_mock_integration_tests_pass: bool,
    production_preflight_pass: bool,
    invariants: Mapping[str, Any],
    production_uses_shared_core_not_fixture_wrapper: bool = True,
    production_callback_bundle_complete: bool = False,
) -> Dict[str, Any]:
    """Derive readiness; production_command_execution_allowed only when PASS."""
    gates = {
        "fixture_regression_tests_pass": bool(fixture_regression_tests_pass),
        "production_mock_integration_tests_pass": bool(
            production_mock_integration_tests_pass
        ),
        "production_uses_shared_core_not_fixture_wrapper": bool(
            production_uses_shared_core_not_fixture_wrapper
        ),
        "production_callback_bundle_complete": bool(production_callback_bundle_complete),
        "production_contract_checked_before_execution": bool(
            invariants.get("production_contract_checked_before_execution", False)
        ),
        "fixture_default_callback_use_count": int(
            invariants.get("fixture_default_callback_use_count", 0)
        ),
        "simulated_production_result_count": int(
            invariants.get("simulated_production_result_count", 0)
        ),
        "simulated_result_is_fatal_in_production": True,
        "schema_dispatch_is_atomic_target_authority": bool(
            invariants.get("schema_dispatch_is_atomic_target_authority", False)
        ),
        "search_holdout_scorers_isolated": bool(
            invariants.get("search_holdout_scorers_isolated", False)
        ),
        "holdout_call_before_candidate_freeze_count": int(
            invariants.get("holdout_call_before_candidate_freeze_count", 0)
        ),
        "selected_candidate_manifest_locked": bool(
            invariants.get("selected_candidate_manifest_locked", False)
        ),
        "evaluation_closure_manifest_locked": bool(
            invariants.get("evaluation_closure_manifest_locked", False)
        ),
        "parent_candidate_added_after_freeze_count": int(
            invariants.get("parent_candidate_added_after_freeze_count", 0)
        ),
        "evaluation_closure_locked_before_parent_completion": bool(
            invariants.get("evaluation_closure_locked_before_parent_completion", False)
        ),
        "search_parent_completion_before_holdout": bool(
            invariants.get("search_parent_completion_before_holdout", False)
        ),
        "parent_completion_search_complete_before_holdout": bool(
            invariants.get("parent_completion_search_complete_before_holdout", False)
        ),
        "holdout_call_before_search_closure_complete_count": int(
            invariants.get("holdout_call_before_search_closure_complete_count", 0)
        ),
        "holdout_set_matches_evaluation_closure": bool(
            invariants.get("holdout_set_matches_evaluation_closure", False)
        ),
        "candidate_identity_projection_locked": bool(
            invariants.get("candidate_identity_projection_locked", False)
        ),
        "parity_resolved_before_effect_finalization": bool(
            invariants.get("parity_resolved_before_effect_finalization", False)
        ),
        "parent_metrics_use_finalized_effects_only": bool(
            invariants.get("parent_metrics_use_finalized_effects_only", False)
        ),
        "pre_fallback_effect_used_for_metric_count": int(
            invariants.get("pre_fallback_effect_used_for_metric_count", 0)
        ),
        "incremental_gain_by_fold_complete": bool(
            invariants.get("incremental_gain_by_fold_complete", False)
        ),
        "search_multi_event_only_strict_majority_applied": bool(
            invariants.get("search_multi_event_only_strict_majority_applied", False)
        ),
        "holdout_multi_event_only_strict_majority_applied": bool(
            invariants.get("holdout_multi_event_only_strict_majority_applied", False)
        ),
        "acceptance_policy_evaluator_active": bool(
            invariants.get("acceptance_policy_evaluator_active", False)
        ),
        "random_matched_subset_statistics_active": bool(
            invariants.get("random_matched_subset_statistics_active", False)
        ),
        "production_preflight_pass": bool(production_preflight_pass),
        "production_dependency_closure_locked": bool(
            invariants.get("production_dependency_closure_locked", False)
        ),
    }

    hard_fail = (
        not gates["fixture_regression_tests_pass"]
        or not gates["production_mock_integration_tests_pass"]
        or not gates["production_preflight_pass"]
        or not gates["production_uses_shared_core_not_fixture_wrapper"]
        or gates["fixture_default_callback_use_count"] != 0
        or gates["simulated_production_result_count"] != 0
        or gates["holdout_call_before_candidate_freeze_count"] != 0
        or gates["holdout_call_before_search_closure_complete_count"] != 0
        or gates["parent_candidate_added_after_freeze_count"] != 0
        or gates["pre_fallback_effect_used_for_metric_count"] != 0
        or not gates["search_parent_completion_before_holdout"]
        or not gates["parent_completion_search_complete_before_holdout"]
    )

    ready = not hard_fail and all(
        [
            gates["production_callback_bundle_complete"],
            gates["production_contract_checked_before_execution"],
            gates["schema_dispatch_is_atomic_target_authority"],
            gates["search_holdout_scorers_isolated"],
            gates["selected_candidate_manifest_locked"],
            gates["evaluation_closure_manifest_locked"],
            gates["holdout_set_matches_evaluation_closure"],
            gates["candidate_identity_projection_locked"],
            gates["parity_resolved_before_effect_finalization"],
            gates["parent_metrics_use_finalized_effects_only"],
            gates["acceptance_policy_evaluator_active"],
            gates["production_dependency_closure_locked"],
        ]
    )

    return {
        **gates,
        "readiness_scope": "READY_FOR_REAL_DEVELOPMENT_SMOKE" if ready else "NOT_READY",
        "production_semantic_results_validated": False,
        "production_command_execution_allowed": bool(ready),
        "primary_32_allowed": False,
        "problem20_allowed": False,
        "final_primary_lock": False,
        "readiness_pass": bool(ready),
    }


def require_readiness_for_production_command(
    readiness: Optional[Mapping[str, Any]],
) -> None:
    if readiness is None or not readiness.get("production_command_execution_allowed"):
        raise RuntimeError(
            "production command blocked: CF1S_PRODUCTION_READINESS not PASS"
        )
