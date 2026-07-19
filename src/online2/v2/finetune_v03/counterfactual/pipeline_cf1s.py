"""CF-1S pipeline: shared core + fixture/production wrappers (non-causal)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

from .cf1s.acceptance import aggregate_random_matched_subset
from .cf1s.code_lock import build_cf1s_lock_artifacts
from .cf1s.execution_context import (
    CF1SCallbackBundle,
    CF1SExecutionContext,
    ProductionContractViolation,
    fixture_identity_runner,
    fixture_parity_runner,
    fixture_pass_through,
    fixture_stage_a_noop,
)
from .cf1s.identity import (
    forced_identity_pass,
    pure_repeat_noise_floors,
    verify_deterministic_contract,
)
from .cf1s.locus_universe import build_common_eligible_universe
from .cf1s.schema_targets import fixture_adjacency_targets
from .cf1s.selector_runtime import run_selector_realization
from .cf1s.selectors import (
    SELECTOR_RANDOM,
    SELECTOR_RECENCY,
    SELECTOR_SALIENCY,
)
from .cf1s.status import (
    control_like_authority,
    primary_feasibility_from_selectors,
    saliency_efficiency_status,
)


EffectFn = Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]
ROOT_DEFAULT = Path(__file__).resolve().parents[5]


def load_yaml_policy(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    text = p.read_text(encoding="utf-8")
    out = dict(data or {})
    out["_policy_path"] = str(p)
    out["_policy_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out


def default_adjacency_targets(observed_raw: Any, schema_kind: str) -> List[Any]:
    """Fixture-only adjacency helper. Production must not call this."""
    return fixture_adjacency_targets(observed_raw, schema_kind)


def _zero_effect_fn(edits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Fixture placeholder — marks effects as NOT model-evaluated."""
    return {
        "delta_risk_search_aggregate": 0.0,
        "delta_r_search": 0.0,
        "delta_risk_search_by_fold": [0.0, 0.0],
        "delta_risk_holdout_by_fold": [0.0],
        "delta_risk_holdout_aggregate": 0.0,
        "model_effects_evaluated": False,
        "atomic_parent_deltas_search": [0.0] * len(edits),
        "atomic_parent_deltas_holdout": [0.0] * len(edits),
        "parent_completion_search_complete": False,
        "parent_completion_holdout_complete": False,
        "simulated": True,
    }


def _search_only_from_combined(effect_fn: EffectFn) -> EffectFn:
    def _fn(edits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        out = dict(effect_fn(edits))
        out.pop("delta_risk_holdout_by_fold", None)
        out.pop("delta_risk_holdout_aggregate", None)
        out.pop("atomic_parent_deltas_holdout", None)
        return out

    return _fn


def _holdout_only_from_combined(effect_fn: EffectFn) -> EffectFn:
    def _fn(edits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        out = dict(effect_fn(edits))
        # Keep holdout fields; search fields may remain for diagnostics but
        # selector_runtime reads holdout_* only from this callback.
        return out

    return _fn


def build_fixture_callback_bundle(
    *,
    effect_fn: Optional[EffectFn] = None,
    counters_holder: Optional[CF1SExecutionContext] = None,
) -> CF1SCallbackBundle:
    eff = effect_fn or _zero_effect_fn

    def atomic_target_fn(locus: Mapping[str, Any], record: Any) -> List[Any]:
        if counters_holder is not None:
            counters_holder.counters.fixture_default_callback_use_count += 1
            counters_holder.counters.default_adjacency_target_use_count += 1
        return default_adjacency_targets(
            locus.get("observed_raw"), getattr(record, "schema_kind", "boolean")
        )

    def retokenize_fn(edit: Mapping[str, Any]) -> Dict[str, Any]:
        n = int(edit.get("original_token_length") or 1)
        return {
            "ok": True,
            "original_token_length": n,
            "retokenized_token_length": n,
            "tokens": ["T"] * n,
            "retokenized_mg_bundle": ["T"] * n,
            "simulated": True,
        }

    return CF1SCallbackBundle(
        atomic_target_fn=atomic_target_fn,
        ground_fn=fixture_pass_through,
        invert_fn=fixture_pass_through,
        gate0_fn=fixture_pass_through,
        gate4_fn=fixture_pass_through,
        retokenize_fn=retokenize_fn,
        stage_a_reencode_fn=fixture_stage_a_noop,
        search_effect_fn=_search_only_from_combined(eff),
        holdout_effect_fn=_holdout_only_from_combined(eff),
        identity_runner=fixture_identity_runner,
        parity_runner=fixture_parity_runner,
    )


def run_cf1s_case_core(ctx: CF1SExecutionContext) -> Dict[str, Any]:
    """Shared core. Production must enter only via production wrapper + contract."""
    if str(ctx.provenance_mode).upper() == "PRODUCTION":
        ctx.assert_production_contract()

    universe = build_common_eligible_universe(
        ctx.events, edit_policy=ctx.edit_policy, arm=ctx.arm
    )
    random_seeds = list((ctx.search_policy.get("random") or {}).get("seed_list") or [11])

    identity_payload = dict(ctx.callbacks.identity_runner())
    if str(ctx.provenance_mode).upper() == "PRODUCTION" and identity_payload.get(
        "simulated"
    ):
        raise ProductionContractViolation("simulated identity in production")

    det_flags = identity_payload.get("deterministic_flags") or {
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
    det = verify_deterministic_contract(det_flags)
    risks = list(identity_payload.get("risks") or [0.1, 0.1, 0.1])
    logits = list(identity_payload.get("logits") or [0.0, 0.0, 0.0])
    noise = pure_repeat_noise_floors(risks, logits)
    forced = forced_identity_pass(
        abs_risk_delta=float(identity_payload.get("forced_risk_delta") or 0.0),
        abs_logit_delta=float(identity_payload.get("forced_logit_delta") or 0.0),
        risk_tol=noise["effective_risk_identity_tolerance"],
        logit_tol=noise["effective_logit_identity_tolerance"],
        forced_path_executed=True,
    )
    id_ok = bool(
        ctx.model_effects_evaluated
        and det["deterministic_contract_verified"]
        and forced["forced_identity_pass"]
    )

    selector_results: Dict[str, Any] = {}
    invariant_acc: Dict[str, Any] = {}

    for selector in (SELECTOR_SALIENCY, SELECTOR_RECENCY):
        # Fresh freeze flags per selector realization.
        ctx.candidate_freeze_complete = False
        ctx.parent_completion_search_complete = False
        ctx.evaluation_closure_locked = False
        ctx.selected_manifest_locked = False
        result = run_selector_realization(
            ctx=ctx,
            selector=selector,
            universe=universe,
            random_seed=None,
            identity_and_determinism_ok=id_ok if ctx.model_effects_evaluated else False,
        )
        invariant_acc = {**invariant_acc, **(result.get("invariants") or {})}
        selector_results[selector] = result

    random_realizations = []
    sal_eval = int(selector_results[SELECTOR_SALIENCY]["evaluable_composite_count"])
    rec_eval = int(selector_results[SELECTOR_RECENCY]["evaluable_composite_count"])
    for seed in random_seeds:
        ctx.candidate_freeze_complete = False
        ctx.parent_completion_search_complete = False
        ctx.evaluation_closure_locked = False
        ctx.selected_manifest_locked = False
        result = run_selector_realization(
            ctx=ctx,
            selector=SELECTOR_RANDOM,
            universe=universe,
            random_seed=int(seed),
            identity_and_determinism_ok=id_ok if ctx.model_effects_evaluated else False,
        )
        matched = min(sal_eval, rec_eval, int(result["evaluable_composite_count"]))
        result["matched_evaluable_count"] = matched
        random_realizations.append(result)

    matched_budget = min(
        sal_eval,
        rec_eval,
        min((r["evaluable_composite_count"] for r in random_realizations), default=0),
    )
    random_stats = aggregate_random_matched_subset(
        random_realizations, matched_budget=matched_budget
    )

    if random_realizations:
        if not ctx.model_effects_evaluated:
            random_status = "NOT_EVALUABLE"
            random_metric = 0.0
        else:
            from .cf1s.status import derive_selector_feasibility_status

            stables = [
                r.get("n_stable_incremental_multi_event", 0)
                + r.get("n_stable_multi_event_only", 0)
                for r in random_realizations
            ]
            random_metric = float(sorted(stables)[len(stables) // 2])
            random_status = derive_selector_feasibility_status(
                model_effects_evaluated=True,
                n_evaluable_composites=max(
                    r["evaluable_composite_count"] for r in random_realizations
                ),
                n_stable_model_sensitivity=int(random_metric),
                identity_and_determinism_ok=id_ok,
                n_stable_incremental_multi_event=int(
                    sorted(
                        r.get("n_stable_incremental_multi_event", 0)
                        for r in random_realizations
                    )[len(random_realizations) // 2]
                ),
                n_stable_multi_event_only=int(
                    sorted(
                        r.get("n_stable_multi_event_only", 0)
                        for r in random_realizations
                    )[len(random_realizations) // 2]
                ),
                n_stable_multi_event=int(
                    sorted(
                        r.get("n_stable_multi_event", 0) for r in random_realizations
                    )[len(random_realizations) // 2]
                ),
                n_stable_single_event=int(
                    sorted(
                        r.get("n_stable_single_event", 0) for r in random_realizations
                    )[len(random_realizations) // 2]
                ),
            )
        selector_results[SELECTOR_RANDOM] = {
            "realizations": [
                {
                    "seed": r["random_seed"],
                    "selected_count": r["selected_count"],
                    "selected_pool_sha256": r["selected_pool_sha256"],
                    "evaluable_composite_count": r["evaluable_composite_count"],
                    "matched_evaluable_count": r.get("matched_evaluable_count"),
                    "n_stable_model_sensitivity": r["n_stable_model_sensitivity"],
                    "feasibility_status": r["feasibility_status"],
                    "funnel_accounting_status": r["funnel_accounting_status"],
                    "beam": r["beam"],
                }
                for r in random_realizations
            ],
            "feasibility_status": random_status,
            "n_stable_model_sensitivity": int(random_metric)
            if ctx.model_effects_evaluated
            else 0,
            "evaluable_composite_count": int(
                sorted(r["evaluable_composite_count"] for r in random_realizations)[
                    len(random_realizations) // 2
                ]
            ),
            "funnel": random_realizations[0]["funnel"],
            "funnel_accounting_status": random_realizations[0]["funnel_accounting_status"],
            "bundles": random_realizations[0]["bundles"],
            "parity": random_realizations[0]["parity"],
            "allocated": random_realizations[0]["allocated"],
            "ranked": random_realizations[0]["ranked"],
            "beam": random_realizations[0]["beam"],
            "selected_count": random_realizations[0]["selected_count"],
            "selected_pool_sha256": random_realizations[0]["selected_pool_sha256"],
            "acceptance": random_realizations[0].get("acceptance"),
            "selection_manifest": random_realizations[0].get("selection_manifest"),
            "evaluation_closure": random_realizations[0].get("evaluation_closure"),
            "random_matched_subset": random_stats,
        }

    by_selector = {
        name: row["feasibility_status"] for name, row in selector_results.items()
    }
    primary = primary_feasibility_from_selectors(
        by_selector,
        saliency_evaluable=any(
            bool(l.get("saliency_evaluable", True)) for l in (universe.get("loci") or [])
        ),
    )
    arm_auth = control_like_authority(
        primary_control_like_status=primary[
            "primary_multi_event_edit_feasibility_status"
        ],
        observational_sensitivity_status="NOT_EVALUABLE",
    )

    sal_metric = float(
        selector_results[SELECTOR_SALIENCY].get("n_stable_incremental_multi_event", 0)
        + selector_results[SELECTOR_SALIENCY].get("n_stable_multi_event_only", 0)
    )
    rnd_metric = float(selector_results[SELECTOR_RANDOM]["n_stable_model_sensitivity"])
    rec_metric = float(
        selector_results[SELECTOR_RECENCY].get("n_stable_incremental_multi_event", 0)
        + selector_results[SELECTOR_RECENCY].get("n_stable_multi_event_only", 0)
    )
    efficiency = saliency_efficiency_status(
        saliency_evaluable=ctx.model_effects_evaluated
        and any(
            bool(l.get("saliency_evaluable", True)) for l in (universe.get("loci") or [])
        ),
        saliency_metric=sal_metric,
        random_metric=rnd_metric,
        recency_metric=rec_metric,
    )

    primary_funnel = selector_results[SELECTOR_SALIENCY]["funnel"]
    lock = build_cf1s_lock_artifacts(
        ROOT_DEFAULT,
        fixture_contract_snapshot=str(ctx.provenance_mode).upper() != "PRODUCTION",
        final_primary_lock=False,
    )
    sal_closure = selector_results[SELECTOR_SALIENCY].get("evaluation_closure") or {}
    sal_sel = selector_results[SELECTOR_SALIENCY].get("selection_manifest") or {}

    if str(ctx.provenance_mode).upper() == "PRODUCTION" and ctx.counters.simulated_result_count:
        raise ProductionContractViolation("simulated_result_count>0 in production")

    return {
        "case_id": ctx.case_id,
        "arm": ctx.arm,
        "execution_mode": (
            "FIXTURE_CONTRACT"
            if str(ctx.provenance_mode).upper() == "FIXTURE"
            else "PRODUCTION"
        ),
        "model_effects_evaluated": ctx.model_effects_evaluated,
        "labels": [
            "MULTI_EVENT_MODEL_SENSITIVITY",
            "SEQUENCE_EDIT_FEASIBILITY",
            "NON_CAUSAL",
            "NOT_AN_ACTION",
            "NOT_RECOMMENDATION",
        ],
        "action_authorization": False,
        "recommendation_authorization": False,
        "cf1b_authorization": False,
        "universe": {
            "common_eligible_locus_pool_sha256": universe[
                "common_eligible_locus_pool_sha256"
            ],
            "common_eligible_locus_count": universe["common_eligible_locus_count"],
            "raw_mg_duplicate_detected_count": universe["counts"].get(
                "raw_mg_duplicate_detected_count", 0
            ),
            "raw_mg_duplicate_remaining_count": universe["counts"].get(
                "raw_mg_duplicate_remaining_count", 0
            ),
            "raw_mg_duplicate_locus_count": universe["counts"].get(
                "raw_mg_duplicate_remaining_count", 0
            ),
            "feature_group_count": len(universe.get("feature_groups") or {}),
        },
        "selector_results": {
            name: {
                "feasibility_status": row["feasibility_status"],
                "selected_count": row.get("selected_count"),
                "selected_pool_sha256": row.get("selected_pool_sha256"),
                "evaluable_composite_count": row.get("evaluable_composite_count"),
                "n_stable_model_sensitivity": row.get("n_stable_model_sensitivity"),
                "n_stable_incremental_multi_event": row.get(
                    "n_stable_incremental_multi_event"
                ),
                "n_stable_multi_event_only": row.get("n_stable_multi_event_only"),
                "funnel_accounting_status": row.get("funnel_accounting_status"),
                "beam": row.get("beam"),
                "parity": row.get("parity"),
                "realizations": row.get("realizations"),
                "acceptance": row.get("acceptance"),
                "selection_manifest": {
                    k: v
                    for k, v in (row.get("selection_manifest") or {}).items()
                    if k != "jsonl"
                },
                "evaluation_closure": {
                    k: v
                    for k, v in (row.get("evaluation_closure") or {}).items()
                    if k != "jsonl"
                },
                "random_matched_subset": row.get("random_matched_subset"),
            }
            for name, row in selector_results.items()
        },
        "bundles": selector_results[SELECTOR_SALIENCY]["bundles"],
        "funnel": primary_funnel,
        "funnel_accounting_status": selector_results[SELECTOR_SALIENCY][
            "funnel_accounting_status"
        ],
        "parity": selector_results[SELECTOR_SALIENCY]["parity"],
        "selection_manifest": {k: v for k, v in sal_sel.items() if k != "jsonl"},
        "evaluation_closure": {k: v for k, v in sal_closure.items() if k != "jsonl"},
        "deterministic_contract": {
            **det,
            "deterministic_contract_simulated": not ctx.model_effects_evaluated,
        },
        "identity": {
            **noise,
            **forced,
            "identity_simulated": bool(identity_payload.get("simulated", False)),
        },
        **primary,
        **arm_auth,
        "saliency_guidance_efficiency_status": efficiency,
        "policy_hashes": {
            "edit": ctx.edit_policy.get("_policy_sha256"),
            "search": ctx.search_policy.get("_policy_sha256"),
            "saliency": ctx.saliency_policy.get("_policy_sha256"),
            "acceptance": ctx.acceptance_policy.get("_policy_sha256"),
        },
        "lock": {
            "fixture_contract_snapshot": lock["fixture_contract_snapshot"],
            "final_primary_lock": lock["final_primary_lock"],
            "CODE_TREE_SHA256": lock["CODE_TREE_SHA256"],
            "POLICY_TREE_SHA256": lock["POLICY_TREE_SHA256"],
            "CF1S_LOCK_TREE_SHA256": lock["CF1S_LOCK_TREE_SHA256"],
        },
        "invariants": {
            "selector_changes_ranking_only": True,
            "mg_dedup_precedes_selector": True,
            "feature_profile_grouping_precedes_selector_truncation": True,
            "saliency_stability_application": str(
                (ctx.saliency_policy.get("saliency_selector") or {}).get(
                    "stability_application", "RANKING_TIER"
                )
            ),
            "primary_feasibility_selector": SELECTOR_SALIENCY,
            "baseline_success_can_rescue_primary_feasibility": False,
            "observational_can_rescue_control_like_status": False,
            "per_selector_independent_evaluation": True,
            "beam_search_enabled": True,
            "production_contract_checked_before_execution": ctx.production_contract_checked_before_candidate_generation,
            "fixture_default_callback_use_count": ctx.counters.fixture_default_callback_use_count
            if str(ctx.provenance_mode).upper() == "PRODUCTION"
            else 0,
            "simulated_production_result_count": ctx.counters.simulated_result_count
            if str(ctx.provenance_mode).upper() == "PRODUCTION"
            else 0,
            "acceptance_policy_evaluator_active": True,
            "random_matched_subset_statistics_active": True,
            **invariant_acc,
        },
        "random_matched_subset": random_stats,
    }


def run_cf1s_case_fixture(
    *,
    case_id: str,
    events: Sequence[Mapping[str, Any]],
    edit_policy: Mapping[str, Any],
    search_policy: Mapping[str, Any],
    saliency_policy: Mapping[str, Any],
    acceptance_policy: Mapping[str, Any],
    arm: str = "CONTROL_LIKE",
    cutoff_time: Optional[str] = None,
    effect_fn: Optional[EffectFn] = None,
    model_effects_evaluated: bool = False,
    deterministic_flags: Optional[Mapping[str, Any]] = None,
    identity_repeats: Optional[Mapping[str, Any]] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fixture/contract smoke. Does NOT claim production model evaluation."""
    _ = root
    model_effects_evaluated = bool(model_effects_evaluated and effect_fn is not None)
    ctx = CF1SExecutionContext(
        case_id=case_id,
        events=events,
        edit_policy=edit_policy,
        search_policy=search_policy,
        saliency_policy=saliency_policy,
        acceptance_policy=acceptance_policy,
        callbacks=CF1SCallbackBundle(
            atomic_target_fn=lambda loc, record: default_adjacency_targets(
                loc.get("observed_raw"), getattr(record, "schema_kind", "boolean")
            ),
            ground_fn=fixture_pass_through,
            invert_fn=fixture_pass_through,
            gate0_fn=fixture_pass_through,
            gate4_fn=fixture_pass_through,
            retokenize_fn=lambda edit: {
                "ok": True,
                "original_token_length": int(edit.get("original_token_length") or 1),
                "retokenized_token_length": int(edit.get("original_token_length") or 1),
                "tokens": ["T"] * int(edit.get("original_token_length") or 1),
                "retokenized_mg_bundle": ["T"]
                * int(edit.get("original_token_length") or 1),
                "simulated": True,
            },
            stage_a_reencode_fn=fixture_stage_a_noop,
            search_effect_fn=_search_only_from_combined(effect_fn or _zero_effect_fn),
            holdout_effect_fn=_holdout_only_from_combined(effect_fn or _zero_effect_fn),
            identity_runner=lambda: {
                **(identity_repeats or fixture_identity_runner()),
                "deterministic_flags": deterministic_flags,
                "simulated": not model_effects_evaluated,
            },
            parity_runner=fixture_parity_runner,
        ),
        provenance_mode="FIXTURE",
        uses_fixture_defaults=True,
        arm=arm,
        cutoff_time=cutoff_time,
        model_effects_evaluated=model_effects_evaluated,
    )
    out = run_cf1s_case_core(ctx)
    if not model_effects_evaluated:
        out["execution_mode"] = "FIXTURE_CONTRACT"
    else:
        out["execution_mode"] = "MODEL_EVAL"
    return out


def run_cf1s_case_production(
    *,
    case_id: str,
    events: Sequence[Mapping[str, Any]],
    edit_policy: Mapping[str, Any],
    search_policy: Mapping[str, Any],
    saliency_policy: Mapping[str, Any],
    acceptance_policy: Mapping[str, Any],
    callbacks: CF1SCallbackBundle,
    arm: str = "CONTROL_LIKE",
    cutoff_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Production wrapper: never calls fixture wrapper; requires full callback bundle."""
    ctx = CF1SExecutionContext(
        case_id=case_id,
        events=events,
        edit_policy=edit_policy,
        search_policy=search_policy,
        saliency_policy=saliency_policy,
        acceptance_policy=acceptance_policy,
        callbacks=callbacks,
        provenance_mode="PRODUCTION",
        uses_fixture_defaults=False,
        arm=arm,
        cutoff_time=cutoff_time,
        model_effects_evaluated=True,
    )
    return run_cf1s_case_core(ctx)


def lock_manifest(
    *,
    root: Path,
    parent_cf0r_correction_sha256: str,
    fixture_contract_snapshot: bool = True,
    final_primary_lock: bool = False,
) -> Dict[str, Any]:
    lock = build_cf1s_lock_artifacts(
        root,
        fixture_contract_snapshot=fixture_contract_snapshot,
        final_primary_lock=final_primary_lock,
    )
    return {
        **lock,
        "parent_cf0r_correction_sha256": parent_cf0r_correction_sha256,
        "development_cases": [
            "F420458_2025-02-16_2025-03-01",
            "F385790_2025-03-20_2025-04-02",
            "F279269_2024-10-17_2024-10-30",
        ],
        "development_case_count": 3,
        "primary_internal_validation_case_count": 32,
        "development_cases_excluded_from_primary": True,
        "primary_32_policy_update_count": 0,
        "primary_32_code_update_count": 0,
        "problem20_policy_update_count": 0,
        "problem20_code_update_count": 0,
    }
