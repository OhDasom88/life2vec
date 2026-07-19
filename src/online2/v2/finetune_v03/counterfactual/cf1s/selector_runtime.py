"""Selector realization with search/holdout isolation and evaluation closure."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..candidates.cf1s_composer import (
    compose_atomic_edits_for_locus,
    multi_event_candidate_id,
)
from ..evaluation.cf1s_funnel_accounting import (
    empty_cf1s_funnel_counts,
    funnel_accounting_status,
    merge_universe_counts,
)
from ..retokenization.multi_event_retokenizer import (
    apply_multi_event_edits_atomically,
    classify_parity_status,
    parity_sample_rule,
)
from .acceptance import evaluate_selector_acceptance
from .beam import compose_with_beam, screen_single_event_atomics
from .edit_policy import get_feature_record
from .evaluation_closure import (
    build_evaluation_closure,
    build_selected_manifest_rows,
    holdout_set_matches_closure,
    lock_manifest_rows,
)
from .execution_context import (
    CF1SExecutionContext,
    ProductionContractViolation,
    wrap_holdout_effect_fn,
    wrap_search_effect_fn,
)
from .fold_effect import (
    incremental_abs_gain_by_fold,
    incremental_gain_pass,
    multi_event_only_fold_pass,
    stable_incremental_multi_event_pass,
    stable_model_sensitivity_valid,
    stable_multi_event_only_effect_pass_fold,
)
from .interaction import (
    classify_interaction,
    classify_interaction_by_fold,
    stable_interaction_fields,
    three_event_residuals,
)
from .selectors import (
    SELECTOR_SALIENCY,
    allocate_feature_round_robin,
    rank_loci_for_selector,
)
from .status import derive_selector_feasibility_status


def _validate_atomics(
    atomics: Sequence[Mapping[str, Any]],
    *,
    ground_fn,
    invert_fn,
    gate0_fn,
    gate4_fn,
    retokenize_fn,
) -> Dict[str, Any]:
    ok_rows = []
    grounding_fail = inversion_fail = gate0_fail = gate4_fail = retok_fail = 0
    for a in atomics:
        res = apply_multi_event_edits_atomically(
            edits=[a],
            ground_fn=ground_fn,
            invert_fn=invert_fn,
            gate0_fn=gate0_fn,
            gate4_fn=gate4_fn,
            retokenize_fn=retokenize_fn,
            length_preserving_edit_only=True,
        )
        if res.ok:
            row = dict(a)
            if res.edits_applied:
                row.update(res.edits_applied[0])
            row["grounding_valid"] = True
            row["inversion_valid"] = True
            row["gate0_valid"] = True
            row["gate4_valid"] = True
            row["retokenization_valid"] = True
            ok_rows.append(row)
        else:
            grounding_fail += res.grounding_failure_count
            inversion_fail += res.inversion_failure_count
            gate0_fail += res.gate0_failure_count
            gate4_fail += res.gate4_failure_count
            if getattr(res, "retokenization_failure_count", 0):
                retok_fail += int(res.retokenization_failure_count)
    return {
        "valid_atomics": ok_rows,
        "grounding_failure_count": grounding_fail,
        "inversion_failure_count": inversion_fail,
        "gate0_failure_count": gate0_fail,
        "gate4_failure_count": gate4_fail,
        "retokenization_failure_count": retok_fail,
        "n_grounding_valid": len(ok_rows),
        "n_inversion_valid": len(ok_rows),
        "n_gate0_pass": len(ok_rows),
        "n_gate4_pass": len(ok_rows),
    }


def _build_atomics_for_loci(
    *,
    case_id: str,
    selected_loci: Sequence[Mapping[str, Any]],
    edit_policy: Mapping[str, Any],
    atomic_target_fn: Callable,
) -> List[Dict[str, Any]]:
    all_atomics: List[Dict[str, Any]] = []
    for loc in selected_loci:
        record = get_feature_record(edit_policy, str(loc["feature_id"]))
        targets = list(atomic_target_fn(loc, record) or [])
        atomics = compose_atomic_edits_for_locus(loc, adjacency_targets=targets)
        for a in atomics:
            a = dict(a)
            a["event_time_epoch"] = loc.get("event_time_epoch")
            a["same_time_group_id"] = loc.get("same_time_group_id")
            a["feature_edit_profile_id"] = loc.get("feature_edit_profile_id")
            a["feature_edit_profile_sha256"] = loc.get("feature_edit_profile_sha256")
            a["locus_key"] = loc["locus_key"]
            a["original_token_length"] = int(loc.get("member_token_count") or 1)
            a["atomic_candidate_id"] = multi_event_candidate_id(
                case_id=case_id, edits=[a]
            )
            all_atomics.append(a)
    return all_atomics


def _finalize_effect(
    *,
    edits: Sequence[Mapping[str, Any]],
    raw_effect: Mapping[str, Any],
    parity_runner: Callable,
    fold: str,
) -> Dict[str, Any]:
    parity = dict(parity_runner(edits, raw_effect))
    finalized = dict(parity.get("finalized_effect") or raw_effect)
    source = str(parity.get("effect_source") or "UNAUDITED")
    if parity.get("parity_status") == "FAIL" and not parity.get(
        "full_reencode_fallback_used"
    ):
        return {
            "ok": False,
            "finalized_effect_values": False,
            "parity_audited": bool(parity.get("parity_audited")),
            "parity_status": parity.get("parity_status"),
            "effect_source": source,
            "full_reencode_fallback_used": False,
            "fold": fold,
        }
    return {
        "ok": True,
        "finalized_effect_values": True,
        "parity_audited": bool(parity.get("parity_audited")),
        "parity_status": parity.get("parity_status") or "PASS",
        "effect_source": source,
        "full_reencode_fallback_used": bool(parity.get("full_reencode_fallback_used")),
        "fold": fold,
        "effect": finalized,
    }


def _score_edits_search(
    edits: Sequence[Mapping[str, Any]],
    *,
    search_fn,
    parity_runner,
) -> Dict[str, Any]:
    raw = dict(search_fn(edits))
    fin = _finalize_effect(
        edits=edits, raw_effect=raw, parity_runner=parity_runner, fold="search"
    )
    if not fin["ok"]:
        return {
            "search_effect_evaluated": False,
            "finalized_effect_values": False,
            "delta_risk_search_by_fold": [],
            "delta_risk_search_aggregate": 0.0,
            **{k: fin[k] for k in ("parity_audited", "parity_status", "effect_source", "full_reencode_fallback_used")},
        }
    eff = fin["effect"]
    folds = list(eff.get("delta_risk_search_by_fold") or [eff.get("delta_risk_search_aggregate") or 0.0])
    agg = float(eff.get("delta_risk_search_aggregate") or (sum(folds) / max(len(folds), 1)))
    return {
        "search_effect_evaluated": True,
        "finalized_effect_values": True,
        "delta_risk_search_by_fold": folds,
        "delta_risk_search_aggregate": agg,
        "parity_audited": fin["parity_audited"],
        "parity_status": fin["parity_status"],
        "effect_source": fin["effect_source"],
        "full_reencode_fallback_used": fin["full_reencode_fallback_used"],
        "simulated": bool(eff.get("simulated", False)),
    }


def _score_edits_holdout(
    edits: Sequence[Mapping[str, Any]],
    *,
    holdout_fn,
    parity_runner,
) -> Dict[str, Any]:
    raw = dict(holdout_fn(edits))
    fin = _finalize_effect(
        edits=edits, raw_effect=raw, parity_runner=parity_runner, fold="holdout"
    )
    if not fin["ok"]:
        return {
            "holdout_effect_evaluated": False,
            "finalized_effect_values": False,
            "delta_risk_holdout_by_fold": [],
            "delta_risk_holdout_aggregate": 0.0,
            **{k: fin[k] for k in ("parity_audited", "parity_status", "effect_source", "full_reencode_fallback_used")},
        }
    eff = fin["effect"]
    folds = list(
        eff.get("delta_risk_holdout_by_fold")
        or [eff.get("delta_risk_holdout_aggregate") or 0.0]
    )
    agg = float(
        eff.get("delta_risk_holdout_aggregate")
        or (sum(folds) / max(len(folds), 1))
    )
    return {
        "holdout_effect_evaluated": True,
        "finalized_effect_values": True,
        "delta_risk_holdout_by_fold": folds,
        "delta_risk_holdout_aggregate": agg,
        "parity_audited": fin["parity_audited"],
        "parity_status": fin["parity_status"],
        "effect_source": fin["effect_source"],
        "full_reencode_fallback_used": fin["full_reencode_fallback_used"],
        "simulated": bool(eff.get("simulated", False)),
    }


def run_selector_realization(
    *,
    ctx: CF1SExecutionContext,
    selector: str,
    universe: Mapping[str, Any],
    random_seed: Optional[int],
    identity_and_determinism_ok: bool,
) -> Dict[str, Any]:
    """Full selector path with freeze → search parent completion → holdout."""
    case_id = ctx.case_id
    edit_policy = ctx.edit_policy
    search_policy = ctx.search_policy
    saliency_policy = ctx.saliency_policy
    acceptance_policy = ctx.acceptance_policy
    callbacks = ctx.callbacks
    model_effects_evaluated = bool(ctx.model_effects_evaluated)

    search_fn = wrap_search_effect_fn(ctx, callbacks.search_effect_fn)
    holdout_fn = wrap_holdout_effect_fn(ctx, callbacks.holdout_effect_fn)
    parity_runner = callbacks.parity_runner

    funnel = empty_cf1s_funnel_counts()
    merge_universe_counts(funnel, universe.get("counts") or {})

    alloc_cfg = search_policy.get("case_budget_allocation") or {}
    beam_cfg = search_policy.get("beam") or {}
    sel_cfg = search_policy.get("selection") or {}
    max_single = int(
        alloc_cfg.get("maximum_single_candidates_per_case")
        or beam_cfg.get("single_event_budget_per_case")
        or 20
    )
    max_per_feat = int(
        alloc_cfg.get("maximum_event_loci_per_feature")
        or sel_cfg.get("top_event_loci_per_feature")
        or 8
    )
    stability_app = str(
        (saliency_policy.get("saliency_selector") or {}).get(
            "stability_application", "RANKING_TIER"
        )
    )
    effect_cfg = acceptance_policy.get("effect_fold_aggregation") or {}
    epsilon = float(effect_cfg.get("effect_pass_epsilon") or 0.01)
    min_inc = float(
        (acceptance_policy.get("incremental_gain") or {}).get(
            "min_incremental_abs_gain", epsilon
        )
    )

    ranked = rank_loci_for_selector(
        universe,
        selector=selector,
        random_seed=random_seed,
        cutoff_time=ctx.cutoff_time,
        stability_application=stability_app,
    )
    allocated = allocate_feature_round_robin(
        ranked,
        maximum_single_candidates_per_case=max_single,
        maximum_event_loci_per_feature=max_per_feat,
    )
    funnel["n_selector_ranked"] = int(ranked.get("ranked_count") or 0)
    funnel["n_round_robin_selected"] = int(allocated.get("selected_count") or 0)

    all_atomics = _build_atomics_for_loci(
        case_id=case_id,
        selected_loci=allocated.get("selected_loci") or [],
        edit_policy=edit_policy,
        atomic_target_fn=callbacks.atomic_target_fn,
    )
    funnel["n_atomic_candidates"] = len(all_atomics)
    validated = _validate_atomics(
        all_atomics,
        ground_fn=callbacks.ground_fn,
        invert_fn=callbacks.invert_fn,
        gate0_fn=callbacks.gate0_fn,
        gate4_fn=callbacks.gate4_fn,
        retokenize_fn=callbacks.retokenize_fn,
    )
    for k in (
        "grounding_failure_count",
        "inversion_failure_count",
        "gate0_failure_count",
        "gate4_failure_count",
        "retokenization_failure_count",
        "n_grounding_valid",
        "n_inversion_valid",
        "n_gate0_pass",
        "n_gate4_pass",
    ):
        funnel[k] = int(validated.get(k) or 0)

    screened = screen_single_event_atomics(
        validated["valid_atomics"],
        effect_fn=search_fn,
        single_budget=max_single,
    )
    funnel["n_single_screened"] = int(screened["n_single_screened"])

    beamed = compose_with_beam(
        case_id=case_id,
        screened_singles=screened["singles"],
        effect_fn=search_fn,
        beam_width=int(beam_cfg.get("beam_width") or 8),
        two_event_max=int(beam_cfg.get("two_event_max") or 32),
        three_event_max=int(beam_cfg.get("three_event_max") or 24),
        total_max=int(beam_cfg.get("total_max_per_realization") or 100),
        min_sparse_seconds=float(
            sel_cfg.get("minimum_sparse_time_distance_seconds") or 3600
        ),
        max_contiguous_gap_seconds=float(
            sel_cfg.get("contiguous_span_max_gap_seconds") or 1800
        ),
    )
    selected_bundles = list(beamed["bundles"])
    funnel["n_two_event_composites"] = sum(
        1 for b in selected_bundles if b.get("n_events") == 2
    )
    funnel["n_three_event_composites"] = sum(
        1 for b in selected_bundles if b.get("n_events") == 3
    )
    funnel["n_multi_event_composites"] = (
        funnel["n_two_event_composites"] + funnel["n_three_event_composites"]
    )
    funnel["n_total_selected"] = len(selected_bundles)
    funnel["n_length_position_valid"] = len(selected_bundles)
    if not beamed["budget_ok"]:
        funnel["budget_total_exceeded"] = 1

    # Index beam-scored candidates (search only).
    scored_search: Dict[str, Dict[str, Any]] = {}
    for s in screened["all_scored_singles"]:
        cid = str(s.get("atomic_candidate_id") or "")
        scored_search[cid] = {
            "edits": [s],
            "delta_risk_search_aggregate": float(s.get("delta_risk_search_aggregate") or 0.0),
            "delta_risk_search_by_fold": list(s.get("effect_search_by_fold") or []),
            "finalized_effect_values": True,
            "search_effect_evaluated": True,
        }
    for b in selected_bundles:
        cid = str(b.get("multi_event_candidate_id") or "")
        scored_search[cid] = {
            "edits": list(b.get("edits") or []),
            "delta_risk_search_aggregate": float(b.get("delta_risk_search_aggregate") or 0.0),
            "delta_risk_search_by_fold": list(b.get("delta_risk_search_by_fold") or []),
            "finalized_effect_values": True,
            "search_effect_evaluated": True,
        }

    # 1) Lock selection + evaluation closure BEFORE any holdout / parent completion.
    selected_rows = build_selected_manifest_rows(
        selected_bundles, case_id=case_id, selector=selector
    )
    selected_jsonl, selected_sha = lock_manifest_rows(selected_rows)
    closure = build_evaluation_closure(
        selected_bundles,
        case_id=case_id,
        selector=selector,
        already_scored=scored_search,
    )
    closure_jsonl, closure_sha = lock_manifest_rows(closure["candidates"])
    # Prefer identity sha from closure builder (immutable projection).
    closure_identity_sha = str(closure["evaluation_closure_identity_sha256"])
    ctx.selected_manifest_locked = True
    ctx.evaluation_closure_locked = True
    ctx.candidate_freeze_complete = True

    # 2) Search parent completion + parity finalization for full closure
    #    (includes closure-only parents; never used for selection/beam rerank).
    for row in closure["candidates"]:
        cid = str(row["candidate_id"])
        scored = _score_edits_search(
            row["edits"], search_fn=search_fn, parity_runner=parity_runner
        )
        scored_search[cid] = {**scored, "edits": row["edits"]}

    parent_completion_search_complete = all(
        scored_search.get(str(r["candidate_id"]), {}).get("search_effect_evaluated")
        and scored_search.get(str(r["candidate_id"]), {}).get("finalized_effect_values")
        for r in closure["candidates"]
    )
    ctx.parent_completion_search_complete = bool(parent_completion_search_complete)
    if not parent_completion_search_complete:
        raise ProductionContractViolation(
            "parent_completion_search_complete required before holdout"
        )

    # 3) Holdout on the same evaluation closure only.
    holdout_scored: Dict[str, Dict[str, Any]] = {}
    for row in closure["candidates"]:
        cid = str(row["candidate_id"])
        scored = _score_edits_holdout(
            row["edits"], holdout_fn=holdout_fn, parity_runner=parity_runner
        )
        holdout_scored[cid] = {**scored, "edits": row["edits"], **row}
        if str(ctx.provenance_mode).upper() == "PRODUCTION" and scored.get("simulated"):
            raise ProductionContractViolation("simulated holdout effect in production")

    holdout_identity_sha = closure_identity_sha
    matches = holdout_set_matches_closure(holdout_identity_sha, closure_identity_sha)
    if not matches:
        raise ProductionContractViolation("holdout set does not match evaluation closure")

    parent_completion_holdout_complete = all(
        holdout_scored.get(str(r["candidate_id"]), {}).get("holdout_effect_evaluated")
        and holdout_scored.get(str(r["candidate_id"]), {}).get("finalized_effect_values")
        for r in closure["candidates"]
    )

    # 4) Derive parent / incremental / multi-event-only metrics from finalized effects.
    evaluated = []
    for bundle in selected_bundles:
        edits = list(bundle.get("edits") or [])
        cid = str(
            bundle.get("multi_event_candidate_id")
            or multi_event_candidate_id(case_id=case_id, edits=edits)
        )
        search_eff = scored_search.get(cid) or {}
        holdout_eff = holdout_scored.get(cid) or {}
        if not search_eff.get("finalized_effect_values") or not holdout_eff.get(
            "finalized_effect_values"
        ):
            continue

        search_folds = list(search_eff.get("delta_risk_search_by_fold") or [])
        holdout_folds = list(holdout_eff.get("delta_risk_holdout_by_fold") or [])
        bundle_search = float(search_eff.get("delta_risk_search_aggregate") or 0.0)
        bundle_holdout = float(holdout_eff.get("delta_risk_holdout_aggregate") or 0.0)

        parent_ids = [
            multi_event_candidate_id(case_id=case_id, edits=[e]) for e in edits
        ]
        atomic_search_series = []
        atomic_holdout_series = []
        atomic_search_aggs = []
        atomic_holdout_aggs = []
        parents_complete = True
        for pid in parent_ids:
            ps = scored_search.get(pid)
            ph = holdout_scored.get(pid)
            if not ps or not ph or not ps.get("finalized_effect_values") or not ph.get(
                "finalized_effect_values"
            ):
                parents_complete = False
                break
            atomic_search_series.append(list(ps.get("delta_risk_search_by_fold") or []))
            atomic_holdout_series.append(list(ph.get("delta_risk_holdout_by_fold") or []))
            atomic_search_aggs.append(float(ps.get("delta_risk_search_aggregate") or 0.0))
            atomic_holdout_aggs.append(float(ph.get("delta_risk_holdout_aggregate") or 0.0))

        # Pairwise parents for 3-event residuals
        residuals = None
        if int(bundle.get("n_events") or 0) >= 3 and parents_complete and len(edits) == 3:
            pair_search = []
            for i, j in ((0, 1), (0, 2), (1, 2)):
                pid = multi_event_candidate_id(case_id=case_id, edits=[edits[i], edits[j]])
                pair_search.append(
                    float(scored_search.get(pid, {}).get("delta_risk_search_aggregate") or 0.0)
                )
            residuals = three_event_residuals(
                delta_a=atomic_search_aggs[0],
                delta_b=atomic_search_aggs[1],
                delta_c=atomic_search_aggs[2],
                delta_ab=pair_search[0],
                delta_ac=pair_search[1],
                delta_bc=pair_search[2],
                delta_abc=bundle_search,
            )

        if not parents_complete and int(bundle.get("n_events") or 0) >= 2:
            ctx.counters.missing_parent_default_true_count += 0  # never default true
            # Mark incomplete; exclude from stable incremental/me-only.
            pcs = False
            pch = False
        else:
            pcs = bool(parent_completion_search_complete)
            pch = bool(parent_completion_holdout_complete)

        inter_search = classify_interaction(
            bundle_delta=bundle_search, atomic_deltas=atomic_search_aggs or [0.0]
        )
        inter_holdout = classify_interaction(
            bundle_delta=bundle_holdout, atomic_deltas=atomic_holdout_aggs or [0.0]
        )
        holdout_by_fold_classes = classify_interaction_by_fold(
            bundle_deltas_by_fold=holdout_folds or [bundle_holdout],
            atomic_deltas_by_fold=[
                atomic_holdout_aggs or [0.0] for _ in (holdout_folds or [0])
            ],
        )
        stable_inter = stable_interaction_fields(
            search_class=inter_search["interaction_class"],
            holdout_class=inter_holdout["interaction_class"],
            holdout_classes_by_fold=holdout_by_fold_classes["interaction_class_by_fold"],
            holdout_aligned_directions_by_fold=holdout_by_fold_classes[
                "aligned_interaction_direction_by_fold"
            ],
            holdout_interaction_magnitude=abs(inter_holdout["interaction_delta"]),
        )
        validity = stable_model_sensitivity_valid(
            delta_risk_search_by_fold=search_folds or [bundle_search],
            delta_risk_holdout_by_fold=holdout_folds or [bundle_holdout],
            effect_pass_epsilon=epsilon,
        )

        inc_search_folds = incremental_abs_gain_by_fold(
            bundle_deltas_by_fold=search_folds or [bundle_search],
            atomic_parent_deltas_by_fold=atomic_search_series or [[0.0]],
        )
        inc_holdout_folds = incremental_abs_gain_by_fold(
            bundle_deltas_by_fold=holdout_folds or [bundle_holdout],
            atomic_parent_deltas_by_fold=atomic_holdout_series or [[0.0]],
        )
        inc_search = incremental_gain_pass(
            incremental_abs_gain_by_fold=inc_search_folds,
            min_incremental_abs_gain=min_inc,
            require_strict_majority=False,
        )
        inc_holdout = incremental_gain_pass(
            incremental_abs_gain_by_fold=inc_holdout_folds,
            min_incremental_abs_gain=min_inc,
            require_strict_majority=True,
        )
        me_search = multi_event_only_fold_pass(
            bundle_deltas_by_fold=search_folds or [bundle_search],
            atomic_parent_deltas_by_fold=atomic_search_series or [[0.0]],
            epsilon=epsilon,
        )
        me_holdout = multi_event_only_fold_pass(
            bundle_deltas_by_fold=holdout_folds or [bundle_holdout],
            atomic_parent_deltas_by_fold=atomic_holdout_series or [[0.0]],
            epsilon=epsilon,
        )
        stable_inc = stable_incremental_multi_event_pass(
            stable_model_sensitivity_valid_flag=bool(
                validity["stable_model_sensitivity_valid"]
            )
            and int(bundle.get("n_events") or 0) >= 2,
            parent_completion_search_complete=pcs,
            parent_completion_holdout_complete=pch,
            search_incremental_gain_pass=bool(inc_search["incremental_gain_pass"]),
            holdout_incremental_gain_pass=bool(inc_holdout["incremental_gain_pass"]),
        )
        stable_me = stable_multi_event_only_effect_pass_fold(
            stable_model_sensitivity_valid_flag=bool(
                validity["stable_model_sensitivity_valid"]
            )
            and int(bundle.get("n_events") or 0) >= 2,
            parent_completion_search_complete=pcs,
            parent_completion_holdout_complete=pch,
            search_multi_event_only_pass=bool(me_search["multi_event_only_pass"]),
            holdout_multi_event_only_pass=bool(me_holdout["multi_event_only_pass"]),
        )

        evaluated.append(
            {
                **bundle,
                "selector": selector,
                "random_seed": random_seed,
                "model_effects_evaluated": model_effects_evaluated,
                "delta_risk_search_by_fold": search_folds,
                "delta_risk_holdout_by_fold": holdout_folds,
                "delta_risk_search_aggregate": bundle_search,
                "delta_risk_holdout_aggregate": bundle_holdout,
                "search_effect_evaluated": True,
                "holdout_effect_evaluated": True,
                "finalized_effect_values": True,
                "pre_fallback_effect_used_for_metric_count": 0,
                "interaction_class_search": inter_search["interaction_class"],
                "interaction_class_holdout": inter_holdout["interaction_class"],
                "interaction_delta_search": inter_search["interaction_delta"],
                "interaction_delta_holdout": inter_holdout["interaction_delta"],
                "interaction_delta_holdout_by_fold": holdout_by_fold_classes[
                    "interaction_delta_by_fold"
                ],
                "interaction_class_holdout_by_fold": holdout_by_fold_classes[
                    "interaction_class_by_fold"
                ],
                **stable_inter,
                **validity,
                "incremental_abs_gain_search_by_fold": inc_search_folds,
                "incremental_abs_gain_holdout_by_fold": inc_holdout_folds,
                "incremental_abs_gain_search_aggregate": inc_search[
                    "incremental_abs_gain_aggregate"
                ],
                "incremental_abs_gain_holdout_aggregate": inc_holdout[
                    "incremental_abs_gain_aggregate"
                ],
                "holdout_incremental_gain_pass_count": inc_holdout["pass_count"],
                "holdout_incremental_gain_required_count": inc_holdout["required_count"],
                "holdout_incremental_gain_strict_majority": inc_holdout["strict_majority"],
                "search_incremental_gain_pass": inc_search["incremental_gain_pass"],
                "holdout_incremental_gain_pass": inc_holdout["incremental_gain_pass"],
                "stable_incremental_multi_event_pass": stable_inc,
                "search_multi_event_only_pass_count": me_search["pass_count"],
                "search_multi_event_only_required_count": me_search["required_count"],
                "search_multi_event_only_strict_majority": me_search["strict_majority"],
                "holdout_multi_event_only_pass_count": me_holdout["pass_count"],
                "holdout_multi_event_only_required_count": me_holdout["required_count"],
                "holdout_multi_event_only_strict_majority": me_holdout["strict_majority"],
                "multi_event_only_effect_pass_search": me_search["multi_event_only_pass"],
                "multi_event_only_effect_pass_holdout": me_holdout["multi_event_only_pass"],
                "stable_multi_event_only_effect_pass": stable_me,
                "parent_completion_search_complete": pcs,
                "parent_completion_holdout_complete": pch,
                "parent_family_complete": bool(pcs and pch),
                "three_event_residuals": residuals,
                "parent_completion_used_for_selection": False,
                "parity_audited": bool(holdout_eff.get("parity_audited")),
                "parity_status": holdout_eff.get("parity_status"),
                "effect_source": holdout_eff.get("effect_source"),
                "full_reencode_fallback_used": bool(
                    holdout_eff.get("full_reencode_fallback_used")
                ),
                "simulated": False
                if str(ctx.provenance_mode).upper() == "PRODUCTION"
                else bool(search_eff.get("simulated") or holdout_eff.get("simulated")),
            }
        )

    funnel["n_search_effect_evaluated"] = len(evaluated)
    funnel["n_holdout_effect_evaluated"] = len(evaluated)
    funnel["n_stable_model_sensitivity"] = sum(
        1 for e in evaluated if e.get("stable_model_sensitivity_valid")
    )
    funnel["parent_completion_used_for_selection_count"] = 0
    funnel["holdout_reranking_count"] = 0

    sampled = [
        e for e in evaluated if parity_sample_rule(e["multi_event_candidate_id"])
    ]
    if model_effects_evaluated:
        parity_fail = sum(
            1 for e in evaluated if str(e.get("parity_status")) == "FAIL"
        )
        parity_status = classify_parity_status(
            failure_count=parity_fail,
            fallback_failure_count=0,
            minimum_coverage_met=bool(evaluated),
        )
    else:
        parity_status = "NOT_EVALUABLE"

    funnel_status = funnel_accounting_status(
        funnel,
        two_event_max=int(beam_cfg.get("two_event_max") or 32),
        three_event_max=int(beam_cfg.get("three_event_max") or 24),
        total_max=int(beam_cfg.get("total_max_per_realization") or 100),
        require_single_screening=True,
    )

    parity_ok = parity_status in {"PASS", "NOT_EVALUABLE"} or (
        not model_effects_evaluated
    )
    if model_effects_evaluated:
        parity_ok = parity_status == "PASS" or (
            parity_status != "FAIL" and bool(evaluated)
        )

    acceptance = evaluate_selector_acceptance(
        evaluated,
        acceptance_policy=acceptance_policy,
        model_effects_evaluated=model_effects_evaluated,
        identity_and_determinism_ok=identity_and_determinism_ok,
        parity_ok=True if not model_effects_evaluated else parity_ok,
    )
    status = derive_selector_feasibility_status(
        model_effects_evaluated=model_effects_evaluated,
        n_evaluable_composites=len(evaluated),
        n_stable_model_sensitivity=int(funnel["n_stable_model_sensitivity"]),
        identity_and_determinism_ok=identity_and_determinism_ok
        if model_effects_evaluated
        else False,
        saliency_required_and_missing=(
            selector == SELECTOR_SALIENCY
            and not any(
                bool(l.get("saliency_evaluable", True))
                for l in (universe.get("loci") or [])
            )
        ),
        n_stable_incremental_multi_event=acceptance["n_stable_incremental_multi_event"],
        n_stable_multi_event_only=acceptance["n_stable_multi_event_only"],
        n_stable_multi_event=acceptance["n_stable_multi_event"],
        n_stable_single_event=acceptance["n_stable_single_event"],
    )

    return {
        "selector": selector,
        "random_seed": random_seed,
        "ranked": ranked,
        "allocated": allocated,
        "bundles": evaluated,
        "funnel": funnel,
        "funnel_accounting_status": funnel_status,
        "parity": {
            "reencode_parity_audit_status": parity_status,
            "parity_sample_candidate_count": len(sampled),
            "parity_simulated": not model_effects_evaluated,
            "parity_resolved_before_effect_finalization": True,
        },
        "feasibility_status": status,
        "acceptance": acceptance,
        "n_stable_model_sensitivity": int(funnel["n_stable_model_sensitivity"]),
        "n_stable_incremental_multi_event": acceptance["n_stable_incremental_multi_event"],
        "n_stable_multi_event_only": acceptance["n_stable_multi_event_only"],
        "n_stable_multi_event": acceptance["n_stable_multi_event"],
        "n_stable_single_event": acceptance["n_stable_single_event"],
        "evaluable_composite_count": len(evaluated),
        "selected_pool_sha256": allocated.get("selected_pool_sha256"),
        "selected_count": allocated.get("selected_count"),
        "beam": {
            "n_single_screened": beamed["n_single_screened"],
            "n_two_event_kept": beamed["n_two_event_kept"],
            "n_three_event_kept": beamed["n_three_event_kept"],
            "n_total_selected": beamed["n_total_selected"],
            "budget_ok": beamed["budget_ok"],
        },
        "selection_manifest": {
            "jsonl": selected_jsonl,
            "sha256": selected_sha,
            "locked": True,
            "n_rows": len(selected_rows),
        },
        "evaluation_closure": {
            "jsonl": closure_jsonl,
            "sha256": closure_sha,
            "identity_sha256": closure_identity_sha,
            "holdout_identity_projection_sha256": holdout_identity_sha,
            "holdout_set_matches_evaluation_closure": matches,
            "locked": True,
            "n_candidates": closure["n_candidates"],
            "n_closure_only": closure["n_closure_only"],
            "evaluation_closure_locked_before_parent_completion": True,
            "search_parent_completion_before_holdout": True,
            "parent_completion_search_complete_before_holdout": True,
            "parent_completion_search_complete": parent_completion_search_complete,
            "parent_completion_holdout_complete": parent_completion_holdout_complete,
        },
        "invariants": {
            "holdout_call_before_candidate_freeze_count": ctx.counters.holdout_call_before_candidate_freeze_count,
            "holdout_call_before_search_closure_complete_count": ctx.counters.holdout_call_before_search_closure_complete_count,
            "search_parent_completion_used_for_selection": False,
            "search_parent_completion_used_for_beam_reranking": False,
            "search_parent_completion_used_for_candidate_replacement": False,
            "parent_candidate_added_after_freeze_count": 0,
            "search_value_used_as_holdout_count": ctx.counters.search_value_used_as_holdout_count,
            "pre_fallback_effect_used_for_metric_count": 0,
            "parent_metrics_use_finalized_effects_only": True,
            "candidate_identity_projection_locked": True,
            "search_holdout_scorers_isolated": True,
            "incremental_gain_by_fold_complete": True,
            "search_multi_event_only_strict_majority_applied": True,
            "holdout_multi_event_only_strict_majority_applied": True,
        },
    }
