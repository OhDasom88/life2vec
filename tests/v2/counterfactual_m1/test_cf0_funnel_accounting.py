"""Unit tests for CF-0 candidate-level funnel accounting."""

from __future__ import annotations

from src.online2.v2.finetune_v03.counterfactual.evaluation.funnel_accounting import (
    assign_primary_terminal,
    build_cf0_case_funnel,
    count_mlm_provenance,
    first_zero_stage,
    validate_cf0_conservation,
    validate_cf0_monotonicity,
)


def test_mlm_provenance_separation():
    raw = [
        {"source": "adjacent_bin", "is_noop": False},
        {"source": "constrained_mlm", "is_noop": False},
        {"source": "constrained_mlm", "is_noop": False},
    ]
    dedup = [
        {
            "source": "adjacent_bin",
            "candidate_sources": ["adjacent_bin", "constrained_mlm"],
            "is_noop": False,
            "cf_valid": False,
        },
        {
            "source": "adjacent_bin",
            "candidate_sources": ["adjacent_bin"],
            "is_noop": False,
            "cf_valid": False,
        },
    ]
    prov = count_mlm_provenance(raw_proposals=raw, dedup_rows=dedup)
    assert prov["n_mlm_raw_proposals"] == 2
    assert prov["n_mlm_origin_candidates"] == 1
    assert prov["n_valid_mlm_cf"] == 0


def test_cf0_funnel_materiality_bottleneck_conservation():
    raw = [
        {"source": "adjacent_bin", "is_noop": False},
        {"source": "constrained_mlm", "is_noop": False},
    ]
    dedup = [
        {
            "candidate_id": "c1",
            "source": "adjacent_bin",
            "candidate_sources": ["adjacent_bin", "constrained_mlm"],
            "is_noop": False,
            "bank_supported": True,
            "decoder_scored": True,
            "inversion_pass": True,
            "gate0_pass": True,
            "gate4_raw_pass": True,
            "gate4_pass": True,
            "hard_constraint_pass": True,
            "plausibility_pass": True,
            "critic_scored": True,
            "effect_pass": False,
            "stability_pass": False,
            "search_material": False,
            "cf_valid": False,
            "delta_r_search": -0.00017,
            "delta_r_folds_search": [-0.0003, -0.00002],
        }
    ]
    funnel = build_cf0_case_funnel(
        case_id="CASE_X",
        cohort="problem_set",
        editable_loci_count=1,
        raw_proposals=raw,
        dedup_rows=dedup
        + [{"candidate_id": "noop::CASE_X", "source": "noop", "is_noop": True}],
        selected={
            "candidate_id": "noop::CASE_X",
            "source": "noop",
            "is_noop": True,
            "selection_reason": "no_material_candidate_canonical_noop",
        },
        bank_unique_bundle_count=41,
        decoder_scored_unique_bundle_count=40,
    )
    assert funnel["n_mlm_raw_proposals"] == 1
    assert funnel["n_mlm_origin_candidates"] == 1
    assert funnel["n_valid_mlm_cf"] == 0
    assert funnel["hard_constraint_pass_count"] == 1
    assert funnel["effect_pass_count"] == 0
    assert funnel["valid_cf_count"] == 0
    assert funnel["model_valid_cf_count"] == 0
    assert funnel["recommendation_valid_cf_count"] == 0
    assert funnel["selected_non_noop_count"] == 0
    assert funnel["selected_actionable_non_noop_count"] == 0
    assert funnel["noop_selected"] is True
    assert funnel["first_zero_stage"] == "effect_pass_count"
    assert funnel["bank_unique_bundle_count"] == 41
    assert funnel["plausibility_definition"] == "BANK_SUPPORT_PROXY"
    assert funnel["pending_downstream_stages"] is False
    assert funnel["monotonicity_pass"] is True
    assert funnel["conservation_pass"] is True
    assert funnel["terminal_reason_counts"].get("NON_MATERIAL_SEARCH_DELTA") == 1


def test_execution_error_separated_from_rejection():
    term = assign_primary_terminal(
        {
            "cf_valid": False,
            "inversion_pass": True,
            "gate0_pass": True,
            "gate4_raw_pass": True,
            "gate4_pass": True,
            "hard_constraint_pass": True,
            "stage_a_pass": False,
            "stage_a_failure_reason": "STAGE_A_FORWARD_ERROR",
        }
    )
    assert term["is_execution_error"] is True
    assert term["terminal_failure_reason"] == "STAGE_A_FORWARD_ERROR"


def test_first_zero_and_invariants():
    counts = {
        "editable_loci_count": 1,
        "raw_generated_candidate_count": 4,
        "deduplicated_candidate_count": 2,
        "bank_supported_candidate_count": 2,
        "decoder_scored_candidate_count": 2,
        "hard_constraint_pass_count": 2,
        "plausibility_pass_count": 2,
        "effect_pass_count": 0,
        "stability_pass_count": 0,
        "valid_cf_count": 0,
        "selected_non_noop_count": 0,
        "candidate_rejection_count": 2,
        "candidate_execution_error_count": 0,
        "gate0_pass_count": 2,
        "inversion_pass_count": 2,
        "gate4_pass_count": 2,
        "hard_equals_gate4_invariant_expected": True,
        "terminal_reason_counts": {"NON_MATERIAL_SEARCH_DELTA": 2},
        "execution_error_reason_counts": {},
    }
    assert first_zero_stage(counts) == "effect_pass_count"
    ok, errs = validate_cf0_monotonicity(counts)
    assert ok, errs
    ok2, errs2 = validate_cf0_conservation(counts)
    assert ok2, errs2
