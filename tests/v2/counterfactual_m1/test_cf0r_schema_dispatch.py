"""Tests for Path A circular/boolean schema dispatch and validity separation."""

from __future__ import annotations

from src.online2.v2.finetune_v03.counterfactual.candidates.path_a_schema_dispatch import (
    COMPASS8_CATEGORIES,
    COMPASS8_CANONICAL_DEG,
    build_boolean_toggle_candidates,
    build_compass8_candidates,
    resolve_path_a_edit_strategy,
)
from src.online2.v2.finetune_v03.counterfactual.evaluation.funnel_accounting import (
    build_cf0_case_funnel,
    classify_case_execution_error_stage,
)
from src.online2.v2.tokenizer import wind_compass


def test_compass8_wraparound_and_canonical():
    cands = build_compass8_candidates(feature="wind_direction_deg", observed_raw=0.0)
    assert wind_compass(0.0) == "N"
    cats = {c["to_category"] for c in cands}
    assert cats == {"NW", "NE"}
    assert all(c["actionability"] is False for c in cands)
    assert all(c["operational_eligibility"] == "OBSERVATIONAL_SENSITIVITY_ONLY" for c in cands)
    assert all(c["source"] == "schema_categorical_adjacent" for c in cands)
    for c in cands:
        assert c["target_raw"] == COMPASS8_CANONICAL_DEG[c["to_category"]]
        assert wind_compass(c["target_raw"]) == c["to_category"]


def test_compass8_roundtrip_categories():
    for cat in COMPASS8_CATEGORIES:
        deg = COMPASS8_CANONICAL_DEG[cat]
        assert wind_compass(deg) == cat


def test_boolean_toggle_and_null_skip():
    c0 = build_boolean_toggle_candidates(feature="rain_detected", observed_raw=0.0)
    assert len(c0) == 1 and c0[0]["target_raw"] == 1.0
    c1 = build_boolean_toggle_candidates(feature="rain_detected", observed_raw=1.0)
    assert len(c1) == 1 and c1[0]["target_raw"] == 0.0
    assert build_boolean_toggle_candidates(feature="rain_detected", observed_raw=None) == []
    assert build_boolean_toggle_candidates(feature="rain_detected", observed_raw=0.3) == []
    assert c0[0]["actionability"] is False
    assert c0[0]["operational_eligibility"] == "OBSERVATIONAL_SENSITIVITY_ONLY"


def test_strategy_dispatch():
    w = resolve_path_a_edit_strategy(
        feature="wind_direction_deg", feature_type="circular", circular_encoding="compass8"
    )
    assert w.kind == "circular_categorical"
    r = resolve_path_a_edit_strategy(feature="rain_detected", feature_type="boolean")
    assert r.kind == "boolean_observation"
    c = resolve_path_a_edit_strategy(
        feature="inside_temp_c", feature_type="continuous", absolute_encoding=True
    )
    assert c.kind == "linear_bins"
    u = resolve_path_a_edit_strategy(feature="circulation_fan", feature_type="boolean")
    # boolean without abs bins is toggle-capable schema path
    assert u.kind == "boolean_observation"


def test_observational_valid_not_recommendation():
    dedup = [
        {
            "candidate_id": "obs1",
            "source": "schema_categorical_adjacent",
            "candidate_sources": ["schema_categorical_adjacent"],
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
            "effect_pass": True,
            "stability_pass": True,
            "search_material": True,
            "cf_valid": True,
            "actionability": False,
            "operational_eligibility": "OBSERVATIONAL_SENSITIVITY_ONLY",
            "validity_labels": ["MODEL_SENSITIVITY_ONLY", "NOT_RECOMMENDATION"],
            "delta_r_folds_search": [-0.02, -0.03],
        }
    ]
    funnel = build_cf0_case_funnel(
        case_id="CASE_OBS",
        cohort="example35_abnormal",
        editable_loci_count=1,
        raw_proposals=[{"source": "schema_categorical_adjacent", "is_noop": False}],
        dedup_rows=dedup
        + [{"candidate_id": "noop", "source": "noop", "is_noop": True}],
        selected={"candidate_id": "obs1", "source": "schema_categorical_adjacent", "is_noop": False},
    )
    assert funnel["model_valid_cf_count"] == 1
    assert funnel["observational_sensitivity_valid_count"] == 1
    assert funnel["recommendation_valid_cf_count"] == 0
    assert funnel["selected_actionable_non_noop_count"] == 0
    assert funnel["conservation_pass"] is True


def test_unsupported_locus_not_in_conservation():
    funnel = build_cf0_case_funnel(
        case_id="CASE_SKIP",
        cohort="example35_abnormal",
        editable_loci_count=1,
        raw_proposals=[],
        dedup_rows=[{"candidate_id": "noop", "source": "noop", "is_noop": True}],
        selected={"candidate_id": "noop", "source": "noop", "is_noop": True},
        unsupported_locus_count=1,
        candidate_generation_skip_reason_counts={"NO_SCHEMA_SUPPORTED_CANDIDATE": 1},
    )
    assert funnel["unsupported_locus_count"] == 1
    assert funnel["deduplicated_candidate_count"] == 0
    assert funnel["candidate_rejection_count"] == 0
    assert "NO_SCHEMA_SUPPORTED_CANDIDATE" not in (funnel.get("terminal_reason_counts") or {})
    assert funnel["conservation_pass"] is True


def test_case_error_stage_classifier():
    assert classify_case_execution_error_stage("no bin edges for feature=wind_direction_deg") == (
        "FEATURE_BIN_LOOKUP"
    )
    assert classify_case_execution_error_stage(RuntimeError("boom")) == "UNHANDLED_EXCEPTION"
