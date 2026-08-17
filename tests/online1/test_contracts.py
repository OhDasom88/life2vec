from __future__ import annotations

from pathlib import Path

from src.online1.time_keys import parse_dat_time, assign_zone_ids
from src.online1.lineage import build_env_lineage, assert_encoder_safe, load_deny_config
from src.online1.paths import load_pipeline_config
from src.online1.regression_anchors import regression_example_id
from src.online1.acceptance import evaluate_acceptance, load_acceptance
import pandas as pd


def test_parse_and_zone():
    cfg = load_pipeline_config()
    s = pd.Series(["DAT109 00:05", "DAT115 12:00"])
    meta = parse_dat_time(s)
    assert meta.iloc[0]["t_ord"] == 109 * 1440 + 5
    df = pd.DataFrame({"dat": [109, 115, 121, 127]})
    z = assign_zone_ids(df, cfg, split="train")
    assert list(z["zone_id"]) == [0, 1, 2, 3]


def test_lineage_denies_targets():
    deny = load_deny_config()
    lin = build_env_lineage(["temperature", "soil_moisture", "soil_ec"], deny)
    safe = assert_encoder_safe([l for l in lin if l.name == "temperature"])
    assert safe["pass"]
    bad = assert_encoder_safe(lin)
    assert bad["target_lineage_violations"] >= 2


def test_regression_id_stable():
    a = regression_example_id("inductive", 109, 0, "DAT109 00:05")
    b = regression_example_id("inductive", 109, 0, "DAT109 00:05")
    assert a == b


def test_acceptance_zero_leakage():
    exp = load_acceptance()
    obs = {
        "leakage": {k: 0 for k in exp["leakage"]},
        "cube": {
            "unverified_zone_mapping": 0,
            "malformed_cube_count": 0,
            "frozen_weight_sha_changed": False,
            "embedding_reproducibility_min_cosine": 0.9999,
            "invalid_pixel_ratio_max": 0.35,
            "cube_env_alignment_max_backward_minutes": 5,
            "future_alignment_in_regression": 0,
        },
        "training": {
            "nan_inf_steps": 0,
            "resume_parity_required": True,
            "wandb_required_fields_missing": 0,
            "oov_required_families": 0,
            "padding_loss_applied": False,
            "cube_missing_cube_loss_applied": False,
            "frozen_encoder_grad_norm_max": 0.0,
        },
        "compute": {
            "forward_backward_resume_smoke": "PASS",
            "default_max_length_locked": True,
            "full_store_cost_estimate_reported": True,
        },
        "champion": {
            "score_formula_locked_before_sweep": True,
            "arm_internal_selection_only": True,
            "tie_break_rule": "lower_compute_then_earlier_trial",
        },
    }
    v = evaluate_acceptance(obs, exp)
    assert v["pass"], v["failures"]
