"""Unit tests for adaptive equal-frequency binning."""

from __future__ import annotations

import numpy as np

from src.online2.v2.binning import BinningPolicy, ChannelBinPlan, compute_edges


def test_equal_frequency_near_equal_occupancy():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=20.0, scale=5.0, size=10_000)
    edges, strategy, diag = compute_edges(values, bins=10, strategy="equal_frequency")
    assert strategy == "equal_frequency"
    assert len(edges) == 11
    shares = np.asarray(diag["occupancy"], dtype=float) / values.size
    assert shares.min() > 0.08
    assert shares.max() < 0.12
    assert diag["occupancy_cv"] < 0.05


def test_zero_inflated_reserves_b00():
    zeros = np.zeros(4000)
    pos = np.linspace(1.0, 100.0, 6000)
    values = np.concatenate([zeros, pos])
    edges, strategy, diag = compute_edges(
        values, bins=11, strategy="zero_inflated_equal_frequency", zero_mass_threshold=0.15
    )
    assert "zero_inflated" in strategy
    assert diag["zero_mass_frac"] > 0.3
    # First interior edge is tiny eps → B00 holds zeros.
    assert edges[1] <= 1e-6
    # Non-zero bins should not be empty.
    assert all(c > 0 for c in diag["occupancy"][1:])


def test_anchors_inserted_for_humidity_like():
    values = np.linspace(20.0, 100.0, 5000)
    edges, strategy, _ = compute_edges(
        values,
        bins=20,
        strategy="equal_frequency_with_anchors",
        anchors=(65.0, 90.0),
    )
    assert strategy == "equal_frequency_with_anchors"
    assert any(abs(e - 90.0) < 1e-9 for e in edges)
    assert any(abs(e - 65.0) < 1e-9 for e in edges)


def test_policy_disease_critical_more_bins_than_growth():
    policy = BinningPolicy(
        meta={"default_bins": 10, "min_bins": 2, "max_bins": 100, "min_samples_per_bin_abs": 40},
        tiers={
            "disease_critical": {"abs_bins": 40, "strategy": "equal_frequency_with_anchors"},
            "growth_continuous": {"abs_bins": 12, "strategy": "equal_frequency"},
        },
        features={
            "inside_humidity_pct": {"tier": "disease_critical", "anchors": [90.0]},
            "plant_height_cm": {"tier": "growth_continuous"},
        },
    )
    hum = policy.plan_for("inside_humidity_pct", channel="ABS", n_values=70000, n_unique=20000)
    growth = policy.plan_for("plant_height_cm", channel="ABS", n_values=440, n_unique=393)
    assert isinstance(hum, ChannelBinPlan)
    assert hum.n_bins > growth.n_bins
    assert hum.n_bins == 40
    # growth clipped by sample floor 440/40=11
    assert growth.n_bins <= 12


def test_line_flow_equal_width_problem_fixed_by_quantile():
    """Reproduce V8 collapse (equal-width B00~92%) vs equal-frequency."""
    rng = np.random.default_rng(1)
    zeros = np.zeros(27000)
    pos = np.clip(rng.lognormal(mean=4.5, sigma=0.8, size=47000), 0.1, 2000)
    outliers = np.array([6098.0] * 20)
    values = np.concatenate([zeros, pos, outliers])

    # equal-width analogue
    lo, hi = 0.0, float(values.max())
    ew = np.linspace(lo, hi, 11)
    ew_counts = np.zeros(10, dtype=int)
    for v in values:
        # right-closed like digitize
        idx = int(np.searchsorted(ew[1:-1], v, side="left"))
        ew_counts[idx] += 1
    assert ew_counts[0] / values.size > 0.85

    edges, strategy, diag = compute_edges(
        values, bins=20, strategy="zero_inflated_equal_frequency", zero_mass_threshold=0.15
    )
    assert "zero_inflated" in strategy
    # After zero split, positive bins should be populated.
    assert diag["occupancy"][0] > 0
    assert max(diag["occupancy"][1:]) > 0
    assert len(edges) >= 10
