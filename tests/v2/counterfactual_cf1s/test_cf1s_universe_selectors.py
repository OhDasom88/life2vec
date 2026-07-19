from __future__ import annotations

from pathlib import Path

from src.online2.v2.finetune_v03.counterfactual.cf1s.edit_policy import (
    EditClass,
    load_edit_policy,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.locus_universe import (
    build_common_eligible_universe,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.selectors import (
    SELECTOR_RANDOM,
    SELECTOR_RECENCY,
    SELECTOR_SALIENCY,
    allocate_feature_round_robin,
    rank_loci_for_selector,
)
from src.online2.v2.finetune_v03.counterfactual.pipeline_cf1s import load_yaml_policy


ROOT = Path(__file__).resolve().parents[3]


def test_edit_policy_classes(policy_dir: Path):
    policy = load_edit_policy(policy_dir / "CF1S_EDIT_POLICY_V1.yaml")
    assert policy["recommendation_eligible"] is False
    assert "circulation_fan" in policy["features"]
    assert policy["features"]["diagnosis_label"]["edit_class"] == EditClass.FORBIDDEN_OUTCOME.value


def test_mg_dedup_precedes_selector(sample_events, policy_dir: Path):
    edit = load_edit_policy(policy_dir / "CF1S_EDIT_POLICY_V1.yaml")
    universe = build_common_eligible_universe(sample_events, edit_policy=edit)
    assert universe["common_universe_unit"] == "DISTINCT_RAW_MG_LOCUS"
    assert universe["mg_dedup_precedes_selector"] is True
    assert universe["counts"]["raw_mg_duplicate_detected_count"] > 0
    assert universe["counts"]["raw_mg_duplicate_remaining_count"] == 0
    assert universe["raw_mg_duplicate_locus_count"] == 0
    # Duplicates detected but not retained.
    keys = [l["locus_key"] for l in universe["loci"]]
    assert len(keys) == len(set(keys))
    assert all(l["feature_id"] != "diagnosis_label" for l in universe["loci"])
    assert all(l.get("event_time_epoch") is not None for l in universe["loci"])


def test_stability_ranking_tier_keeps_unstable(sample_events, policy_dir: Path):
    edit = load_edit_policy(policy_dir / "CF1S_EDIT_POLICY_V1.yaml")
    search = load_yaml_policy(policy_dir / "CF1S_SEARCH_POLICY_V1.yaml")
    universe = build_common_eligible_universe(sample_events, edit_policy=edit)
    ranked = rank_loci_for_selector(
        universe, selector=SELECTOR_SALIENCY, stability_application="RANKING_TIER"
    )
    assert ranked["selector_changes_ranking_only"] is True
    assert ranked["ranked_count"] == universe["common_eligible_locus_count"]
    tiers = [l["stability_tier"] for l in ranked["ranked_loci"]]
    assert 0 in tiers and 1 in tiers
    # Stable loci appear before unstable.
    first_unstable = next(i for i, t in enumerate(tiers) if t == 1)
    assert all(t == 0 for t in tiers[:first_unstable])

    allocated = allocate_feature_round_robin(
        ranked,
        maximum_single_candidates_per_case=int(
            search["case_budget_allocation"]["maximum_single_candidates_per_case"]
        ),
        maximum_event_loci_per_feature=int(
            search["case_budget_allocation"]["maximum_event_loci_per_feature"]
        ),
    )
    assert allocated["selected_count"] <= 20


def test_selectors_share_feature_round_robin_budget(sample_events, policy_dir: Path):
    edit = load_edit_policy(policy_dir / "CF1S_EDIT_POLICY_V1.yaml")
    universe = build_common_eligible_universe(sample_events, edit_policy=edit)
    counts = {}
    for selector, seed in (
        (SELECTOR_SALIENCY, None),
        (SELECTOR_RECENCY, None),
        (SELECTOR_RANDOM, 11),
    ):
        ranked = rank_loci_for_selector(
            universe,
            selector=selector,
            random_seed=seed,
            cutoff_time="2025-01-01T05:00:00",
            stability_application="RANKING_TIER",
        )
        # Random/recency do not require saliency.
        assert ranked["ranked_count"] > 0
        alloc = allocate_feature_round_robin(
            ranked,
            maximum_single_candidates_per_case=20,
            maximum_event_loci_per_feature=8,
        )
        counts[selector] = alloc["selected_count"]
        # Per-feature cap respected
        assert max(alloc["per_feature_selected_counts"].values()) <= 8
    # Same universe size and same budget method → same selected count when pool rich.
    assert counts[SELECTOR_SALIENCY] == counts[SELECTOR_RANDOM] == counts[SELECTOR_RECENCY]


def test_random_does_not_require_saliency(sample_events, policy_dir: Path):
    edit = load_edit_policy(policy_dir / "CF1S_EDIT_POLICY_V1.yaml")
    # Mark all saliency unevaluable — common universe still built.
    events = []
    for e in sample_events:
        row = dict(e)
        row["saliency_evaluable"] = False
        row["absolute_saliency"] = 0.0
        events.append(row)
    universe = build_common_eligible_universe(events, edit_policy=edit)
    ranked = rank_loci_for_selector(
        universe, selector=SELECTOR_RANDOM, random_seed=23
    )
    assert ranked["ranked_count"] == universe["common_eligible_locus_count"]
