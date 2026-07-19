from __future__ import annotations

from src.online2.v2.finetune_v03.counterfactual.cf1s.fold_effect import (
    multi_event_only_effect_pass,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.selectors import (
    SELECTOR_RANDOM,
    SELECTOR_RECENCY,
    SELECTOR_SALIENCY,
    allocate_feature_round_robin,
    rank_loci_for_selector,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.locus_universe import (
    build_common_eligible_universe,
)
from src.online2.v2.finetune_v03.counterfactual.cf1s.edit_policy import load_edit_policy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_matched_budget_per_random_seed(sample_events):
    edit = load_edit_policy(ROOT / "conf/m1/cf1s_policies/CF1S_EDIT_POLICY_V1.yaml")
    universe = build_common_eligible_universe(sample_events, edit_policy=edit)
    sal = allocate_feature_round_robin(
        rank_loci_for_selector(universe, selector=SELECTOR_SALIENCY),
        maximum_single_candidates_per_case=20,
        maximum_event_loci_per_feature=8,
    )
    rec = allocate_feature_round_robin(
        rank_loci_for_selector(
            universe, selector=SELECTOR_RECENCY, cutoff_time="2025-01-01T05:00:00"
        ),
        maximum_single_candidates_per_case=20,
        maximum_event_loci_per_feature=8,
    )
    matched_by_seed = {}
    for seed in (11, 23, 37):
        rand = allocate_feature_round_robin(
            rank_loci_for_selector(
                universe, selector=SELECTOR_RANDOM, random_seed=seed
            ),
            maximum_single_candidates_per_case=20,
            maximum_event_loci_per_feature=8,
        )
        matched_by_seed[seed] = min(
            sal["selected_count"], rec["selected_count"], rand["selected_count"]
        )
    # Per-seed matching — do not sum random seeds before comparing.
    assert all(v == matched_by_seed[11] for v in matched_by_seed.values())
    assert sum(matched_by_seed.values()) != matched_by_seed[11]  # multiple seeds exist


def test_multi_event_only_search_holdout_split():
    assert multi_event_only_effect_pass(
        bundle_abs_delta=0.02, atomic_parent_abs_deltas=[0.001, 0.002]
    )
    assert not multi_event_only_effect_pass(
        bundle_abs_delta=0.02, atomic_parent_abs_deltas=[0.02, 0.001]
    )
