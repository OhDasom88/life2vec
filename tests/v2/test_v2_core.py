"""Core V2 unit tests (schema, circular, grouped masking, determinism)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs/online2/v2_audit/feature_semantics_and_units.csv"
OUT = ROOT / "outputs/online2/v2_build"


@pytest.fixture(scope="module")
def built_registries():
    if not (OUT / "vocab_v2.json").exists():
        pytest.skip("V2 build registries missing; run scripts/online2_v2/build_v2.py")
    from src.online2.v2.feature_schema import FeatureSchema
    from src.online2.v2.binning import BinningRegistryV2
    from src.online2.v2.vocab import VocabV2

    return (
        FeatureSchema.load(OUT / "feature_schema_v2.yaml"),
        BinningRegistryV2.load(OUT / "binning_registry_v2_transductive.json"),
        VocabV2.load(OUT / "vocab_v2.json"),
    )


def test_schema_covers_audit_features():
    from src.online2.v2.feature_schema import build_feature_schema_from_audit
    import pandas as pd

    schema = build_feature_schema_from_audit(AUDIT)
    audit = pd.read_csv(AUDIT).drop_duplicates("feature_name")
    for name in audit["feature_name"]:
        assert name in schema.features
    # unresolved actuators must not invent ON/OFF state_mapping
    for name, spec in schema.features.items():
        if spec.unit_confidence in {"LOW", "UNRESOLVED"} and spec.type in {
            "boolean",
            "flow",
            "ordinal_actuator",
        }:
            assert not spec.state_mapping
            assert spec.threshold_tokens_enabled is False


def test_wind_circular_adjacency():
    from src.online2.v2.tokenizer import wind_compass

    assert wind_compass(1) == wind_compass(359) == "N"
    assert wind_compass(0) == "N"
    assert wind_compass(180) == "S"


def test_vocab_deterministic(built_registries):
    schema, binning, vocab = built_registries
    from src.online2.v2.vocab import build_vocab_v2

    again = build_vocab_v2(schema, binning, code_commit_hash=vocab.meta.get("git_commit_hash", ""))
    assert again.token_to_id == vocab.token_to_id
    assert again.get("[PAD]") == 0


def test_no_narrative_in_core_input_families(built_registries):
    _, _, vocab = built_registries
    assert not any(t.startswith("NARRATIVE|") for t in vocab.token_to_id)
    assert not any(t.startswith("CATEGORY|") for t in vocab.token_to_id)
    assert not any(t.startswith("DISEASE|") for t in vocab.token_to_id)


def test_grouped_masking_siblings(built_registries):
    _, _, vocab = built_registries
    from src.online2.v2.masking import GroupedMLMMasker

    masker = GroupedMLMMasker(vocab, mask_ratio=1.0, seed=0)
    tokens = [
        "FEATURE|INSIDE_TEMP_C",
        "VALUE_ABS|ABS_B01",
        "VALUE_GLOBAL_REL|GLOBAL_REL_B02",
        "VALUE_FARM_REL|FARM_REL_B03",
    ]
    # ensure tokens exist or map via get -> unk; add if missing by skipping
    ids = []
    roles = ["feature_identity", "value_abs", "value_global", "value_farm"]
    for t in tokens:
        if t not in vocab.token_to_id:
            pytest.skip(f"missing {t} in vocab")
        ids.append(vocab.get(t))
    groups = ["g"] * 4
    masked, pos, tgt, report = masker.mask(ids, groups, roles)
    assert report.masked_group_count == 1
    assert masked[0] == ids[0]  # identity preserved
    assert set(pos.tolist()) == {1, 2, 3}
    # no narrative replacement
    for mid in masked.tolist():
        tok = vocab.id_to_token[int(mid)]
        assert not tok.startswith("NARRATIVE|")


def test_binning_includes_problem_scope(built_registries):
    _, binning, _ = built_registries
    assert binning.meta["fit_scope"] == "transductive_public_pretrain_pool"
    assert binning.meta["contains_problem_observations"] is True
    assert binning.meta["contains_problem_hidden_targets"] is False
    assert binning.meta["problem_observation_count"] > 0
    assert binning.meta["example_observation_count"] > 0
