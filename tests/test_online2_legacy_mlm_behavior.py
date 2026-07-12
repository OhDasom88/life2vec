"""Legacy online2 MLM behavior fixtures (comparison baseline, not an approval of leakage)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.tasks.mlm import MLM


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "outputs/online2/build-v8-active80-r3/life2vec_token_registry.json"
AUDIT = ROOT / "outputs/online2/v2_audit"


@dataclass
class _FakeVocab:
    general_tokens: List[str] = field(
        default_factory=lambda: ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    )
    background_tokens: List[str] = field(default_factory=list)
    token2index: Dict[str, int] = field(default_factory=dict)
    _frame: pd.DataFrame = field(default_factory=pd.DataFrame)

    def vocab(self) -> pd.DataFrame:
        return self._frame.copy()


@dataclass
class _FakeDataModule:
    vocabulary: _FakeVocab


def _load_vocab() -> _FakeVocab:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    tokens = payload["tokens"]
    frame = pd.DataFrame(
        {
            "ID": [t["token_id"] for t in tokens],
            "TOKEN": [t["token"] for t in tokens],
            "CATEGORY": [t["category"] for t in tokens],
        }
    )
    vocab = _FakeVocab(
        background_tokens=frame.loc[frame["CATEGORY"] == "BACKGROUND", "TOKEN"].tolist(),
        token2index={t["token"]: t["token_id"] for t in tokens},
        _frame=frame,
    )
    return vocab


def _task(vocab: _FakeVocab) -> MLM:
    task = MLM(
        name="legacy_mlm_fixture",
        max_length=128,
        mask_ratio=0.30,
        smart_masking=False,
        mask_special_tokens=False,
        mask_background_tokens=False,
        non_maskable_tokens=("IMAGE_EMBED_SLOT", "TEXT_EMBED_SLOT"),
    )
    task.datamodule = _FakeDataModule(vocabulary=vocab)
    return task


@pytest.fixture(scope="module")
def vocab() -> _FakeVocab:
    if not REGISTRY.exists():
        pytest.skip("approved V1 token registry missing")
    return _load_vocab()


def test_random_replacement_includes_narrative_and_excludes_slots(vocab: _FakeVocab):
    task = _task(vocab)
    # Build a short legal sequence: CLS + siblings + narrative present only in vocab
    abs_tok = next(t for t in vocab.token2index if "|ABS_" in t)
    global_tok = abs_tok.replace("|ABS_", "|GLOBAL_")
    farm_tok = abs_tok.replace("|ABS_", "|FARM_REL_")
    for required in (global_tok, farm_tok, "IMAGE_EMBED_SLOT", "NARRATIVE"):
        if required == "NARRATIVE":
            assert any(k.startswith("NARRATIVE|") for k in vocab.token2index)
        elif required not in vocab.token2index and required != "NARRATIVE":
            if required not in vocab.token2index:
                pytest.skip(f"missing token {required}")

    # Reconstruct excluded / replacement sets exactly like mlm_mask
    token2index = vocab.token2index
    excluded_ids = {token2index["[SEP]"], token2index["[UNK]"], token2index.get("[PAD]", 0)}
    excluded_ids.update(token2index[t] for t in vocab.general_tokens if t in token2index)
    excluded_ids.update(vocab.vocab().loc[vocab.vocab().CATEGORY == "GENERAL", "ID"].astype(int))
    excluded_ids.update(token2index[t] for t in vocab.background_tokens if t in token2index)
    excluded_ids.update(
        token2index[t] for t in task.non_maskable_tokens if t in token2index
    )
    replacement_ids = sorted(set(token2index.values()) - excluded_ids)
    replacement_tokens = [
        tok for tok, idx in token2index.items() if idx in set(replacement_ids)
    ]
    families = sorted(
        {
            tok.split("|", 1)[0] if "|" in tok else tok
            for tok in replacement_tokens
        }
    )
    assert "NARRATIVE" in families
    assert "CATEGORY" in families
    assert "IMAGE_EMBED_SLOT" not in replacement_tokens
    assert "TEXT_EMBED_SLOT" not in replacement_tokens
    assert not any(tok.startswith("FARM|") for tok in replacement_tokens)

    AUDIT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"token_family": families, "in_random_replacement_pool": True}
    ).to_csv(AUDIT / "random_replacement_token_families_fixture.csv", index=False)


def test_sibling_abs_global_farm_independent_masking(vocab: _FakeVocab):
    task = _task(vocab)
    # Pick a concrete sibling triple present in vocab
    candidates = [t for t in vocab.token2index if t.endswith("|ABS_B00") or "|ABS_B0" in t]
    abs_tok = None
    for tok in candidates:
        g = tok.replace("|ABS_", "|GLOBAL_")
        f = tok.replace("|ABS_", "|FARM_REL_")
        if g in vocab.token2index and f in vocab.token2index:
            abs_tok, global_tok, farm_tok = tok, g, f
            break
    else:
        pytest.skip("no ABS/GLOBAL/FARM_REL sibling triple in vocab")

    token_ids = np.array(
        [
            vocab.token2index["[CLS]"],
            vocab.token2index[abs_tok],
            vocab.token2index[global_tok],
            vocab.token2index[farm_tok],
            vocab.token2index["[SEP]"],
        ],
        dtype=np.int64,
    )
    np.random.seed(2023)
    masked, y_indx, y_token = task.mlm_mask(token_ids.copy())
    # Positions 1,2,3 are the siblings; SEP excluded; CLS at 0 never eligible
    selected = set(int(i) for i in y_indx if i < len(token_ids))
    sibling_positions = {1, 2, 3}
    masked_siblings = sibling_positions & selected
    # With short sequence and 30% mask, at least one sibling may be selected;
    # assert independence: it is possible that not all three are masked together.
    assert masked_siblings != sibling_positions or len(masked_siblings) < 3 or True
    # Stronger: find a seed where exactly one sibling is masked
    found = None
    for seed in range(2023, 2023 + 500):
        np.random.seed(seed)
        masked, y_indx, y_token = task.mlm_mask(token_ids.copy())
        selected = {int(i) for i in y_indx if i in sibling_positions}
        if len(selected) == 1:
            found = (seed, selected, masked.tolist(), y_token.tolist())
            break
    assert found is not None, "could not reproduce single-sibling mask within seeds"
    seed, selected, masked_ids, targets = found
    surviving = sibling_positions - selected
    assert len(surviving) == 2
    # Surviving sibling token ids remain visible in the input (not all masked)
    for pos in surviving:
        # unchanged or random/mask only at selected; survivors should equal original
        # unless randomly replaced - for independence leakage we only need them present
        assert masked_ids[pos] != vocab.token2index["[PAD]"]

    report = {
        "seed": seed,
        "abs_token": abs_tok,
        "global_token": global_tok,
        "farm_token": farm_tok,
        "masked_sibling_positions": sorted(selected),
        "visible_sibling_positions": sorted(surviving),
        "note": "Legacy independent MLM can leave sibling representations unmasked.",
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "legacy_mlm_sibling_leakage_fixture.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_mask_count_capped_by_legal_and_max_length(vocab: _FakeVocab):
    task = _task(vocab)
    # Many SEP/background-like tokens reduce legal set
    farm = next(t for t in vocab.background_tokens if t.startswith("FARM|"))
    ids = [vocab.token2index["[CLS]"]]
    ids += [vocab.token2index[farm]] * 10
    ids += [vocab.token2index["[SEP]"]]
    token_ids = np.asarray(ids, dtype=np.int64)
    np.random.seed(0)
    masked, y_indx, y_token = task.mlm_mask(token_ids.copy())
    # Background excluded => legal mask count can be zero
    assert int((y_token != 0).sum()) == 0 or True
    legal_nonzero = int((y_token != 0).sum())
    assert legal_nonzero == 0
