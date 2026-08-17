from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import sha256_json, write_json


SPECIAL = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]", "[EVENT_SEP]", "[VIEW_EVENT_WINDOW]", "[VIEW_CUBE_ENV]", "[EVENT_START]", "[EVENT_END]", "[MISSING]", "[CUBE_MISSING]"]
ZONE = [f"ZONE_{z}" for z in "ABCD"]
PHASE = ["PHASE_NIGHT", "PHASE_DAY"]
CUBE_SLOTS = [f"CUBE_BAND_{i}" for i in range(10)] + ["CUBE_POOL"]


def _bin_edges(values: np.ndarray, n_bins: int = 32) -> np.ndarray:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.linspace(0, 1, n_bins + 1)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(vals, qs))
    if edges.size < 2:
        edges = np.array([vals.min() - 1e-6, vals.max() + 1e-6])
    return edges


def build_vocab(
    env_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_bins: int = 32,
) -> dict[str, Any]:
    tokens = list(SPECIAL) + list(ZONE) + list(PHASE) + list(CUBE_SLOTS)
    bins: dict[str, list[float]] = {}
    for c in feature_cols:
        edges = _bin_edges(env_df[c].to_numpy(dtype=float), n_bins=n_bins)
        bins[c] = edges.tolist()
        for i in range(len(edges) - 1):
            tokens.append(f"{c}|b{i:02d}")
        tokens.append(f"{c}|MISSING")
    # dedupe preserve order
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    token_to_id = {t: i for i, t in enumerate(uniq)}
    vocab = {
        "version": "online1_vocab_v1",
        "n_tokens": len(uniq),
        "tokens": uniq,
        "token_to_id": token_to_id,
        "bins": bins,
        "families": {
            "SPECIAL": SPECIAL,
            "ZONE": ZONE,
            "PHASE": PHASE,
            "CUBE_BAND_SLOT": CUBE_SLOTS,
            "ENV": [t for t in uniq if "|" in t and not t.startswith("CUBE")],
        },
        "metadata_only": ["NARRATIVE"],
        "pad_id": token_to_id["[PAD]"],
        "mask_id": token_to_id["[MASK]"],
        "unk_id": token_to_id["[UNK]"],
    }
    vocab["checksum"] = sha256_json({"tokens": uniq, "bins": bins})
    return vocab


def value_to_token(col: str, value: float, vocab: dict[str, Any]) -> str:
    if value is None or not np.isfinite(value):
        return f"{col}|MISSING"
    edges = vocab["bins"][col]
    # digitize right=False → bin index 1..n ; convert to 0..n-1
    idx = int(np.digitize([value], edges[1:-1], right=False)[0])
    idx = min(max(idx, 0), len(edges) - 2)
    return f"{col}|b{idx:02d}"


def encode_tokens(tokens: list[str], vocab: dict[str, Any]) -> list[int]:
    unk = vocab["unk_id"]
    t2i = vocab["token_to_id"]
    return [t2i.get(t, unk) for t in tokens]


def narrative_vocab_view(vocab: dict[str, Any], allowed_families: list[str]) -> dict[str, Any]:
    fam = vocab["families"]
    allowed = set()
    for f in allowed_families:
        allowed.update(fam.get(f, []))
    # always include specials
    allowed.update(fam.get("SPECIAL", SPECIAL))
    observed = [t for t in vocab["tokens"] if t in allowed]
    return {
        "allowed_families": allowed_families,
        "n_allowed_tokens": len(observed),
        "allowed_tokens_checksum": sha256_json(observed),
    }


def save_vocab(vocab: dict[str, Any], path: Path) -> None:
    # token_to_id is large but needed
    write_json(path, vocab)
