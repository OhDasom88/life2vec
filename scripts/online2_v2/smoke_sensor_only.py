#!/usr/bin/env python3
"""Sensor-only smoke for Online2 V2 grouped MLM using exported parquet."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.online2.v2.masking import GroupedMLMMasker
from src.online2.v2.vocab import VocabV2


ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("ONLINE2_V2_BUILD", ROOT / "outputs/online2/v2_build"))


def main() -> None:
    vocab = VocabV2.load(BUILD / "vocab_v2.json")
    smoke = BUILD / "training_events_v2_smoke.parquet"
    export = pd.read_parquet(smoke if smoke.exists() else BUILD / "training_events_v2.parquet")
    assert len(export) > 0
    row = export.iloc[0]
    tokens = str(row["SENTENCE"]).split()
    groups = json.loads(row["measurement_group_ids"]) if row["measurement_group_ids"] else ["NONE"] * len(tokens)
    roles = json.loads(row["token_roles"]) if row["token_roles"] else ["meta"] * len(tokens)
    # pad align
    n = min(len(tokens), len(groups), len(roles), 1024)
    tokens, groups, roles = tokens[:n], groups[:n], roles[:n]
    ids = [vocab.get(t) for t in tokens]
    masker = GroupedMLMMasker(vocab, mask_ratio=0.3, seed=2023)
    masked, pos, tgt, report = masker.mask(ids, groups, roles)
    # tiny embedding table forward/backward
    emb = torch.nn.Embedding(vocab.size(), 32)
    x = torch.tensor(masked[:n], dtype=torch.long)
    y = emb(x).mean()
    y.backward()
    assert torch.isfinite(y).item()
    assert emb.weight.grad is not None
    out = {
        "rows": int(len(export)),
        "vocab_size": vocab.size(),
        "mask_report": report.__dict__,
        "loss_proxy": float(y.detach()),
        "grad_norm": float(emb.weight.grad.norm().item()),
        "training_mode": "transductive_public_pretraining",
        "pass": True,
    }
    path = BUILD / "smoke_sensor_only_v2.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
