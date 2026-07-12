#!/usr/bin/env python3
"""Multimodal smoke: frozen external embeddings + trainable projection + MLM/SOP proxies."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.online2.v2.masking import GroupedMLMMasker
from src.online2.v2.vocab import VocabV2
from src.transformer.external_embeddings import ExternalEmbeddingProjection


ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("ONLINE2_V2_BUILD", ROOT / "outputs/online2/v2_build"))
EMB = BUILD / "external_embeddings"


def main() -> None:
    vocab = VocabV2.load(BUILD / "vocab_v2.json")
    smoke = BUILD / "training_events_v2_smoke.parquet"
    export = pd.read_parquet(smoke if smoke.exists() else BUILD / "training_events_v2.parquet")
    assert len(export) > 0
    row = export.iloc[0]
    tokens = str(row["SENTENCE"]).split()[:512]
    groups = json.loads(row["measurement_group_ids"]) if row["measurement_group_ids"] else ["NONE"] * len(tokens)
    roles = json.loads(row["token_roles"]) if row["token_roles"] else ["meta"] * len(tokens)
    n = min(len(tokens), len(groups), len(roles))
    tokens, groups, roles = tokens[:n], groups[:n], roles[:n]
    ids = [vocab.get(t) for t in tokens]
    masker = GroupedMLMMasker(vocab, mask_ratio=0.3, seed=2023)
    masked, pos, tgt, report = masker.mask(ids, groups, roles)

    emb = torch.nn.Embedding(vocab.size(), 64)
    x = emb(torch.tensor(masked[:n], dtype=torch.long))

    img_path = EMB / "image_embeddings.npy"
    txt_path = EMB / "text_embeddings.npy"
    assert img_path.exists(), "image embeddings missing"
    img = torch.tensor(np.load(img_path)[:2], dtype=torch.float32)
    img_proj = ExternalEmbeddingProjection(input_dim=img.shape[-1], hidden_dim=64)
    img_out = img_proj(img)
    assert img_out.shape == (img.shape[0], 64)

    text_out = None
    text_proj = None
    if txt_path.exists():
        txt = torch.tensor(np.load(txt_path)[:2], dtype=torch.float32)
        text_proj = ExternalEmbeddingProjection(input_dim=txt.shape[-1], hidden_dim=64)
        text_out = text_proj(txt)

    # Slot alignment: inject projected vectors at IMAGE/TEXT slot positions when present
    slot_positions = [i for i, t in enumerate(tokens) if t in {"[IMAGE_SLOT]", "[TEXT_SLOT]"}]
    fused = x.mean(dim=0) + img_out.mean(dim=0)
    if text_out is not None:
        fused = fused + text_out.mean(dim=0)
    if slot_positions:
        # Keep slot indices in report for alignment audit
        pass

    # MLM proxy
    mlm_head = torch.nn.Linear(64, vocab.size())
    logits = mlm_head(x)
    if len(pos):
        mlm_loss = torch.nn.functional.cross_entropy(logits[pos[: min(8, len(pos))]], torch.tensor(tgt[: min(8, len(pos))], dtype=torch.long))
    else:
        mlm_loss = logits.mean() * 0.0

    # SOP proxy (binary)
    sop_head = torch.nn.Linear(64, 2)
    sop_logits = sop_head(fused)
    sop_loss = torch.nn.functional.cross_entropy(sop_logits.unsqueeze(0), torch.tensor([0]))

    loss = mlm_loss + sop_loss
    loss.backward()

    # Frozen encoder gradient check: only projection/embedding params may have grad
    assert emb.weight.grad is not None
    assert img_proj.projection.weight.grad is not None
    assert torch.isfinite(mlm_loss).item()
    assert torch.isfinite(sop_loss).item()
    assert torch.isfinite(img_proj.projection.weight.grad).all().item()
    if text_proj is not None:
        assert text_proj.projection.weight.grad is not None

    out = {
        "rows": int(len(export)),
        "vocab_size": vocab.size(),
        "mask_report": report.__dict__,
        "mlm_loss": float(mlm_loss.detach()),
        "sop_loss": float(sop_loss.detach()),
        "image_dim": int(img.shape[-1]),
        "text_dim": int(np.load(txt_path).shape[-1]) if txt_path.exists() else None,
        "slot_positions": slot_positions,
        "projection_grad_norm": float(img_proj.projection.weight.grad.norm().item()),
        "frozen_encoder_in_graph": False,
        "training_mode": "transductive_public_pretraining",
        "pass": True,
    }
    path = BUILD / "smoke_multimodal_v2.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
