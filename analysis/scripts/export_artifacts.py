#!/usr/bin/env python3
"""Phase 0: export token embeddings, plant representations, metadata, saliency."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.scripts.loaders import (
    iter_all_batches,
    load_finetune_model,
    load_pretrain_model,
    load_vocab,
)
from analysis.scripts.paths import (
    FIGURES_DIR,
    META_PKL,
    OUTPUT_ROOT,
    PLANT_REPR_NPY,
    SALIENCY_PKL,
    TOKEN_EMBEDDINGS_TSV,
)


def export_token_embeddings(device: str = "cpu") -> Path:
    model, _ = load_pretrain_model(device)
    vocab = load_vocab()
    weights = model.transformer.embedding.token.weight.detach().cpu().numpy()
    df = pd.DataFrame(weights, index=vocab.index)
    df.index.name = "ID"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(TOKEN_EMBEDDINGS_TSV, sep="\t")
    print(f"Wrote {TOKEN_EMBEDDINGS_TSV} ({len(df)} tokens, dim={df.shape[1]})")
    return TOKEN_EMBEDDINGS_TSV


@torch.no_grad()
def export_plant_repr_and_meta(device: str = "cpu") -> tuple[Path, Path]:
    model, _ = load_finetune_model(device)
    from analysis.scripts.loaders import load_datamodule

    dm, _ = load_datamodule()
    reprs = []
    preds = []
    targets = []
    sequence_ids = []
    token_rows = []

    for batch in iter_all_batches(dm, device):
        with torch.no_grad():
            out = model(batch)
        if out.ndim == 3:
            out = out[:, 0, 0]
        else:
            out = out.reshape(-1)
        hidden = model.encoder_f(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long(),
        )
        cls_hidden = hidden[:, 0].detach().cpu().numpy()

        reprs.append(cls_hidden)
        preds.append(out.detach().cpu().numpy())
        targets.extend(batch["target"].detach().cpu().tolist())
        sequence_ids.extend(batch["sequence_id"].tolist())
        token_rows.append(batch["input_ids"][:, 0].detach().cpu().numpy())

    plant_repr = np.vstack(reprs)
    meta = {
        "plant_repr": plant_repr,
        "predictions": np.concatenate(preds),
        "targets": np.array(targets, dtype=float),
        "sequence_ids": sequence_ids,
        "metadata": np.concatenate(token_rows),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(PLANT_REPR_NPY, plant_repr)
    with open(META_PKL, "wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote {PLANT_REPR_NPY} shape={plant_repr.shape}")
    print(f"Wrote {META_PKL} n={len(sequence_ids)}")
    return PLANT_REPR_NPY, META_PKL


def _summarize_attr(attr: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    attr = attr.sum(dim=-1)
    attr = attr[mask]
    norm = torch.norm(attr)
    if norm > 0:
        attr = attr / norm
    return attr.detach().cpu().numpy()


def export_saliency(device: str = "cpu") -> Path:
    try:
        from captum.attr import InputXGradient
    except ImportError as exc:
        raise SystemExit("captum required: pip install captum") from exc

    model, _ = load_finetune_model(device)
    from analysis.scripts.loaders import load_datamodule

    dm, _ = load_datamodule()

    def forward_for_attr(embeddings, meta):
        padding_mask = meta["padding_mask"].long()
        x = model.transformer.forward_finetuning_with_embeddings(
            embeddings, padding_mask
        )
        out = model.decoder(x)
        if out.ndim == 3:
            out = out[:, 0, 0]
        return out.unsqueeze(-1)

    attr_fn = InputXGradient(forward_for_attr)
    saliency = {}

    for batch in iter_all_batches(dm, device):
        embeddings, _ = model.transformer.get_sequence_embedding(
            batch["input_ids"].long()
        )
        embeddings = embeddings.detach().requires_grad_(True)
        padding_mask = batch["padding_mask"].long()
        out = attr_fn.attribute(
            embeddings,
            target=0,
            additional_forward_args={"padding_mask": padding_mask},
        )
        for i, sid in enumerate(batch["sequence_id"].tolist()):
            mask = padding_mask[i].bool()
            saliency[sid] = _summarize_attr(out[i], mask)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SALIENCY_PKL, "wb") as f:
        pickle.dump(saliency, f)
    print(f"Wrote {SALIENCY_PKL} n={len(saliency)}")
    return SALIENCY_PKL


def main():
    parser = argparse.ArgumentParser(description="Berry2Vec Phase 0 exports")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["token", "plant", "saliency"],
        choices=["token", "plant", "saliency", "all"],
    )
    args = parser.parse_args()
    steps = args.steps
    if "all" in steps:
        steps = ["token", "plant", "saliency"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if "token" in steps:
        export_token_embeddings(args.device)
    if "plant" in steps:
        export_plant_repr_and_meta(args.device)
    if "saliency" in steps:
        export_saliency(args.device)


if __name__ == "__main__":
    main()
