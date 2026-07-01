#!/usr/bin/env python3
"""Generate Berry2Vec analysis figures from Phase 0 artifacts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.vq import whiten
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from analysis.scripts.loaders import load_population, load_vocab
from analysis.scripts.paths import (
    FIGURES_DIR,
    META_PKL,
    OUTPUT_ROOT,
    SALIENCY_PKL,
    TOKEN_EMBEDDINGS_TSV,
)

sns.set(style="white")


AGRI_CATEGORY_COLORS = {
    "GENERAL": "#999999",
    "BACKGROUND": "#e69f00",
    "MONTH": "#0072b2",
    "YEAR": "#0072b2",
    "standardCategoryName": "#009E73",
    "standardItemName": "#d55e00",
    "value_token": "#000000",
}


def _project_2d(x: np.ndarray, method: str = "tsne") -> np.ndarray:
    if method == "pacmap":
        try:
            import pacmap

            projector = pacmap.PaCMAP(
                n_components=2,
                n_neighbors=None,
                random_state=0,
                MN_ratio=1,
                FP_ratio=10,
                distance="angular",
                lr=0.5,
            )
            return projector.fit_transform(x)
        except ImportError:
            print("pacmap not installed, falling back to t-SNE")
    return TSNE(n_components=2, random_state=0, perplexity=min(30, len(x) - 1)).fit_transform(
        x
    )


def plot_concept_space(method: str = "tsne") -> Path:
    vocab = load_vocab()
    emb = pd.read_csv(TOKEN_EMBEDDINGS_TSV, sep="\t", index_col="ID")
    x = emb.values.astype(float)
    pad_mask = ~vocab["TOKEN"].str.contains("PLCH|PAD|MASK|UNK", regex=True)
    x = x[pad_mask.values]
    vocab_f = vocab.loc[pad_mask]
    x = x - x.mean(0)
    xh = whiten(x)

    coords = _project_2d(xh, method=method)
    colors = [AGRI_CATEGORY_COLORS.get(c, "#666666") for c in vocab_f["CATEGORY"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=40, edgecolors="white", linewidths=0.3)
    for i, (x0, y0) in enumerate(coords):
        tok = vocab_f.iloc[i]["TOKEN"]
        if tok.startswith(("CAT_", "ITEM_", "GH_", "LN_")) or tok.startswith("VAL_"):
            label = tok.replace("_", "-")
            ax.annotate(label, (x0, y0), fontsize=5, alpha=0.8)

    ax.set_title("Agri concept space (token embeddings)")
    ax.set_xticks([])
    ax.set_yticks([])
    out_csv = OUTPUT_ROOT / "concept_embedding_projection.csv"
    out_pdf = FIGURES_DIR / "fig01_concept_space.pdf"
    pd.DataFrame(
        {
            "x_coord": coords[:, 0],
            "y_coord": coords[:, 1],
            "category": vocab_f["CATEGORY"].values,
            "token": vocab_f["TOKEN"].values,
        }
    ).to_csv(out_csv, index=False)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_csv}, {out_pdf}")
    return out_pdf


def plot_plant_space() -> Path:
    try:
        import umap
    except ImportError:
        umap = None

    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)
    pop = load_population()
    ids = meta["sequence_ids"]
    act = meta["plant_repr"]
    pop = pop.loc[ids]

    if umap is not None:
        prj = umap.UMAP(
            n_components=2,
            min_dist=0.3,
            n_neighbors=min(15, len(act) - 1),
            metric="euclidean",
            random_state=0,
        )
        xp = prj.fit_transform(act)
    else:
        xp = TSNE(
            n_components=2, random_state=0, perplexity=min(10, len(act) - 1)
        ).fit_transform(act)

    targets = np.asarray(meta["targets"]).reshape(-1)
    preds = np.asarray(meta["predictions"]).reshape(-1)
    gh = pop["RES_ORIGIN"].values
    ln = pop["GENDER"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sc0 = axes[0].scatter(xp[:, 0], xp[:, 1], c=targets, cmap="viridis", s=60, edgecolors="k", linewidths=0.3)
    axes[0].set_title("TARGET (last fruiting count)")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046)

    sc1 = axes[1].scatter(xp[:, 0], xp[:, 1], c=preds, cmap="plasma", s=60, edgecolors="k", linewidths=0.3)
    axes[1].set_title("Predicted fruiting count")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046)

    gh_codes = LabelEncoder().fit_transform(gh)
    axes[2].scatter(xp[:, 0], xp[:, 1], c=gh_codes, cmap="Set1", s=60, edgecolors="k", linewidths=0.3)
    axes[2].set_title("Greenhouse (GH)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    out_pdf = FIGURES_DIR / "fig02_plant_summary_space.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    plt.close(fig)

    out_csv = OUTPUT_ROOT / "plant_embedding_projection.csv"
    pd.DataFrame(
        {
            "PERSON_ID": ids,
            "x_coord": xp[:, 0],
            "y_coord": xp[:, 1],
            "TARGET": targets,
            "prediction": preds,
            "GH": gh,
            "LN": ln,
        }
    ).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}, {out_pdf}")
    return out_pdf


def plot_saliency_examples(top_n: int = 6) -> Path:
    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)
    with open(SALIENCY_PKL, "rb") as f:
        saliency = pickle.load(f)

    vocab = load_vocab()
    id2tok = vocab["TOKEN"].to_dict()
    ids = meta["sequence_ids"]
    tokens = meta["metadata"]
    targets = np.asarray(meta["targets"]).reshape(-1)
    preds = np.asarray(meta["predictions"]).reshape(-1)

    errors = np.abs(preds - targets)
    order = np.argsort(-errors)[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for ax_i, idx in enumerate(order):
        sid = ids[idx]
        toks = tokens[idx]
        sal = saliency[sid]
        labels = [id2tok.get(int(t), str(t)) for t in toks[: len(sal)]]
        y = np.arange(len(sal))
        ax = axes[ax_i]
        colors = plt.cm.Reds(np.abs(sal) / (np.abs(sal).max() + 1e-8))
        ax.barh(y, sal, color=colors)
        ax.set_yticks(y[:: max(1, len(y) // 12)])
        ax.set_yticklabels([labels[i] for i in y[:: max(1, len(y) // 12)]], fontsize=6)
        ax.set_title(
            f"id={sid} true={targets[idx]:.1f} pred={preds[idx]:.1f}",
            fontsize=9,
        )
        ax.invert_yaxis()

    fig.suptitle("Token saliency (Input×Gradient) — highest error plants")
    out_pdf = FIGURES_DIR / "fig03_saliency_top_errors.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    return out_pdf


def plot_pred_vs_true() -> Path:
    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)
    targets = np.asarray(meta["targets"]).reshape(-1)
    preds = np.asarray(meta["predictions"]).reshape(-1)
    mae = np.mean(np.abs(preds - targets))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, c="steelblue", s=70, edgecolors="k", linewidths=0.4)
    lims = [min(targets.min(), preds.min()) - 0.5, max(targets.max(), preds.max()) + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True last fruiting count")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Finetune predictions (MAE={mae:.3f}, n={len(targets)})")
    out_pdf = FIGURES_DIR / "fig04_pred_vs_true.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser(description="Berry2Vec figure generation")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["concept", "plant", "saliency", "scatter"],
        choices=["concept", "plant", "saliency", "scatter", "all"],
    )
    parser.add_argument("--projection", default="tsne", choices=["tsne", "pacmap"])
    args = parser.parse_args()
    figures = args.figures
    if "all" in figures:
        figures = ["concept", "plant", "saliency", "scatter"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if "concept" in figures:
        plot_concept_space(method=args.projection)
    if "plant" in figures:
        plot_plant_space()
    if "saliency" in figures:
        plot_saliency_examples()
    if "scatter" in figures:
        plot_pred_vs_true()


if __name__ == "__main__":
    main()
