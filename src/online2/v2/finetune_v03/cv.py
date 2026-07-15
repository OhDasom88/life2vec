"""CV split helpers: repeated group-stratified K-fold for tuning."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.online2.v2.diagnosis_dataset import make_stratified_farm_folds


def make_repeated_group_folds(
    labels_df: pd.DataFrame,
    *,
    n_folds: int = 3,
    seeds: List[int] | None = None,
    n_repeats: int | None = None,
) -> List[Dict[str, Any]]:
    """Return list of {repeat, fold, seed, train, val, val_farms}."""
    seeds = list(seeds or [2023, 2024, 2025])
    if n_repeats is not None:
        seeds = seeds[: int(n_repeats)]
    out: List[Dict[str, Any]] = []
    for ri, seed in enumerate(seeds):
        folds = make_stratified_farm_folds(labels_df, n_folds=n_folds, seed=int(seed))
        for fi, fold in enumerate(folds):
            out.append(
                {
                    "repeat": ri,
                    "fold": fi,
                    "seed": int(seed),
                    "train": list(fold["train"]),
                    "val": list(fold["val"]),
                    "val_farms": list(fold.get("val_farms", [])),
                    "split_id": f"r{ri}_f{fi}",
                }
            )
    return out
