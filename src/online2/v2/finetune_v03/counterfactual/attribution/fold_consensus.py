"""Fold-level L1 / percentile normalization and sign-agreement locus helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def robust_scale(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        return x - med
    return (x - med) / (1.4826 * mad)


def l1_normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    s = float(np.abs(x).sum())
    if s < 1e-12:
        return np.zeros_like(x)
    return x / s


def percentile_normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    ranks = pd.Series(x).rank(method="average", pct=True).to_numpy()
    return ranks * 2.0 - 1.0


def normalize_fold_scores(
    fold_scores: np.ndarray,
    *,
    method: str = "l1",
) -> np.ndarray:
    """Normalize each fold row independently. fold_scores: [n_folds, n_tokens]."""
    arr = np.asarray(fold_scores, dtype=np.float64)
    if arr.ndim == 1:
        if method == "l1":
            return l1_normalize(arr)
        if method == "percentile":
            return percentile_normalize(arr)
        return robust_scale(arr)
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        if method == "l1":
            out[i] = l1_normalize(arr[i])
        elif method == "percentile":
            out[i] = percentile_normalize(arr[i])
        else:
            out[i] = robust_scale(arr[i])
    return out


def consensus_mean(normalized_fold_scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(normalized_fold_scores, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=0)


def fold_agreement(
    signed_by_fold: Sequence[float],
    *,
    min_agree: int = 2,
    min_abs_score_per_fold: float = 1.0e-5,
) -> Dict[str, float]:
    arr = np.asarray(signed_by_fold, dtype=np.float64)
    if arr.size == 0:
        return {
            "median": 0.0,
            "mean": 0.0,
            "iqr": 0.0,
            "folds_agree_sign": 0,
            "agreement_pass": 0.0,
            "n_above_min_abs": 0,
        }
    med = float(np.median(arr))
    mean = float(np.mean(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    above = np.abs(arr) >= float(min_abs_score_per_fold)
    n_above = int(above.sum())
    if n_above < min_agree:
        agree = 0
    else:
        subset = arr[above]
        if abs(med) < 1e-12:
            agree = int(np.sum(np.abs(subset) < float(min_abs_score_per_fold)))
        else:
            agree = int(np.sum(np.sign(subset) == np.sign(med)))
    return {
        "median": med,
        "mean": mean,
        "iqr": iqr,
        "folds_agree_sign": float(agree),
        "agreement_pass": float(agree >= min_agree and n_above >= min_agree),
        "n_above_min_abs": float(n_above),
    }


def sign_agreement_mask(
    fold_scores: np.ndarray,
    *,
    min_abs_score_per_fold: float = 1.0e-5,
    require_all_folds: bool = True,
) -> np.ndarray:
    """Per-token mask: all (or enough) folds above min abs AND same sign."""
    arr = np.asarray(fold_scores, dtype=np.float64)
    if arr.ndim == 1:
        return np.abs(arr) >= float(min_abs_score_per_fold)
    above = np.abs(arr) >= float(min_abs_score_per_fold)
    if require_all_folds:
        ok_mag = above.all(axis=0)
    else:
        ok_mag = above.sum(axis=0) >= 2
    signs = np.sign(arr)
    # treat near-zero as 0 already filtered by above
    ref = signs[0]
    ok_sign = (signs == ref).all(axis=0) & (ref != 0)
    return ok_mag & ok_sign


def aggregate_fold_frame(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str = "signed_attribution",
    *,
    min_agree: int = 2,
    min_abs_score_per_fold: float = 1.0e-5,
) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stats = fold_agreement(
            g[value_col].tolist(),
            min_agree=min_agree,
            min_abs_score_per_fold=min_abs_score_per_fold,
        )
        row = {c: v for c, v in zip(group_cols, keys)}
        row.update(stats)
        row["n_folds"] = int(g["fold_id"].nunique()) if "fold_id" in g.columns else int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)
