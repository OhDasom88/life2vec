"""5-fold saliency consensus utilities (finetune v0.2)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.online2.v2.finetune_v02.config import SaliencyConfig


def normalize_fold_scores(scores: np.ndarray, *, method: str = "percentile") -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    if method == "percentile":
        # rank percentile in [0, 1]
        order = np.argsort(np.argsort(x))
        return order / max(len(x) - 1, 1)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)


def consensus_table(
    event_ids: Sequence[str],
    fold_scores: Sequence[np.ndarray],
    *,
    cfg: SaliencyConfig,
) -> pd.DataFrame:
    """Build consensus dataframe for one case across folds.

    fold_scores: list of length F, each [T] aligned to event_ids.
    """
    mats = np.stack([normalize_fold_scores(s) for s in fold_scores], axis=0)  # [F, T]
    raw = np.stack([np.asarray(s, dtype=np.float64) for s in fold_scores], axis=0)
    median = np.median(mats, axis=0)
    mean = mats.mean(axis=0)
    iqr = np.percentile(mats, 75, axis=0) - np.percentile(mats, 25, axis=0)
    positive = (raw > 0).sum(axis=0).astype(int)

    t = len(event_ids)
    top_k = max(1, int(np.ceil(t * cfg.top_rank_frac)))
    top_sets = []
    for f in range(mats.shape[0]):
        top_sets.append(set(np.argsort(-mats[f])[:top_k].tolist()))
    top_agree = np.zeros(t, dtype=int)
    for i in range(t):
        top_agree[i] = sum(1 for s in top_sets if i in s)

    rows = []
    for i, eid in enumerate(event_ids):
        rows.append(
            {
                "event_id": eid,
                "median_saliency": float(median[i]),
                "mean_saliency": float(mean[i]),
                "saliency_iqr": float(iqr[i]),
                "positive_agreement_count": int(positive[i]),
                "top_rank_agreement_count": int(top_agree[i]),
                **{f"fold_{f}_score": float(raw[f, i]) for f in range(raw.shape[0])},
            }
        )
    return pd.DataFrame(rows)


def select_report_events(df: pd.DataFrame, *, cfg: SaliencyConfig) -> Dict[str, pd.DataFrame]:
    """Split primary vs strong vs disputed evidence."""
    primary = df[
        (df["positive_agreement_count"] >= cfg.positive_agree_min)
        & (df["top_rank_agreement_count"] >= cfg.top_rank_agree_min)
    ].copy()
    strong = df[
        (df["positive_agreement_count"] >= cfg.strong_agree)
        & (df["top_rank_agreement_count"] >= max(cfg.top_rank_agree_min, 4))
    ].copy()
    disputed = df[
        (df["positive_agreement_count"] > 0)
        & (df["positive_agreement_count"] < cfg.positive_agree_min)
    ].copy()
    return {"primary": primary, "strong": strong, "disputed": disputed}
