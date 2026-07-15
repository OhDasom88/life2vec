"""Metrics helpers for finetune v0.2 (present-label fold F1 vs full OOF F1)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import f1_score


def macro_f1_present_labels(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> float:
    """Macro-F1 over labels that appear in y_true (fold validation)."""
    y_true_a = np.asarray(y_true, dtype=np.int64)
    y_pred_a = np.asarray(y_pred, dtype=np.int64)
    if y_true_a.size == 0:
        return float("nan")
    labels = sorted(set(y_true_a.tolist()))
    return float(
        f1_score(y_true_a, y_pred_a, labels=labels, average="macro", zero_division=0)
    )


def macro_f1_all_classes(
    y_true: Sequence[int], y_pred: Sequence[int], *, num_classes: int = 10
) -> float:
    """Full 10-class macro-F1 (use on concatenated 35 OOF predictions)."""
    y_true_a = np.asarray(y_true, dtype=np.int64)
    y_pred_a = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(num_classes))
    return float(
        f1_score(y_true_a, y_pred_a, labels=labels, average="macro", zero_division=0)
    )


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    if y_true_a.size == 0:
        return float("nan")
    return float((y_true_a == y_pred_a).mean())


def fold_selection_ok(
    train_f1: float,
    val_f1: float,
    *,
    eps: float = 1e-9,
) -> bool:
    """Final candidate requires train/val F1 = 1.0 on present labels."""
    return train_f1 >= 1.0 - eps and val_f1 >= 1.0 - eps


def oof_selection_ok(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    num_classes: int = 10,
    eps: float = 1e-9,
) -> Dict[str, float | bool]:
    acc = accuracy(y_true, y_pred)
    mf1 = macro_f1_all_classes(y_true, y_pred, num_classes=num_classes)
    return {
        "oof_accuracy": acc,
        "oof_macro_f1": mf1,
        "accepted": acc >= 1.0 - eps and mf1 >= 1.0 - eps,
    }


def average_probability_ensemble(
    fold_probs: Iterable[np.ndarray],
) -> np.ndarray:
    """Mean of softmax probability arrays [N, C] across folds."""
    stacked = np.stack(list(fold_probs), axis=0)
    return stacked.mean(axis=0)
