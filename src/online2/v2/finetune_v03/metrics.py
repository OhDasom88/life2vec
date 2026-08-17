"""Metrics + analysis tables for v0.3 (binary thresholds, head consistency)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def fine_metrics(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> Dict[str, Any]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    class_ids = list(range(n_classes))
    per_class_f1 = (
        f1_score(yt, yp, labels=class_ids, average=None, zero_division=0).tolist()
        if len(yt)
        else [float("nan")] * n_classes
    )
    confmat = (
        confusion_matrix(yt, yp, labels=class_ids).tolist()
        if len(yt)
        else [[0] * n_classes for _ in class_ids]
    )
    return {
        "acc": float((yt == yp).mean()) if len(yt) else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(yt, yp)) if len(yt) else float("nan"),
        "macro_f1": float(
            f1_score(yt, yp, labels=class_ids, average="macro", zero_division=0)
        ),
        "per_class_f1": per_class_f1,
        "confusion_matrix": confmat,
    }


def _safe_auc(yt, ys) -> float:
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, ys))


def _safe_ap(yt, ys) -> float:
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(average_precision_score(yt, ys))


def _ece(yt: np.ndarray, ys: np.ndarray, n_bins: int = 10) -> float:
    if len(yt) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (ys >= bins[i]) & (ys < bins[i + 1] if i < n_bins - 1 else ys <= bins[i + 1])
        if not np.any(m):
            continue
        ece += abs(float(yt[m].mean()) - float(ys[m].mean())) * float(m.mean())
    return float(ece)


def reliability_bins(
    y_true: Sequence[float], y_score: Sequence[float], n_bins: int = 10
) -> List[Dict[str, float]]:
    yt = np.asarray(y_true, dtype=np.float64)
    ys = np.asarray(y_score, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (ys >= lo) & (ys < hi if i < n_bins - 1 else ys <= hi)
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "mean_pred": float(ys[m].mean()) if np.any(m) else float("nan"),
                "mean_true": float(yt[m].mean()) if np.any(m) else float("nan"),
                "count": int(m.sum()),
            }
        )
    return out


def binary_metrics_at_threshold(
    y_true: Sequence[float],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    ys = np.asarray(y_score, dtype=np.float64)
    pred = (ys >= float(threshold)).astype(np.float64)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    out: Dict[str, float] = {
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": float(accuracy_score(yt, pred)) if len(yt) else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(yt, pred)) if len(np.unique(yt)) > 1 else float("nan"),
        "abnormal_recall": float(recall_score(yt, pred, pos_label=1, zero_division=0)),
        "normal_recall": float(recall_score(yt, pred, pos_label=0, zero_division=0)),
        "abnormal_precision": float(precision_score(yt, pred, pos_label=1, zero_division=0)),
        "normal_precision": float(precision_score(yt, pred, pos_label=0, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "f1_abnormal": float(f1_score(yt, pred, pos_label=1, zero_division=0)),
        "f1_normal": float(f1_score(yt, pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(yt, pred)) if len(np.unique(yt)) > 1 else float("nan"),
        "coverage": float(len(yt)),
        "npv": float(tn / max(tn + fn, 1)),
    }
    return out


def binary_metrics(
    y_true: Sequence[float],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.float64)
    ys = np.asarray(y_score, dtype=np.float64)
    out: Dict[str, Any] = {
        "n_pos": float(yt.sum()),
        "n_neg": float((1 - yt).sum()),
        "prob_mean": float(ys.mean()) if len(ys) else float("nan"),
        "prob_std": float(ys.std()) if len(ys) else float("nan"),
    }
    out["auroc"] = _safe_auc(yt, ys)
    out["auprc_abnormal"] = _safe_ap(yt, ys)
    out["auprc_normal"] = _safe_ap(1.0 - yt, 1.0 - ys)
    # backward-compatible alias
    out["auprc"] = out["auprc_abnormal"]
    try:
        out["brier"] = float(brier_score_loss(yt, ys)) if len(yt) else float("nan")
    except Exception:
        out["brier"] = float("nan")
    out["ece"] = _ece(yt, ys)
    out["reliability_bins"] = reliability_bins(yt, ys)
    # NLL of Bernoulli
    eps = 1e-7
    ys_c = np.clip(ys, eps, 1 - eps)
    out["nll"] = float(-(yt * np.log(ys_c) + (1 - yt) * np.log(1 - ys_c)).mean()) if len(yt) else float("nan")

    thr_list = list(thresholds) if thresholds is not None else [float(threshold)]
    if float(threshold) not in thr_list:
        thr_list = sorted(set(thr_list + [float(threshold)]))
    table = [binary_metrics_at_threshold(yt, ys, threshold=t) for t in thr_list]
    out["threshold_table"] = table
    at = binary_metrics_at_threshold(yt, ys, threshold=threshold)
    out.update({k: v for k, v in at.items() if k != "threshold"})
    out["balanced_acc"] = at["balanced_acc"]
    return out


def head_consistency_metrics(
    *,
    y_abnormal: Sequence[float],
    p_bin: Sequence[float],
    p_fine_abn: Sequence[float],
    fine_pred_is_normal: Sequence[bool],
    tau_binary: float = 0.5,
) -> Dict[str, Any]:
    yt = np.asarray(y_abnormal, dtype=np.float64)
    pb = np.asarray(p_bin, dtype=np.float64)
    pf = np.asarray(p_fine_abn, dtype=np.float64)
    fine_n = np.asarray(fine_pred_is_normal, dtype=bool)
    bin_abn = pb >= float(tau_binary)
    fine_abn = ~fine_n
    gap = np.abs(pb - pf)
    out: Dict[str, Any] = {
        "mean_absolute_gap": float(gap.mean()) if len(gap) else float("nan"),
        "max_gap": float(gap.max()) if len(gap) else float("nan"),
        "agreement_normal": int((~fine_abn & ~bin_abn).sum()),
        "agreement_abnormal": int((fine_abn & bin_abn).sum()),
        "binary_only_abnormal": int((~fine_abn & bin_abn).sum()),
        "fine_only_abnormal": int((fine_abn & ~bin_abn).sum()),
    }
    n = max(len(yt), 1)
    out["head_agreement_rate"] = float(
        (out["agreement_normal"] + out["agreement_abnormal"]) / n
    )
    out["head_contradiction_rate"] = float(
        (out["binary_only_abnormal"] + out["fine_only_abnormal"]) / n
    )
    if len(pb) >= 2 and np.std(pb) > 1e-12 and np.std(pf) > 1e-12:
        out["pearson_corr"] = float(np.corrcoef(pb, pf)[0, 1])
        # Spearman without scipy
        rb = np.argsort(np.argsort(pb)).astype(np.float64)
        rf = np.argsort(np.argsort(pf)).astype(np.float64)
        out["spearman_corr"] = float(np.corrcoef(rb, rf)[0, 1])
    else:
        out["pearson_corr"] = float("nan")
        out["spearman_corr"] = float("nan")
    return out


def score_distribution_stats(
    y_abnormal: Sequence[float],
    p_bin: Sequence[float],
    *,
    fine_correct: Optional[Sequence[bool]] = None,
) -> Dict[str, float]:
    yt = np.asarray(y_abnormal, dtype=np.float64)
    pb = np.asarray(p_bin, dtype=np.float64)
    out = {
        "prob_mean_normal_cases": float(pb[yt == 0].mean()) if np.any(yt == 0) else float("nan"),
        "prob_mean_abnormal_cases": float(pb[yt == 1].mean()) if np.any(yt == 1) else float("nan"),
        "prob_std_normal_cases": float(pb[yt == 0].std()) if np.any(yt == 0) else float("nan"),
        "prob_std_abnormal_cases": float(pb[yt == 1].std()) if np.any(yt == 1) else float("nan"),
    }
    if fine_correct is not None:
        fc = np.asarray(fine_correct, dtype=bool)
        out["prob_mean_fine_correct"] = float(pb[fc].mean()) if np.any(fc) else float("nan")
        out["prob_mean_fine_wrong"] = float(pb[~fc].mean()) if np.any(~fc) else float("nan")
    return out
