"""§9.1/§9.7-style quantitative check: does a pooled case representation
actually separate 정상_운영 (normal) from abnormal cases?

Answers a narrower, more falsifiable question than the 3D projection module
(`projection.py`) does on purpose: `metadata.py`'s own disclaimer says 3D
distance/clusters must not be used to confirm a concept exists. A
cross-validated linear-probe AUROC is a real, checkable claim about the
representation itself (can a simple boundary separate these two label
groups), still only `E2_PREDICTIVE` per the §9.7 evidence ladder (not causal),
but stronger evidence than eyeballing a scatterplot.

Representations must come from OUT-OF-FOLD model outputs (i.e. every case's
representation was produced by a checkpoint that never trained on that case)
-- an in-sample probe would trivially separate the labels the model was
fit to distinguish and prove nothing about representation quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold

DISCLAIMER = (
    "AUROC/실루엣은 표현이 라벨과 얼마나 갈리는지에 대한 예측적(E2_PREDICTIVE) 신호일 뿐, "
    "그 자체로 개념의 존재나 인과성을 확정하지 않는다 (계획서 §9.7 E0-E5 증거 등급 참조). "
    "반드시 out-of-fold 표현에 대해서만 계산한다 — in-sample 표현으로는 분리가 자명하게 좋아 보여 무의미하다."
)


@dataclass(frozen=True)
class SeparationProbeMetadata:
    representation_name: str  # e.g. "h_binary", "z_proj"
    checkpoint_ids: tuple[str, ...]
    method: str
    seed: int
    label_source: str
    n_samples: int
    n_positive: int

    def __post_init__(self) -> None:
        if not self.representation_name:
            raise ValueError("representation_name must not be empty")
        if not self.checkpoint_ids:
            raise ValueError("checkpoint_ids must not be empty — an OOF probe needs a known source per case")
        if not self.label_source:
            raise ValueError("label_source must not be empty")
        if self.n_samples < 4:
            raise ValueError(f"n_samples={self.n_samples} too small for a meaningful CV probe (need >=4)")
        if not (0 < self.n_positive < self.n_samples):
            raise ValueError(
                f"n_positive={self.n_positive} of n_samples={self.n_samples} — need both classes present"
            )


@dataclass(frozen=True)
class SeparationProbeReport:
    metadata: SeparationProbeMetadata
    cv_auroc_per_fold: tuple[float, ...]
    silhouette: float
    disclaimer: str = field(default=DISCLAIMER)

    def __post_init__(self) -> None:
        if self.disclaimer != DISCLAIMER:
            raise ValueError("disclaimer must not be overridden or removed")

    @property
    def cv_auroc_mean(self) -> float:
        return float(np.mean(self.cv_auroc_per_fold))

    @property
    def cv_auroc_std(self) -> float:
        return float(np.std(self.cv_auroc_per_fold))


def probe_label_separation(
    representations: np.ndarray,
    labels: Sequence[int],
    *,
    representation_name: str,
    checkpoint_ids: Sequence[str],
    label_source: str,
    n_splits: int = 5,
    seed: int = 0,
) -> SeparationProbeReport:
    x = np.asarray(representations, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("representations and labels must have the same length")

    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError("not enough examples of the minority class for cross-validated probing")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aurocs: list[float] = []
    for train_idx, test_idx in skf.split(x, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x[train_idx], y[train_idx])
        if len(np.unique(y[test_idx])) < 2:
            continue
        p = clf.predict_proba(x[test_idx])[:, 1]
        fold_aurocs.append(float(roc_auc_score(y[test_idx], p)))
    if not fold_aurocs:
        raise ValueError("no fold had both classes in its held-out split; cannot compute AUROC")

    sil = float(silhouette_score(x, y)) if len(np.unique(y)) > 1 else float("nan")

    meta = SeparationProbeMetadata(
        representation_name=representation_name,
        checkpoint_ids=tuple(checkpoint_ids),
        method="logistic_regression_stratified_kfold",
        seed=seed,
        label_source=label_source,
        n_samples=int(x.shape[0]),
        n_positive=int(y.sum()),
    )
    return SeparationProbeReport(
        metadata=meta,
        cv_auroc_per_fold=tuple(fold_aurocs),
        silhouette=sil,
    )
