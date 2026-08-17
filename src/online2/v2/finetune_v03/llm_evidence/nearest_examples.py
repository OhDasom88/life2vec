"""Cosine-similarity nearest labeled training examples -- the "학습 예시" evidence.

Uses the same OOF representations `representation_explorer/label_separation_probe.py`
consumes (`*_oof_representations.npz` from `run_diagnosis_finetune_v03.py`), so
"nearest example" means "nearest in the same out-of-fold representation space
already used to check normal/abnormal separation," not a separately invented
similarity metric.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def nearest_labeled_examples(
    query_rep: np.ndarray,
    train_reps: np.ndarray,
    train_case_ids: Sequence[str],
    train_labels: Sequence[str],
    *,
    k: int = 3,
    exclude_case_id: str | None = None,
) -> List[Dict[str, object]]:
    """Top-k cosine-nearest labeled cases to `query_rep`."""
    q = np.asarray(query_rep, dtype=np.float64)
    x = np.asarray(train_reps, dtype=np.float64)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sims = x_norm @ q_norm

    order = np.argsort(-sims)
    out: List[Dict[str, object]] = []
    for idx in order:
        cid = str(train_case_ids[idx])
        if exclude_case_id is not None and cid == str(exclude_case_id):
            continue
        out.append({"case_id": cid, "diagnosis": str(train_labels[idx]), "cosine_similarity": float(sims[idx])})
        if len(out) >= k:
            break
    return out
