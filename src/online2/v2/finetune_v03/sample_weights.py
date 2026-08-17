"""Class-balanced (inverse-frequency) per-sample weights for WeightedRandomSampler.

Same formula as original life2vec's `CLSDataModule.get_train_weights()`
(`src/data_new/datamodule.py`): `class_weight = 1/count(class)`, normalized to
sum 1, mapped back to a per-sample weight. life2vec's survival task
(`eos.yaml`) uses this to oversample the minority (death) class during
training via `WeightedRandomSampler(replacement=True)` -- its own docstring
says "mostly to upsample the minority class". No PU-style asymmetric loss is
ported here (that's a separate, more involved technique tied to life2vec's
positive-unlabeled framing of survival, not diagnosis).

Weighted by the BINARY normal/abnormal label (`y_abnormal`), not the fine
10-class diagnosis -- checked empirically first (2026-07-24): the fine
classes in `labels_example_score90.csv` are already near-balanced (3-5 cases
each across all 10), so weighting by fine label reproduces almost the same
per-batch normal/abnormal ratio as no weighting at all (measured: 10.2%
normal sampled vs 11.4% normal in the raw data -- no real correction). The
severe imbalance (4 normal vs 31 abnormal cases) is a binary-level artifact
of collapsing 9 fine abnormal classes into one "abnormal" bucket, so only
binary-level weighting actually fixes it. This is a deliberate departure
from life2vec's literal single-`TARGET` weighting (`get_train_weights()` in
`src/data_new/datamodule.py`) -- life2vec's survival TARGET *is* binary, so
weighting by it and weighting by "the primary target" are the same thing
there; here they are not, and binary is the one that matches the actual
problem (see `run_diagnosis_finetune_v03.py --pos-weight` discussion).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch


def class_balanced_sample_weights(
    case_ids: Sequence[str],
    labels_df: pd.DataFrame,
    normal_class_name: str = "정상_운영",
) -> torch.Tensor:
    diag_by_case = labels_df.set_index(labels_df["case_id"].astype(str))["diagnosis_normalized"]
    targets = np.array(
        [0 if diag_by_case.loc[str(cid)] == normal_class_name else 1 for cid in case_ids]
    )

    uniq = np.unique(targets)
    counts = np.array([int((targets == t).sum()) for t in uniq])
    class_weight = 1.0 / counts
    class_weight = class_weight / class_weight.sum()
    class_weight_by_id = dict(zip(uniq.tolist(), class_weight.tolist()))

    weights = np.array([class_weight_by_id[int(t)] for t in targets], dtype=np.float64)
    return torch.tensor(weights, dtype=torch.double)
