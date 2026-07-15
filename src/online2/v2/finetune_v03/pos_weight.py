"""Resolve absolute BCE pos_weight (PLAN: no scale; auto = N_normal/N_abnormal)."""

from __future__ import annotations

from typing import Any, Union

import pandas as pd
import torch


def resolve_pos_weight(
    labels_df: pd.DataFrame,
    label_map: dict,
    normal_id: int,
    pos_weight: Union[str, float, None],
) -> torch.Tensor:
    """Return shape [1] float tensor for BCEWithLogits pos_weight."""
    if pos_weight is not None and str(pos_weight).lower() != "auto":
        return torch.tensor([float(pos_weight)], dtype=torch.float32)
    name_to_id = {n: i for i, n in enumerate(label_map["labels"])}
    y = labels_df["diagnosis_normalized"].map(name_to_id)
    n_abn = int((y != normal_id).sum())
    n_norm = int((y == normal_id).sum())
    w = float(n_norm) / max(float(n_abn), 1.0)
    return torch.tensor([w], dtype=torch.float32)


def parse_pos_weight_arg(raw: Any) -> Union[str, float]:
    if raw is None:
        return "auto"
    s = str(raw).strip().lower()
    if s == "auto":
        return "auto"
    return float(raw)
