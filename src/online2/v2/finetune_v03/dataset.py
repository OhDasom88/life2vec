"""Dataset for v0.3 — reuses v0.2 caches + binary label."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from src.online2.v2.finetune_v02.dataset import (
    DiagnosisEventDatasetV02,
    collate_diagnosis_batch_v02,
    load_label_map,
    make_stratified_farm_folds,
)
from src.online2.v2.finetune_v03.cv import make_repeated_group_folds

__all__ = [
    "DiagnosisEventDatasetV03",
    "collate_diagnosis_batch_v03",
    "load_label_map",
    "make_stratified_farm_folds",
    "make_repeated_group_folds",
    "normal_class_id_from_map",
]


def normal_class_id_from_map(label_map: Dict[str, Any], name: str = "정상_운영") -> int:
    labels = list(label_map["labels"])
    if name not in labels:
        raise KeyError(f"{name} not in label_map.labels={labels}")
    return int(labels.index(name))


class DiagnosisEventDatasetV03(DiagnosisEventDatasetV02):
    def __init__(self, *args, normal_class_id: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal_class_id = int(normal_class_id)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        y = int(sample["label"])
        sample["y_abnormal"] = float(0.0 if y == self.normal_class_id else 1.0)
        return sample


def collate_diagnosis_batch_v03(
    samples: List[Dict[str, Any]], *, max_events: int = 4096
) -> Dict[str, Any]:
    batch = collate_diagnosis_batch_v02(samples, max_events=max_events)
    batch["y_abnormal"] = torch.tensor(
        [float(s["y_abnormal"]) for s in samples], dtype=torch.float32
    )
    return batch
