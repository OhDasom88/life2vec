"""Online2 diagnosis finetune v0.2 (semantic alignment + ensemble + saliency).

Does not modify v0.1 modules. Reuses PAD blocks via import composition.
See docs/online2/FINETUNE_VERSIONS.md and DIAGNOSIS_FINETUNE_V02.md.
"""

from src.online2.v2.finetune_v02.config import FinetuneV02Config, LossWeights
from src.online2.v2.finetune_v02.model import EventPoolingDiagnosisModelV02
from src.online2.v2.finetune_v02.losses import total_finetune_loss
from src.online2.v2.finetune_v02.dataset import (
    DiagnosisEventDatasetV02,
    collate_diagnosis_batch_v02,
    make_stratified_farm_folds,
    load_label_map,
)
from src.online2.v2.finetune_v02.version import FINETUNE_VERSION, DEFAULT_OUTPUT_ROOT

__all__ = [
    "FINETUNE_VERSION",
    "DEFAULT_OUTPUT_ROOT",
    "FinetuneV02Config",
    "LossWeights",
    "EventPoolingDiagnosisModelV02",
    "total_finetune_loss",
    "DiagnosisEventDatasetV02",
    "collate_diagnosis_batch_v02",
    "make_stratified_farm_folds",
    "load_label_map",
]
