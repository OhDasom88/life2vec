"""v0.3 finetune hyper-parameters (binary + fine + projection + open-set).

See docs/online2/DIAGNOSIS_FINETUNE_V03_PLAN.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.online2.v2.event_pooling_finetune import EventPoolingConfig
from src.online2.v2.finetune_v02.config import ImageAdapterConfig, SaliencyConfig
from src.online2.v2.finetune_v03.version import FINETUNE_VERSION

# Binary operating points (eval / freeze); not training hyperparameters except logging.
BINARY_THRESHOLDS_DEFAULT: List[float] = [0.25, 0.50, 0.75]


@dataclass
class LossWeightsV03:
    """All λ may be 0 so sweeps can disable a term."""

    fine_ce: float = 1.0
    binary_bce: float = 1.0
    consistency: float = 0.05
    proj_supcon: float = 0.0
    prototype: float = 0.05


@dataclass
class ArchitectureV03:
    task_specific_query: bool = True
    token_attention_pool: bool = False  # needs token cache; event residual still ok
    attention_residual: bool = True
    use_image_adapter: bool = True  # fixed on — not a sweep axis
    use_mean_max: bool = True


@dataclass
class OpenSetTrainCfg:
    """Leak-safe defaults during CV / tuning."""

    fold_local_prototype: bool = True
    oof_only_calibration: bool = True
    merge_fold_embeddings: bool = False
    crossfit_disagreement: bool = True
    use_reject: bool = True
    use_open_set: bool = True


@dataclass
class OpenSetInferCfg:
    """Final 5-fold inference: release OOF-only training gates; never merge raw z."""

    fold_local_prototype: bool = True
    oof_only_calibration: bool = False
    merge_fold_embeddings: bool = False
    crossfit_disagreement: bool = False
    use_reject: bool = True
    use_open_set: bool = True


@dataclass
class RoutingThresholds:
    tau_binary: float = 0.5
    tau_fine: float = 0.5
    prototype_percentile: float = 95.0
    min_fold_agreement: int = 4
    energy_tau: Optional[float] = None  # set after calibration


@dataclass
class CvPlanV03:
    tuning_folds: int = 3
    repeats: int = 3
    final_inference_folds: int = 5
    group_key: str = "farm_id"
    seeds: List[int] = field(default_factory=lambda: [2023, 2024, 2025])


@dataclass
class FinetuneV03Config:
    finetune_version: str = FINETUNE_VERSION
    input_dim: int = 384
    hidden_dim: int = 192
    num_heads: int = 4
    ff_dim: int = 384
    dropout: float = 0.2
    num_classes: int = 10
    max_case_age_hours: float = 336.0
    use_mean_max: bool = True
    proj_dim: int = 128
    max_events: int = 4096
    n_folds: int = 5
    normal_class_name: str = "정상_운영"
    # "auto" | float absolute pos_weight for BCEWithLogits
    pos_weight: Union[str, float] = "auto"
    binary_threshold: float = 0.5
    binary_thresholds: List[float] = field(default_factory=lambda: list(BINARY_THRESHOLDS_DEFAULT))
    supcon_temperature: float = 0.1
    consistency_mode: str = "stopgrad_bce"  # or "js"
    loss: LossWeightsV03 = field(default_factory=LossWeightsV03)
    arch: ArchitectureV03 = field(default_factory=ArchitectureV03)
    open_set_train: OpenSetTrainCfg = field(default_factory=OpenSetTrainCfg)
    open_set_infer: OpenSetInferCfg = field(default_factory=OpenSetInferCfg)
    routing: RoutingThresholds = field(default_factory=RoutingThresholds)
    cv: CvPlanV03 = field(default_factory=CvPlanV03)
    image: ImageAdapterConfig = field(default_factory=ImageAdapterConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    monitor: str = "val_loss"

    def pad_config(self) -> EventPoolingConfig:
        return EventPoolingConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim if self.ff_dim else self.hidden_dim * 2,
            dropout=self.dropout,
            num_classes=self.num_classes,
            max_case_age_hours=self.max_case_age_hours,
            use_mean_max=self.use_mean_max and self.arch.use_mean_max,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
