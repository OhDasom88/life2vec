"""v0.2 finetune hyper-parameters (classification + semantic alignment)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from src.online2.v2.event_pooling_finetune import EventPoolingConfig
from src.online2.v2.finetune_v02.version import FINETUNE_VERSION


@dataclass
class LossWeights:
    classification_ce: float = 1.0
    state_alignment: float = 0.10
    cause_alignment: float = 0.15
    quality_ranking: float = 0.0  # enable after classification stabilizes


@dataclass
class QualityWeights:
    high: float = 1.0
    medium: float = 0.5
    low: float = 0.1


@dataclass
class ImageAdapterConfig:
    dino_dim: int = 1024
    model_dim: int = 192  # matches PAD hidden by default
    gate_init: float = 0.01
    freeze_dino: bool = True
    normalize_l2: bool = True


@dataclass
class SaliencyConfig:
    top_frac_per_model: float = 0.08
    occlusion_candidate_cap: int = 400
    consensus_events_min: int = 10
    consensus_events_max: int = 30
    positive_agree_min: int = 4  # of 5
    strong_agree: int = 5
    top_rank_frac: float = 0.10
    top_rank_agree_min: int = 3


@dataclass
class FinetuneV02Config:
    """Full Stage B/C (+ semantic) config. PAD knobs mirror v0.1 EventPoolingConfig."""

    finetune_version: str = FINETUNE_VERSION
    input_dim: int = 384
    hidden_dim: int = 192
    num_heads: int = 4
    ff_dim: int = 384
    dropout: float = 0.2
    num_classes: int = 10
    max_case_age_hours: float = 336.0
    use_mean_max: bool = True
    semantic_dim: int = 192  # z_state / z_cause output dim (L2-normalized)
    max_events: int = 4096
    n_folds: int = 5
    loss: LossWeights = field(default_factory=LossWeights)
    quality: QualityWeights = field(default_factory=QualityWeights)
    image: ImageAdapterConfig = field(default_factory=ImageAdapterConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)

    def pad_config(self) -> EventPoolingConfig:
        """Build an immutable-compatible v0.1 PAD config (no v0.1 mutation)."""
        return EventPoolingConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim if self.ff_dim else self.hidden_dim * 2,
            dropout=self.dropout,
            num_classes=self.num_classes,
            max_case_age_hours=self.max_case_age_hours,
            use_mean_max=self.use_mean_max,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
