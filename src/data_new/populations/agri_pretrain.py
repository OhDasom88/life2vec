from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..decorators import save_pickle
from ..serialize import DATA_ROOT
from ..sources.agri_environment import GH_PERSON_ID_OFFSET
from ..sources.agri_growth import AgriGrowthTokens
from .base import DataSplit, Population
from .from_agri_growth import FromAgriGrowth


@dataclass
class SanitySplitMixin:
    """Use identical train/val/test ids (sanity-check pretraining)."""

    def data_split(self) -> DataSplit:
        ids = self.population().index.to_numpy()
        return DataSplit(train=ids.copy(), val=ids.copy(), test=ids.copy())


@dataclass
class AgriPretrainPlantPopulation(SanitySplitMixin, FromAgriGrowth):
    """Plants used for growth-only and episode corpora."""

    name: str = "agri_pretrain_plants"
    train_val_test: Tuple[float, float, float] = (1.0, 0.0, 0.0)


@dataclass
class AgriPretrainGreenhousePopulation(SanitySplitMixin, Population):
    """Greenhouse documents (PERSON_ID = 1000 + greenhouseId)."""

    growth_data: AgriGrowthTokens
    name: str = "agri_pretrain_greenhouses"
    greenhouse_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                "RES_ORIGIN": ["GH_" + str(g) for g in self.greenhouse_ids],
                "GENDER": ["LN_0"] * len(self.greenhouse_ids),
                "BIRTHDAY": pd.to_datetime(["2024-11-01"] * len(self.greenhouse_ids)),
            },
            index=pd.Index(
                [GH_PERSON_ID_OFFSET + g for g in self.greenhouse_ids],
                name="PERSON_ID",
            ),
        )
        assert isinstance(result, pd.DataFrame)
        return result


@dataclass
class AgriPretrainUnionPopulation(SanitySplitMixin, Population):
    """Union of plant and greenhouse ids for shared vocabulary fitting."""

    plants: AgriPretrainPlantPopulation
    greenhouses: AgriPretrainGreenhousePopulation
    name: str = "agri_pretrain_union"

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        result = pd.concat(
            [self.plants.population(), self.greenhouses.population()]
        )
        result.index.name = "PERSON_ID"
        assert isinstance(result, pd.DataFrame)
        return result
