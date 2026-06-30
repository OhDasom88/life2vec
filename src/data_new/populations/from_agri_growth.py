from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from ..decorators import save_pickle
from ..serialize import DATA_ROOT
from ..sources.agri_growth import AgriGrowthTokens
from .base import DataSplit, Population


@dataclass
class FromAgriGrowth(Population):
    """Population of plants (idx) from AgriChallenge growth data."""

    growth_data: AgriGrowthTokens
    name: str = "agri_growth_set"
    min_events: int = 5
    seed: int = 2023
    train_val_test: Tuple[float, float, float] = (0.7, 0.15, 0.15)

    def __post_init__(self) -> None:
        assert sum(self.train_val_test) == 1.0

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        combined = self.combined()
        event_counts = combined.groupby("PERSON_ID").size()
        valid_ids = event_counts[event_counts >= self.min_events].index

        beta = (
            combined.reset_index()
            .groupby("PERSON_ID")
            .agg(
                {
                    "greenhouseId": "first",
                    "measurementLine": "first",
                    "sampleId": "first",
                }
            )
            .loc[valid_ids]
        )

        result = pd.DataFrame(
            {
                "RES_ORIGIN": "GH_" + beta["greenhouseId"].astype(str),
                "GENDER": "LN_" + beta["measurementLine"].astype(str),
            },
            index=beta.index.rename("PERSON_ID"),
        )
        result.index.name = "PERSON_ID"
        sample_month = beta["sampleId"].clip(1, 12).astype(int)
        result["BIRTHDAY"] = pd.to_datetime(
            "2024-" + sample_month.astype(str).str.zfill(2) + "-01"
        )
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(
        DATA_ROOT / "interim/populations/{self.name}/combined",
        on_validation_error="recompute",
    )
    def combined(self) -> pd.DataFrame:
        df = pd.read_csv(self.growth_data.input_csv)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"idx": "PERSON_ID", "time": "START_DATE"})
        df["PERSON_ID"] = df["PERSON_ID"].astype(int)
        result = df.set_index("PERSON_ID")[
            ["START_DATE", "greenhouseId", "measurementLine", "sampleId"]
        ]
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(DATA_ROOT / "processed/populations/{self.name}/data_split")
    def data_split(self) -> DataSplit:
        ids = self.population().index.to_numpy()
        np.random.default_rng(self.seed).shuffle(ids)
        split_idxs = np.round(np.cumsum(self.train_val_test) * len(ids))[:2].astype(int)
        train_ids, val_ids, test_ids = np.split(ids, split_idxs)
        return DataSplit(
            train=train_ids,
            val=val_ids,
            test=test_ids,
        )
