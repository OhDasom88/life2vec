from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..decorators import save_pickle
from ..serialize import DATA_ROOT
from ..sources.hackathon_environment import HackathonEnvironmentTokens
from .base import DataSplit, Population


@dataclass
class HackathonPopulation(Population):
    """Hackathon greenhouse sequences (seq_id as PERSON_ID).

    Train/holdout ratio defaults to 9:1. Validation and test share the same
    holdout IDs (the remaining 10%).
    """

    env_data: HackathonEnvironmentTokens
    name: str = "hackathon_set"
    min_events: int = 1
    seed: int = 2023
    train_val_test: Tuple[float, float, float] = (0.9, 0.05, 0.05)

    def __post_init__(self) -> None:
        assert abs(sum(self.train_val_test) - 1.0) < 1e-6

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        combined = self.combined()
        event_counts = combined.groupby("PERSON_ID").size()
        valid_ids = event_counts[event_counts >= self.min_events].index

        result = pd.DataFrame(
            {
                "RES_ORIGIN": "GH_1",
                "GENDER": "LN_0",
                "BIRTHDAY": pd.Timestamp("2024-01-01"),
            },
            index=pd.Index(valid_ids.astype(int), name="PERSON_ID"),
        )
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(
        DATA_ROOT / "interim/populations/{self.name}/combined",
        on_validation_error="recompute",
    )
    def combined(self) -> pd.DataFrame:
        pdf = self.env_data._load_pandas()
        result = pdf.set_index("PERSON_ID")[["START_DATE"]]
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(DATA_ROOT / "processed/populations/{self.name}/data_split")
    def data_split(self) -> DataSplit:
        ids = self.population().index.to_numpy()
        rng = np.random.default_rng(self.seed)
        rng.shuffle(ids)

        n_train = int(round(self.train_val_test[0] * len(ids)))
        train_ids = ids[:n_train]
        holdout_ids = ids[n_train:]

        return DataSplit(
            train=train_ids,
            val=holdout_ids,
            test=holdout_ids,
        )

    def prepare(self) -> None:
        self.combined()
        self.population()
        self.data_split()


@dataclass
class HackathonClassificationPopulation(HackathonPopulation):
    """Hackathon sequences with binary TARGET from targets.csv."""

    name: str = "hackathon_cls_set"
    targets_csv: Path = DATA_ROOT / "rawdata" / "hackathon" / "targets.csv"

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        base = HackathonPopulation.population(self)
        targets = self._targets()
        base["TARGET"] = targets.loc[base.index, "TARGET"].astype(int)
        assert isinstance(base, pd.DataFrame)
        return base

    @save_pickle(
        DATA_ROOT / "interim/populations/{self.name}/target",
        on_validation_error="recompute",
    )
    def _targets(self) -> pd.DataFrame:
        df = pd.read_csv(self.targets_csv)
        result = (
            df.rename(columns={"seq_id": "PERSON_ID", "target": "TARGET"})
            .astype({"PERSON_ID": int, "TARGET": int})
            .set_index("PERSON_ID")[["TARGET"]]
        )
        assert isinstance(result, pd.DataFrame)
        return result
