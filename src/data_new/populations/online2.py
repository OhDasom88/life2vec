"""Leakage-resistant population split for online2 sequence entities."""

import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from ..sources.online2 import Online2ParquetTokenSource
from .base import DataSplit, Population


@dataclass
class Online2Population(Population):
    source: Online2ParquetTokenSource
    name: str = "online2"
    seed: int = 2023
    train_val_test: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_group_column: str = "farm_id"
    default_birthday: str = "2000-01-01"

    def _entities(self) -> pd.DataFrame:
        frame = self.source.tokenized().compute().reset_index()
        if self.split_group_column not in frame:
            frame[self.split_group_column] = frame["PERSON_ID"].astype(str)
        return frame.sort_values(
            ["PERSON_ID", "START_DATE", "event_position"], kind="mergesort"
        ).drop_duplicates("PERSON_ID", keep="first")

    def population(self) -> pd.DataFrame:
        entities = self._entities()
        result = pd.DataFrame(index=pd.Index(entities["PERSON_ID"], name="PERSON_ID"))
        result["BIRTHDAY"] = pd.to_datetime(
            entities.get("BIRTHDAY", self.default_birthday).values
            if "BIRTHDAY" in entities
            else [self.default_birthday] * len(entities)
        )
        result["GENDER"] = (
            entities["GENDER"].fillna("[UNK]").astype(str).values
            if "GENDER" in entities
            else "[UNK]"
        )
        result["RES_ORIGIN"] = (
            entities["RES_ORIGIN"].fillna("[UNK]").astype(str).values
            if "RES_ORIGIN" in entities
            else "[UNK]"
        )
        result["_SPLIT_GROUP"] = entities[self.split_group_column].astype(str).values
        return result

    def data_split(self) -> DataSplit:
        ratios = np.asarray(self.train_val_test, dtype=float)
        if len(ratios) != 3 or np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
            raise ValueError("train_val_test must contain three non-negative ratios summing to 1")
        population = self.population()
        thresholds = np.cumsum(ratios)
        buckets = {}
        for group in population["_SPLIT_GROUP"].unique():
            digest = hashlib.sha256(f"{self.seed}:{group}".encode("utf-8")).digest()
            buckets[group] = int.from_bytes(digest[:8], "big") / float(2**64)
        values = population["_SPLIT_GROUP"].map(buckets)
        ids = population.index.to_numpy()
        return DataSplit(
            train=ids[(values < thresholds[0]).to_numpy()],
            val=ids[
                ((values >= thresholds[0]) & (values < thresholds[1])).to_numpy()
            ],
            test=ids[(values >= thresholds[1]).to_numpy()],
        )
