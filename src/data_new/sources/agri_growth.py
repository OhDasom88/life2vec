from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, TokenSource


@dataclass
class AgriGrowthTokens(TokenSource):
    """Growth measurement tokens from the AgriChallenge dataset."""

    name: str = "agri_growth"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: [
            "standardCategoryName",
            "standardItemName",
            "value_token",
        ]
    )

    input_csv: Path = (
        DATA_ROOT / "rawdata" / "agrichallenge2023" / "growth_info_2024_1.csv"
    )

    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
        verify_index=False,
    )
    def tokenized(self) -> dd.DataFrame:
        result = (
            self.indexed()
            .assign(
                standardCategoryName=lambda x: "CAT_"
                + x.standardCategoryName.astype("string"),
                standardItemName=lambda x: "ITEM_"
                + x.standardItemName.astype("string"),
                value_token=lambda x: "VAL_"
                + x.surveyItemValue.round(1).astype("string"),
            )
            .pipe(sort_partitions, columns=["START_DATE"])[
                ["START_DATE", "AGE", *self.field_labels()]
            ]
        )
        assert isinstance(result, dd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/indexed",
        on_validation_error="recompute",
        verify_index=False,
    )
    def indexed(self) -> dd.DataFrame:
        pdf = self._load_pandas().sort_values(["PERSON_ID", "START_DATE"])
        result = dd.from_pandas(pdf.set_index("PERSON_ID"), npartitions=1)
        assert isinstance(result, dd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/parsed",
        on_validation_error="error",
        verify_index=False,
    )
    def parsed(self) -> dd.DataFrame:
        pdf = self._load_pandas()
        result = dd.from_pandas(pdf, npartitions=1)
        assert isinstance(result, dd.DataFrame)
        return result

    def _load_pandas(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_csv)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"idx": "PERSON_ID", "time": "START_DATE"})
        df["PERSON_ID"] = df["PERSON_ID"].astype(int)
        first_dates = df.groupby("PERSON_ID")["START_DATE"].transform("min")
        df["AGE"] = (df["START_DATE"] - first_dates).dt.days.astype(float)
        return df
