import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, Binned, TokenSource

_TIME_RE = re.compile(r"DAT(\d+)\s+(\d+):(\d+)")


def _parse_hackathon_time(value: str) -> pd.Timestamp:
    match = _TIME_RE.match(str(value).strip())
    if match is None:
        raise ValueError(f"Unrecognized hackathon time format: {value!r}")
    day, hour, minute = (int(match.group(i)) for i in range(1, 4))
    return pd.Timestamp("2024-01-01") + pd.Timedelta(
        days=day - 1, hours=hour, minutes=minute
    )


def _default_continuous_fields() -> List[FIELD_TYPE]:
    return [
        Binned("temperature_outside", "TO", n_bins=20),
        Binned("humidity_outside", "HO", n_bins=20),
        Binned("solar_radiation", "SOL", n_bins=20),
        Binned("wind_direction_outside", "WD", n_bins=20),
        Binned("wind_speed_outside", "WS", n_bins=10),
        Binned("temperature", "TI", n_bins=20),
        Binned("humidity", "HI", n_bins=20),
        Binned("co2", "CO2", n_bins=20),
    ]


def _default_discrete_fields() -> List[str]:
    return [
        "rainfall",
        "greenhouse_roof_vent1",
        "greenhouse_roof_vent2",
        "shading_curtain",
        "thermal_curtain",
        "fcu_fan",
        "fcu_pump",
        "circ_fan",
        "co2_supply",
        "tube_rail_valve",
        "fogging",
    ]


_DISCRETE_PREFIX = {
    "rainfall": "RF",
    "greenhouse_roof_vent1": "GRV1",
    "greenhouse_roof_vent2": "GRV2",
    "shading_curtain": "SHD",
    "thermal_curtain": "THM",
    "fcu_fan": "FCUF",
    "fcu_pump": "FCUP",
    "circ_fan": "CIRC",
    "co2_supply": "CO2S",
    "tube_rail_valve": "TRV",
    "fogging": "FOG",
}


@dataclass
class HackathonEnvironmentTokens(TokenSource):
    """Greenhouse environment tokens from the hackathon train_X.csv."""

    name: str = "hackathon_environment"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: ["ENV_TAG"]
        + _default_continuous_fields()
        + _default_discrete_fields()
    )

    input_csv: Path = DATA_ROOT / "rawdata" / "hackathon" / "train_X.csv"

    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
        verify_index=False,
    )
    def tokenized(self) -> dd.DataFrame:
        indexed = self.indexed()
        discrete_assign = {
            column: (lambda x, col=column: _DISCRETE_PREFIX[col] + "_" + x[col].astype("string"))
            for column in _default_discrete_fields()
        }
        result = (
            indexed.assign(ENV_TAG="ENV_H", **discrete_assign)
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
        df["START_DATE"] = df["time"].map(_parse_hackathon_time)
        df = df.rename(columns={"seq_id": "PERSON_ID"})
        df["PERSON_ID"] = df["PERSON_ID"].astype(int)
        first_dates = df.groupby("PERSON_ID")["START_DATE"].transform("min")
        df["AGE"] = (
            (df["START_DATE"] - first_dates).dt.total_seconds() / 60.0
        )
        return df
