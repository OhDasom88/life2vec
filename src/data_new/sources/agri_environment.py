from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dask.dataframe as dd
import pandas as pd

from ..decorators import save_parquet
from ..ops import sort_partitions
from ..serialize import DATA_ROOT
from .base import FIELD_TYPE, Binned, TokenSource

GH_PERSON_ID_OFFSET = 1000


def _default_env_fields() -> List[FIELD_TYPE]:
    return [
        Binned("internalTemperature", "TEMP", n_bins=20),
        Binned("internalHumidity", "HUM", n_bins=20),
        Binned("internalCo2", "CO2", n_bins=20),
        Binned("externalSolarRadiation", "SOL", n_bins=20),
        Binned("supplyEc", "EC", n_bins=10),
        Binned("supplyPh", "PH", n_bins=10),
        Binned("externalTemperature", "ET", n_bins=20),
        Binned("externalWindSpeed", "WS", n_bins=10),
    ]


@dataclass
class AgriEnvironmentTokens(TokenSource):
    """Greenhouse environment tokens for plant or greenhouse documents.

    :param narrative: ``hourly`` or ``episode`` (env on growth observation days).
    :param person_scope: ``plant`` (idx) or ``greenhouse`` (1000 + greenhouseId).
    """

    name: str = "agri_environment"
    fields: List[FIELD_TYPE] = field(
        default_factory=lambda: ["ENV_TAG"] + _default_env_fields()
    )
    narrative: str = "hourly"
    person_scope: str = "plant"

    env_csv: Path = (
        DATA_ROOT / "rawdata" / "agrichallenge2023" / "environment_total_2024.csv"
    )
    growth_csv: Path = (
        DATA_ROOT / "rawdata" / "agrichallenge2023" / "growth_info_2024_1.csv"
    )

    @save_parquet(
        DATA_ROOT / "processed/sources/{self.name}/tokenized",
        on_validation_error="error",
        verify_index=False,
    )
    def tokenized(self) -> dd.DataFrame:
        pdf = self._build_events().assign(ENV_TAG="ENV_H")
        cols = ["PERSON_ID", "START_DATE", "AGE", *self.field_labels()]
        pdf = pdf[cols]
        result = (
            dd.from_pandas(pdf, npartitions=1)
            .set_index("PERSON_ID", sorted=False)
            .pipe(sort_partitions, columns=["START_DATE"])
        )
        assert isinstance(result, dd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/indexed",
        on_validation_error="recompute",
        verify_index=False,
    )
    def indexed(self) -> dd.DataFrame:
        pdf = self._build_events().set_index("PERSON_ID")
        result = dd.from_pandas(pdf, npartitions=1)
        assert isinstance(result, dd.DataFrame)
        return result

    @save_parquet(
        DATA_ROOT / "interim/sources/{self.name}/parsed",
        on_validation_error="error",
        verify_index=False,
    )
    def parsed(self) -> dd.DataFrame:
        result = dd.from_pandas(self._build_events(), npartitions=1)
        assert isinstance(result, dd.DataFrame)
        return result

    def _build_events(self) -> pd.DataFrame:
        env = pd.read_csv(self.env_csv)
        env["time"] = pd.to_datetime(env["time"])
        growth = pd.read_csv(self.growth_csv)
        growth["time"] = pd.to_datetime(growth["time"])

        g_start = growth["time"].min()
        g_end = growth["time"].max()
        env = env[(env["time"] >= g_start) & (env["time"] <= g_end + pd.Timedelta(days=1))]

        if self.person_scope == "greenhouse":
            env["PERSON_ID"] = GH_PERSON_ID_OFFSET + env["greenhouseId"].astype(int)
            if self.narrative == "episode":
                gh_obs_days = (
                    growth.groupby("greenhouseId")["time"]
                    .apply(lambda s: set(s.dt.normalize()))
                    .to_dict()
                )
                env = env[
                    env.apply(
                        lambda r: r["time"].normalize()
                        in gh_obs_days.get(int(r["greenhouseId"]), set()),
                        axis=1,
                    )
                ]
        elif self.person_scope == "plant":
            if self.narrative != "episode":
                raise ValueError("plant scope only supports narrative=episode")
            frames: List[pd.DataFrame] = []
            plant_gh = growth.groupby("idx")["greenhouseId"].first()
            for plant_id, gh in plant_gh.items():
                obs_days = growth.loc[
                    growth["idx"] == plant_id, "time"
                ].dt.normalize().unique()
                sub = env[env["greenhouseId"] == gh].copy()
                sub = sub[sub["time"].dt.normalize().isin(obs_days)]
                sub["PERSON_ID"] = int(plant_id)
                frames.append(sub)
            env = pd.concat(frames, ignore_index=True) if frames else env.iloc[0:0]
        else:
            raise ValueError(f"Unknown person_scope: {self.person_scope}")

        env = env.rename(columns={"time": "START_DATE"})
        env = env.sort_values(["PERSON_ID", "START_DATE"])
        first_dates = env.groupby("PERSON_ID")["START_DATE"].transform("min")
        env["AGE"] = (env["START_DATE"] - first_dates).dt.total_seconds() / 3600.0
        return env.reset_index(drop=True)
