import pandas as pd
from dataclasses import dataclass

from ..decorators import save_pickle
from ..serialize import DATA_ROOT
from .base import DataSplit
from .from_agri_growth import FromAgriGrowth


@dataclass
class AgriFruitingPopulation(FromAgriGrowth):
    """Agri growth population with last-observation fruiting count as TARGET."""

    name: str = "agri_fruiting_set"

    @save_pickle(
        DATA_ROOT / "processed/populations/{self.name}/population",
        on_validation_error="error",
    )
    def population(self) -> pd.DataFrame:
        result = super().population()
        targets = self._last_fruiting_targets()
        result["TARGET"] = targets.loc[result.index, "TARGET"]
        assert isinstance(result, pd.DataFrame)
        return result

    @save_pickle(
        DATA_ROOT / "interim/populations/{self.name}/target",
        on_validation_error="recompute",
    )
    def _last_fruiting_targets(self) -> pd.DataFrame:
        df = pd.read_csv(self.growth_data.input_csv)
        df["time"] = pd.to_datetime(df["time"])
        fruit = df[df["standardItemName"] == "fruitingnumber"].copy()
        last = fruit.sort_values("time").groupby("idx").tail(1)
        result = pd.DataFrame(
            {"TARGET": last["surveyItemValue"].astype(float).values},
            index=pd.Index(last["idx"].astype(int), name="PERSON_ID"),
        )
        assert isinstance(result, pd.DataFrame)
        return result


@dataclass
class AgriFruitingOverfitPopulation(AgriFruitingPopulation):
    """Same cohort as AgriFruitingPopulation, but train/val/test all use every sample."""

    name: str = "agri_fruiting_overfit"

    def data_split(self) -> DataSplit:
        ids = self.population().index.to_numpy()
        return DataSplit(train=ids, val=ids, test=ids)
