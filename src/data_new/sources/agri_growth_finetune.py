from dataclasses import dataclass

import pandas as pd

from .agri_growth import AgriGrowthTokens


@dataclass
class AgriGrowthFinetuneTokens(AgriGrowthTokens):
    """Growth tokens for finetuning: drop only the last observation's fruiting count."""

    name: str = "agri_growth_finetune"

    def _load_pandas(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_csv)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(columns={"idx": "PERSON_ID", "time": "START_DATE"})
        df["PERSON_ID"] = df["PERSON_ID"].astype(int)

        last_dates = df.groupby("PERSON_ID")["START_DATE"].transform("max")
        is_last_fruiting = (df["standardItemName"] == "fruitingnumber") & (
            df["START_DATE"] == last_dates
        )
        df = df.loc[~is_last_fruiting].copy()

        first_dates = df.groupby("PERSON_ID")["START_DATE"].transform("min")
        df["AGE"] = (df["START_DATE"] - first_dates).dt.days.astype(float)
        return df
