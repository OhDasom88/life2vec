"""Deterministic adapter from a validated online2 parquet export."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import dask.dataframe as dd
import pandas as pd

from .base import FIELD_TYPE, TokenSource


@dataclass
class Online2ParquetTokenSource(TokenSource):
    """Expose a frozen online2 build through the existing ``TokenSource`` contract.

    The upstream exporter remains responsible for graph/materialization validation.
    This adapter verifies build identity and deterministic event ordering.
    """

    name: str = "online2"
    fields: List[FIELD_TYPE] = field(default_factory=lambda: ["SENTENCE"])
    downsample: bool = False
    path: str = ""
    build_id: str = ""
    registry_version: str = ""
    token_fields: Tuple[str, ...] = ("SENTENCE",)
    expected_checksum: str = ""

    @staticmethod
    def _files(path: Path) -> List[Path]:
        if path.is_file():
            return [path]
        return sorted(path.rglob("*.parquet"))

    def checksum(self) -> str:
        digest = hashlib.sha256()
        files = self._files(Path(self.path))
        if not files:
            raise FileNotFoundError(f"No parquet files found at {self.path}")
        for file in files:
            digest.update(file.name.encode("utf-8"))
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def tokenized(self) -> dd.DataFrame:
        if not self.path:
            raise ValueError("Online2 parquet path must be configured")
        if self.expected_checksum and self.checksum() != self.expected_checksum:
            raise ValueError("Online2 parquet checksum does not match configuration")

        frame = pd.read_parquet(self.path)
        required = {
            "PERSON_ID",
            "START_DATE",
            "event_id",
            "same_time_group_id",
            "event_kind",
            "event_position",
            "time_group_rank",
            "sequence_id",
            *self.token_fields,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Online2 export is missing required columns: {missing}")

        for column, expected in (
            ("build_id", self.build_id),
            ("registry_version", self.registry_version),
        ):
            if expected:
                if column not in frame:
                    raise ValueError(f"Online2 export has no {column} column")
                actual = set(frame[column].dropna().astype(str).unique())
                if actual != {expected}:
                    raise ValueError(
                        f"{column} mismatch: expected {expected!r}, found {sorted(actual)}"
                    )

        frame["START_DATE"] = pd.to_datetime(frame["START_DATE"], utc=True).dt.tz_localize(
            None
        )
        if "AGE" not in frame:
            frame["AGE"] = 0.0
        # Events may be reused across sequences; uniqueness is per PERSON_ID/sequence.
        person_event = frame.duplicated(["PERSON_ID", "event_id"])
        if person_event.any():
            duplicates = frame.loc[person_event, ["PERSON_ID", "event_id"]].head()
            raise ValueError(
                f"Duplicate online2 (PERSON_ID, event_id) pairs: {duplicates.to_dict('records')}"
            )

        order = [
            "PERSON_ID",
            "START_DATE",
            "time_group_rank",
            "event_position",
            "event_id",
        ]
        frame = frame.sort_values(order, kind="mergesort").set_index("PERSON_ID")
        frame.index.name = "PERSON_ID"
        result = dd.from_pandas(frame, npartitions=1, sort=True)
        if self.downsample:
            result = self.downsample_persons(result)
        return result
