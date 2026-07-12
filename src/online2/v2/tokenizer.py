"""Type-aware V2 tokenizer with measurement groups."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from .binning import BinningRegistryV2
from .feature_schema import FeatureSchema, FeatureSpec
from .vocab import VocabV2
from ..canonical import stable_id


def wind_compass(deg: float) -> str:
    # 0/360 -> N; bins of 45 degrees
    x = deg % 360.0
    idx = int((x + 22.5) // 45.0) % 8
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][idx]


def literal_state(value: Optional[float], raw: str) -> list[str]:
    if value is None or raw == "":
        return ["OBSERVED_VALUE|NULL", "STATE_SEMANTICS|UNRESOLVED"]
    tokens = ["STATE_SEMANTICS|UNRESOLVED"]
    if value == 0:
        tokens.insert(0, "OBSERVED_VALUE|ZERO")
    elif value > 0:
        tokens.insert(0, "OBSERVED_VALUE|POSITIVE")
    else:
        tokens.insert(0, "OBSERVED_VALUE|NEGATIVE")
    if abs(value - round(value)) < 1e-9:
        tokens.append("RAW_CODE_CLASS|INTEGER")
    else:
        tokens.append("RAW_CODE_CLASS|FRACTIONAL")
    return tokens


@dataclass
class MeasurementTokens:
    measurement_group_id: str
    feature: str
    tokens: list[str]
    roles: list[str]  # feature_identity | value_abs | value_global | value_farm | literal | quality | circular
    source_cell_id: str
    source_atomic_value_id: str
    raw_value: Optional[float]


class TokenizerV2:
    def __init__(
        self,
        schema: FeatureSchema,
        binning: BinningRegistryV2,
        vocab: VocabV2,
        farm_relative_enabled: bool = True,
    ) -> None:
        self.schema = schema
        self.binning = binning
        self.vocab = vocab
        self.farm_relative_enabled = farm_relative_enabled

    def tokenize_value(
        self,
        feature: str,
        raw: str,
        *,
        farm_id: str = "",
        cell_id: str = "",
        atomic_value_id: str = "",
    ) -> MeasurementTokens:
        spec = self.schema.features.get(feature)
        if spec is None:
            mg = stable_id("meas", {"feature": feature, "raw": raw, "cell": cell_id})
            return MeasurementTokens(mg, feature, ["[UNK]"], ["unk"], cell_id, atomic_value_id, None)

        value: Optional[float]
        try:
            value = float(raw) if raw != "" else None
        except ValueError:
            value = None

        mg = stable_id(
            "meas",
            {
                "feature": feature,
                "raw": raw,
                "farm": farm_id,
                "cell": cell_id,
                "atomic": atomic_value_id,
            },
        )
        tokens: list[str] = []
        roles: list[str] = []
        tokens.append(f"FEATURE|{feature.upper()}")
        roles.append("feature_identity")

        if value is None:
            tokens.append("OBSERVED_VALUE|NULL")
            roles.append("literal")
            tokens.append("QUALITY|MISSING")
            roles.append("quality")
            return MeasurementTokens(mg, feature, tokens, roles, cell_id, atomic_value_id, None)

        if spec.type == "circular" or spec.circular_encoding == "compass8":
            tokens.append(f"WIND_DIR|{wind_compass(value)}")
            roles.append("circular")
            tokens.append("QUALITY|OK")
            roles.append("quality")
            return MeasurementTokens(mg, feature, tokens, roles, cell_id, atomic_value_id, value)

        if spec.type in {"boolean", "categorical", "ordinal_actuator", "flow"} and not (
            spec.absolute_encoding or spec.global_relative_encoding
        ):
            for lit in literal_state(value, raw):
                tokens.append(lit)
                roles.append("literal")
            tokens.append("QUALITY|OK")
            roles.append("quality")
            return MeasurementTokens(mg, feature, tokens, roles, cell_id, atomic_value_id, value)

        # continuous-like multi-channel encoding
        if spec.absolute_encoding and (feature, "ABS", None) in self.binning.rules:
            suffix = self.binning.encode_suffix(feature, value, "ABS")
            tokens.append(f"VALUE_ABS|{suffix}")
            roles.append("value_abs")
            tokens.append(f"{feature.upper()}|{suffix}")
            roles.append("value_abs_combined")
        if spec.global_relative_encoding and (feature, "GLOBAL_REL", None) in self.binning.rules:
            suffix = self.binning.encode_suffix(feature, value, "GLOBAL_REL")
            tokens.append(f"VALUE_GLOBAL_REL|{suffix}")
            roles.append("value_global")
            tokens.append(f"{feature.upper()}|{suffix}")
            roles.append("value_global_combined")
        if (
            self.farm_relative_enabled
            and spec.farm_relative_encoding
            and farm_id
            and (feature, "FARM_REL", farm_id) in self.binning.rules
        ):
            suffix = self.binning.encode_suffix(feature, value, "FARM_REL", farm_id)
            tokens.append(f"VALUE_FARM_REL|{suffix}")
            roles.append("value_farm")
            tokens.append(f"{feature.upper()}|{suffix}")
            roles.append("value_farm_combined")

        tokens.append("QUALITY|OK")
        roles.append("quality")
        return MeasurementTokens(mg, feature, tokens, roles, cell_id, atomic_value_id, value)
