"""Type-aware V2 tokenizer with measurement groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .binning import BinningRegistryV2
from .feature_schema import FeatureSchema
from .vocab import VocabV2
from ..canonical import stable_id


def wind_compass(deg: float) -> str:
    # 0/360 -> N; bins of 45 degrees
    x = deg % 360.0
    idx = int((x + 22.5) // 45.0) % 8
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][idx]


@dataclass
class MeasurementTokens:
    measurement_group_id: str
    feature: str
    tokens: list[str]
    roles: list[str]  # value_abs_combined | circular | unk
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

        # 2026-07-26: 셀 하나 = 토큰 하나(`FEATURE|value`)로 단순화. 이전에는
        # FEATURE(식별)+OBSERVED_VALUE+STATE_SEMANTICS+RAW_CODE_CLASS+QUALITY를
        # 따로 냈는데, STATE_SEMANTICS는 항상 상수(UNRESOLVED)라 정보가 없었고
        # QUALITY는 OBSERVED_VALUE가 NULL인지 아닌지로 이미 100% 유도되는
        # 중복이었다. RAW_CODE_CLASS(정수/소수 구분)는 정보가 있었지만
        # 사용자가 "심플하게"를 택해 같이 제거하기로 함. feature 식별 토큰을
        # 값 토큰과 분리해서 항상 보이게 하던 것도(GroupedMLM이 feature
        # 정체성은 마스킹 안 하고 값만 마스킹하게 하려던 설계) 없앤다 —
        # life2vec 원본 MLM(tasks_mlm.py)은 애초에 그런 구분 없이
        # 평평한 토큰 시퀀스를 그냥 80/10/10 규칙으로 마스킹하므로, 이 쪽이
        # 오히려 life2vec 원본에 더 가깝다.
        #
        # missing/circular/binned 셋 다 결국 "{FEATURE}|{suffix}" 한 토큰으로
        # 귀결된다. binned 경로는 이제 거의 모든 feature type에 적용된다 —
        # unit_confidence(LOW/UNRESOLVED)로 binning 여부를 가르던 게이트를
        # feature_schema.py에서 없앴다(binning은 물리 단위를 안 쓰므로 confidence
        # 문제와 무관하고, binning.py의 _clip_bins()가 고유값 적은 feature는
        # 알아서 raw값과 동등한 해상도로 자동 축소한다).
        has_abs_rule = (feature, "ABS", None) in self.binning.rules
        if spec.circular_encoding == "compass8" and value is not None:
            suffix = wind_compass(value)
            role = "circular"
        elif has_abs_rule:
            suffix = self.binning.encode_suffix(feature, value, "ABS")  # value=None -> "ABS_NULL"
            role = "value_abs_combined"
        else:
            # 방어적 fallback(정상 스키마라면 여기 안 옴): binning 규칙이 없는
            # feature는 raw 문자열을 그대로 값으로 쓴다.
            suffix = raw if raw else "NULL"
            role = "value_abs_combined"

        token = f"{feature.upper()}|{suffix}"
        return MeasurementTokens(mg, feature, [token], [role], cell_id, atomic_value_id, value)
