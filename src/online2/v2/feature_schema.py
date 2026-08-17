"""Typed feature schema for Online2 V2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
import yaml

SCHEMA_VERSION = "v2.0.0"


@dataclass
class FeatureSpec:
    feature_name: str
    type: str
    raw_type: str = "unknown"
    canonical_unit: str = "unresolved"
    unit_confidence: str = "UNRESOLVED"
    valid_min: Optional[float] = None
    valid_max: Optional[float] = None
    missing_policy: str = "MISSING"
    zero_semantics: str = "UNRESOLVED"
    noise_epsilon: Optional[float] = None
    absolute_encoding: bool = False
    global_relative_encoding: bool = False
    farm_relative_encoding: bool = False
    state_mapping: Dict[str, str] = field(default_factory=dict)
    circular_encoding: str = "none"  # none|compass8|sincos
    threshold_tokens_enabled: bool = False
    domain_threshold_ready: bool = False
    # Optional binning overrides (None → BinningPolicy / adaptive defaults).
    n_bins_abs: Optional[int] = None
    n_bins_global_rel: Optional[int] = None
    n_bins_farm_rel: Optional[int] = None
    binning_tier: str = ""
    evidence: str = ""
    schema_version: str = SCHEMA_VERSION
    event_view: str = ""
    source_file_family: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureSchema:
    def __init__(self, features: Mapping[str, FeatureSpec], meta: Optional[dict] = None) -> None:
        self.features = dict(features)
        self.meta = meta or {
            "schema_version": SCHEMA_VERSION,
            "training_mode": "transductive_public_pretraining",
        }

    def get(self, name: str) -> FeatureSpec:
        if name not in self.features:
            raise KeyError(name)
        return self.features[name]

    def names(self) -> list[str]:
        return sorted(self.features)

    def measurable(self) -> list[FeatureSpec]:
        skip = {"identifier", "image", "text"}
        return [self.features[n] for n in self.names() if self.features[n].type not in skip]

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "features": {name: spec.to_dict() for name, spec in sorted(self.features.items())},
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.to_yaml_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_yaml_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureSchema":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        features = {
            name: FeatureSpec(**{k: v for k, v in spec.items() if k in FeatureSpec.__dataclass_fields__})
            for name, spec in payload["features"].items()
        }
        return cls(features, payload.get("meta"))


def build_feature_schema_from_audit(audit_csv: Path) -> FeatureSchema:
    frame = pd.read_csv(audit_csv).drop_duplicates("feature_name", keep="first")
    features: dict[str, FeatureSpec] = {}
    for row in frame.itertuples(index=False):
        name = str(row.feature_name)
        ftype = str(row.proposed_v2_type)
        conf = str(row.unit_confidence)
        # 2026-07-26 사용자 결정: binning 여부를 unit_confidence(단위 라벨 확신도)로 가르던
        # 기존 게이트를 없앤다. equal-frequency binning은 물리 단위를 전혀 안 쓰고 그 feature
        # 자체의 관측 분포 안에서의 상대적 위치만 표현하므로, "이 숫자가 정확히 몇 %/어떤
        # 스케일인지 확신 없음"(LOW/UNRESOLVED)이 binning 자체를 막을 이유가 안 된다.
        # 대신 binning.py의 _clip_bins()가 이미 n_unique로 자동으로 bin 수를 제한하므로
        # (예: 고유값 2개인 boolean은 자동으로 2 bin = raw 값과 동등한 해상도, 고유값
        # 수천 개인 ordinal_actuator는 제대로 압축됨) 사람이 type/confidence로 미리
        # binning 여부를 나눌 필요가 없다. identifier/image/text(측정값이 아님)와
        # circular(각도, 등빈도 binning이 의미 없음 -- compass8 유지)만 예외.
        if ftype == "circular":
            abs_enc = False
            circular = "compass8"
            zero_sem = "N/A"
        elif ftype in {"identifier", "image", "text"}:
            abs_enc = False
            circular = "none"
            zero_sem = "N/A"
        else:
            abs_enc = True
            circular = "none"
            zero_sem = "PHYSICAL_ZERO" if conf in {"CONFIRMED", "HIGH"} else "UNRESOLVED"
        # GLOBAL_REL/FARM_REL 상대 bin은 tokenizer.py Phase 2b에서 이미 안 쓰게 됐으므로
        # (VALUE_ABS combined 토큰 하나만 유지) 여기서도 계산 자체를 끈다 -- FARM_REL은
        # 농장별로 반복 계산되는 비용이 커서, 안 쓰는 걸 계속 fit하는 낭비를 없앤다.
        global_enc = False
        farm_enc = False

        noise = None
        if bool(getattr(row, "noise_epsilon_ready", False)):
            noise = 1e-3
        features[name] = FeatureSpec(
            feature_name=name,
            type=ftype,
            raw_type=str(getattr(row, "raw_type", "unknown")),
            canonical_unit=str(getattr(row, "canonical_unit", "unresolved")),
            unit_confidence=conf,
            valid_min=None if pd.isna(getattr(row, "raw_min", None)) else float(row.raw_min),
            valid_max=None if pd.isna(getattr(row, "raw_max", None)) else float(row.raw_max),
            missing_policy="MISSING",
            zero_semantics=zero_sem,
            noise_epsilon=noise,
            absolute_encoding=abs_enc,
            global_relative_encoding=global_enc,
            farm_relative_encoding=farm_enc,
            circular_encoding=circular,
            threshold_tokens_enabled=bool(getattr(row, "threshold_ready", False)),
            domain_threshold_ready=bool(getattr(row, "threshold_ready", False)),
            evidence=str(getattr(row, "unit_evidence", "")),
            event_view=str(getattr(row, "event_view", "")),
            source_file_family=str(getattr(row, "source_file_family", "")),
        )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "training_mode": "transductive_public_pretraining",
        "contains_problem_observations": True,
        "contains_problem_images": True,
        "contains_problem_hidden_targets": False,
        "source_audit": str(audit_csv),
    }
    return FeatureSchema(features, meta)


def load_feature_schema(path: Path) -> FeatureSchema:
    return FeatureSchema.load(path)
