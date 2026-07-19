"""CF-1S feature edit policy loader and feature-edit-profile helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


class EditClass(str, Enum):
    EDITABLE_CONTROL_LIKE = "EDITABLE_CONTROL_LIKE"
    OBSERVATIONAL_ONLY = "OBSERVATIONAL_ONLY"
    FORBIDDEN_OUTCOME = "FORBIDDEN_OUTCOME"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class FeatureEditRecord:
    feature_id: str
    edit_class: EditClass
    schema_kind: str
    adjacency: str
    semantic_confidence: str
    recommendation_eligible: bool
    use_policy: str
    farm_scope_checked: bool
    cross_farm_semantic_assumption: str
    direction_inferred_from_saliency: bool
    feature_edit_profile_id: str
    feature_edit_profile_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["edit_class"] = self.edit_class.value
        return d


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def feature_edit_profile_id(
    *,
    feature_id: str,
    schema_kind: str,
    adjacency: str,
    tokenization_contract: str = "production_tokenizer_v2",
    adjacency_policy_version: str = "CF1S_EDIT_POLICY_V1",
) -> str:
    payload = {
        "feature_id": feature_id,
        "schema_kind": schema_kind,
        "adjacency": adjacency,
        "tokenization_contract": tokenization_contract,
        "adjacency_policy_version": adjacency_policy_version,
    }
    return "fep_" + _sha256_text(json.dumps(payload, sort_keys=True))[:16]


def feature_edit_profile_sha256(
    *,
    feature_id: str,
    schema_kind: str,
    adjacency: str,
    tokenization_contract: str = "production_tokenizer_v2",
    adjacency_policy_version: str = "CF1S_EDIT_POLICY_V1",
) -> str:
    payload = {
        "feature_id": feature_id,
        "schema_kind": schema_kind,
        "adjacency": adjacency,
        "tokenization_contract": tokenization_contract,
        "adjacency_policy_version": adjacency_policy_version,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True))


def load_edit_policy(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"invalid edit policy: {p}")
    data = dict(data)
    data["_policy_path"] = str(p)
    data["_policy_sha256"] = _sha256_text(p.read_text(encoding="utf-8"))
    return data


def get_feature_record(
    policy: Mapping[str, Any],
    feature_id: str,
) -> FeatureEditRecord:
    features = policy.get("features") or {}
    defaults = policy.get("default_record_flags") or {}
    feat = str(feature_id).strip()
    row = features.get(feat)
    if row is None:
        edit_class = EditClass.UNRESOLVED
        schema_kind = "unsupported"
        adjacency = "none"
    else:
        edit_class = EditClass(str(row.get("edit_class") or "UNRESOLVED"))
        schema_kind = str(row.get("schema_kind") or "unsupported")
        adjacency = str(row.get("adjacency") or "none")
    profile_id = feature_edit_profile_id(
        feature_id=feat, schema_kind=schema_kind, adjacency=adjacency
    )
    profile_sha = feature_edit_profile_sha256(
        feature_id=feat, schema_kind=schema_kind, adjacency=adjacency
    )
    return FeatureEditRecord(
        feature_id=feat,
        edit_class=edit_class,
        schema_kind=schema_kind,
        adjacency=adjacency,
        semantic_confidence=str(defaults.get("semantic_confidence") or "MEDIUM"),
        recommendation_eligible=bool(defaults.get("recommendation_eligible", False)),
        use_policy=str(defaults.get("use_policy") or "MODEL_SENSITIVITY_ONLY"),
        farm_scope_checked=bool(defaults.get("farm_scope_checked", False)),
        cross_farm_semantic_assumption=str(
            defaults.get("cross_farm_semantic_assumption") or "UNVERIFIED"
        ),
        direction_inferred_from_saliency=bool(
            defaults.get("direction_inferred_from_saliency", False)
        ),
        feature_edit_profile_id=profile_id,
        feature_edit_profile_sha256=profile_sha,
    )


def is_edit_class_allowed(edit_class: EditClass, *, arm: str) -> bool:
    arm_u = str(arm).upper()
    if edit_class == EditClass.FORBIDDEN_OUTCOME:
        return False
    if edit_class == EditClass.UNRESOLVED:
        return False
    if arm_u == "CONTROL_LIKE":
        return edit_class == EditClass.EDITABLE_CONTROL_LIKE
    if arm_u == "OBSERVATIONAL":
        return edit_class == EditClass.OBSERVATIONAL_ONLY
    return False
