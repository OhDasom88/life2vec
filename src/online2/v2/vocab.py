"""Deterministic V2 vocabulary builder."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .feature_schema import FeatureSchema
from .binning import BinningRegistryV2

SPECIAL = [
    "[PAD]",
    "[UNK]",
    "[MASK]",
    "[CLS]",
    "[SEP]",
    "[SEQ_SEP]",
    "[GROUP_SEP]",
    "[EVENT_SEP]",
    "[IMAGE_SLOT]",
    "[TEXT_SLOT]",
    "[MISSING]",
    "[INVALID]",
]

VIEWS = ["ENVIRONMENT", "ROOTZONE", "ACTUATOR", "GROWTH", "IMAGE", "INTERPRETATION"]
EVENT_KINDS = ["OBSERVATION", "STATE_CHANGE", "DERIVED", "IMAGE", "INTERPRETATION"]
IMAGE_ROLES = ["HISTORY", "QUERY", "UNRESOLVED"]
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DELTA_T = ["1H", "3H", "6H", "12H", "1D", "3D", "7D", "GT_7D"]
DAY_FROM_START = ["D00_02", "D03_07", "D08_14", "D15_21", "D22_PLUS"]
LOCAL_HOURS = [f"H{h:02d}" for h in range(24)]
FARM_LOCAL_SLOTS = 24  # crossfarm_hourly/crossfarm_growth 서사는 max_events=20까지 서로 다른 농장을 한 시퀀스에 담는다(기존 8칸으로는 부족)
SPATIAL = [f"FARM_LOCAL|{i}" for i in range(FARM_LOCAL_SLOTS)] + [f"ZONE_LOCAL|{i}" for i in range(8)]
DYNAMIC = [
    "STATE_TRANSITION|OFF_TO_ON",
    "STATE_TRANSITION|ON_TO_OFF",
    "TREND_1H|UP",
    "TREND_1H|DOWN",
    "TREND_3H|STRONG_UP",
    "TREND_3H|STRONG_DOWN",
    "RUN_HIGH|3H",
    "RUN_HIGH|6H",
    "RUN_LOW|3H",
    "RUN_LOW|6H",
    "FLOW|ONSET",
    "FLOW|OFFSET",
    "ACTUATOR|ONSET",
    "ACTUATOR|OFFSET",
    "DELTA_1H|UP",
    "DELTA_1H|DOWN",
    "DELTA_1H|FLAT",
]


TOKEN_USAGE = {
    "NARRATIVE": {
        "input_allowed": False,
        "target_allowed": False,
        "mask_allowed": False,
        "random_replacement_allowed": False,
        "metadata_only": True,
    },
    "CATEGORY": {
        "input_allowed": False,
        "target_allowed": False,
        "mask_allowed": False,
        "random_replacement_allowed": False,
        "metadata_only": True,
    },
    "DATASET": {
        "input_allowed": False,
        "target_allowed": False,
        "mask_allowed": False,
        "random_replacement_allowed": False,
        "metadata_only": True,
    },
    "VALUE_ABS": {
        "input_allowed": True,
        "target_allowed": True,
        "mask_allowed": True,
        "random_replacement_allowed": True,
        "metadata_only": False,
    },
    "VALUE_GLOBAL_REL": {
        "input_allowed": True,
        "target_allowed": True,
        "mask_allowed": True,
        "random_replacement_allowed": True,
        "metadata_only": False,
    },
    "VALUE_FARM_REL": {
        "input_allowed": True,
        "target_allowed": True,
        "mask_allowed": True,
        "random_replacement_allowed": True,
        "metadata_only": False,
    },
    "FEATURE": {
        "input_allowed": True,
        "target_allowed": False,
        "mask_allowed": False,
        "random_replacement_allowed": False,
        "metadata_only": False,
    },
    "LITERAL_STATE": {
        "input_allowed": True,
        "target_allowed": True,
        "mask_allowed": True,
        "random_replacement_allowed": True,
        "metadata_only": False,
    },
    "IMAGE_SLOT": {
        "input_allowed": True,
        "target_allowed": False,
        "mask_allowed": False,
        "random_replacement_allowed": False,
        "metadata_only": False,
    },
    "TEXT_SLOT": {
        "input_allowed": True,
        "target_allowed": False,
        "mask_allowed": False,
        "random_replacement_allowed": False,
        "metadata_only": False,
    },
    "SPECIAL": {
        "input_allowed": True,
        "target_allowed": False,
        "mask_allowed": False,
        "random_replacement_allowed": False,
        "metadata_only": False,
    },
}


def _family(token: str) -> str:
    if token in SPECIAL:
        return "SPECIAL"
    if token.startswith("FEATURE|"):
        return "FEATURE"
    if token.startswith("VALUE_ABS|") or "|ABS_" in token:
        return "VALUE_ABS"
    if token.startswith("VALUE_GLOBAL_REL|") or "|GLOBAL_REL_" in token:
        return "VALUE_GLOBAL_REL"
    if token.startswith("VALUE_FARM_REL|") or "|FARM_REL_" in token:
        return "VALUE_FARM_REL"
    if token.startswith("OBSERVED_VALUE|") or token.startswith("RAW_CODE_CLASS|") or token.startswith("STATE_SEMANTICS|"):
        return "LITERAL_STATE"
    if token in {"[IMAGE_SLOT]", "IMAGE_SLOT"} or token.startswith("IMAGE_ROLE|"):
        return "IMAGE_SLOT"
    if token in {"[TEXT_SLOT]", "TEXT_SLOT"}:
        return "TEXT_SLOT"
    if token.startswith("LOCAL_HOUR|"):
        return "LOCAL_HOUR"
    if token.startswith("DELTA_T|"):
        return "DELTA_T"
    if token.startswith("NARRATIVE|"):
        return "NARRATIVE"
    if token.startswith("CATEGORY|"):
        return "CATEGORY"
    if token.startswith("DATASET|"):
        return "DATASET"
    if token.startswith("FARM|") or token.startswith("ZONE|"):
        return "RAW_SPATIAL"
    return token.split("|", 1)[0]


@dataclass
class VocabV2:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    family_to_ids: Dict[str, List[int]]
    meta: Dict[str, Any]

    def size(self) -> int:
        return len(self.token_to_id)

    def get(self, token: str, default: Optional[int] = None) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if default is not None:
            return default
        return self.token_to_id["[UNK]"]

    def save(self, path: Path) -> None:
        payload = {
            "meta": self.meta,
            "tokens": [
                {
                    "token_id": i,
                    "token": self.id_to_token[i],
                    "family": _family(self.id_to_token[i]),
                    "registry_version": self.meta.get("vocab_version", "v2"),
                }
                for i in range(self.size())
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VocabV2":
        payload = json.loads(path.read_text(encoding="utf-8"))
        token_to_id = {row["token"]: int(row["token_id"]) for row in payload["tokens"]}
        id_to_token = {int(row["token_id"]): row["token"] for row in payload["tokens"]}
        family_to_ids: Dict[str, List[int]] = {}
        for row in payload["tokens"]:
            family_to_ids.setdefault(row.get("family") or _family(row["token"]), []).append(int(row["token_id"]))
        return cls(token_to_id, id_to_token, family_to_ids, payload.get("meta", {}))


def build_vocab_v2(
    schema: FeatureSchema,
    binning: BinningRegistryV2,
    *,
    code_commit_hash: str = "",
    corpus_scope: str = "transductive_public_pretrain_pool",
    narrative_ids: Iterable[str] = (),
) -> VocabV2:
    tokens: list[str] = []
    tokens.extend(SPECIAL)
    for narrative_id in sorted(set(narrative_ids)):
        tokens.append(f"NARRATIVE|{narrative_id}")
    for view in VIEWS:
        tokens.append(f"VIEW|{view}")
    for kind in EVENT_KINDS:
        tokens.append(f"EVENT_KIND|{kind}")
    for role in IMAGE_ROLES:
        tokens.append(f"IMAGE_ROLE|{role}")
    for item in DELTA_T:
        tokens.append(f"DELTA_T|{item}")
    for item in DAY_FROM_START:
        tokens.append(f"DAY_FROM_START|{item}")
    for item in LOCAL_HOURS:
        tokens.append(f"LOCAL_HOUR|{item}")
    tokens.extend(SPATIAL)
    tokens.extend(DYNAMIC)
    tokens.append("STATIC_SEMANTICS|UNRESOLVED")
    tokens.append("STATIC_OBSERVED_VALUE|ZERO")
    tokens.append("STATIC_OBSERVED_VALUE|POSITIVE")

    # 2026-07-26: 셀 토큰이 이제 전부 "{FEATURE}|{value}" 하나뿐이라, 바깥의 bare
    # FEATURE|{name} 식별 토큰은 안 쓴다(STATIC_FEATURE는 별개 메커니즘이라 유지).
    # circular feature(compass8)는 binning.rules에 안 잡히므로 여기서 직접
    # {FEATURE}|{방위}(+NULL) 조합 토큰을 만들어 준다 -- ABS/GLOBAL_REL/FARM_REL
    # bin 토큰과 같은 "combined" 패턴.
    for name, spec in sorted(schema.features.items()):
        if spec.type in {"identifier", "image", "text"}:
            continue
        tokens.append(f"STATIC_FEATURE|{name.upper()}")
        if spec.circular_encoding == "compass8":
            for c in COMPASS:
                tokens.append(f"{name.upper()}|{c}")
            tokens.append(f"{name.upper()}|NULL")

    value_family_tokens: set[str] = set()
    combined_tokens: set[str] = set()
    for key in sorted(binning.rules, key=lambda item: (item[0], item[1], item[2] or "")):
        rule = binning.rules[key]
        prefix = rule.feature.upper()
        if rule.constant:
            suffixes = [f"{rule.kind}_CONSTANT", f"{rule.kind}_NULL"]
        else:
            n_bins = max(1, len(rule.edges) - 1)
            suffixes = [f"{rule.kind}_B{i:02d}" for i in range(n_bins)] + [f"{rule.kind}_NULL"]
        kind_map = {
            "ABS": "VALUE_ABS",
            "GLOBAL_REL": "VALUE_GLOBAL_REL",
            "FARM_REL": "VALUE_FARM_REL",
        }
        fam = kind_map[rule.kind]
        for suffix in suffixes:
            value_family_tokens.add(f"{fam}|{suffix}")
            combined_tokens.add(f"{prefix}|{suffix}")
    tokens.extend(sorted(value_family_tokens))
    tokens.extend(sorted(combined_tokens))

    # Deduplicate preserving order
    seen = set()
    ordered = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)

    token_to_id = {token: idx for idx, token in enumerate(ordered)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    family_to_ids: Dict[str, List[int]] = {}
    for token, idx in token_to_id.items():
        family_to_ids.setdefault(_family(token), []).append(idx)

    meta = {
        "vocab_version": "v2",
        "feature_schema_version": schema.meta.get("schema_version"),
        "config_hash": schema.config_hash(),
        "binning_registry_hash": binning.meta.get("registry_hash"),
        "git_commit_hash": code_commit_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_scope": corpus_scope,
        "training_mode": "transductive_public_pretraining",
        "contains_problem_observations": True,
        "contains_problem_images": True,
        "contains_problem_hidden_targets": False,
        "vocab_fit_scope": corpus_scope,
        "size": len(ordered),
    }
    # ensure PAD=0
    assert token_to_id["[PAD]"] == 0
    return VocabV2(token_to_id, id_to_token, family_to_ids, meta)


def save_token_usage_policy(path: Path) -> None:
    path.write_text(json.dumps(TOKEN_USAGE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
