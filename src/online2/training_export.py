"""Create the deterministic, build-pinned life2vec training export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .catalog import file_hash, write_json

SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]


def _load_json(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in json.loads(str(value))]


def _person_id(sequence_id: str) -> int:
    digest = hashlib.sha256(sequence_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _category(token: str) -> str:
    if token in SPECIAL_TOKENS:
        return "GENERAL"
    if token in {"IMAGE_EMBED_SLOT", "TEXT_EMBED_SLOT"}:
        return "MODALITY"
    if token.startswith(("FARM|", "ZONE|")):
        return "BACKGROUND"
    return token.split("|", 1)[0] or "TOKEN"


def export_training_manifest(build_dir: Path) -> dict[str, Any]:
    """Join lineage tables without querying Neo4j during training."""
    build_dir = build_dir.resolve()
    manifest_path = build_dir / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_id = str(manifest["build_id"])

    sequences = pd.read_parquet(build_dir / "sequences.parquet")
    segments = pd.read_parquet(build_dir / "sequence_segments.parquet")
    events = pd.read_parquet(build_dir / "events.parquet")
    event_tokens = pd.read_parquet(build_dir / "event_tokens.parquet")

    ordered_tokens = event_tokens.sort_values(
        ["event_id", "position", "token_string"], kind="mergesort"
    )
    sentences = (
        ordered_tokens.groupby("event_id", sort=False)["token_string"]
        .agg(lambda values: " ".join(map(str, values)))
        .rename("SENTENCE")
    )
    if (sentences.str.len() == 0).any():
        raise ValueError("An exported event has an empty token sentence")

    event_columns = [
        "event_id",
        "event_kind",
        "observation_timestamp",
        "same_time_group_id",
        "farm_ids",
        "zone_ids",
        "modalities",
        "embedding_status",
    ]
    sequence_columns = [
        "sequence_id",
        "narrative_id",
        "background_tokens",
        "order_semantics",
        "op_eligible",
    ]
    frame = (
        segments.merge(events[event_columns], on="event_id", validate="many_to_one")
        .merge(sequences[sequence_columns], on="sequence_id", validate="many_to_one")
        .join(sentences, on="event_id", validate="many_to_one")
    )
    if frame.duplicated(["sequence_id", "event_id"]).any():
        raise ValueError("Duplicate event occurrence within a sequence")

    frame = frame.sort_values(
        ["sequence_id", "position", "time_group_rank", "event_id"],
        kind="mergesort",
    )
    frame["PERSON_ID"] = frame["sequence_id"].map(_person_id)
    if frame[["PERSON_ID", "sequence_id"]].drop_duplicates()["PERSON_ID"].duplicated().any():
        raise ValueError("Stable PERSON_ID collision")
    frame["START_DATE"] = pd.to_datetime(
        frame["observation_timestamp"], utc=True
    ).dt.tz_localize(None)
    first_time = frame.groupby("sequence_id")["START_DATE"].transform("min")
    frame["AGE"] = (frame["START_DATE"] - first_time).dt.total_seconds() / 3600.0
    frame["event_position"] = frame["position"].astype("int64")
    frame["modality_ref"] = frame["modalities"]
    frame["farm_id"] = frame["farm_ids"].map(
        lambda value: (_load_json(value) or [""])[0]
    )
    frame["zone_id"] = frame["zone_ids"].map(
        lambda value: (_load_json(value) or [""])[0]
    )
    frame["BACKGROUND_TOKENS"] = frame["background_tokens"]
    frame["SEGMENT"] = (frame["event_position"] % 3 + 1).astype("int64")
    frame["build_id"] = build_id
    frame["registry_version"] = "1"

    output_columns = [
        "PERSON_ID",
        "START_DATE",
        "AGE",
        "SENTENCE",
        "event_id",
        "same_time_group_id",
        "event_kind",
        "event_position",
        "time_group_rank",
        "sequence_id",
        "narrative_id",
        "order_semantics",
        "op_eligible",
        "modality_ref",
        "SEGMENT",
        "farm_id",
        "zone_id",
        "BACKGROUND_TOKENS",
        "build_id",
        "registry_version",
        "embedding_status",
    ]
    export_path = build_dir / "training_events.parquet"
    frame[output_columns].to_parquet(
        export_path, index=False, compression="zstd"
    )

    tokens = set(ordered_tokens["token_string"].astype(str))
    for value in sequences["background_tokens"]:
        tokens.update(_load_json(value))
    sorted_tokens = SPECIAL_TOKENS + sorted(tokens - set(SPECIAL_TOKENS))
    registry = {
        "schema_version": manifest["schema_version"],
        "registry_version": "1",
        "build_id": build_id,
        "tokens": [
            {
                "token_id": index,
                "token": token,
                "category": _category(token),
                "registry_version": "1",
            }
            for index, token in enumerate(sorted_tokens)
        ],
    }
    registry_path = build_dir / "life2vec_token_registry.json"
    write_json(registry_path, registry)

    result = {
        "build_id": build_id,
        "row_count": int(len(frame)),
        "sequence_count": int(frame["sequence_id"].nunique()),
        "event_count": int(frame["event_id"].nunique()),
        "vocabulary_size": len(sorted_tokens),
        "training_events_sha256": file_hash(export_path),
        "token_registry_sha256": file_hash(registry_path),
    }
    write_json(build_dir / "training_export_manifest.json", result)
    return result
