"""Shared I/O and hashing helpers for M1 artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def artifact_meta(
    *,
    model_hash: Optional[str] = None,
    tokenizer_hash: Optional[str] = None,
    registry_hash: Optional[str] = None,
    config_path: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    from . import SCHEMA_VERSION

    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "model_hash": model_hash,
        "tokenizer_hash": tokenizer_hash,
        "registry_hash": registry_hash,
        "config_path": config_path,
    }
    if extra:
        meta.update(dict(extra))
    return meta


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
