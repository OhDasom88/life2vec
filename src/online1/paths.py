from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path("/home/dasom/life2vec")
CONFIG_DIR = REPO_ROOT / "configs" / "online1"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "online1"
DEFAULT_DATA = Path("/data/datasets/agrichallenge/online1/dataset")


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipeline_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else CONFIG_DIR / "pipeline.yaml"
    cfg = load_yaml(cfg_path)
    cfg["_config_dir"] = str(CONFIG_DIR)
    cfg["_pipeline_config_path"] = str(cfg_path)
    return cfg


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


@dataclass(frozen=True)
class Online1Paths:
    data_root: Path = DEFAULT_DATA
    output_root: Path = DEFAULT_OUTPUT

    @property
    def train_x(self) -> Path:
        return self.data_root / "train" / "env" / "train_X.csv"

    @property
    def train_y(self) -> Path:
        return self.data_root / "train" / "env" / "train_y.csv"

    @property
    def test_x(self) -> Path:
        return self.data_root / "test" / "env" / "test_X.csv"

    @property
    def train_ms(self) -> Path:
        return self.data_root / "train" / "ms"

    @property
    def test_ms(self) -> Path:
        return self.data_root / "test" / "ms"

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "Online1Paths":
        return cls(
            data_root=Path(cfg.get("data_root", DEFAULT_DATA)),
            output_root=Path(cfg.get("output_root", DEFAULT_OUTPUT)),
        )
