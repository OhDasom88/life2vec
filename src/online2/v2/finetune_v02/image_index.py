"""Load frozen DINOv2 vectors keyed by IMAGE event_id."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_dino_index(
    *,
    meta_path: Path,
    npy_path: Path,
) -> Dict[str, np.ndarray]:
    """Map event_id → float32[1024] (row-aligned parquet ↔ npy)."""
    meta = pd.read_parquet(meta_path)
    arr = np.load(npy_path)
    if len(meta) != int(arr.shape[0]):
        raise ValueError(
            f"image meta/npy length mismatch: {len(meta)} vs {arr.shape[0]}"
        )
    out: Dict[str, np.ndarray] = {}
    for i, eid in enumerate(meta["event_id"].astype(str).tolist()):
        out[eid] = np.asarray(arr[i], dtype=np.float32)
    return out


def default_dino_paths(root: Path) -> tuple[Path, Path]:
    base = root / "outputs/online2/v2_build/external_embeddings"
    return base / "image_embeddings.parquet", base / "image_embeddings.npy"


def lookup_dino(
    index: Dict[str, np.ndarray], event_id: str
) -> Optional[np.ndarray]:
    return index.get(str(event_id))
