"""Load diagnosis case batch + embedding sidecar metadata for M1 critic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from src.online2.v2.finetune_v03.dataset import (
    DiagnosisEventDatasetV03,
    collate_diagnosis_batch_v03,
    load_label_map,
    normal_class_id_from_map,
)


def load_embedding_sidecar(embeddings_dir: Path, case_id: str) -> pd.DataFrame:
    path = Path(embeddings_dir) / f"{case_id}.parquet"
    cols = [
        "event_id",
        "event_order",
        "same_time_group_id",
        "timestamp",
        "view",
        "zone",
        "case_age_hours",
        "local_hour",
    ]
    df = pd.read_parquet(path, columns=[c for c in cols if c])
    df = df.sort_values("event_order").reset_index(drop=True)
    df["event_index"] = df.index.astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def load_case_batch(
    *,
    case_id: str,
    embeddings_dir: Path,
    labels_path: Path,
    label_map_path: Path,
    max_events: int = 4096,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    label_map = load_label_map(Path(label_map_path))
    normal_id = normal_class_id_from_map(label_map)
    labels_df = pd.read_csv(labels_path)
    ds = DiagnosisEventDatasetV03(
        case_ids=[case_id],
        labels_df=labels_df,
        label_map=label_map,
        emb_dir=Path(embeddings_dir),
        normal_class_id=normal_id,
        require_complete=False,
        max_events=max_events,
    )
    sample = ds[0]
    batch = collate_diagnosis_batch_v03([sample], max_events=max_events)
    side = load_embedding_sidecar(Path(embeddings_dir), case_id)
    # align length to padding mask
    t = int(batch["padding_mask"].shape[1])
    if len(side) > t:
        side = side.iloc[:t].copy()
    batch["_normal_class_id"] = normal_id
    batch["_label_map"] = label_map
    batch["_event_ids"] = side["event_id"].astype(str).tolist()
    batch["_event_index_map"] = {
        str(r.event_id): int(r.event_index) for r in side.itertuples()
    }
    return batch, side


def tensor_batch_only(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    keep = {
        "event_mean",
        "event_max",
        "case_age_hours",
        "view_id",
        "zone_id",
        "local_hour",
        "padding_mask",
        "dino_vec",
        "dino_mask",
    }
    return {k: v for k, v in batch.items() if k in keep and torch.is_tensor(v)}
