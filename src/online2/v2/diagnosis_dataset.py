"""Case-level diagnosis dataset over Stage A event embedding caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.online2.v2.event_pooling_finetune import VIEW_TO_ID


def load_label_map(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def view_to_id(view: str) -> int:
    return int(VIEW_TO_ID.get(str(view), VIEW_TO_ID["UNKNOWN"]))


def zone_to_id(zone: Any) -> int:
    try:
        z = int(float(zone))
    except (TypeError, ValueError):
        return 0
    return max(0, min(z, 7))


class DiagnosisEventDataset(Dataset):
    """One sample = one case event-summary sequence + diagnosis label."""

    def __init__(
        self,
        case_ids: Sequence[str],
        labels_df: pd.DataFrame,
        label_map: Dict[str, Any],
        emb_dir: Path,
        *,
        max_events: int = 4096,
        require_complete: bool = True,
    ):
        self.emb_dir = Path(emb_dir)
        self.max_events = int(max_events)
        self.name_to_id = {str(k): int(v) for k, v in label_map["name_to_id"].items()}
        labels_df = labels_df.copy()
        labels_df["case_id"] = labels_df["case_id"].astype(str)
        self.labels = labels_df.set_index("case_id")

        self.case_ids: List[str] = []
        for cid in case_ids:
            cid = str(cid)
            path = self.emb_dir / f"{cid}.parquet"
            if not path.exists():
                if require_complete:
                    raise FileNotFoundError(f"missing event cache: {path}")
                continue
            if cid not in self.labels.index:
                raise KeyError(f"no label for case_id={cid}")
            # quick completeness check
            df = pd.read_parquet(path, columns=["event_mean"])
            if df.empty or df["event_mean"].iloc[0] is None:
                if require_complete:
                    raise RuntimeError(f"incomplete cache: {path}")
                continue
            self.case_ids.append(cid)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id = self.case_ids[idx]
        row = self.labels.loc[case_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        diag = str(row["diagnosis_normalized"])
        label = self.name_to_id[diag]
        farm_id = str(row["farm_id"]) if "farm_id" in row else case_id.split("_")[0]

        df = pd.read_parquet(self.emb_dir / f"{case_id}.parquet")
        df = df.sort_values("event_order").reset_index(drop=True)
        if len(df) > self.max_events:
            df = df.iloc[: self.max_events].copy()

        means = np.stack([np.asarray(x, dtype=np.float32) for x in df["event_mean"]])
        maxes = np.stack([np.asarray(x, dtype=np.float32) for x in df["event_max"]])
        t = means.shape[0]
        h = means.shape[1]

        if "case_age_hours" in df.columns:
            ages = df["case_age_hours"].astype(np.float32).to_numpy()
        else:
            ts = pd.to_datetime(df["timestamp"])
            ages = ((ts - ts.iloc[0]) / pd.Timedelta(hours=1)).astype(np.float32).to_numpy()

        if "local_hour" in df.columns:
            hours = df["local_hour"].astype(np.int64).to_numpy()
        else:
            hours = pd.to_datetime(df["timestamp"]).dt.hour.astype(np.int64).to_numpy()

        views = np.asarray([view_to_id(v) for v in df["view"]], dtype=np.int64)
        zones = np.asarray([zone_to_id(z) for z in df["zone"]], dtype=np.int64)

        return {
            "case_id": case_id,
            "farm_id": farm_id,
            "label": int(label),
            "diagnosis_normalized": diag,
            "event_ids": df["event_id"].astype(str).tolist(),
            "event_mean": means,  # [T, H]
            "event_max": maxes,
            "case_age_hours": ages.astype(np.float32),
            "view_id": views,
            "zone_id": zones,
            "local_hour": hours,
            "length": t,
            "hidden_size": h,
        }


def collate_diagnosis_batch(
    samples: List[Dict[str, Any]], *, max_events: int = 4096
) -> Dict[str, torch.Tensor]:
    """Pad variable-length event sequences. Prefer batch_size=1."""
    bsz = len(samples)
    lengths = [int(s["length"]) for s in samples]
    t_max = min(max(lengths), max_events)
    h = int(samples[0]["hidden_size"])

    event_mean = torch.zeros(bsz, t_max, h, dtype=torch.float32)
    event_max = torch.zeros(bsz, t_max, h, dtype=torch.float32)
    case_age = torch.zeros(bsz, t_max, dtype=torch.float32)
    view_id = torch.zeros(bsz, t_max, dtype=torch.long)
    zone_id = torch.zeros(bsz, t_max, dtype=torch.long)
    local_hour = torch.zeros(bsz, t_max, dtype=torch.long)
    padding_mask = torch.zeros(bsz, t_max, dtype=torch.bool)
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)

    for i, s in enumerate(samples):
        t = min(int(s["length"]), t_max)
        event_mean[i, :t] = torch.from_numpy(s["event_mean"][:t])
        event_max[i, :t] = torch.from_numpy(s["event_max"][:t])
        case_age[i, :t] = torch.from_numpy(s["case_age_hours"][:t])
        view_id[i, :t] = torch.from_numpy(s["view_id"][:t])
        zone_id[i, :t] = torch.from_numpy(s["zone_id"][:t])
        local_hour[i, :t] = torch.from_numpy(s["local_hour"][:t])
        padding_mask[i, :t] = True

    return {
        "event_mean": event_mean,
        "event_max": event_max,
        "case_age_hours": case_age,
        "view_id": view_id,
        "zone_id": zone_id,
        "local_hour": local_hour,
        "padding_mask": padding_mask,
        "label": labels,
        "case_ids": [s["case_id"] for s in samples],
        "farm_ids": [s["farm_id"] for s in samples],
    }


def make_stratified_farm_folds(
    labels_df: pd.DataFrame, *, n_folds: int = 5, seed: int = 2023
) -> List[Dict[str, List[str]]]:
    """Approximate stratified group K-fold by farm (group) and diagnosis.

    Each fold's val set is a set of farms; diagnosis distribution is balanced
    greedily across folds.
    """
    rng = np.random.RandomState(seed)
    df = labels_df.copy()
    df["case_id"] = df["case_id"].astype(str)
    df["farm_id"] = df["farm_id"].astype(str)

    # One diagnosis per farm in example_set (1 case/farm typically).
    farm_diag = (
        df.groupby("farm_id")["diagnosis_normalized"].agg(lambda s: s.iloc[0]).to_dict()
    )
    farms = list(farm_diag.keys())
    rng.shuffle(farms)

    folds: List[List[str]] = [[] for _ in range(n_folds)]
    fold_diag_counts = [ {} for _ in range(n_folds)]

    # Sort farms by rare diagnosis first for better balance.
    diag_freq = df["diagnosis_normalized"].value_counts().to_dict()
    farms_sorted = sorted(farms, key=lambda f: (diag_freq.get(farm_diag[f], 0), f))

    for farm in farms_sorted:
        diag = farm_diag[farm]
        # Choose fold with fewest of this diagnosis, then fewest farms.
        best = min(
            range(n_folds),
            key=lambda i: (
                fold_diag_counts[i].get(diag, 0),
                len(folds[i]),
                i,
            ),
        )
        folds[best].append(farm)
        fold_diag_counts[best][diag] = fold_diag_counts[best].get(diag, 0) + 1

    case_by_farm = df.groupby("farm_id")["case_id"].apply(list).to_dict()
    out: List[Dict[str, List[str]]] = []
    for i in range(n_folds):
        val_farms = set(folds[i])
        val_cases = [c for f in val_farms for c in case_by_farm[f]]
        train_cases = [
            c for f, cases in case_by_farm.items() if f not in val_farms for c in cases
        ]
        out.append({"train": train_cases, "val": val_cases, "val_farms": sorted(val_farms)})
    return out
