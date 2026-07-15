"""Dataset wrappers for finetune v0.2 (extends v0.1 caches; does not alter v0.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.online2.v2.diagnosis_dataset import (
    DiagnosisEventDataset,
    collate_diagnosis_batch,
    load_label_map,
    make_stratified_farm_folds,
)

# Re-export v0.1 helpers so v02 scripts import from one place.
__all__ = [
    "DiagnosisEventDataset",
    "DiagnosisEventDatasetV02",
    "collate_diagnosis_batch",
    "collate_diagnosis_batch_v02",
    "load_label_map",
    "make_stratified_farm_folds",
    "load_interpretation_bank",
]


def load_interpretation_bank(path: Path) -> pd.DataFrame:
    """Parquet with columns: case_id, quality, axis, embedding (list[float]).

    axis ∈ {state, cause}; quality ∈ {high, medium, low}.
    """
    df = pd.read_parquet(path)
    required = {"case_id", "quality", "axis", "embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interpretation bank missing columns: {sorted(missing)}")
    return df


class DiagnosisEventDatasetV02(DiagnosisEventDataset):
    """v0.1 event cache + optional High/Medium/Low state/cause embeddings."""

    def __init__(
        self,
        case_ids: Sequence[str],
        labels_df: pd.DataFrame,
        label_map: Dict[str, Any],
        emb_dir: Path,
        *,
        interpretation_bank: Optional[pd.DataFrame] = None,
        semantic_dim: int = 192,
        max_events: int = 4096,
        require_complete: bool = True,
    ):
        super().__init__(
            case_ids,
            labels_df,
            label_map,
            emb_dir,
            max_events=max_events,
            require_complete=require_complete,
        )
        self.semantic_dim = int(semantic_dim)
        self._bank_index: Dict[tuple, np.ndarray] = {}
        if interpretation_bank is not None and len(interpretation_bank):
            for _, row in interpretation_bank.iterrows():
                key = (str(row["case_id"]), str(row["axis"]), str(row["quality"]))
                self._bank_index[key] = np.asarray(row["embedding"], dtype=np.float32)

    def _pack_axis(self, case_id: str, axis: str) -> tuple[np.ndarray, np.ndarray]:
        """Return targets [3, D] and present mask [3] for H/M/L."""
        order = ("high", "medium", "low")
        dim = self.semantic_dim
        for q in order:
            vec = self._bank_index.get((case_id, axis, q))
            if vec is not None:
                dim = int(vec.shape[0])
                break
        targets = np.zeros((3, dim), dtype=np.float32)
        present = np.zeros(3, dtype=np.bool_)
        for i, q in enumerate(order):
            vec = self._bank_index.get((case_id, axis, q))
            if vec is None:
                continue
            targets[i] = vec
            n = float(np.linalg.norm(targets[i]) + 1e-8)
            targets[i] = targets[i] / n
            present[i] = True
        return targets, present

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        case_id = sample["case_id"]
        state_t, state_p = self._pack_axis(case_id, "state")
        cause_t, cause_p = self._pack_axis(case_id, "cause")
        sample["state_targets"] = state_t
        sample["state_present"] = state_p
        sample["cause_targets"] = cause_t
        sample["cause_present"] = cause_p
        sample["has_semantic"] = bool(state_p.any() or cause_p.any())

        # Optional IMAGE event DINO vectors from Stage A v02 cache
        df = pd.read_parquet(self.emb_dir / f"{case_id}.parquet")
        df = df.sort_values("event_order").reset_index(drop=True)
        if len(df) > self.max_events:
            df = df.iloc[: self.max_events].copy()
        t = int(sample["length"])
        h_dino = 1024
        dino = np.zeros((t, h_dino), dtype=np.float32)
        dino_mask = np.zeros(t, dtype=np.bool_)
        if "dino_vec" in df.columns:
            for i in range(min(t, len(df))):
                vec = df["dino_vec"].iloc[i]
                if vec is None:
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                if arr.ndim != 1 or arr.size == 0:
                    continue
                if float(np.linalg.norm(arr)) < 1e-8:
                    continue
                h_dino = int(arr.shape[0])
                if dino.shape[1] != h_dino:
                    dino = np.zeros((t, h_dino), dtype=np.float32)
                dino[i] = arr
                dino_mask[i] = True
        sample["dino_vec"] = dino
        sample["dino_mask"] = dino_mask
        sample["dino_dim"] = int(h_dino)
        return sample


def collate_diagnosis_batch_v02(
    samples: List[Dict[str, Any]], *, max_events: int = 4096
) -> Dict[str, Any]:
    batch = collate_diagnosis_batch(samples, max_events=max_events)
    if not samples:
        return batch

    if "state_targets" in samples[0]:
        state_dim = int(samples[0]["state_targets"].shape[-1])
        cause_dim = int(samples[0]["cause_targets"].shape[-1])
        bsz = len(samples)
        state_targets = torch.zeros(bsz, 3, state_dim, dtype=torch.float32)
        cause_targets = torch.zeros(bsz, 3, cause_dim, dtype=torch.float32)
        state_present = torch.zeros(bsz, 3, dtype=torch.bool)
        cause_present = torch.zeros(bsz, 3, dtype=torch.bool)
        for i, s in enumerate(samples):
            state_targets[i] = torch.from_numpy(s["state_targets"])
            cause_targets[i] = torch.from_numpy(s["cause_targets"])
            state_present[i] = torch.from_numpy(s["state_present"])
            cause_present[i] = torch.from_numpy(s["cause_present"])
        batch["state_targets"] = state_targets
        batch["cause_targets"] = cause_targets
        batch["state_present"] = state_present
        batch["cause_present"] = cause_present
        batch["has_semantic"] = torch.tensor(
            [bool(s.get("has_semantic")) for s in samples], dtype=torch.bool
        )

    if "dino_vec" in samples[0]:
        bsz = len(samples)
        t_max = int(batch["padding_mask"].size(1))
        dino_dim = int(samples[0].get("dino_dim", 1024))
        dino = torch.zeros(bsz, t_max, dino_dim, dtype=torch.float32)
        dino_mask = torch.zeros(bsz, t_max, dtype=torch.bool)
        for i, s in enumerate(samples):
            t = min(int(s["length"]), t_max)
            dv = s["dino_vec"][:t]
            if dv.shape[-1] != dino_dim:
                # pad/truncate rare mismatch
                fixed = np.zeros((t, dino_dim), dtype=np.float32)
                w = min(dino_dim, dv.shape[-1])
                fixed[:, :w] = dv[:, :w]
                dv = fixed
            dino[i, :t] = torch.from_numpy(np.asarray(dv, dtype=np.float32))
            dino_mask[i, :t] = torch.from_numpy(s["dino_mask"][:t])
        batch["dino_vec"] = dino
        batch["dino_mask"] = dino_mask

    return batch
