from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

from analysis.scripts.paths import FINETUNE_CKPT, PRETRAIN_CKPT, PROJECT_ROOT, VOCAB_PATH


def _config_dir() -> str:
    return str((PROJECT_ROOT / "conf").resolve())


def load_experiment_config(experiment: str) -> DictConfig:
    with initialize_config_dir(config_dir=_config_dir(), version_base="1.3"):
        return compose(config_name="config", overrides=[f"experiment={experiment}"])


def load_datamodule(experiment: str = "finetune_agri_fruiting"):
    cfg = load_experiment_config(experiment)
    dm = instantiate(cfg.datamodule, _convert_="all")
    dm.setup()
    return dm, cfg


def _load_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["state_dict"]


def load_pretrain_model(device: str = "cpu"):
    cfg = load_experiment_config("pretrain_agri")
    model = instantiate(cfg.model, _convert_="all")
    model.load_state_dict(_load_state_dict(PRETRAIN_CKPT), strict=False)
    model.eval()
    model.to(device)
    return model, cfg


def load_finetune_model(device: str = "cpu"):
    cfg = load_experiment_config("finetune_agri_fruiting")
    model = instantiate(cfg.model, _convert_="all")
    model.load_state_dict(_load_state_dict(FINETUNE_CKPT), strict=False)
    model.eval()
    model.to(device)
    return model, cfg


def load_vocab() -> pd.DataFrame:
    return pd.read_csv(VOCAB_PATH, sep="\t").set_index("ID")


def load_population() -> pd.DataFrame:
    with open(
        PROJECT_ROOT
        / "data/processed/populations/agri_fruiting_set/population/result.pkl",
        "rb",
    ) as f:
        pop = pickle.load(f)
    return pop


def iter_all_batches(dm, device: str = "cpu"):
    from torch.utils.data import ConcatDataset, DataLoader

    from src.data_new.datamodule import collate_encoded_documents

    dataset = ConcatDataset([dm.train, dm.val, dm.test])
    loader = DataLoader(
        dataset,
        batch_size=dm.batch_size,
        shuffle=False,
        collate_fn=collate_encoded_documents,
        num_workers=0,
    )
    for batch in loader:
        yield {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
