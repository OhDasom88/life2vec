"""W&B sweep agent entrypoint for Agri multi-corpus pretrain."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import wandb
from hydra import compose, initialize_config_dir
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _register_resolvers() -> None:
    def project_root(*_args):
        try:
            return get_original_cwd()
        except Exception:
            return str(PROJECT_ROOT)

    try:
        OmegaConf.register_new_resolver("project_root", project_root)
    except Exception:
        pass


def _config_dir() -> str:
    return str((PROJECT_ROOT / "conf").resolve())


def _apply_vocab_size(cfg, data) -> None:
    if hasattr(data, "get_vocab_size"):
        size = data.get_vocab_size()
    else:
        size = data.vocabulary.size()
    OmegaConf.update(cfg, "model.hparams.vocab_size", int(size), merge=True)


def _sweep_overrides(wc) -> list[str]:
    run_id = wandb.run.id
    return [
        "experiment=sweep_agri_pretrain",
        f"version=sweep_{run_id}",
        f"model.hparams.learning_rate={wc.learning_rate}",
        f"model.hparams.att_dropout={wc.dropout}",
        f"model.hparams.fw_dropout={wc.dropout}",
        f"model.hparams.dc_dropout={wc.dropout}",
        f"model.hparams.emb_dropout={wc.dropout}",
        f"model.hparams.weight_decay={wc.weight_decay}",
        f"datamodule.batch_size={wc.batch_size}",
        f"model.hparams.batch_size={wc.batch_size}",
        f"trainer.logger.1.name=pretrain_sweep_{run_id}",
    ]


def train_from_cfg(cfg) -> None:
    seed_everything(cfg.seed)
    data = instantiate(cfg.datamodule, _convert_="all")
    _apply_vocab_size(cfg, data)
    model = instantiate(cfg.model, _convert_="all")
    if hasattr(model, "mlm_w") and "mlm_weight" in wandb.config:
        model.mlm_w.fill_(float(wandb.config.mlm_weight))
        model.cls_w.fill_(1.0 - float(wandb.config.mlm_weight))
    trainer: Trainer = instantiate(cfg.trainer, _convert_="all")

    hparams = OmegaConf.to_container(cfg, resolve=True)
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

    trainer.fit(model, data, ckpt_path=cfg.ckpt_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    os.chdir(PROJECT_ROOT)
    _register_resolvers()

    wandb.init()
    wc = wandb.config

    with initialize_config_dir(config_dir=_config_dir(), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=_sweep_overrides(wc),
        )

    log.info("Pretrain sweep run %s", wandb.run.id)
    train_from_cfg(cfg)
    wandb.finish()


if __name__ == "__main__":
    main()
