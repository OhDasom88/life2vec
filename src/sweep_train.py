"""W&B sweep agent entrypoint for Agri fruiting finetune."""

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


def _sweep_overrides(wc) -> list[str]:
    loss_type = wc.loss_type
    num_classes = 6 if loss_type == "entropy" else 1
    run_id = wandb.run.id
    pooled = str(wc.pooled).lower()
    freeze = str(wc.freeze_embeddings).lower()

    return [
        "experiment=sweep_agri_fruiting",
        f"version=sweep_{run_id}",
        f"model.hparams.pooled={pooled}",
        f"model.hparams.loss_type={loss_type}",
        f"model.hparams.num_classes={num_classes}",
        f"model.hparams.learning_rate={wc.learning_rate}",
        f"model.hparams.att_dropout={wc.dropout}",
        f"model.hparams.fw_dropout={wc.dropout}",
        f"model.hparams.dc_dropout={wc.dropout}",
        f"model.hparams.emb_dropout={wc.dropout}",
        f"model.hparams.weight_decay={wc.weight_decay}",
        f"model.hparams.freeze_embeddings={freeze}",
        f"datamodule.batch_size={wc.batch_size}",
        f"model.hparams.batch_size={wc.batch_size}",
        f"trainer.accumulate_grad_batches={wc.accumulate_grad_batches}",
        f"trainer.logger.1.name=fruiting_sweep_{run_id}",
    ]


def train_from_cfg(cfg) -> None:
    seed_everything(cfg.seed)
    data = instantiate(cfg.datamodule, _convert_="all")
    model = instantiate(cfg.model, _convert_="all")
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

    log.info("Sweep run %s overrides: %s", wandb.run.id, _sweep_overrides(wc))
    train_from_cfg(cfg)
    wandb.finish()


if __name__ == "__main__":
    main()
