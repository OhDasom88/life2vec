"""W&B sweep agent entrypoint for Hackathon binary CLS finetune."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import wandb
from hydra import compose, initialize_config_dir
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from src.wandb_pretrain import (
    HACKATHON_IMPLEMENTATION,
    HACKATHON_PRETRAIN_SWEEP_ID,
    list_pretrain_sweep_runs,
    pretrain_weight_hparam,
    pretrain_weight_path,
)

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


def _resolve_pretrained_run_id(wc) -> str:
    if hasattr(wc, "pretrained_run_id") and wc.pretrained_run_id:
        return str(wc.pretrained_run_id)
    if hasattr(wc, "pretrained_model_path") and wc.pretrained_model_path:
        return str(wc.pretrained_model_path)
    return "none"


def _pretrain_weight_path(run_id: str) -> str:
    if run_id.lower() == "none":
        return "none"
    local_weight = pretrain_weight_path(run_id, implementation=HACKATHON_IMPLEMENTATION)
    if not local_weight.is_file():
        raise FileNotFoundError(
            f"Missing hackathon pretrain weights for run {run_id}: {local_weight}. "
            "Finish the pretrain sweep or set pretrained_run_id=none."
        )
    return pretrain_weight_hparam(run_id, implementation=HACKATHON_IMPLEMENTATION)


def _sweep_overrides(wc) -> list[str]:
    run_id = wandb.run.id
    pretrain_run_id = _resolve_pretrained_run_id(wc)
    weight_path = _pretrain_weight_path(pretrain_run_id)

    overrides = [
        "experiment=sweep_hackathon_finetune",
        f"version=sweep_{run_id}",
        f"pretrained_model_path={weight_path}",
        f"model.hparams.pretrained_model_path={weight_path}",
        f"model.hparams.pooled={str(wc.pooled).lower()}",
        f"model.hparams.loss_type={wc.loss_type}",
        f"model.hparams.num_targets=2",
        f"model.hparams.learning_rate={wc.learning_rate}",
        f"model.hparams.att_dropout={wc.dropout}",
        f"model.hparams.fw_dropout={wc.dropout}",
        f"model.hparams.dc_dropout={wc.dropout}",
        f"model.hparams.emb_dropout={wc.dropout}",
        f"model.hparams.weight_decay={wc.weight_decay}",
        f"model.hparams.freeze_embeddings={str(wc.freeze_embeddings).lower()}",
        f"datamodule.batch_size={wc.batch_size}",
        f"model.hparams.batch_size={wc.batch_size}",
        f"trainer.accumulate_grad_batches={wc.accumulate_grad_batches}",
        f"trainer.logger.1.name=hackathon_cls_sweep_{run_id}",
    ]

    if wc.loss_type == "asymmetric" and hasattr(wc, "pos_weight"):
        overrides.append(f"model.hparams.pos_weight={wc.pos_weight}")

    return overrides


def _log_pretrain_metadata(wc, pretrain_run_id: str) -> None:
    if pretrain_run_id.lower() == "none":
        wandb.config.update({"pretrain_run_id": "none"}, allow_val_change=True)
        return

    runs = list_pretrain_sweep_runs(
        sweep_id=HACKATHON_PRETRAIN_SWEEP_ID,
        implementation=HACKATHON_IMPLEMENTATION,
        top_k=200,
        finished_only=True,
    )
    meta = next((r for r in runs if r["run_id"] == pretrain_run_id), None)
    payload = {
        "pretrain_sweep_id": HACKATHON_PRETRAIN_SWEEP_ID,
        "pretrain_run_id": pretrain_run_id,
        "pretrain_weight_path": str(
            pretrain_weight_path(pretrain_run_id, implementation=HACKATHON_IMPLEMENTATION)
        ),
    }
    if meta:
        payload["pretrain_score"] = meta["score"]
        for key, value in meta["config"].items():
            payload[f"pretrain/{key}"] = value
    wandb.config.update(payload, allow_val_change=True)


def _apply_vocab_size(cfg, data) -> None:
    if hasattr(data, "get_vocab_size"):
        size = data.get_vocab_size()
    else:
        size = data.vocabulary.size()
    OmegaConf.update(cfg, "model.hparams.vocab_size", int(size), merge=True)


def train_from_cfg(cfg) -> None:
    seed_everything(cfg.seed)
    data = instantiate(cfg.datamodule, _convert_="all")
    data.setup()
    _apply_vocab_size(cfg, data)

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
    pretrain_run_id = _resolve_pretrained_run_id(wc)
    _log_pretrain_metadata(wc, pretrain_run_id)

    with initialize_config_dir(config_dir=_config_dir(), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=_sweep_overrides(wc),
        )

    log.info("Hackathon CLS sweep run %s pretrain=%s", wandb.run.id, pretrain_run_id)
    train_from_cfg(cfg)
    wandb.finish()


if __name__ == "__main__":
    main()
