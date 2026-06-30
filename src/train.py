import re
import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
import sys
from pathlib import Path
import logging

HOME_PATH = str(Path.home())

log = logging.getLogger(__name__)


def last_ckpt(dir_):
    ckpt_path = Path(dir_) / "last.ckpt"
    if ckpt_path.exists():
        log.info("Checkpoint exists:\n\t%s" % str(ckpt_path))
        return str(ckpt_path)
    else:
        log.info("Checkpoint DOES NOT exists:\n\t%s" % str(ckpt_path))
        return None

def project_root(*_args):
    return get_original_cwd()

def home_path():
    return HOME_PATH

try:
    OmegaConf.register_new_resolver("project_root", project_root)
    OmegaConf.register_new_resolver("last_ckpt", last_ckpt)
except Exception as e:
    print(e)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):

    # Workaround for hydra breaking import of local package - don't need if running as module "-m src.train"
    # sys.path.append(hydra.utils.get_original_cwd())
    
    ##GLOBAL SEED
    seed_everything(cfg.seed)
    
    data = instantiate(cfg.datamodule, _convert_="all")
    ##MODEL
    model = instantiate(cfg.model, _convert_="all")

    ##TRAINER
    trainer: Trainer = instantiate(cfg.trainer, _convert_="all")
    hparams = OmegaConf.to_container(cfg, resolve=True)
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

    ##TRAINING
    trainer.fit(model, data, ckpt_path = cfg.ckpt_path)


if __name__ == "__main__":
    main()
