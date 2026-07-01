import logging

import torch

from src.transformer.hexaco_model import REG_LOSS, Transformer_PSY

log = logging.getLogger(__name__)


class Transformer_AgriFruiting(Transformer_PSY):
    """Regression head for AgriChallenge fruiting-count prediction."""

    def transform_targets(self, targets, seq, stage: str):
        if self.hparams.loss_type in REG_LOSS:
            if targets.dim() == 1:
                return targets.float().unsqueeze(-1)
            return targets.float()
        return super().transform_targets(targets, seq, stage)

    def _reg_metrics(
        self,
        predictions,
        targets,
        loss,
        stage,
        on_step: bool = True,
        on_epoch: bool = True,
        sid=None,
    ):
        if stage == "train":
            self.log("train/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "train/mae",
                self.train_mae(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/mse",
                self.train_mse(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
        elif stage == "val":
            self.log("val/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "val/mae",
                self.val_mae(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/mse",
                self.val_mse(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
        elif stage == "test":
            self.log("test/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "test/mae",
                self.test_mae(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def on_validation_epoch_end(self):
        return

    def on_test_epoch_end(self):
        return
