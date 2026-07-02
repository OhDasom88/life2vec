import logging

import numpy as np
import torch

from src.transformer.hexaco_model import CLS_LOSS, REG_LOSS, Transformer_PSY

log = logging.getLogger(__name__)

# Bucket labels: ≤3, 4, 5, 6, 7, ≥8
FRUITING_BUCKET_REPRESENTATIVES = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
FRUITING_NUM_CLASSES = 6


def fruiting_count_to_bucket(count: torch.Tensor) -> torch.Tensor:
    """Map raw fruiting counts to 6 ordinal buckets."""
    count = count.float()
    buckets = torch.zeros_like(count, dtype=torch.long)
    buckets[count <= 3] = 0
    buckets[count == 4] = 1
    buckets[count == 5] = 2
    buckets[count == 6] = 3
    buckets[count == 7] = 4
    buckets[count >= 8] = 5
    return buckets


def fruiting_bucket_to_count(bucket: torch.Tensor) -> torch.Tensor:
    """Map bucket indices back to representative counts for MAE."""
    reps = torch.tensor(
        FRUITING_BUCKET_REPRESENTATIVES,
        device=bucket.device,
        dtype=torch.float32,
    )
    return reps[bucket.long()]


class Transformer_AgriFruiting(Transformer_PSY):
    """Fruiting-count finetune with regression or bucket classification."""

    def __init__(self, hparams):
        self._raw_fruiting_targets = None
        super().__init__(hparams)
        if self.hparams.loss_type in CLS_LOSS:
            assert self.hparams.num_classes == FRUITING_NUM_CLASSES

    def transform_targets(self, targets, seq, stage: str):
        if targets.dim() == 1:
            raw = targets.float().unsqueeze(-1)
        else:
            raw = targets.float()
        self._raw_fruiting_targets = raw

        if self.hparams.loss_type in REG_LOSS:
            return raw
        if self.hparams.loss_type in CLS_LOSS:
            buckets = fruiting_count_to_bucket(raw.squeeze(-1))
            return buckets.unsqueeze(-1)
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
                "train/mae_count",
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
                "val/mae_count",
                self.val_mae(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
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
            self.log(
                "test/mae_count",
                self.test_mae(predictions, targets),
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def _cls_metrics(
        self,
        predictions,
        targets,
        loss,
        stage,
        on_step: bool = True,
        on_epoch: bool = True,
        sid=None,
    ):
        pred_logits = predictions[:, 0]
        tgt_buckets = targets[:, 0].long()
        scores = self.sigsoftmax(pred_logits)
        pred_buckets = torch.argmax(scores, dim=-1)
        raw = self._raw_fruiting_targets.squeeze(-1)
        pred_counts = fruiting_bucket_to_count(pred_buckets)

        if stage == "train":
            self.log("train/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "train/acc",
                self.train_acc(scores, tgt_buckets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "train/mae_count",
                self.train_mae(pred_counts, raw),
                on_step=on_step,
                on_epoch=on_epoch,
            )
        elif stage == "val":
            self.log("val/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "val/acc",
                self.val_acc(scores, tgt_buckets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "val/mae_count",
                self.val_mae(pred_counts, raw),
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
            )
        elif stage == "test":
            self.log("test/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "test/mae_count",
                self.test_mae(pred_counts, raw),
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def on_validation_epoch_end(self):
        return

    def on_test_epoch_end(self):
        return
