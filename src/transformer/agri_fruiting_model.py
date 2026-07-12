import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.transformer.cls_model import Transformer_CLS
from src.transformer.transformer import AttentionDecoder, CLS_Decoder_FT2
from src.transformer.transformer_utils import AsymmetricMulticlassCrossEntropyLoss

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


def compute_fruiting_bucket_weights(targets: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights over 6 fruiting buckets."""
    buckets = fruiting_count_to_bucket(torch.tensor(targets, dtype=torch.float32)).numpy()
    counts = np.bincount(buckets, minlength=FRUITING_NUM_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * FRUITING_NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


class Transformer_AgriFruiting(Transformer_CLS):
    """Fruiting-count bucket classifier (Transformer_CLS-based)."""

    def __init__(self, hparams):
        self._raw_fruiting_targets = None
        super().__init__(hparams)
        assert self.hparams.num_targets == FRUITING_NUM_CLASSES

    @property
    def num_outputs(self):
        return FRUITING_NUM_CLASSES

    def init_decoder(self):
        if self.hparams.pooled:
            log.info("Fruiting classifier with pooled representation")
            self.decoder = AttentionDecoder(self.hparams, num_outputs=FRUITING_NUM_CLASSES)
            self.encoder_f = self.transformer.forward_finetuning
        else:
            log.info("Fruiting classifier with CLS token representation")
            self.decoder = CLS_Decoder_FT2(
                self.hparams, num_outputs=FRUITING_NUM_CLASSES
            )
            self.encoder_f = self.transformer.forward_finetuning_cls

    def init_loss(self):
        if self.hparams.loss_type == "asymmetric":
            self.loss = AsymmetricMulticlassCrossEntropyLoss(
                under_penalty=getattr(self.hparams, "asym_beta", 1.5),
                over_penalty=getattr(self.hparams, "asym_alpha", 1.0),
            )
        elif self.hparams.loss_type == "entropy":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                f"Unsupported fruiting loss_type: {self.hparams.loss_type}"
            )

    def init_metrics(self):
        acc_cls = torchmetrics.classification.MulticlassAccuracy
        self.train_acc = acc_cls(num_classes=FRUITING_NUM_CLASSES)
        self.val_acc = acc_cls(num_classes=FRUITING_NUM_CLASSES)
        self.test_acc = acc_cls(num_classes=FRUITING_NUM_CLASSES)
        self.train_mae_count = torchmetrics.MeanAbsoluteError()
        self.val_mae_count = torchmetrics.MeanAbsoluteError()
        self.test_mae_count = torchmetrics.MeanAbsoluteError()

    def init_collector(self):
        return

    def on_fit_start(self):
        if not getattr(self.hparams, "auto_class_weights", True):
            return
        if self.hparams.loss_type not in ("asymmetric", "entropy"):
            return
        dm = self.trainer.datamodule
        if not hasattr(dm, "get_bucket_class_weights"):
            return
        weights = dm.get_bucket_class_weights()
        if self.hparams.loss_type == "asymmetric":
            self.loss.class_weights = weights.to(self.device)
        elif self.hparams.loss_type == "entropy":
            self.loss.weight = weights.to(self.device)

    def on_train_epoch_end(self, *args):
        if (
            self.hparams.attention_type == "performer"
            and getattr(self.hparams, "redraw_projections", True)
        ):
            log.info("Redraw Projection Matrices")
            self.transformer.redraw_projection_matrix(-1)

    def transform_targets(self, targets, seq, stage: str):
        if targets.dim() == 1:
            raw = targets.float()
        else:
            raw = targets.float().view(-1)
        self._raw_fruiting_targets = raw
        return fruiting_count_to_bucket(raw).long()

    def forward(self, batch):
        predicted = self.encoder_f(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long(),
        )
        if self.hparams.pooled:
            predicted = self.decoder(predicted, mask=batch["padding_mask"].long())
        else:
            predicted = self.decoder(predicted)
        return predicted

    def log_metrics(
        self,
        predictions,
        targets,
        loss,
        stage,
        on_step: bool = True,
        on_epoch: bool = True,
        sid=None,
    ):
        scores = F.softmax(predictions, dim=-1)
        pred_buckets = torch.argmax(scores, dim=-1)
        tgt_buckets = targets.long().view(-1)
        raw = self._raw_fruiting_targets
        pred_counts = fruiting_bucket_to_count(pred_buckets)

        if stage == "train":
            self.log("train/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "train/acc",
                self.train_acc(scores, tgt_buckets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            if pred_counts.shape[0] == raw.shape[0]:
                self.log(
                    "train/mae_count",
                    self.train_mae_count(pred_counts, raw),
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
                self.val_mae_count(pred_counts, raw),
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=True,
            )
        elif stage == "test":
            self.log("test/loss", loss, on_step=on_step, on_epoch=on_epoch)
            self.log(
                "test/acc",
                self.test_acc(scores, tgt_buckets),
                on_step=on_step,
                on_epoch=on_epoch,
            )
            self.log(
                "test/mae_count",
                self.test_mae_count(pred_counts, raw),
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def on_validation_epoch_end(self):
        return

    def on_test_epoch_end(self):
        return
