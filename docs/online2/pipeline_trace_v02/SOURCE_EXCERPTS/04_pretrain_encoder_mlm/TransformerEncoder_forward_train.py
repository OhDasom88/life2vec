# Excerpt from src/transformer/models.py forward + training_step

# --- lines 182-250 ---
    def forward(self, batch):
        """Forward pass"""
        ## 1. ENCODER INPUT
        predicted = self.transformer(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long()
        )
        ## 2. MASKED LANGUAGE MODEL
        mlm_pred = self.mlm_decoder(predicted, batch)
        ## 3. CLS TASK
        cls_pred  = self.cls_decoder(predicted[:,0])
        return mlm_pred, cls_pred

    def training_step(self, batch, batch_idx):
        """Training Iteration"""
        ## 1. ENCODER-DECODER
        mlm_preds, cls_preds = self(batch)
        ## 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        cls_targs = batch["target_cls"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        cls_loss = masked_sop_loss(
            self.cls_loss(cls_preds, target=cls_targs),
            batch.get("target_cls_mask"),
        )

        self.log("train/loss_mlm", mlm_loss.detach(), on_step=True, on_epoch=True)
        self.log("train/loss_cls", cls_loss.detach(), on_step=True, on_epoch=True)

        loss = self.cls_w * cls_loss + self.mlm_w * mlm_loss
        ## 3. METRICS
        if (self.global_step + 1) % (self.trainer.log_every_n_steps) == 0:
            self.log_metrics(
                predictions=(mlm_preds.detach(), cls_preds.detach()),
                targets=(mlm_targs.detach(),  cls_targs.detach()),
                loss=loss.detach(),
                stage="train",
                on_step=True,
                on_epoch=True,
            )
        return loss

    def train_epoch_start(self, *args):
        """On Epoch Start"""
        self.last_global_step = self.global_step
        seed_everything(self.hparams.seed + self.trainer.current_epoch)

    def training_epoch_end(self, output):
        """On Epoch End"""
        if self.hparams.attention_type == "performer":
            self.transformer.redraw_projection_matrix(-1)

    def validation_epoch_end(self, outputs) -> None:
        """Save the embedding on validation epoch end"""
        return super().validation_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        """Validation Step"""
        ## 1. ENCODER-DECODER
        mlm_preds, cls_preds = self(batch)
        ## 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        cls_targs = batch["target_cls"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        cls_loss = masked_sop_loss(
            self.cls_loss(cls_preds, target=cls_targs),
            batch.get("target_cls_mask"),
        )

