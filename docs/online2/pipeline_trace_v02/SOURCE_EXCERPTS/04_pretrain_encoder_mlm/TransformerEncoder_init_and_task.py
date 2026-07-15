# Excerpt from src/transformer/models.py


# --- TransformerEncoder __init__ (MLM+SOP heads wiring) (lines 37-80) ---
class TransformerEncoder(pl.LightningModule):
    """Transformer with Masked Language Model"""

    def __init__(self, hparams):
        super(TransformerEncoder, self).__init__()
        if "vocabulary" in hparams:
            hparams = dict(hparams)
            vocabulary = hparams.pop("vocabulary")
            actual_size = vocabulary.size()
            configured_size = hparams.get("vocab_size")
            if configured_size is not None and int(configured_size) != actual_size:
                raise ValueError(
                    f"Configured vocab_size={configured_size} does not match "
                    f"vocabulary size {actual_size}"
                )
            hparams["vocab_size"] = actual_size
        self.hparams.update(hparams)
        #self.idx2token = self.load_lookup(HOME_PATH + self.hparams.dict_path)
        self.last_global_step = 0
        # 1. ENCODER
        self.transformer = Transformer(self.hparams)

        # 2. DECODER BLOCK
        self.task = self.hparams.training_task
        log.info("Training task: %s" %self.task)
        if "mlm" in self.task:
            self.register_buffer("cls_w", torch.tensor(0.2))
            self.register_buffer("mlm_w", torch.tensor(0.8))
            self.register_buffer("cls_a", torch.tensor([1/0.9, 1/0.1, 1/0.1]))
            self.num_outputs = self.hparams.vocab_size
            ### 2.1. DECODERS
            self.mlm_decoder = MaskedLanguageModel(self.hparams, self.transformer.embedding, act="tanh")
            self.cls_decoder = CLS_Decoder(self.hparams)
            ## 2.2. LOSS
            self.cls_loss = nn.CrossEntropyLoss(
                weight=self.cls_a, label_smoothing=0.1, reduction="none"
            )
            self.mlm_loss = nn.CrossEntropyLoss(ignore_index = 0)
        else:
            raise NotImplementedError()

        # 3. METRICS
        self.init_metrics()

