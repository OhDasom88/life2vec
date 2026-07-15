# Excerpt from src/transformer/transformer.py — DO NOT EDIT; see repo source


# --- forward_finetuning* + get_sequence_embedding + MaskedLanguageModel (lines 75-174) ---
    def forward_finetuning(self,x, padding_mask=None):

        x, _ = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:,3]
        )

        for i, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)

        return x

    def forward_finetuning_cls(self, x, padding_mask):
        logits = list()
        x, _ = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:,3]
        )
        for i, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)
            if  i == (self.hparams.n_encoders - 1)//2 or i == 1 or i == (self.hparams.n_encoders - 1): ## we extract CLS embeddings after 0th and last encoder block and average those
                logits.append(x[:, 0])
        return x[:,0]
        return torch.stack(logits, dim=0).mean(dim=0)

    def forward_finetuning_with_embeddings(self, x, padding_mask):
        ### Inputs are the embeddings (not sequence of tokens)
        for _, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)
        return x

    def forward_finetuning_with_embeddings_cls(self, x, padding_mask):
        ### Inputs are the embeddings (not sequence of tokens)
        logits = list()
        for i, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)
            if  i == (self.hparams.n_encoders - 1)//2 or i == 1 or i == (self.hparams.n_encoders - 1): ## we extract CLS embeddings after 0th and last encoder block and average those
                logits.append(x[:, 0])
        return torch.stack(logits, dim=0).mean(dim=0)

    def get_sequence_embedding(self, x):
        """Get only embeddings"""
        return self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )

    def redraw_projection_matrix(self, batch_idx: int):
        """Redraw projection Matrices for each layer (only valid for Performer)"""
        if batch_idx == -1:
            #log.info("Redrawing projections for the encoder layers")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()

        elif batch_idx > 0 and batch_idx % self.hparams.feature_redraw_interval == 0:
            log.info("Redrawing projections for the encoder layers")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()

class MaskedLanguageModel(nn.Module):
    """Masked Language Head for Predictions"""

    def __init__(self, hparams, embedding, act:str = "tanh"):
        super(MaskedLanguageModel, self).__init__()
        self.hparams = hparams
        self.act = ACT2FN[act]
        self.dropout = nn.Dropout(p=self.hparams.emb_dropout)

        self.V = nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)
        self.g = nn.Parameter(torch.tensor([hparams.hidden_size**0.5]))
        self.out = nn.Linear(
                self.hparams.hidden_size, 
                self.hparams.vocab_size, 
                bias=False
            )
        if self.hparams.weight_tying == "wt":
            log.info("MLM decoder WITH Wight Tying")
            try:
                self.out.weight = embedding.token.parametrizations.weight.original
            except:
                log.warning("MLM decoder parametrization failed")
                self.out.weight = embedding.token.weight

        if self.hparams.parametrize_emb:
            ignore_index = torch.LongTensor([0,4,5,6,7,8])
            log.info("(MLM Decoder) centering: true normalisation: %s" %hparams.norm_output_emb)
            parametrize.register_parametrization(self.out, "weight", Center(ignore_index = ignore_index, norm=hparams.norm_output_emb))

    def batched_index_select(self, x, dim, indx):
        """Gather the embeddings of tokens that we should make prediction on"""
        indx_ = indx.unsqueeze(2).expand(indx.size(0), indx.size(1), x.size(-1))
        return x.gather(dim, indx_)

    def forward(self, logits, batch): ##before 2.97
        indx = batch["target_pos"].long()
        logits = self.dropout(self.batched_index_select(logits, 1, indx))
        logits = self.dropout(l2_norm(self.act(self.V(logits))))
        return  self.g * self.out(logits)

