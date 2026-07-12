#!/usr/bin/env python
"""CPU/GPU 1-batch MLM+SOP smoke for an online2 training export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def main() -> None:
    root = Path(os.environ.get("ONLINE2_SMOKE_DIR", "outputs/online2/smoke_train"))
    parquet = root / "training_events.parquet"
    registry_path = root / "life2vec_token_registry.json"
    build_id = json.loads((root / "smoke_manifest.json").read_text())["build_id"]

    from src.data_new.sources.online2 import Online2ParquetTokenSource
    from src.data_new.vocabulary import RegistryVocabulary
    from src.tasks.base import collate_encoded_documents
    from src.tasks.mlm import MLM
    from src.transformer.models import TransformerEncoder

    source = Online2ParquetTokenSource(
        path=str(parquet),
        build_id=build_id,
        registry_version="1",
    )
    frame = source.tokenized().compute().reset_index()
    vocab = RegistryVocabulary(
        registry_path=str(registry_path),
        registry_version="1",
    )
    task = MLM(
        name="online2_mlm_smoke",
        max_length=512,
        mask_ratio=0.3,
        sop_reverse_probability=0.5,
        sop_shuffle_probability=0.25,
        evaluation_seed=2023,
        non_maskable_tokens=["IMAGE_EMBED_SLOT", "TEXT_EMBED_SLOT"],
    )
    task.datamodule = SimpleNamespace(vocabulary=vocab)

    person_ids = sorted(frame["PERSON_ID"].unique().tolist())[:4]
    batch_docs = []
    for person_id in person_ids:
        person = frame[frame["PERSON_ID"] == person_id].copy()
        person.name = int(person_id)
        person["BIRTHDAY"] = np.datetime64("2000-01-01")
        person["GENDER"] = "U"
        person["RES_ORIGIN"] = "UNK"
        person["AFTER_THRESHOLD"] = False
        person["AGE"] = person["AGE"].astype(float)
        start = person["START_DATE"].min()
        person["START_DATE"] = (
            (person["START_DATE"] - start) / np.timedelta64(1, "h")
        ).astype(int)
        if "SEGMENT" not in person.columns:
            person["SEGMENT"] = (person["event_position"] % 3 + 1).astype(int)
        doc = task.get_document(person)
        preprocessor = task.get_preprocessor(is_train=True)
        encoded = preprocessor(doc)
        batch_docs.append(encoded)

    batch = collate_encoded_documents(batch_docs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)

    hparams = {
        "vocabulary": vocab,
        "batch_size": len(person_ids),
        "max_length": task.max_length,
        "hidden_size": 128,
        "hidden_ff": 512,
        "hidden_act": "swish",
        "n_encoders": 2,
        "n_heads": 4,
        "n_local": 3,
        "local_window_size": 16,
        "norm_type": "rezero",
        "att_dropout": 0.1,
        "fw_dropout": 0.1,
        "dc_dropout": 0.1,
        "emb_dropout": 0.1,
        "parametrize_emb": False,
        "norm_input_emb": False,
        "norm_output_emb": True,
        "weight_tying": "wt",
        "training_task": "mlm",
        "experiment_name": "online2_smoke",
        "experiment_version": "8.0",
        "attention_type": "performer",
        "multihead_dc": False,
        "num_random_features": 64,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "cls_num_targs": 3,
        "epsilon": 1e-6,
        "stage": "pre_training",
        "implementation": "online2",
        "version": "8.0",
    }
    model = TransformerEncoder(hparams).to(device)
    assert int(model.hparams.vocab_size) == len(vocab.tokens())
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    mlm_preds, cls_preds = model(batch)
    mlm_targs = batch["target_tokens"].long()
    cls_targs = batch["target_cls"].long()
    mlm_loss = model.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
    from src.transformer.models import masked_sop_loss

    sop_loss = masked_sop_loss(
        model.cls_loss(cls_preds, target=cls_targs),
        batch.get("target_cls_mask"),
    )
    loss = model.cls_w * sop_loss + model.mlm_w * mlm_loss

    report = {
        "device": str(device),
        "batch_persons": len(person_ids),
        "vocab_size": int(model.hparams.vocab_size),
        "loss": float(loss.detach().cpu()),
        "mlm_loss": float(mlm_loss.detach().cpu()),
        "sop_loss": float(sop_loss.detach().cpu()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_mlm": bool(torch.isfinite(mlm_loss)),
        "finite_sop": bool(torch.isfinite(sop_loss)),
        "sop_mask_sum": float(batch["target_cls_mask"].detach().cpu().sum())
        if "target_cls_mask" in batch
        else None,
    }
    print(json.dumps(report, indent=2))
    assert torch.isfinite(loss), "non-finite loss"
    loss.backward()
    grad_norm = sum(
        float(p.grad.detach().norm().cpu())
        for p in model.parameters()
        if p.grad is not None
    )
    # Expect gradients on MLM/SOP heads and encoder
    head_grads = {
        "mlm_decoder": any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for n, p in model.named_parameters()
            if n.startswith("mlm_decoder")
        ),
        "cls_decoder": any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for n, p in model.named_parameters()
            if n.startswith("cls_decoder")
        ),
        "transformer": any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for n, p in model.named_parameters()
            if n.startswith("transformer")
        ),
    }
    optimizer.step()
    print(json.dumps({"grad_norm_sum": grad_norm, "head_grads": head_grads, "status": "PASS"}, indent=2))
    out = root / "smoke_pretrain_report.json"
    out.write_text(
        json.dumps(
            {**report, "grad_norm_sum": grad_norm, "head_grads": head_grads, "status": "PASS"},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
