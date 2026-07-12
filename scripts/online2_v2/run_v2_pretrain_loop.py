#!/usr/bin/env python3
"""Online2 V2 grouped-MLM sanity / resumable pretrain without HDF5 prepare_data.

Reads training_events parquet directly, encodes with GroupedMLM, trains TransformerEncoder.
Supports checkpoint resume and OOM batch-size fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def iter_person_batches(parquet: Path, batch_size: int, max_rows: int | None):
    """Yield dataframe chunks without requiring the full corpus in RAM."""
    pf = pq.ParquetFile(parquet)
    seen = 0
    for i in range(pf.metadata.num_row_groups):
        part = pf.read_row_group(i).to_pandas()
        if max_rows is not None:
            remain = max_rows - seen
            if remain <= 0:
                break
            part = part.head(remain)
        seen += len(part)
        # yield person-sized microbatches from this row group
        ids = part["PERSON_ID"].astype(int).unique().tolist()
        for start in range(0, len(ids), batch_size):
            chosen = ids[start : start + batch_size]
            yield part[part["PERSON_ID"].isin(chosen)].copy(), chosen
        if max_rows is not None and seen >= max_rows:
            break


def encode_batch(task, frame: pd.DataFrame, person_ids: list[int], device: torch.device):
    from src.tasks.base import collate_encoded_documents

    docs = []
    for person_id in person_ids:
        person = frame[frame["PERSON_ID"] == person_id].copy()
        if person.empty:
            continue
        person.name = int(person_id)
        person["BIRTHDAY"] = np.datetime64("2000-01-01")
        person["GENDER"] = "U"
        person["RES_ORIGIN"] = "UNK"
        person["AFTER_THRESHOLD"] = False
        person["AGE"] = person["AGE"].astype(float)
        start = person["START_DATE"].min()
        person["START_DATE"] = ((person["START_DATE"] - start) / np.timedelta64(1, "h")).astype(int)
        if "SEGMENT" not in person.columns:
            person["SEGMENT"] = 1
        doc = task.get_document(person)
        encoded = task.get_preprocessor(is_train=True)(doc)
        docs.append(encoded)
    if not docs:
        raise RuntimeError("empty batch")
    batch = collate_encoded_documents(docs)
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "outputs/online2/v2_build")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/online2/v2_runs/sanity")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--mode", choices=["sanity", "full"], default="sanity")
    args = parser.parse_args()

    build = args.build_dir
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    parquet = args.parquet or (
        build / "training_events_v2_smoke.parquet"
        if args.mode == "sanity" and (build / "training_events_v2_smoke.parquet").exists()
        else build / "training_events_v2.parquet"
    )
    registry = build / "life2vec_token_registry_v2.json"
    vocab_v2 = build / "vocab_v2.json"

    from src.data_new.vocabulary import RegistryVocabulary
    from src.tasks.grouped_mlm import GroupedMLM
    from src.transformer.models import TransformerEncoder

    max_rows = None if args.mode == "full" and args.max_rows <= 0 else args.max_rows
    print(f"streaming parquet={parquet} max_rows={max_rows}", flush=True)

    vocab = RegistryVocabulary(registry_path=str(registry), registry_version="v2")
    task = GroupedMLM(
        name="online2_v2_grouped_mlm",
        max_length=args.max_length,
        mask_ratio=0.3,
        vocab_v2_path=str(vocab_v2),
        mask_feature_identity=False,
        non_maskable_tokens=["[IMAGE_SLOT]", "[TEXT_SLOT]", "IMAGE_EMBED_SLOT", "TEXT_EMBED_SLOT"],
        sop_reverse_probability=0.05,
        sop_shuffle_probability=0.05,
        evaluation_seed=2023,
        shuffle_within_sentences=False,
    )
    task.datamodule = SimpleNamespace(vocabulary=vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = {
        "vocabulary": vocab,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "hidden_size": 128 if args.mode == "sanity" else 256,
        "hidden_ff": 512 if args.mode == "sanity" else 1024,
        "hidden_act": "swish",
        "n_encoders": 2 if args.mode == "sanity" else 4,
        "n_heads": 4 if args.mode == "sanity" else 8,
        "n_local": 2,
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
        "experiment_name": f"online2_v2_{args.mode}",
        "experiment_version": "2.0",
        "attention_type": "performer",
        "multihead_dc": False,
        "num_random_features": 64,
        "learning_rate": 1e-3 if args.mode == "sanity" else 5e-4,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "cls_num_targs": 3,
        "epsilon": 1e-6,
        "stage": "pre_training",
        "implementation": "online2_v2",
        "version": "2.0",
        "vocab_size": vocab.size(),
    }
    model = TransformerEncoder(hparams).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=hparams["learning_rate"], weight_decay=0.01)

    start_step = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume} step={start_step}", flush=True)

    batch_size = args.batch_size
    steps = args.steps if args.mode == "sanity" else max(args.steps, 50)
    history = []
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "training_mode": "transductive_public_pretraining",
        "contains_problem_observations": True,
        "contains_problem_images": True,
        "contains_problem_hidden_targets": False,
        "parquet": str(parquet),
        "vocab_size": vocab.size(),
        "device": str(device),
        "status": "running",
        "attempts": [],
        "max_rows": max_rows,
    }
    write_json(run_dir / "run_manifest_v2.json", manifest)

    from src.transformer.models import masked_sop_loss

    step = start_step
    data_iter = iter_person_batches(parquet, batch_size, max_rows)
    while step < start_step + steps:
        try:
            try:
                frame, chosen = next(data_iter)
            except StopIteration:
                data_iter = iter_person_batches(parquet, batch_size, max_rows)
                frame, chosen = next(data_iter)
            batch = encode_batch(task, frame, chosen, device)
            opt.zero_grad(set_to_none=True)
            mlm_preds, cls_preds = model(batch)
            mlm_targs = batch["target_tokens"].long()
            cls_targs = batch["target_cls"].long()
            mlm_loss = model.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
            sop_loss = masked_sop_loss(
                model.cls_loss(cls_preds, target=cls_targs),
                batch.get("target_cls_mask"),
            )
            loss = model.cls_w * sop_loss + model.mlm_w * mlm_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {loss}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            rec = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "mlm_loss": float(mlm_loss.detach().cpu()),
                "sop_loss": float(sop_loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
                "batch_size": batch_size,
            }
            history.append(rec)
            print(rec, flush=True)
            if step % args.ckpt_every == 0 or step == start_step + steps:
                ckpt_path = run_dir / f"checkpoint_step_{step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                        "hparams": {k: v for k, v in hparams.items() if k != "vocabulary"},
                    },
                    ckpt_path,
                )
                last = run_dir / "last.ckpt"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, last)
                print(f"saved {ckpt_path}", flush=True)
        except RuntimeError as exc:
            msg = str(exc).lower()
            manifest["attempts"].append({"step": step, "error": str(exc), "batch_size": batch_size})
            write_json(run_dir / "run_manifest_v2.json", manifest)
            if "out of memory" in msg:
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    data_iter = iter_person_batches(parquet, batch_size, max_rows)
                    print(f"OOM_RECOVERY batch_size->{batch_size}", flush=True)
                    torch.cuda.empty_cache()
                    continue
            raise

    # Resume verification
    verify = torch.load(run_dir / "last.ckpt", map_location=device)
    model.load_state_dict(verify["model"])
    print({"resume_verify_step": verify["step"], "pass": True}, flush=True)

    manifest.update(
        {
            "status": "PASS",
            "final_step": step,
            "history_tail": history[-5:],
            "batch_size_final": batch_size,
            "checkpoint": str(run_dir / "last.ckpt"),
            "resume_command": (
                f"python scripts/online2_v2/run_v2_pretrain_loop.py --mode {args.mode} "
                f"--build-dir {build} --run-dir {run_dir} --resume {run_dir/'last.ckpt'} "
                f"--steps {steps} --batch-size {batch_size} --max-rows {args.max_rows}"
            ),
        }
    )
    write_json(run_dir / "run_manifest_v2.json", manifest)
    write_json(run_dir / "train_history.json", {"history": history})
    print("DONE", manifest["status"], manifest["final_step"], flush=True)


if __name__ == "__main__":
    main()
