#!/usr/bin/env python3
"""Online2 V2 grouped-MLM sanity / resumable pretrain without HDF5 prepare_data.

Reads training_events parquet directly, encodes with GroupedMLM, trains TransformerEncoder.
Supports checkpoint resume, OOM batch-size fallback, global abspos, and wandb logging.

Time channels (life2vec contract):
  - abspos: hours since corpus-global reference (abspos_reference.json)
  - age:    sequence-relative hours from parquet AGE (first event in sequence)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_abspos_reference(build: Path) -> dict[str, Any]:
    path = build / "abspos_reference.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Compute with min(START_DATE) over training_events_v2 first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    ref = pd.Timestamp(payload["reference_timestamp"])
    if ref.tzinfo is not None:
        ref = ref.tz_convert("UTC").tz_localize(None)
    payload["_reference_ts"] = ref
    return payload


def person_hash_unit(person_id: int, seed: int = 2023) -> float:
    digest = hashlib.sha256(f"{seed}:{int(person_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def make_split_predicate(
    *,
    use_shadow_split: bool,
    want: str,
    val_frac: float,
    seed: int,
) -> Callable[[pd.DataFrame, list[int]], list[int]]:
    """Return a filter that keeps PERSON_IDs for train or val.

    Prefer parquet ``shadow_split`` when it has a real val set; otherwise fall back to
    a deterministic PERSON_ID hash split (current V2 export is all-train).
    """

    def _filter(frame: pd.DataFrame, ids: list[int]) -> list[int]:
        if use_shadow_split and "shadow_split" in frame.columns:
            split = frame.drop_duplicates("PERSON_ID").set_index("PERSON_ID")["shadow_split"]
            keep = []
            for pid in ids:
                label = str(split.get(pid, "train")).lower()
                if want == "train" and label == "train":
                    keep.append(pid)
                elif want == "val" and label in {"val", "valid", "validation"}:
                    keep.append(pid)
            return keep
        keep = []
        for pid in ids:
            is_val = person_hash_unit(int(pid), seed=seed) < val_frac
            if want == "val" and is_val:
                keep.append(pid)
            elif want == "train" and not is_val:
                keep.append(pid)
        return keep

    return _filter


def detect_shadow_val(parquet: Path, max_row_groups: int = 8) -> bool:
    pf = pq.ParquetFile(parquet)
    for i in range(min(max_row_groups, pf.metadata.num_row_groups)):
        part = pf.read_row_group(i, columns=["shadow_split"]).to_pandas()
        labels = {str(x).lower() for x in part["shadow_split"].unique()}
        if labels & {"val", "valid", "validation"}:
            return True
    return False


def iter_person_batches(
    parquet: Path,
    batch_size: int,
    max_rows: int | None,
    *,
    id_filter: Callable[[pd.DataFrame, list[int]], list[int]] | None = None,
):
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
        ids = part["PERSON_ID"].astype(int).unique().tolist()
        if id_filter is not None:
            ids = id_filter(part, ids)
        if not ids:
            continue
        for start in range(0, len(ids), batch_size):
            chosen = ids[start : start + batch_size]
            yield part[part["PERSON_ID"].isin(chosen)].copy(), chosen
        if max_rows is not None and seen >= max_rows:
            break


def encode_batch(
    task,
    frame: pd.DataFrame,
    person_ids: list[int],
    device: torch.device,
    abspos_reference: pd.Timestamp,
    *,
    is_train: bool = True,
):
    """Encode persons with global abspos and sequence-relative AGE.

    START_DATE → integer hours since ``abspos_reference`` (get_document adds +1 → abspos).
    AGE is left as parquet sequence-relative hours — never rebase START_DATE to person.min().
    """
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
        # AGE: sequence-relative hours (already exported)
        person["AGE"] = person["AGE"].astype(float)
        # abspos path: global hour offset; get_document uses abspos = START_DATE + 1
        sd = pd.to_datetime(person["START_DATE"])
        if getattr(sd.dt, "tz", None) is not None:
            sd = sd.dt.tz_convert("UTC").dt.tz_localize(None)
        person["START_DATE"] = (
            (sd - abspos_reference) / np.timedelta64(1, "h")
        ).astype(int)
        if "SEGMENT" not in person.columns:
            person["SEGMENT"] = 1
        doc = task.get_document(person)
        encoded = task.get_preprocessor(is_train=is_train)(doc)
        docs.append(encoded)
    if not docs:
        raise RuntimeError("empty batch")
    batch = collate_encoded_documents(docs)
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    encode_fn,
    data_iter,
    device: torch.device,
    *,
    max_batches: int,
) -> dict[str, float]:
    """Average val losses over up to ``max_batches`` person-batches."""
    from src.transformer.models import masked_sop_loss

    model.eval()
    totals = {
        "loss": 0.0,
        "mlm_loss": 0.0,
        "sop_loss": 0.0,
        "sop_mask_mean": 0.0,
        "n": 0.0,
    }
    for _ in range(max_batches):
        try:
            frame, chosen = next(data_iter)
        except StopIteration:
            break
        if not chosen:
            continue
        batch = encode_fn(frame, chosen, device)
        mlm_preds, cls_preds = model(batch)
        mlm_loss = model.mlm_loss(
            mlm_preds.permute(0, 2, 1), target=batch["target_tokens"].long()
        )
        sop_loss = masked_sop_loss(
            model.cls_loss(cls_preds, target=batch["target_cls"].long()),
            batch.get("target_cls_mask"),
        )
        loss = model.cls_w * sop_loss + model.mlm_w * mlm_loss
        totals["loss"] += float(loss.detach().cpu())
        totals["mlm_loss"] += float(mlm_loss.detach().cpu())
        totals["sop_loss"] += float(sop_loss.detach().cpu())
        if torch.is_tensor(batch.get("target_cls_mask")):
            totals["sop_mask_mean"] += float(batch["target_cls_mask"].float().mean().cpu())
        totals["n"] += 1.0
        del batch, mlm_preds, cls_preds, loss, mlm_loss, sop_loss
    model.train()
    n = max(totals["n"], 1.0)
    return {
        "val/loss": totals["loss"] / n,
        "val/mlm_loss": totals["mlm_loss"] / n,
        "val/sop_loss": totals["sop_loss"] / n,
        "val/sop_mask_mean": totals["sop_mask_mean"] / n,
        "val/batches": totals["n"],
    }


def calibrate_batch_size(
    model: torch.nn.Module,
    encode_fn,
    data_iter,
    device: torch.device,
    start_bs: int,
    target_frac: float = 0.70,
    max_bs: int = 64,
) -> int:
    """Grow batch size until reserved VRAM reaches target_frac of device memory."""
    if device.type != "cuda":
        return start_bs
    total = torch.cuda.get_device_properties(device).total_memory
    target = int(total * target_frac)
    bs = max(1, start_bs)
    best = bs
    while bs <= max_bs:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            frame, chosen = next(data_iter)
            people = list(chosen)
            while len(people) < bs:
                people.extend(chosen)
            people = people[:bs]
            batch = encode_fn(frame, people, device)
            opt_probe = torch.optim.AdamW(model.parameters(), lr=1e-4)
            opt_probe.zero_grad(set_to_none=True)
            mlm_preds, cls_preds = model(batch)
            mlm_targs = batch["target_tokens"].long()
            cls_targs = batch["target_cls"].long()
            from src.transformer.models import masked_sop_loss

            mlm_loss = model.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
            sop_loss = masked_sop_loss(
                model.cls_loss(cls_preds, target=cls_targs),
                batch.get("target_cls_mask"),
            )
            loss = model.cls_w * sop_loss + model.mlm_w * mlm_loss
            loss.backward()
            opt_probe.step()
            del batch, mlm_preds, cls_preds, loss, mlm_loss, sop_loss, opt_probe
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            used = peak
            print(
                {
                    "vram_calibrate_bs": bs,
                    "peak_alloc_gb": round(peak / 1e9, 2),
                    "reserved_gb": round(reserved / 1e9, 2),
                    "target_gb": round(target / 1e9, 2),
                },
                flush=True,
            )
            best = bs
            if used >= target:
                break
            if used >= int(total * 0.85):
                break
            if used >= int(total * 0.40):
                next_bs = min(max_bs, bs + 8)
            else:
                next_bs = min(max_bs, bs * 2 if bs >= 4 else bs + 2)
            if (
                next_bs > bs
                and used > 0
                and used * (next_bs / max(bs, 1)) > int(total * 0.88)
            ):
                next_bs = min(max_bs, bs + 8)
            if next_bs <= bs:
                break
            if used > 0 and used * (next_bs / max(bs, 1)) > int(total * 0.88):
                break
            bs = next_bs
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(f"VRAM_CALIBRATE_OOM at bs={bs}, using {best}", flush=True)
                break
            raise
    return max(1, best)


def init_wandb(
    args,
    manifest: dict,
    abspos_ref: dict,
    model: Optional[torch.nn.Module] = None,
) -> Optional[Any]:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed. `pip install wandb` or pass --no-wandb."
        ) from exc
    if not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        raise SystemExit(
            "wandb login missing (WANDB_API_KEY / ~/.netrc). "
            "Run `wandb login` or pass --no-wandb."
        )
    tags = [t for t in (args.wandb_tags or "").split(",") if t.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name
        or args.run_dir.name
        or "online2_v2_event_grain_global_abspos",
        tags=tags
        or ["online2_v2", "event_grain", "global_abspos", args.mode],
        config={
            **{k: v for k, v in manifest.items() if k not in {"attempts", "history_tail"}},
            "abspos_reference": {
                k: abspos_ref[k]
                for k in (
                    "reference_timestamp",
                    "unit",
                    "method",
                    "source_parquet",
                    "start_date_max",
                )
                if k in abspos_ref
            },
            "sop_reverse": args.sop_reverse,
            "sop_shuffle": args.sop_shuffle,
            "wandb_watch": getattr(args, "wandb_watch", False),
            "wandb_watch_log": getattr(args, "wandb_watch_log", "all"),
            "wandb_watch_freq": getattr(args, "wandb_watch_freq", 100),
        },
        reinit=True,
    )
    if model is not None and getattr(args, "wandb_watch", False):
        # Histograms of parameters + gradients (wandb UI: Model → Gradients / Parameters)
        wandb.watch(
            model,
            log=args.wandb_watch_log,
            log_freq=max(1, int(args.wandb_watch_freq)),
            log_graph=False,
        )
        print(
            {
                "wandb_watch": True,
                "log": args.wandb_watch_log,
                "log_freq": args.wandb_watch_freq,
            },
            flush=True,
        )
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "outputs/online2/v2_build")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/online2/v2_runs/sanity")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Max training steps (sanity default 30; full default 100000 with early stop).",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--mode", choices=["sanity", "full"], default="sanity")
    parser.add_argument("--target-vram-frac", type=float, default=0.70)
    parser.add_argument("--sop-reverse", type=float, default=0.20)
    parser.add_argument("--sop-shuffle", type=float, default=0.20)
    parser.add_argument("--skip-vram-calibrate", action="store_true")
    parser.add_argument("--abspos-reference", type=Path, default=None)
    parser.add_argument("--val-frac", type=float, default=0.05, help="Hash val frac if no shadow val")
    parser.add_argument("--val-seed", type=int, default=2023)
    parser.add_argument("--val-every", type=int, default=100, help="Validate every N train steps")
    parser.add_argument("--val-batches", type=int, default=32, help="Val batches per evaluation")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Val checks without improve")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="Berry2Vec")
    parser.add_argument("--wandb-entity", type=str, default="dasom-oh")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="online2_v2,event_grain,global_abspos",
    )
    parser.add_argument(
        "--wandb-watch",
        dest="wandb_watch",
        action="store_true",
        default=None,
        help="Track parameter/gradient histograms via wandb.watch (default: on when --wandb).",
    )
    parser.add_argument("--no-wandb-watch", dest="wandb_watch", action="store_false")
    parser.add_argument(
        "--wandb-watch-log",
        choices=["gradients", "parameters", "all"],
        default="all",
        help="What wandb.watch logs (default: all = params + grads).",
    )
    parser.add_argument(
        "--wandb-watch-freq",
        type=int,
        default=100,
        help="Log param/grad histograms every N backward steps.",
    )
    args = parser.parse_args()
    if args.wandb is None:
        args.wandb = args.mode == "full"
    if args.wandb_watch is None:
        args.wandb_watch = bool(args.wandb)
    if args.steps is None:
        args.steps = 30 if args.mode == "sanity" else 100_000
    if args.mode == "sanity":
        args.val_every = min(args.val_every, 10)
        args.val_batches = min(args.val_batches, 8)
        args.early_stop_patience = min(args.early_stop_patience, 5)
        args.wandb_watch_freq = min(args.wandb_watch_freq, 10)

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

    if args.abspos_reference is not None:
        payload = json.loads(args.abspos_reference.read_text(encoding="utf-8"))
        ref_ts = pd.Timestamp(payload["reference_timestamp"])
        if ref_ts.tzinfo is not None:
            ref_ts = ref_ts.tz_convert("UTC").tz_localize(None)
        payload["_reference_ts"] = ref_ts
        abspos_ref = payload
    else:
        abspos_ref = load_abspos_reference(build)
    global_ref: pd.Timestamp = abspos_ref["_reference_ts"]
    print(
        {
            "abspos_reference": abspos_ref["reference_timestamp"],
            "unit": abspos_ref.get("unit", "hour"),
        },
        flush=True,
    )

    from src.data_new.vocabulary import RegistryVocabulary
    from src.tasks.grouped_mlm import GroupedMLM
    from src.transformer.models import TransformerEncoder

    max_rows = None if args.max_rows <= 0 else args.max_rows
    print(f"streaming parquet={parquet} max_rows={max_rows}", flush=True)

    vocab = RegistryVocabulary(registry_path=str(registry), registry_version="v2")
    if args.mode == "full" and args.max_length == 512:
        max_length = 1024
    else:
        max_length = args.max_length

    task = GroupedMLM(
        name="online2_v2_grouped_mlm",
        max_length=max_length,
        mask_ratio=0.3,
        vocab_v2_path=str(vocab_v2),
        mask_feature_identity=False,
        non_maskable_tokens=["[IMAGE_SLOT]", "[TEXT_SLOT]", "IMAGE_EMBED_SLOT", "TEXT_EMBED_SLOT"],
        sop_reverse_probability=args.sop_reverse,
        sop_shuffle_probability=args.sop_shuffle,
        evaluation_seed=2023,
        shuffle_within_sentences=False,
    )
    task.datamodule = SimpleNamespace(vocabulary=vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = args.mode == "full"
    hparams = {
        "vocabulary": vocab,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "hidden_size": 128 if args.mode == "sanity" else 384,
        "hidden_ff": 512 if args.mode == "sanity" else 1536,
        "hidden_act": "swish",
        "n_encoders": 2 if args.mode == "sanity" else 6,
        "n_heads": 4 if args.mode == "sanity" else 8,
        "n_local": 2 if args.mode == "sanity" else 4,
        "local_window_size": 16 if args.mode == "sanity" else 64,
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
        "experiment_version": "2.2",
        "attention_type": "performer",
        "multihead_dc": False,
        "num_random_features": 64 if args.mode == "sanity" else 128,
        "learning_rate": 1e-3 if args.mode == "sanity" else 5e-4,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "cls_num_targs": 3,
        "epsilon": 1e-6,
        "stage": "pre_training",
        "implementation": "online2_v2",
        "version": "2.2",
        "vocab_size": vocab.size(),
        "abspos_reference": abspos_ref["reference_timestamp"],
        "abspos_unit": "hour",
    }
    model = TransformerEncoder(hparams).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=hparams["learning_rate"], weight_decay=0.01)

    use_shadow_val = detect_shadow_val(parquet)
    train_filter = make_split_predicate(
        use_shadow_split=use_shadow_val,
        want="train",
        val_frac=args.val_frac,
        seed=args.val_seed,
    )
    val_filter = make_split_predicate(
        use_shadow_split=use_shadow_val,
        want="val",
        val_frac=args.val_frac,
        seed=args.val_seed,
    )
    print(
        {
            "val_split": "shadow_split" if use_shadow_val else f"hash_person_frac={args.val_frac}",
            "val_every": args.val_every,
            "val_batches": args.val_batches,
            "early_stop_patience": None if args.no_early_stop else args.early_stop_patience,
            "max_steps": args.steps,
        },
        flush=True,
    )

    def _encode_train(frame, people, dev):
        return encode_batch(task, frame, people, dev, global_ref, is_train=True)

    def _encode_val(frame, people, dev):
        return encode_batch(task, frame, people, dev, global_ref, is_train=False)

    start_step = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {args.resume} step={start_step}", flush=True)

    batch_size = args.batch_size if args.mode == "sanity" else max(args.batch_size, 8)
    if full and not args.skip_vram_calibrate and device.type == "cuda":
        cal_iter = iter_person_batches(
            parquet, max(2, batch_size), max_rows, id_filter=train_filter
        )
        batch_size = calibrate_batch_size(
            model,
            _encode_train,
            cal_iter,
            device,
            start_bs=batch_size,
            target_frac=args.target_vram_frac,
            max_bs=64,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=hparams["learning_rate"], weight_decay=0.01)
        torch.cuda.empty_cache()
        print({"batch_size_final": batch_size, "max_length": max_length}, flush=True)

    steps = args.steps if args.mode == "sanity" else max(args.steps, 50)
    history = []
    val_history = []
    sop_nonzero = 0
    best_val = float("inf")
    best_step = 0
    patience_left = args.early_stop_patience
    stopped_early = False
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
        "max_length": max_length,
        "batch_size": batch_size,
        "sop_reverse_probability": args.sop_reverse,
        "sop_shuffle_probability": args.sop_shuffle,
        "target_vram_frac": args.target_vram_frac,
        "hidden_size": hparams["hidden_size"],
        "n_encoders": hparams["n_encoders"],
        "abspos_reference": abspos_ref["reference_timestamp"],
        "abspos_unit": "hour",
        "age_semantics": "sequence_relative_hours",
        "wandb_enabled": bool(args.wandb),
        "val_split": "shadow_split" if use_shadow_val else "hash_person",
        "val_frac": args.val_frac if not use_shadow_val else None,
        "val_every": args.val_every,
        "val_batches": args.val_batches,
        "early_stop_patience": None if args.no_early_stop else args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "max_steps": steps,
        "wandb_watch": bool(args.wandb_watch),
        "wandb_watch_log": args.wandb_watch_log,
        "wandb_watch_freq": args.wandb_watch_freq,
    }
    write_json(run_dir / "run_manifest_v2.json", manifest)

    wb_run = None
    try:
        wb_run = init_wandb(args, manifest, abspos_ref, model=model)
        if wb_run is not None:
            manifest["wandb_run_id"] = wb_run.id
            manifest["wandb_url"] = wb_run.url
            write_json(run_dir / "run_manifest_v2.json", manifest)
            print({"wandb_url": wb_run.url, "wandb_run_id": wb_run.id}, flush=True)
    except SystemExit:
        raise
    except Exception as exc:
        if args.wandb:
            raise SystemExit(f"wandb.init failed: {exc}. Use --no-wandb to skip.") from exc

    from src.transformer.models import masked_sop_loss

    step = start_step
    data_iter = iter_person_batches(
        parquet, batch_size, max_rows, id_filter=train_filter
    )
    val_iter = iter_person_batches(
        parquet, batch_size, max_rows, id_filter=val_filter
    )
    try:
        while step < start_step + steps:
            try:
                try:
                    frame, chosen = next(data_iter)
                except StopIteration:
                    data_iter = iter_person_batches(
                        parquet, batch_size, max_rows, id_filter=train_filter
                    )
                    frame, chosen = next(data_iter)
                people = list(chosen)
                if len(people) < batch_size:
                    while len(people) < batch_size:
                        people.extend(chosen)
                    people = people[:batch_size]
                batch = _encode_train(frame, people, device)
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
                sop_v = float(sop_loss.detach().cpu())
                if sop_v > 0:
                    sop_nonzero += 1
                rec = {
                    "step": step,
                    "train/loss": float(loss.detach().cpu()),
                    "train/mlm_loss": float(mlm_loss.detach().cpu()),
                    "train/sop_loss": sop_v,
                    "train/sop_mask_mean": float(batch["target_cls_mask"].float().mean().cpu())
                    if torch.is_tensor(batch.get("target_cls_mask"))
                    else None,
                    "train/grad_norm": float(grad_norm.detach().cpu())
                    if torch.is_tensor(grad_norm)
                    else float(grad_norm),
                    "batch_size": batch_size,
                }
                if device.type == "cuda" and step % 20 == 0:
                    rec["train/vram_alloc_gb"] = round(
                        torch.cuda.memory_allocated(device) / 1e9, 2
                    )
                    rec["train/vram_reserved_gb"] = round(
                        torch.cuda.memory_reserved(device) / 1e9, 2
                    )
                    rec["train/vram_frac"] = round(
                        torch.cuda.memory_reserved(device)
                        / torch.cuda.get_device_properties(device).total_memory,
                        3,
                    )
                history.append(rec)
                print(rec, flush=True)
                if wb_run is not None:
                    import wandb

                    wandb.log({k: v for k, v in rec.items() if k != "step"}, step=step)

                do_val = step % args.val_every == 0 or step == start_step + steps
                if do_val:
                    val_metrics = evaluate_validation(
                        model,
                        _encode_val,
                        val_iter,
                        device,
                        max_batches=args.val_batches,
                    )
                    if val_metrics["val/batches"] == 0:
                        # Restart val iterator if exhausted mid-run.
                        val_iter = iter_person_batches(
                            parquet, batch_size, max_rows, id_filter=val_filter
                        )
                        val_metrics = evaluate_validation(
                            model,
                            _encode_val,
                            val_iter,
                            device,
                            max_batches=args.val_batches,
                        )
                    val_rec = {"step": step, **val_metrics}
                    val_history.append(val_rec)
                    print(val_rec, flush=True)
                    if wb_run is not None:
                        import wandb

                        wandb.log(
                            {k: v for k, v in val_rec.items() if k != "step"},
                            step=step,
                        )

                    val_loss = float(val_metrics["val/loss"])
                    improved = val_loss < (best_val - args.early_stop_min_delta)
                    if improved:
                        best_val = val_loss
                        best_step = step
                        patience_left = args.early_stop_patience
                        best_path = run_dir / "best.ckpt"
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "opt": opt.state_dict(),
                                "step": step,
                                "best_val_loss": best_val,
                                "abspos_reference": abspos_ref["reference_timestamp"],
                                "hparams": {
                                    k: v for k, v in hparams.items() if k != "vocabulary"
                                },
                            },
                            best_path,
                        )
                        print(
                            {
                                "best_ckpt": str(best_path),
                                "best_val_loss": best_val,
                                "step": step,
                            },
                            flush=True,
                        )
                    elif not args.no_early_stop:
                        patience_left -= 1
                        print(
                            {
                                "early_stop_patience_left": patience_left,
                                "best_val_loss": best_val,
                                "val_loss": val_loss,
                            },
                            flush=True,
                        )
                        if patience_left <= 0:
                            stopped_early = True
                            print(
                                {
                                    "EARLY_STOP": True,
                                    "best_step": best_step,
                                    "best_val_loss": best_val,
                                    "step": step,
                                },
                                flush=True,
                            )

                if step % args.ckpt_every == 0 or step == start_step + steps or stopped_early:
                    ckpt_path = run_dir / f"checkpoint_step_{step}.pt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "step": step,
                            "hparams": {
                                k: v for k, v in hparams.items() if k != "vocabulary"
                            },
                            "abspos_reference": abspos_ref["reference_timestamp"],
                            "best_val_loss": best_val if best_val < float("inf") else None,
                        },
                        ckpt_path,
                    )
                    last = run_dir / "last.ckpt"
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "step": step,
                            "abspos_reference": abspos_ref["reference_timestamp"],
                            "best_val_loss": best_val if best_val < float("inf") else None,
                        },
                        last,
                    )
                    print(f"saved {ckpt_path}", flush=True)
                if stopped_early:
                    break
            except RuntimeError as exc:
                msg = str(exc).lower()
                manifest["attempts"].append(
                    {"step": step, "error": str(exc), "batch_size": batch_size}
                )
                write_json(run_dir / "run_manifest_v2.json", manifest)
                if "out of memory" in msg:
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        data_iter = iter_person_batches(
                            parquet, batch_size, max_rows, id_filter=train_filter
                        )
                        print(f"OOM_RECOVERY batch_size->{batch_size}", flush=True)
                        torch.cuda.empty_cache()
                        continue
                raise

        # Prefer best.ckpt for resume verify when early-stopped / validated.
        verify_path = run_dir / "best.ckpt"
        if not verify_path.exists():
            verify_path = run_dir / "last.ckpt"
        verify = torch.load(verify_path, map_location=device)
        model.load_state_dict(verify["model"])
        print(
            {
                "resume_verify_step": verify["step"],
                "verify_ckpt": str(verify_path),
                "pass": True,
            },
            flush=True,
        )

        manifest.update(
            {
                "status": "PASS",
                "final_step": step,
                "stopped_early": stopped_early,
                "best_val_loss": best_val if best_val < float("inf") else None,
                "best_step": best_step if best_step else None,
                "history_tail": history[-5:],
                "val_history_tail": val_history[-5:],
                "batch_size_final": batch_size,
                "sop_nonzero_steps": sop_nonzero,
                "sop_nonzero_rate": round(sop_nonzero / max(len(history), 1), 4),
                "checkpoint": str(run_dir / "last.ckpt"),
                "best_checkpoint": str(run_dir / "best.ckpt")
                if (run_dir / "best.ckpt").exists()
                else None,
                "resume_command": (
                    f"python scripts/online2_v2/run_v2_pretrain_loop.py --mode {args.mode} "
                    f"--build-dir {build} --run-dir {run_dir} --resume {run_dir/'last.ckpt'} "
                    f"--steps {steps} --batch-size {batch_size} --max-rows {args.max_rows} "
                    f"--max-length {max_length}"
                ),
            }
        )
        write_json(run_dir / "run_manifest_v2.json", manifest)
        write_json(
            run_dir / "train_history.json",
            {"history": history, "val_history": val_history},
        )
        print("DONE", manifest["status"], manifest["final_step"], flush=True)
    finally:
        if wb_run is not None:
            import wandb

            try:
                wandb.unwatch()
            except Exception:
                pass
            wandb.finish()


if __name__ == "__main__":
    main()
