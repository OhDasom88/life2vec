#!/usr/bin/env python3
"""Train Stage B/C diagnosis classifier on Stage A event embedding caches.

Modes:
  --mode overfit   Fit all available cached cases (sanity)
  --mode cv        Stratified farm K-fold CV
  --mode smoke     Tiny overfit on whatever caches exist (few epochs)

Examples:
  python scripts/online2_v2/run_diagnosis_event_pooling_finetune.py --mode smoke --epochs 20
  python scripts/online2_v2/run_diagnosis_event_pooling_finetune.py --mode overfit --epochs 50
  python scripts/online2_v2/run_diagnosis_event_pooling_finetune.py \\
    --mode cv --n-folds 5 --epochs 40 --wandb --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.diagnosis_dataset import (
    DiagnosisEventDataset,
    collate_diagnosis_batch,
    load_label_map,
    make_stratified_farm_folds,
)
from src.online2.v2.event_pooling_finetune import (
    EventPoolingConfig,
    EventPoolingDiagnosisModel,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--emb-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/event_embeddings",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/labels_example_score90.csv",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/label_map.json",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Default: outputs/online2/v2_finetune/runs/<timestamp>",
    )
    p.add_argument("--mode", choices=["smoke", "overfit", "cv"], default="smoke")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=None, help="Run a single CV fold index")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument(
        "--batch-size",
        type=int,
        default=28,
        help="Cases per forward (3090 fits 35@T=4096; CV train fold max≈28)",
    )
    p.add_argument("--max-events", type=int, default=4096)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="PAD attention heads (must divide hidden-dim)",
    )
    p.add_argument(
        "--ff-dim",
        type=int,
        default=None,
        help="PAD FFN dim (default: 2 * hidden-dim)",
    )
    p.add_argument(
        "--mean-only",
        action="store_true",
        help="Use event_mean only (no max concat)",
    )
    p.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any selected case cache is missing (default: use available)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Stop after this many epochs without val improvement (default: 20)",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val_loss decrease to count as improvement",
    )
    p.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=30,
        help="Do not early-stop before this epoch (avoids underfit stops)",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping (run all --epochs)",
    )
    p.add_argument("--wandb", dest="wandb", action="store_true", default=False)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", type=str, default="Berry2Vec")
    p.add_argument("--wandb-entity", type=str, default="dasom-oh")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument(
        "--wandb-tags",
        type=str,
        default="online2_v2,diagnosis,event_pooling,pad,finetuning",
    )
    p.add_argument(
        "--wandb-watch",
        dest="wandb_watch",
        action="store_true",
        default=None,
        help="Track param/grad histograms (default: on when --wandb)",
    )
    p.add_argument("--no-wandb-watch", dest="wandb_watch", action="store_false")
    p.add_argument("--wandb-watch-log", type=str, default="all")
    p.add_argument("--wandb-watch-freq", type=int, default=50)
    args = p.parse_args()
    if args.wandb_watch is None:
        args.wandb_watch = bool(args.wandb)
    return args


def init_wandb(args: argparse.Namespace, run_dir: Path, config: Dict[str, Any]):
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
    tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_dir.name,
        tags=tags,
        config=config,
        dir=str(run_dir),
        reinit=True,
    )
    print({"wandb_url": getattr(run, "url", None), "wandb_id": run.id}, flush=True)
    return run


def maybe_wandb_watch(args: argparse.Namespace, model: torch.nn.Module, watched: list) -> None:
    if not args.wandb or not args.wandb_watch or watched:
        return
    import wandb

    wandb.watch(
        model,
        log=args.wandb_watch_log,
        log_freq=max(1, int(args.wandb_watch_freq)),
        log_graph=False,
    )
    watched.append(True)
    print(
        {
            "wandb_watch": True,
            "log": args.wandb_watch_log,
            "log_freq": args.wandb_watch_freq,
        },
        flush=True,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_cached_case_ids(emb_dir: Path, labels_df: pd.DataFrame) -> List[str]:
    available = []
    for cid in labels_df["case_id"].astype(str):
        path = emb_dir / f"{cid}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["event_mean"])
            if df.empty or df["event_mean"].iloc[0] is None:
                continue
            available.append(cid)
        except Exception:
            continue
    return available


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for batch in loader:
        labels = batch["label"].to(device)
        out = model(
            event_mean=batch["event_mean"].to(device),
            event_max=batch["event_max"].to(device),
            case_age_hours=batch["case_age_hours"].to(device),
            view_id=batch["view_id"].to(device),
            zone_id=batch["zone_id"].to(device),
            local_hour=batch["local_hour"].to(device),
            padding_mask=batch["padding_mask"].to(device),
        )
        loss = model.loss(out["logits"], labels)
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)
        preds = out["logits"].argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())
    model.train()
    if n == 0:
        return {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan")}
    acc = float(np.mean([p == y for p, y in zip(all_preds, all_labels)]))
    # macro-F1
    f1s = []
    for c in sorted(set(all_labels)):
        tp = sum(p == c and y == c for p, y in zip(all_preds, all_labels))
        fp = sum(p == c and y != c for p, y in zip(all_preds, all_labels))
        fn = sum(p != c and y == c for p, y in zip(all_preds, all_labels))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return {
        "loss": total_loss / n,
        "acc": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "n": float(n),
    }


def train_one_split(
    *,
    train_cases: List[str],
    val_cases: List[str],
    labels_df: pd.DataFrame,
    label_map: Dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    split_name: str,
    global_step_start: int = 0,
    wandb_watched: Optional[list] = None,
) -> Dict[str, Any]:
    device = torch.device(args.device)
    train_ds = DiagnosisEventDataset(
        train_cases,
        labels_df,
        label_map,
        args.emb_dir,
        max_events=args.max_events,
        require_complete=False,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"{split_name}: no train caches available")

    val_ds = None
    if val_cases:
        val_ds = DiagnosisEventDataset(
            val_cases,
            labels_df,
            label_map,
            args.emb_dir,
            max_events=args.max_events,
            require_complete=False,
        )

    # Infer H from first sample
    h = int(train_ds[0]["hidden_size"])
    ff_dim = int(args.ff_dim) if args.ff_dim is not None else int(args.hidden_dim) * 2
    cfg = EventPoolingConfig(
        input_dim=h,
        hidden_dim=args.hidden_dim,
        num_heads=int(args.num_heads),
        ff_dim=ff_dim,
        dropout=args.dropout,
        num_classes=len(label_map["labels"]),
        use_mean_max=not bool(args.mean_only),
    )
    model = EventPoolingDiagnosisModel(cfg).to(device)
    if wandb_watched is not None:
        maybe_wandb_watch(args, model, wandb_watched)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def make_loader(ds, shuffle: bool):
        bs = max(1, min(int(args.batch_size), len(ds)))
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=lambda xs: collate_diagnosis_batch(xs, max_events=args.max_events),
            num_workers=0,
            drop_last=False,
        )

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False) if val_ds and len(val_ds) else None

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_path = run_dir / f"{split_name}_best.pt"
    best_row: Optional[Dict[str, Any]] = None
    accum = max(1, int(args.grad_accum))
    global_step = int(global_step_start)
    use_early_stop = (
        (not args.no_early_stop)
        and val_loader is not None
        and int(args.early_stop_patience) > 0
    )
    patience_left = int(args.early_stop_patience)
    stopped_early = False
    stop_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        n_seen = 0
        correct = 0
        step_in_accum = 0
        t0 = time.time()
        for batch in train_loader:
            labels = batch["label"].to(device)
            out = model(
                event_mean=batch["event_mean"].to(device),
                event_max=batch["event_max"].to(device),
                case_age_hours=batch["case_age_hours"].to(device),
                view_id=batch["view_id"].to(device),
                zone_id=batch["zone_id"].to(device),
                local_hour=batch["local_hour"].to(device),
                padding_mask=batch["padding_mask"].to(device),
            )
            loss = model.loss(out["logits"], labels) / accum
            loss.backward()
            step_in_accum += 1
            if step_in_accum >= accum:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step_in_accum = 0
            running += float(loss.item()) * accum * labels.size(0)
            n_seen += labels.size(0)
            correct += int((out["logits"].argmax(-1) == labels).sum().item())

        if step_in_accum > 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        train_metrics = {
            "loss": running / max(n_seen, 1),
            "acc": correct / max(n_seen, 1),
        }
        val_metrics = (
            evaluate(model, val_loader, device)
            if val_loader is not None
            else {"loss": float("nan"), "acc": float("nan"), "macro_f1": float("nan")}
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics.get("macro_f1", float("nan")),
            "sec": time.time() - t0,
            "early_stop_patience_left": patience_left if use_early_stop else None,
        }
        history.append(row)
        print(
            f"[{split_name}] epoch={epoch} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.3f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.3f} "
            f"val_f1={row['val_macro_f1']:.3f} ({row['sec']:.1f}s)"
            + (f" patience={patience_left}" if use_early_stop else ""),
            flush=True,
        )

        global_step += 1
        if args.wandb:
            import wandb

            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "split": split_name,
                f"{split_name}/train_loss": row["train_loss"],
                f"{split_name}/train_acc": row["train_acc"],
                f"{split_name}/epoch_sec": row["sec"],
                "train/loss": row["train_loss"],
                "train/acc": row["train_acc"],
            }
            if val_loader is not None and np.isfinite(row["val_loss"]):
                payload.update(
                    {
                        f"{split_name}/val_loss": row["val_loss"],
                        f"{split_name}/val_acc": row["val_acc"],
                        f"{split_name}/val_macro_f1": row["val_macro_f1"],
                        "val/loss": row["val_loss"],
                        "val/acc": row["val_acc"],
                        "val/macro_f1": row["val_macro_f1"],
                    }
                )
            if use_early_stop:
                payload[f"{split_name}/early_stop_patience_left"] = patience_left
                payload["early_stop/patience_left"] = patience_left
            wandb.log(payload, step=global_step)

        if val_loader is not None and np.isfinite(val_metrics["loss"]):
            improved = val_metrics["loss"] < (
                best_val - float(args.early_stop_min_delta)
            )
            if improved:
                best_val = float(val_metrics["loss"])
                best_row = row
                patience_left = int(args.early_stop_patience)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "epoch": epoch,
                        "train_cases": train_ds.case_ids,
                        "val_cases": val_ds.case_ids if val_ds else [],
                        "metrics": row,
                    },
                    best_path,
                )
            elif use_early_stop and epoch >= int(args.early_stop_min_epochs):
                patience_left -= 1
                if patience_left <= 0:
                    stopped_early = True
                    stop_epoch = epoch
                    print(
                        {
                            "early_stop": True,
                            "split": split_name,
                            "epoch": epoch,
                            "best_val_loss": best_val,
                            "best_epoch": best_row["epoch"] if best_row else None,
                        },
                        flush=True,
                    )
                    if args.wandb:
                        import wandb

                        wandb.log(
                            {
                                f"{split_name}/early_stopped": 1,
                                f"{split_name}/stop_epoch": epoch,
                                "early_stop/triggered": 1,
                            },
                            step=global_step,
                        )
                    break
        else:
            score = -train_metrics["acc"]
            if score < best_val:
                best_val = score
                best_row = row
                torch.save(
                    {
                        "model": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "epoch": epoch,
                        "train_cases": train_ds.case_ids,
                        "val_cases": [],
                        "metrics": row,
                    },
                    best_path,
                )

    (run_dir / f"{split_name}_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "split": split_name,
        "n_train": len(train_ds),
        "n_val": len(val_ds) if val_ds else 0,
        "best_path": str(best_path),
        "best_val": best_val,
        "best_metrics": best_row,
        "history_tail": history[-3:],
        "global_step_end": global_step,
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch,
        "early_stop_enabled": use_early_stop,
    }


def apply_wandb_config(args: argparse.Namespace) -> argparse.Namespace:
    """Override CLI args from wandb.config when running under a sweep agent."""
    if not args.wandb:
        return args
    try:
        import wandb
    except ImportError:
        return args
    if wandb.run is None:
        return args
    cfg = dict(wandb.config)
    mapping = {
        "lr": "lr",
        "learning_rate": "lr",
        "weight_decay": "weight_decay",
        "dropout": "dropout",
        "hidden_dim": "hidden_dim",
        "grad_accum": "grad_accum",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "seed": "seed",
        "num_heads": "num_heads",
        "ff_dim": "ff_dim",
        "mean_only": "mean_only",
        "early_stop_patience": "early_stop_patience",
        "early_stop_min_epochs": "early_stop_min_epochs",
    }
    applied = {}
    for src, dst in mapping.items():
        if src not in cfg or cfg[src] is None:
            continue
        val = cfg[src]
        cur = getattr(args, dst, None)
        if isinstance(cur, bool):
            setattr(args, dst, bool(val))
        elif isinstance(cur, int) and not isinstance(cur, bool):
            setattr(args, dst, int(val))
        elif isinstance(cur, float):
            setattr(args, dst, float(val))
        elif dst == "ff_dim":
            setattr(args, dst, None if val in (None, "null") else int(val))
        else:
            setattr(args, dst, val)
        applied[dst] = getattr(args, dst)
    if applied:
        print({"wandb_config_applied": applied}, flush=True)
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    labels_df = pd.read_csv(args.labels)
    label_map = load_label_map(args.label_map)
    cached = list_cached_case_ids(args.emb_dir, labels_df)
    if not cached:
        raise SystemExit(
            f"No Stage A caches found under {args.emb_dir}. "
            "Run cache_stage_a_event_embeddings.py first."
        )
    print(f"cached cases={len(cached)}/{len(labels_df)}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (
        args.root / "outputs/online2/v2_finetune/runs" / f"{args.mode}_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    wb_run = init_wandb(
        args,
        run_dir,
        config={
            "mode": args.mode,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_accum": args.grad_accum,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "num_heads": args.num_heads,
            "ff_dim": args.ff_dim,
            "mean_only": bool(args.mean_only),
            "max_events": args.max_events,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "n_cached": len(cached),
            "run_dir": str(run_dir),
            "architecture": "PAD_event_pooling_diagnosis",
            "stage_a": "frozen_cached_mean_max",
            "num_classes": len(label_map["labels"]),
            "early_stop_patience": None
            if args.no_early_stop
            else args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "early_stop_min_epochs": args.early_stop_min_epochs,
            "no_early_stop": args.no_early_stop,
        },
    )
    args = apply_wandb_config(args)
    set_seed(args.seed)

    manifest = {
        "mode": args.mode,
        "created_utc": stamp,
        "cached_cases": cached,
        "n_cached": len(cached),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }

    results = []
    global_step = 0
    wandb_watched: list = []
    try:
        print(
            {
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            flush=True,
        )
        if args.mode in ("smoke", "overfit"):
            train_cases = cached
            val_cases = cached if args.mode == "smoke" else []
            results.append(
                train_one_split(
                    train_cases=train_cases,
                    val_cases=val_cases,
                    labels_df=labels_df,
                    label_map=label_map,
                    args=args,
                    run_dir=run_dir,
                    split_name=args.mode,
                    global_step_start=global_step,
                    wandb_watched=wandb_watched,
                )
            )
            global_step = results[-1]["global_step_end"]
        else:
            folds = make_stratified_farm_folds(
                labels_df, n_folds=args.n_folds, seed=args.seed
            )
            fold_idxs = (
                [args.fold] if args.fold is not None else list(range(args.n_folds))
            )
            for i in fold_idxs:
                fold = folds[i]
                train_cases = [c for c in fold["train"] if c in set(cached)]
                val_cases = [c for c in fold["val"] if c in set(cached)]
                if not train_cases:
                    print(f"[fold{i}] skip: no train caches", flush=True)
                    continue
                print(
                    f"[fold{i}] train={len(train_cases)} val={len(val_cases)} "
                    f"val_farms={fold['val_farms']} "
                    f"batch_size={min(args.batch_size, len(train_cases))}",
                    flush=True,
                )
                if args.wandb:
                    import wandb

                    wandb.log(
                        {
                            f"fold{i}/n_train": len(train_cases),
                            f"fold{i}/n_val": len(val_cases),
                        },
                        step=max(global_step, 1),
                    )
                results.append(
                    train_one_split(
                        train_cases=train_cases,
                        val_cases=val_cases,
                        labels_df=labels_df,
                        label_map=label_map,
                        args=args,
                        run_dir=run_dir,
                        split_name=f"fold{i}",
                        global_step_start=global_step,
                        wandb_watched=wandb_watched,
                    )
                )
                global_step = results[-1]["global_step_end"]

        if args.mode == "cv" and results:
            val_losses = [
                r["best_metrics"]["val_loss"]
                for r in results
                if r.get("best_metrics")
                and np.isfinite(r["best_metrics"].get("val_loss", float("nan")))
            ]
            val_accs = [
                r["best_metrics"]["val_acc"]
                for r in results
                if r.get("best_metrics")
                and np.isfinite(r["best_metrics"].get("val_acc", float("nan")))
            ]
            val_f1s = [
                r["best_metrics"]["val_macro_f1"]
                for r in results
                if r.get("best_metrics")
                and np.isfinite(r["best_metrics"].get("val_macro_f1", float("nan")))
            ]
            summary = {
                "cv/mean_best_val_loss": float(np.mean(val_losses)) if val_losses else None,
                "cv/mean_best_val_acc": float(np.mean(val_accs)) if val_accs else None,
                "cv/mean_best_val_macro_f1": float(np.mean(val_f1s)) if val_f1s else None,
                "cv/n_folds_run": len(results),
            }
            manifest["cv_summary"] = summary
            print("CV_SUMMARY", summary, flush=True)
            if args.wandb:
                import wandb

                wandb.log(
                    {k: v for k, v in summary.items() if v is not None},
                    step=global_step + 1,
                )
                wandb.summary.update(
                    {k: v for k, v in summary.items() if v is not None}
                )

        manifest["results"] = results
        if wb_run is not None:
            manifest["wandb_url"] = getattr(wb_run, "url", None)
            manifest["wandb_id"] = getattr(wb_run, "id", None)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"DONE run_dir={run_dir}", flush=True)
        if wb_run is not None:
            print(f"WANDB {getattr(wb_run, 'url', None)}", flush=True)
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
