#!/usr/bin/env python3
"""Finetune v0.2: PAD + diagnosis + state/cause semantic alignment.

Does NOT modify v0.1 training script. Reads Stage A caches from v0.1 path by
default. Checkpoints use v0.1-compatible names:

  <run_dir>/fold{i}_best.pt

with payload keys: model, cfg, epoch, train_cases, val_cases, metrics,
finetune_version, model_class — so run_evaluation_pipeline.py can load them.

Examples:
  python scripts/online2_v2/v02/build_interpretation_banks.py
  python scripts/online2_v2/v02/run_diagnosis_finetune_v02.py --mode smoke --epochs 5
  python scripts/online2_v2/v02/run_diagnosis_finetune_v02.py --mode cv --device cuda
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v02.checkpoint import (  # noqa: E402
    build_v02_ckpt_payload,
    fold_ckpt_path,
)
from src.online2.v2.finetune_v02.config import FinetuneV02Config, LossWeights  # noqa: E402
from src.online2.v2.finetune_v02.dataset import (  # noqa: E402
    DiagnosisEventDatasetV02,
    collate_diagnosis_batch_v02,
    load_interpretation_bank,
    load_label_map,
    make_stratified_farm_folds,
)
from src.online2.v2.finetune_v02.losses import total_finetune_loss  # noqa: E402
from src.online2.v2.finetune_v02.metrics import (  # noqa: E402
    accuracy,
    fold_selection_ok,
    macro_f1_all_classes,
    macro_f1_present_labels,
    oof_selection_ok,
)
from src.online2.v2.finetune_v02.model import EventPoolingDiagnosisModelV02  # noqa: E402
from src.online2.v2.finetune_v02.version import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    FINETUNE_VERSION,
)

DEFAULT_BANK = ROOT / DEFAULT_OUTPUT_ROOT / "interpretation_banks" / "state_cause_bank.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--emb-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune_v02/event_embeddings",
        help="Stage A cache dir (prefer IMAGE+dino_vec from cache_stage_a_image_slots_v02)",
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
        "--interpretation-bank",
        type=Path,
        default=DEFAULT_BANK,
        help="Parquet bank (case_id,quality,axis,embedding). Missing → CE-only.",
    )
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--mode", choices=["smoke", "overfit", "cv"], default="smoke")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=28)
    p.add_argument("--max-events", type=int, default=4096)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--semantic-dim", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lambda-state", type=float, default=0.10)
    p.add_argument("--lambda-cause", type=float, default=0.15)
    p.add_argument("--lambda-rank", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--early-stop-min-epochs", type=int, default=30)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument(
        "--monitor",
        choices=["val_loss", "val_macro_f1"],
        default="val_macro_f1",
        help="Checkpoint selection metric (v0.2 default: maximize val_macro_f1)",
    )
    p.add_argument("--require-complete", action="store_true")
    p.add_argument(
        "--allow-missing-bank",
        action="store_true",
        help="If bank missing, train CE-only instead of failing",
    )
    p.add_argument("--wandb", dest="wandb", action="store_true", default=False)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", type=str, default="Berry2Vec")
    p.add_argument("--wandb-entity", type=str, default="dasom-oh")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument(
        "--wandb-tags",
        type=str,
        default="online2_v2,diagnosis,finetune_v02,event_pooling,semantic_align,image_dino",
    )
    p.add_argument(
        "--wandb-watch",
        dest="wandb_watch",
        action="store_true",
        default=None,
    )
    p.add_argument("--no-wandb-watch", dest="wandb_watch", action="store_false")
    p.add_argument("--wandb-watch-log", type=str, default="gradients")
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
        raise SystemExit("wandb not installed") from exc
    if not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        raise SystemExit("wandb login missing (WANDB_API_KEY / ~/.netrc)")
    tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
    under_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=None if under_sweep else (args.wandb_run_name or run_dir.name),
        tags=tags,
        config=config,
        dir=str(run_dir),
        reinit=True,
    )
    print({"wandb_url": getattr(run, "url", None), "wandb_id": run.id, "sweep": under_sweep}, flush=True)
    return run


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
        "batch_size": "batch_size",
        "epochs": "epochs",
        "seed": "seed",
        "num_heads": "num_heads",
        "early_stop_patience": "early_stop_patience",
        "early_stop_min_epochs": "early_stop_min_epochs",
        "lambda_state": "lambda_state",
        "lambda_cause": "lambda_cause",
        "lambda_rank": "lambda_rank",
        "monitor": "monitor",
        "n_folds": "n_folds",
    }
    applied: Dict[str, Any] = {}
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
        elif dst == "monitor":
            m = str(val)
            if m not in {"val_loss", "val_macro_f1"}:
                continue
            setattr(args, dst, m)
        else:
            setattr(args, dst, val)
        applied[dst] = getattr(args, dst)
    if applied:
        print({"wandb_config_applied": applied}, flush=True)
    return args


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


@torch.no_grad()
def evaluate_probability_ensemble_all(
    *,
    run_dir: Path,
    case_ids: List[str],
    labels_df: pd.DataFrame,
    label_map: Dict[str, Any],
    bank: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """v0.1-style: each fold predicts ALL cases, mean softmax → Acc/F1."""
    from src.online2.v2.finetune_v02.checkpoint import load_five_fold_models

    device = torch.device(args.device)
    models, _metas, _ver = load_five_fold_models(run_dir, device=device)
    if not models:
        return {"n_folds": 0, "n": 0, "acc": float("nan"), "macro_f1": float("nan"), "perfect": False}

    ds = DiagnosisEventDatasetV02(
        case_ids,
        labels_df=labels_df,
        label_map=label_map,
        emb_dir=args.emb_dir,
        interpretation_bank=bank,
        semantic_dim=args.semantic_dim,
        max_events=args.max_events,
        require_complete=bool(args.require_complete),
    )
    fold_probs: List[np.ndarray] = []
    labels: List[int] = []
    ids: List[str] = []
    for i in range(len(ds)):
        sample = ds[i]
        batch = collate_diagnosis_batch_v02([sample], max_events=args.max_events)
        ids.append(sample["case_id"])
        labels.append(int(sample["label"]))
        probs_f = []
        for model in models:
            model.eval()
            out = model(
                event_mean=batch["event_mean"].to(device),
                event_max=batch["event_max"].to(device),
                case_age_hours=batch["case_age_hours"].to(device),
                view_id=batch["view_id"].to(device),
                zone_id=batch["zone_id"].to(device),
                local_hour=batch["local_hour"].to(device),
                padding_mask=batch["padding_mask"].to(device),
                dino_vec=batch["dino_vec"].to(device) if "dino_vec" in batch else None,
                dino_mask=batch["dino_mask"].to(device) if "dino_mask" in batch else None,
            )
            probs_f.append(torch.softmax(out["logits"], dim=-1).float().cpu().numpy()[0])
        fold_probs.append(np.stack(probs_f, axis=0))  # [F,C]

    if not fold_probs:
        return {"n_folds": len(models), "n": 0, "acc": float("nan"), "macro_f1": float("nan"), "perfect": False}

    # [N,F,C] → mean over folds
    stacked = np.stack(fold_probs, axis=0)  # [N,F,C]
    ens = stacked.mean(axis=1)
    preds = ens.argmax(axis=-1).tolist()
    acc = accuracy(labels, preds)
    mf1 = macro_f1_all_classes(labels, preds, num_classes=len(label_map["labels"]))
    out = {
        "n_folds": len(models),
        "n": len(labels),
        "acc": acc,
        "macro_f1": mf1,
        "perfect": bool(acc >= 1.0 - 1e-9 and mf1 >= 1.0 - 1e-9),
        "case_ids": ids,
        "labels": labels,
        "preds": preds,
    }
    (run_dir / "ensemble_all35.json").write_text(
        json.dumps({k: v for k, v in out.items() if k not in {"case_ids", "labels", "preds"}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return out


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_cached_case_ids(emb_dir: Path, labels_df: pd.DataFrame) -> List[str]:
    out = []
    for cid in labels_df["case_id"].astype(str):
        path = emb_dir / f"{cid}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["event_mean"])
            if df.empty or df["event_mean"].iloc[0] is None:
                continue
            out.append(cid)
        except Exception:
            continue
    return out


def _maybe_semantic(batch: Dict[str, Any], device: torch.device):
    if "state_targets" not in batch:
        return None, None, None, None
    if not bool(batch.get("has_semantic", torch.tensor([False])).any()):
        return None, None, None, None
    return (
        batch["state_targets"].to(device),
        batch["cause_targets"].to(device),
        batch["state_present"].to(device),
        batch["cause_present"].to(device),
    )


@torch.no_grad()
def evaluate(
    model: EventPoolingDiagnosisModelV02,
    loader: DataLoader,
    device: torch.device,
    cfg: FinetuneV02Config,
) -> Dict[str, Any]:
    model.eval()
    total = 0.0
    ce_sum = 0.0
    state_sum = 0.0
    cause_sum = 0.0
    n = 0
    preds: List[int] = []
    labels: List[int] = []
    case_ids: List[str] = []
    n_dino = 0
    for batch in loader:
        y = batch["label"].to(device)
        dino_vec = batch["dino_vec"].to(device) if "dino_vec" in batch else None
        dino_mask = batch["dino_mask"].to(device) if "dino_mask" in batch else None
        if dino_mask is not None:
            n_dino += int(dino_mask.sum().item())
        out = model(
            event_mean=batch["event_mean"].to(device),
            event_max=batch["event_max"].to(device),
            case_age_hours=batch["case_age_hours"].to(device),
            view_id=batch["view_id"].to(device),
            zone_id=batch["zone_id"].to(device),
            local_hour=batch["local_hour"].to(device),
            padding_mask=batch["padding_mask"].to(device),
            dino_vec=dino_vec,
            dino_mask=dino_mask,
        )
        st, ct, sp, cp = _maybe_semantic(batch, device)
        loss_parts = total_finetune_loss(
            out, y, cfg=cfg, state_targets=st, cause_targets=ct, state_present=sp, cause_present=cp
        )
        bs = y.size(0)
        total += float(loss_parts["total"].item()) * bs
        ce_sum += float(loss_parts["classification_ce"].item()) * bs
        if "state_alignment" in loss_parts:
            state_sum += float(loss_parts["state_alignment"].item()) * bs
        if "cause_alignment" in loss_parts:
            cause_sum += float(loss_parts["cause_alignment"].item()) * bs
        n += bs
        preds.extend(out["logits"].argmax(dim=-1).cpu().tolist())
        labels.extend(y.cpu().tolist())
        case_ids.extend(batch["case_ids"])
    model.train()
    if n == 0:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "macro_f1": float("nan"),
            "ce": float("nan"),
            "state_align": float("nan"),
            "cause_align": float("nan"),
            "n_dino": 0,
            "preds": [],
            "labels": [],
            "case_ids": [],
        }
    return {
        "loss": total / n,
        "ce": ce_sum / n,
        "state_align": state_sum / n,
        "cause_align": cause_sum / n,
        "acc": accuracy(labels, preds),
        "macro_f1": macro_f1_present_labels(labels, preds),
        "n": n,
        "n_dino": n_dino,
        "preds": preds,
        "labels": labels,
        "case_ids": case_ids,
    }


def _is_better(monitor: str, cur: Dict[str, Any], best_score: float) -> bool:
    if monitor == "val_macro_f1":
        score = float(cur.get("val_macro_f1", float("nan")))
        if not np.isfinite(score):
            return False
        return score > best_score + 1e-6
    score = float(cur.get("val_loss", float("nan")))
    if not np.isfinite(score):
        return False
    return score < best_score - 1e-4


def _init_best(monitor: str) -> float:
    return -float("inf") if monitor == "val_macro_f1" else float("inf")


def train_one_split(
    *,
    train_cases: List[str],
    val_cases: List[str],
    labels_df: pd.DataFrame,
    label_map: Dict[str, Any],
    bank: Optional[pd.DataFrame],
    args: argparse.Namespace,
    run_dir: Path,
    split_name: str,
    fold_index: Optional[int] = None,
    global_step_start: int = 0,
    wandb_watched: Optional[list] = None,
) -> Dict[str, Any]:
    device = torch.device(args.device)
    common = dict(
        labels_df=labels_df,
        label_map=label_map,
        emb_dir=args.emb_dir,
        interpretation_bank=bank,
        semantic_dim=args.semantic_dim,
        max_events=args.max_events,
        require_complete=bool(args.require_complete),
    )
    train_ds = DiagnosisEventDatasetV02(train_cases, **common)
    if len(train_ds) == 0:
        raise RuntimeError(f"{split_name}: no train caches")
    val_ds = DiagnosisEventDatasetV02(val_cases, **common) if val_cases else None

    h = int(train_ds[0]["hidden_size"])
    cfg = FinetuneV02Config(
        input_dim=h,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ff_dim=args.hidden_dim * 2,
        dropout=args.dropout,
        num_classes=len(label_map["labels"]),
        semantic_dim=args.semantic_dim,
        max_events=args.max_events,
        loss=LossWeights(
            classification_ce=1.0,
            state_alignment=float(args.lambda_state),
            cause_alignment=float(args.lambda_cause),
            quality_ranking=float(args.lambda_rank),
        ),
    )
    if bank is None:
        cfg.loss.state_alignment = 0.0
        cfg.loss.cause_alignment = 0.0
        cfg.loss.quality_ranking = 0.0

    model = EventPoolingDiagnosisModelV02(cfg).to(device)
    if wandb_watched is not None:
        maybe_wandb_watch(args, model, wandb_watched)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    global_step = int(global_step_start)

    def make_loader(ds, shuffle: bool) -> DataLoader:
        bs = max(1, min(int(args.batch_size), len(ds)))
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=lambda xs: collate_diagnosis_batch_v02(xs, max_events=args.max_events),
            num_workers=0,
        )

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False) if val_ds and len(val_ds) else None

    if fold_index is not None:
        best_path = fold_ckpt_path(run_dir, fold_index)
    else:
        best_path = run_dir / f"{split_name}_best.pt"

    best_score = _init_best(args.monitor if val_loader else "val_loss")
    best_row: Optional[Dict[str, Any]] = None
    patience = int(args.early_stop_patience)
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            y = batch["label"].to(device)
            out = model(
                event_mean=batch["event_mean"].to(device),
                event_max=batch["event_max"].to(device),
                case_age_hours=batch["case_age_hours"].to(device),
                view_id=batch["view_id"].to(device),
                zone_id=batch["zone_id"].to(device),
                local_hour=batch["local_hour"].to(device),
                padding_mask=batch["padding_mask"].to(device),
                dino_vec=batch["dino_vec"].to(device) if "dino_vec" in batch else None,
                dino_mask=batch["dino_mask"].to(device) if "dino_mask" in batch else None,
            )
            st, ct, sp, cp = _maybe_semantic(batch, device)
            loss_parts = total_finetune_loss(
                out, y, cfg=cfg, state_targets=st, cause_targets=ct, state_present=sp, cause_present=cp
            )
            loss = loss_parts["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
            global_step += 1

        train_eval = evaluate(model, train_loader, device, cfg)
        val_eval = (
            evaluate(model, val_loader, device, cfg)
            if val_loader is not None
            else {
                "loss": float("nan"),
                "acc": float("nan"),
                "macro_f1": float("nan"),
                "ce": float("nan"),
                "state_align": float("nan"),
                "cause_align": float("nan"),
                "n_dino": 0,
            }
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(n_seen, 1),
            "train_ce": train_eval.get("ce"),
            "train_state_align": train_eval.get("state_align"),
            "train_cause_align": train_eval.get("cause_align"),
            "train_acc": train_eval["acc"],
            "train_macro_f1": train_eval["macro_f1"],
            "val_loss": val_eval["loss"],
            "val_ce": val_eval.get("ce"),
            "val_state_align": val_eval.get("state_align"),
            "val_cause_align": val_eval.get("cause_align"),
            "val_acc": val_eval.get("acc"),
            "val_macro_f1": val_eval.get("macro_f1"),
            "n_dino_train": train_eval.get("n_dino"),
            "n_dino_val": val_eval.get("n_dino"),
            "sec": time.time() - t0,
        }
        history.append(row)
        print({split_name: row}, flush=True)

        if args.wandb:
            import wandb

            payload = {
                f"{split_name}/{k}": v
                for k, v in row.items()
                if isinstance(v, (int, float)) and v is not None and np.isfinite(float(v))
            }
            payload["epoch"] = epoch
            payload["global_step"] = global_step
            wandb.log(payload, step=global_step)

        improved = _is_better(args.monitor, row, best_score) if val_loader is not None else (
            float(train_eval["macro_f1"]) > best_score
            if args.monitor == "val_macro_f1"
            else float(train_eval["loss"]) < best_score
        )
        if improved:
            if val_loader is not None:
                best_score = (
                    float(row["val_macro_f1"])
                    if args.monitor == "val_macro_f1"
                    else float(row["val_loss"])
                )
            else:
                best_score = (
                    float(train_eval["macro_f1"])
                    if args.monitor == "val_macro_f1"
                    else float(train_eval["loss"])
                )
            best_row = {
                **row,
                "train_preds": train_eval["preds"],
                "train_labels": train_eval["labels"],
                "val_preds": val_eval.get("preds", []),
                "val_labels": val_eval.get("labels", []),
                "val_case_ids": val_eval.get("case_ids", []),
            }
            torch.save(
                build_v02_ckpt_payload(
                    model=model,
                    cfg=cfg,
                    epoch=epoch,
                    train_cases=list(train_ds.case_ids),
                    val_cases=list(val_ds.case_ids) if val_ds else [],
                    metrics=row,
                ),
                best_path,
            )
            patience = int(args.early_stop_patience)
            if args.wandb:
                import wandb

                best_payload = {
                    f"{split_name}/best_epoch": epoch,
                    f"{split_name}/best_val_macro_f1": row.get("val_macro_f1"),
                    f"{split_name}/best_val_loss": row.get("val_loss"),
                }
                wandb.log(
                    {
                        k: float(v)
                        for k, v in best_payload.items()
                        if isinstance(v, (int, float)) and np.isfinite(float(v))
                    },
                    step=global_step,
                )
        elif (
            not args.no_early_stop
            and val_loader is not None
            and epoch >= int(args.early_stop_min_epochs)
        ):
            patience -= 1
            if patience <= 0:
                print({split_name: "early_stop", "epoch": epoch}, flush=True)
                break

    assert best_row is not None
    val_f1 = best_row.get("val_macro_f1")
    if val_f1 is None or not np.isfinite(float(val_f1)):
        val_f1 = best_row["train_macro_f1"]
    accepted = fold_selection_ok(float(best_row["train_macro_f1"]), float(val_f1))
    (run_dir / f"{split_name}_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "split": split_name,
        "best_path": str(best_path),
        "accepted_fold": accepted,
        "global_step_end": global_step,
        "best": {
            k: v
            for k, v in best_row.items()
            if not str(k).endswith("preds")
            and not str(k).endswith("labels")
            and k != "val_case_ids"
        },
        "history_tail": history[-3:],
        "oof": {
            "case_ids": best_row.get("val_case_ids", []),
            "labels": best_row.get("val_labels", []),
            "preds": best_row.get("val_preds", []),
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    labels_df = pd.read_csv(args.labels)
    label_map = load_label_map(args.label_map)

    bank = None
    bank_path = Path(args.interpretation_bank) if args.interpretation_bank else None
    if bank_path and bank_path.exists():
        bank = load_interpretation_bank(bank_path)
        emb0 = np.asarray(bank.iloc[0]["embedding"])
        args.semantic_dim = int(emb0.shape[0])
    elif not args.allow_missing_bank:
        raise SystemExit(
            f"interpretation bank missing: {bank_path}\n"
            "Run: python scripts/online2_v2/v02/build_interpretation_banks.py\n"
            "Or pass --allow-missing-bank for CE-only."
        )

    cached = list_cached_case_ids(args.emb_dir, labels_df)
    if args.mode == "smoke":
        cached = cached[: min(4, len(cached))]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (
        args.root / DEFAULT_OUTPUT_ROOT / "runs" / f"{args.mode}_{ts}"
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    wb = init_wandb(
        args,
        run_dir,
        config={
            "finetune_version": FINETUNE_VERSION,
            "mode": args.mode,
            "n_cached": len(cached),
            "emb_dir": str(args.emb_dir),
            "interpretation_bank": str(bank_path) if bank is not None else None,
            "semantic_dim": args.semantic_dim,
            "monitor": args.monitor,
            "architecture": "PAD_event_pooling_v02_semantic_align_image",
        },
    )
    args = apply_wandb_config(args)
    set_seed(args.seed)

    meta = {
        "finetune_version": FINETUNE_VERSION,
        "mode": args.mode,
        "n_cached": len(cached),
        "emb_dir": str(args.emb_dir),
        "interpretation_bank": str(bank_path) if bank is not None else None,
        "ckpt_naming": "fold{i}_best.pt",
        "monitor": args.monitor,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        {
            "hyper": {
                "lr": args.lr,
                "dropout": args.dropout,
                "hidden_dim": args.hidden_dim,
                "weight_decay": args.weight_decay,
                "lambda_state": args.lambda_state,
                "lambda_cause": args.lambda_cause,
                "monitor": args.monitor,
            }
        },
        flush=True,
    )

    wandb_watched: list = []
    global_step = 0

    results: List[Dict[str, Any]] = []
    try:
        if args.mode in {"smoke", "overfit"}:
            res = train_one_split(
                train_cases=cached,
                val_cases=cached,
                labels_df=labels_df,
                label_map=label_map,
                bank=bank,
                args=args,
                run_dir=run_dir,
                split_name=args.mode,
                global_step_start=global_step,
                wandb_watched=wandb_watched,
            )
            results.append(res)
            global_step = int(res.get("global_step_end", global_step))
        else:
            folds = make_stratified_farm_folds(labels_df, n_folds=args.n_folds, seed=args.seed)
            oof_cases: List[str] = []
            oof_labels: List[int] = []
            oof_preds: List[int] = []
            for i, fold in enumerate(folds):
                if args.fold is not None and i != int(args.fold):
                    continue
                train_cases = [c for c in fold["train"] if c in set(cached)]
                val_cases = [c for c in fold["val"] if c in set(cached)]
                res = train_one_split(
                    train_cases=train_cases,
                    val_cases=val_cases,
                    labels_df=labels_df,
                    label_map=label_map,
                    bank=bank,
                    args=args,
                    run_dir=run_dir,
                    split_name=f"fold{i}",
                    fold_index=i,
                    global_step_start=global_step,
                    wandb_watched=wandb_watched,
                )
                results.append(res)
                global_step = int(res.get("global_step_end", global_step))
                oof_cases.extend(res["oof"]["case_ids"])
                oof_labels.extend(res["oof"]["labels"])
                oof_preds.extend(res["oof"]["preds"])
                if args.wandb:
                    import wandb

                    wandb.log(
                        {
                            f"fold{i}/accepted": int(bool(res["accepted_fold"])),
                            f"fold{i}/best_train_macro_f1": res["best"].get("train_macro_f1"),
                            f"fold{i}/best_val_macro_f1": res["best"].get("val_macro_f1"),
                            f"fold{i}/best_val_acc": res["best"].get("val_acc"),
                        },
                        step=global_step,
                    )

            val_f1s = [
                float(r["best"]["val_macro_f1"])
                for r in results
                if r.get("best") and np.isfinite(float(r["best"].get("val_macro_f1", float("nan"))))
            ]
            val_accs = [
                float(r["best"]["val_acc"])
                for r in results
                if r.get("best") and np.isfinite(float(r["best"].get("val_acc", float("nan"))))
            ]
            n_val_f1_1 = sum(1 for x in val_f1s if x >= 1.0 - 1e-9)
            cv_summary = {
                "cv/mean_best_val_macro_f1": float(np.mean(val_f1s)) if val_f1s else float("nan"),
                "cv/min_best_val_macro_f1": float(np.min(val_f1s)) if val_f1s else float("nan"),
                "cv/mean_best_val_acc": float(np.mean(val_accs)) if val_accs else float("nan"),
                "cv/n_folds_val_f1_1": n_val_f1_1,
                "cv/n_folds_run": len(results),
                "cv/n_folds_accepted": sum(1 for r in results if r["accepted_fold"]),
            }
            print({"CV_SUMMARY": cv_summary}, flush=True)

            oof_stats = None
            if oof_labels:
                oof_stats = oof_selection_ok(
                    oof_labels, oof_preds, num_classes=len(label_map["labels"])
                )
                pd.DataFrame(
                    {"case_id": oof_cases, "label": oof_labels, "pred": oof_preds}
                ).to_parquet(run_dir / "oof_predictions.parquet", index=False)
                ens_dir = args.root / DEFAULT_OUTPUT_ROOT / "ensemble"
                ens_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {"case_id": oof_cases, "label": oof_labels, "pred": oof_preds}
                ).to_parquet(ens_dir / "oof_predictions_35.parquet", index=False)
                (run_dir / "oof_metrics.json").write_text(
                    json.dumps(oof_stats, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    {
                        "oof": oof_stats,
                        "macro_f1_all": macro_f1_all_classes(oof_labels, oof_preds),
                        "n_folds_accepted": sum(1 for r in results if r["accepted_fold"]),
                    },
                    flush=True,
                )
                cv_summary.update(
                    {
                        "oof/acc": float(oof_stats["oof_accuracy"]),
                        "oof/macro_f1_all": float(oof_stats["oof_macro_f1"]),
                        "oof/gate_ok": int(bool(oof_stats.get("accepted", False))),
                    }
                )

            # v0.1-style: mean-softmax ensemble on all labeled cached cases
            ens = None
            if len(results) >= args.n_folds and args.fold is None:
                try:
                    ens = evaluate_probability_ensemble_all(
                        run_dir=run_dir,
                        case_ids=cached,
                        labels_df=labels_df,
                        label_map=label_map,
                        bank=bank,
                        args=args,
                    )
                    print(
                        {
                            "ensemble_all35": {
                                "acc": ens["acc"],
                                "macro_f1": ens["macro_f1"],
                                "perfect": ens["perfect"],
                                "n": ens["n"],
                            }
                        },
                        flush=True,
                    )
                    cv_summary.update(
                        {
                            "ensemble/acc": float(ens["acc"]),
                            "ensemble/macro_f1": float(ens["macro_f1"]),
                            "ensemble/perfect": int(bool(ens["perfect"])),
                        }
                    )
                except Exception as exc:
                    print({"ensemble_all35_error": str(exc)}, flush=True)

            if args.wandb:
                import wandb

                payload = {k: v for k, v in cv_summary.items() if v is not None and np.isfinite(float(v))}
                wandb.log(payload, step=global_step + 1)
                wandb.summary.update(payload)

        summary = {
            "finetune_version": FINETUNE_VERSION,
            "run_dir": str(run_dir),
            "wandb_url": getattr(wb, "url", None) if wb is not None else None,
            "results": [
                {k: v for k, v in r.items() if k not in {"history", "oof", "history_tail"}}
                for r in results
            ],
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            {
                "done": str(run_dir),
                "finetune_version": FINETUNE_VERSION,
                "wandb_url": summary.get("wandb_url"),
            },
            flush=True,
        )
    finally:
        if wb is not None:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
