#!/usr/bin/env python3
"""Finetune v0.3 (PLAN): multitask + logs + repeated-3fold + open-set hooks.

Canonical plan: docs/online2/DIAGNOSIS_FINETUNE_V03_PLAN.md
Isolation: writes only under outputs/online2/v2_finetune_v03/
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
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.checkpoint import (  # noqa: E402
    build_v03_ckpt_payload,
    fold_ckpt_path,
)
from src.online2.v2.finetune_v03.config import (  # noqa: E402
    BINARY_THRESHOLDS_DEFAULT,
    ArchitectureV03,
    FinetuneV03Config,
    LossWeightsV03,
    RoutingThresholds,
)
from src.online2.v2.finetune_v03.dataset import (  # noqa: E402
    DiagnosisEventDatasetV03,
    collate_diagnosis_batch_v03,
    load_label_map,
    make_repeated_group_folds,
    make_stratified_farm_folds,
    normal_class_id_from_map,
)
from src.online2.v2.finetune_v03.losses import (  # noqa: E402
    fine_abnormal_prob,
    total_finetune_loss_v03,
)
from src.online2.v2.finetune_v03.metrics import (  # noqa: E402
    binary_metrics,
    fine_metrics,
    head_consistency_metrics,
    score_distribution_stats,
)
from src.online2.v2.finetune_v03.model import EventPoolingDiagnosisModelV03  # noqa: E402
from src.online2.v2.finetune_v03.open_set import energy_from_logits, route_case  # noqa: E402
from src.online2.v2.finetune_v03.pos_weight import (  # noqa: E402
    parse_pos_weight_arg,
    resolve_pos_weight,
)
from src.online2.v2.finetune_v03.sample_weights import (  # noqa: E402
    class_balanced_sample_weights,
)
from src.online2.v2.finetune_v03.version import DEFAULT_OUTPUT_ROOT, FINETUNE_VERSION  # noqa: E402


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--emb-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune_v02/event_embeddings",
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
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument(
        "--mode",
        choices=["smoke", "cv", "cv_repeated", "cv_final5"],
        default="cv_repeated",
    )
    p.add_argument("--n-folds", "--n_folds", type=int, default=3, help="tuning default 3; final5 uses 5")
    p.add_argument("--n-repeats", "--n_repeats", type=int, default=3)
    p.add_argument("--cv-seeds", "--cv_seeds", type=str, default="2023,2024,2025")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch-size", "--batch_size", type=int, default=8)
    p.add_argument("--max-events", "--max_events", type=int, default=4096)
    p.add_argument("--hidden-dim", "--hidden_dim", type=int, default=192)
    p.add_argument("--proj-dim", "--proj_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num-heads", "--num_heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--lambda-fine", "--lambda_fine", type=float, default=1.0)
    p.add_argument("--lambda-binary", "--lambda_binary", type=float, default=1.0)
    p.add_argument("--lambda-consistency", "--lambda_consistency", type=float, default=0.05)
    p.add_argument("--lambda-supcon", "--lambda_supcon", type=float, default=0.0)
    p.add_argument("--lambda-prototype", "--lambda_prototype", type=float, default=0.05)
    p.add_argument("--supcon-temperature", "--supcon_temperature", type=float, default=0.1)
    p.add_argument("--consistency-mode", "--consistency_mode", choices=["stopgrad_bce", "js"], default="stopgrad_bce")
    p.add_argument(
        "--pos-weight",
        "--pos_weight",
        type=str,
        default="auto",
        help="auto (=N_normal/N_abnormal on train split) or absolute float",
    )
    p.add_argument(
        "--weighted-sampler",
        "--weighted_sampler",
        type=str2bool,
        default=True,
        help="class-balanced WeightedRandomSampler on the fine label for train batches "
        "(mirrors original life2vec CLSDataModule.get_train_weights(), replacement=True). "
        "Independent of --pos-weight (loss reweighting) -- both act on train only, "
        "val/test are never resampled.",
    )
    p.add_argument("--binary-threshold", "--binary_threshold", type=float, default=0.5)
    p.add_argument(
        "--binary-thresholds",
        "--binary_thresholds",
        type=str,
        default="0.25,0.5,0.75",
        help="eval threshold grid (comma)",
    )

    p.add_argument("--task-specific-query", "--task_specific_query", type=str2bool, default=True)
    p.add_argument("--attention-residual", "--attention_residual", type=str2bool, default=True)
    p.add_argument("--token-attention-pool", "--token_attention_pool", type=str2bool, default=False)
    p.add_argument("--use-image-adapter", "--use_image_adapter", type=str2bool, default=True)

    p.add_argument("--use-reject", "--use_reject", type=str2bool, default=True)
    p.add_argument("--use-open-set", "--use_open_set", type=str2bool, default=True)
    p.add_argument("--tau-fine", "--tau_fine", type=float, default=0.5)
    p.add_argument("--min-fold-agreement", "--min_fold_agreement", type=int, default=4)

    p.add_argument("--early-stop-patience", "--early_stop_patience", type=int, default=20)
    p.add_argument("--early-stop-min-epochs", "--early_stop_min_epochs", type=int, default=30)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument(
        "--monitor",
        choices=["val_loss", "val_macro_f1", "val_bin_auprc", "val_bin_balanced_acc"],
        default="val_loss",
    )
    p.add_argument("--require-complete", action="store_true")

    p.add_argument("--wandb", type=str2bool, default=False)
    p.add_argument("--wandb-project", "--wandb_project", type=str, default="Berry2Vec")
    p.add_argument("--wandb-entity", "--wandb_entity", type=str, default="dasom-oh")
    p.add_argument("--wandb-run-name", "--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb-group", "--wandb_group", type=str, default=None)
    p.add_argument("--wandb-job-type", "--wandb_job_type", type=str, default=None)
    p.add_argument(
        "--wandb-tags",
        "--wandb_tags",
        type=str,
        default="v03,multitask,repeated3fold,open-set,attention-pooling",
    )
    p.add_argument("--wandb-watch", "--wandb_watch", type=str2bool, default=None)
    p.add_argument("--wandb-watch-log", "--wandb_watch_log", type=str, default="gradients")
    p.add_argument("--wandb-watch-freq", "--wandb_watch_freq", type=int, default=50)
    args = p.parse_args()
    if args.wandb_watch is None:
        args.wandb_watch = bool(args.wandb)
    args.pos_weight = parse_pos_weight_arg(args.pos_weight)
    args.binary_thresholds = [
        float(x) for x in str(args.binary_thresholds).split(",") if x.strip()
    ] or list(BINARY_THRESHOLDS_DEFAULT)
    args.cv_seeds = [int(x) for x in str(args.cv_seeds).split(",") if x.strip()]
    return args


def append_progress(msg: Dict[str, Any]) -> None:
    log = ROOT / DEFAULT_OUTPUT_ROOT / "logs" / "PROGRESS.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    msg = dict(msg)
    msg["ts"] = datetime.now(timezone.utc).isoformat()
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
    print(msg, flush=True)


def _finite_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and v is not None and np.isfinite(float(v))


def init_wandb(args: argparse.Namespace, run_dir: Path, config: Dict[str, Any]):
    if not args.wandb:
        return None
    import wandb

    if not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        raise SystemExit("wandb login missing")
    tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
    group = args.wandb_group or os.environ.get("WANDB_GROUP")
    job_type = args.wandb_job_type or os.environ.get("WANDB_JOB_TYPE")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or run_dir.name,
        tags=tags,
        group=group,
        job_type=job_type,
        config=config,
        dir=str(run_dir),
        reinit=True,
    )
    info = {"wandb_url": getattr(run, "url", None), "wandb_id": run.id, "group": group, "job_type": job_type}
    print(info, flush=True)
    append_progress({"phase": "wandb_init", **info})
    return run


def apply_wandb_config(args: argparse.Namespace) -> argparse.Namespace:
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
        "proj_dim": "proj_dim",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "seed": "seed",
        "num_heads": "num_heads",
        "lambda_binary": "lambda_binary",
        "lambda_fine": "lambda_fine",
        "lambda_supcon": "lambda_supcon",
        "lambda_consistency": "lambda_consistency",
        "lambda_prototype": "lambda_prototype",
        "supcon_temperature": "supcon_temperature",
        "pos_weight": "pos_weight",
        "weighted_sampler": "weighted_sampler",
        "binary_threshold": "binary_threshold",
        "monitor": "monitor",
        "n_folds": "n_folds",
        "task_specific_query": "task_specific_query",
        "attention_residual": "attention_residual",
        "token_attention_pool": "token_attention_pool",
        "use_reject": "use_reject",
        "use_open_set": "use_open_set",
        "tau_fine": "tau_fine",
    }
    applied = {}
    for src, dst in mapping.items():
        if src not in cfg or cfg[src] is None:
            continue
        val = cfg[src]
        if dst == "pos_weight":
            args.pos_weight = parse_pos_weight_arg(val)
        elif dst == "monitor":
            m = str(val)
            if m in {"val_loss", "val_macro_f1", "val_bin_auprc", "val_bin_balanced_acc"}:
                args.monitor = m
        elif isinstance(getattr(args, dst, None), bool):
            setattr(args, dst, bool(val))
        elif isinstance(getattr(args, dst, None), int) and not isinstance(getattr(args, dst), bool):
            setattr(args, dst, int(val))
        elif isinstance(getattr(args, dst, None), float):
            setattr(args, dst, float(val))
        else:
            setattr(args, dst, val)
        applied[dst] = getattr(args, dst)
    if applied:
        print({"wandb_config_applied": applied}, flush=True)
    return args


def build_model_cfg(args: argparse.Namespace, input_dim: int, n_classes: int) -> FinetuneV03Config:
    return FinetuneV03Config(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ff_dim=args.hidden_dim * 2,
        dropout=args.dropout,
        num_classes=n_classes,
        proj_dim=args.proj_dim,
        max_events=args.max_events,
        pos_weight=args.pos_weight,
        binary_threshold=float(args.binary_threshold),
        binary_thresholds=list(args.binary_thresholds),
        supcon_temperature=float(args.supcon_temperature),
        consistency_mode=str(args.consistency_mode),
        monitor=str(args.monitor),
        loss=LossWeightsV03(
            fine_ce=float(args.lambda_fine),
            binary_bce=float(args.lambda_binary),
            consistency=float(args.lambda_consistency),
            proj_supcon=float(args.lambda_supcon),
            prototype=float(args.lambda_prototype),
        ),
        arch=ArchitectureV03(
            task_specific_query=bool(args.task_specific_query),
            attention_residual=bool(args.attention_residual),
            token_attention_pool=bool(args.token_attention_pool),
            use_image_adapter=bool(args.use_image_adapter),
        ),
        routing=RoutingThresholds(
            tau_binary=float(args.binary_threshold),
            tau_fine=float(args.tau_fine),
            min_fold_agreement=int(args.min_fold_agreement),
        ),
    )


@torch.no_grad()
def eval_and_analyze(
    model,
    loader,
    device,
    *,
    normal_class_id: int,
    n_classes: int,
    cfg: FinetuneV03Config,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.eval()
    ys, preds, ybin, scores, p_fine_abn, fine_is_norm, logits_all = [], [], [], [], [], [], []
    case_rows = []
    loss_sum, n = 0.0, 0
    for batch in loader:
        y = batch["label"].to(device)
        y_abn = batch["y_abnormal"].to(device)
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
        out["y_abnormal"] = y_abn
        parts = total_finetune_loss_v03(
            out, y, cfg=cfg, binary_pos_weight=None, normal_class_id=normal_class_id
        )
        bs = int(y.size(0))
        loss_sum += float(parts["total"].item()) * bs
        n += bs
        prob = torch.softmax(out["logits"], dim=-1)
        pred = out["logits"].argmax(dim=-1)
        p_abn = torch.sigmoid(out["abnormal_logit"])
        p_fa = fine_abnormal_prob(out["logits"], normal_class_id)
        for i in range(bs):
            ys.append(int(y[i].item()))
            preds.append(int(pred[i].item()))
            ybin.append(float(y_abn[i].item()))
            scores.append(float(p_abn[i].item()))
            p_fine_abn.append(float(p_fa[i].item()))
            fine_is_norm.append(bool(pred[i].item() == normal_class_id))
            logits_all.append(out["logits"][i].detach().cpu().numpy())
            cid = batch["case_ids"][i]
            pf = prob[i].detach().cpu().numpy()
            route = route_case(
                p_abnormal=float(p_abn[i].item()),
                fine_probs=pf,
                normal_class_id=normal_class_id,
                energy=energy_from_logits(logits_all[-1]),
                head_contradiction=(fine_is_norm[-1] != (scores[-1] < args.binary_threshold)),
                use_reject=bool(args.use_reject),
                use_open_set=bool(args.use_open_set),
                thr=cfg.routing,
            )
            case_rows.append(
                {
                    "case_id": cid,
                    "y": int(y[i].item()),
                    "y_abnormal": float(y_abn[i].item()),
                    "fine_pred": int(pred[i].item()),
                    "fine_normal_prob": float(pf[normal_class_id]),
                    "binary_prob": float(p_abn[i].item()),
                    "fine_abnormal_prob": float(p_fa[i].item()),
                    "probability_gap": abs(float(p_abn[i].item()) - float(p_fa[i].item())),
                    "route": route.decision,
                    "route_reasons": ",".join(route.reasons),
                    "z_proj": out["z_proj"][i].detach().cpu().numpy(),
                    "h_binary": out["h_binary"][i].detach().cpu().numpy(),
                }
            )

    fine = fine_metrics(ys, preds, n_classes)
    binm = binary_metrics(
        ybin, scores, threshold=args.binary_threshold, thresholds=args.binary_thresholds
    )
    cons = head_consistency_metrics(
        y_abnormal=ybin,
        p_bin=scores,
        p_fine_abn=p_fine_abn,
        fine_pred_is_normal=fine_is_norm,
        tau_binary=args.binary_threshold,
    )
    dist = score_distribution_stats(
        ybin, scores, fine_correct=[a == b for a, b in zip(ys, preds)]
    )
    return {
        "loss": loss_sum / max(n, 1),
        "n": n,
        **{f"fine_{k}": v for k, v in fine.items()},
        **{f"bin_{k}": v for k, v in binm.items() if k not in ("threshold_table", "reliability_bins")},
        "bin_threshold_table": binm.get("threshold_table"),
        "bin_reliability_bins": binm.get("reliability_bins"),
        **{f"cons_{k}": v for k, v in cons.items()},
        **{f"dist_{k}": v for k, v in dist.items()},
        "case_rows": case_rows,
        "image_stats": getattr(model, "_last_image_stats", {}),
    }


def train_one_split(
    *,
    train_cases: List[str],
    val_cases: List[str],
    labels_df: pd.DataFrame,
    label_map: Dict[str, Any],
    normal_class_id: int,
    args: argparse.Namespace,
    run_dir: Path,
    split_name: str,
    fold_index: Optional[int] = None,
    repeat_index: Optional[int] = None,
    global_step: Optional[List[int]] = None,
) -> Dict[str, Any]:
    device = torch.device(args.device)
    step_box = global_step if global_step is not None else [0]
    common = dict(
        labels_df=labels_df,
        label_map=label_map,
        emb_dir=args.emb_dir,
        interpretation_bank=None,
        semantic_dim=192,
        max_events=args.max_events,
        require_complete=bool(args.require_complete),
        normal_class_id=normal_class_id,
    )
    train_ds = DiagnosisEventDatasetV03(train_cases, **common)
    if len(train_ds) == 0:
        raise RuntimeError(f"{split_name}: no train caches")
    val_ds = DiagnosisEventDatasetV03(val_cases, **common) if val_cases else None
    h = int(train_ds[0]["hidden_size"])
    cfg = build_model_cfg(args, h, len(label_map["labels"]))
    model = EventPoolingDiagnosisModelV03(cfg, use_image_adapter=bool(args.use_image_adapter)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_w = resolve_pos_weight(
        labels_df.loc[labels_df["case_id"].isin(train_cases)],
        label_map,
        normal_class_id,
        args.pos_weight,
    ).to(device)
    effective_pw = float(pos_w.item())

    def make_loader(ds, *, shuffle: bool = False, sampler=None) -> DataLoader:
        bs = max(1, min(int(args.batch_size), len(ds)))
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=lambda xs: collate_diagnosis_batch_v03(xs, max_events=args.max_events),
            num_workers=0,
        )

    if args.weighted_sampler:
        sample_weights = class_balanced_sample_weights(train_ds.case_ids, labels_df)
        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = make_loader(train_ds, sampler=train_sampler)
    else:
        train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False) if val_ds and len(val_ds) else None
    best_path = fold_ckpt_path(run_dir, fold_index or 0, repeat=repeat_index)

    best_score = 1e18 if args.monitor == "val_loss" else -1e18
    best_row: Optional[Dict[str, Any]] = None
    patience_left = int(args.early_stop_patience)
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_seen = 0.0, 0
        for batch in train_loader:
            y = batch["label"].to(device)
            y_abn = batch["y_abnormal"].to(device)
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
            out["y_abnormal"] = y_abn
            parts = total_finetune_loss_v03(
                out, y, cfg=cfg, binary_pos_weight=pos_w, normal_class_id=normal_class_id
            )
            opt.zero_grad(set_to_none=True)
            parts["total"].backward()
            opt.step()
            bs = int(y.size(0))
            running += float(parts["total"].item()) * bs
            n_seen += bs
            step_box[0] += 1
        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running / max(n_seen, 1),
            "sec": time.time() - t0,
            "split": split_name,
            "binary_pos_weight_effective": effective_pw,
        }
        improved = False
        per_class_f1 = confmat = reliability = None
        if val_loader is not None:
            ev = eval_and_analyze(
                model,
                val_loader,
                device,
                normal_class_id=normal_class_id,
                n_classes=cfg.num_classes,
                cfg=cfg,
                args=args,
            )
            # drop heavy case_rows from epoch history
            case_rows = ev.pop("case_rows", [])
            thr_table = ev.pop("bin_threshold_table", None)
            per_class_f1 = ev.pop("fine_per_class_f1", None)
            confmat = ev.pop("fine_confusion_matrix", None)
            reliability = ev.pop("bin_reliability_bins", None)
            row.update({f"val_{k}": v for k, v in ev.items() if not isinstance(v, (dict, list))})
            row["val_bin_threshold_table"] = thr_table
            row["val_fine_per_class_f1"] = per_class_f1
            row["val_fine_confusion_matrix"] = confmat
            row["val_bin_reliability_bins"] = reliability

            if args.monitor == "val_loss":
                score = float(ev["loss"])
                better = score < best_score
            elif args.monitor == "val_macro_f1":
                score = float(ev["fine_macro_f1"])
                better = score > best_score
            elif args.monitor == "val_bin_auprc":
                score = float(ev.get("bin_auprc_abnormal", ev.get("bin_auprc", float("nan"))))
                better = score > best_score if np.isfinite(score) else False
            else:
                score = float(ev.get("bin_balanced_acc", float("nan")))
                better = score > best_score if np.isfinite(score) else False

            if better:
                best_score = score if np.isfinite(score) else best_score
                best_row = {**row, "case_rows": case_rows}
                improved = True
                torch.save(
                    build_v03_ckpt_payload(
                        model=model,
                        cfg=cfg,
                        epoch=epoch,
                        train_cases=train_cases,
                        val_cases=val_cases,
                        metrics={k: v for k, v in row.items() if k != "val_bin_threshold_table"},
                        normal_class_id=normal_class_id,
                        extra={"binary_pos_weight_effective": effective_pw, "repeat": repeat_index},
                    ),
                    best_path,
                )
                patience_left = int(args.early_stop_patience)
            else:
                patience_left -= 1

        history.append({k: v for k, v in row.items() if k != "case_rows"})
        if args.wandb:
            import wandb

            payload = {
                f"{split_name}/{k}": float(v)
                for k, v in row.items()
                if k not in {"split", "val_bin_threshold_table", "case_rows"} and _finite_num(v)
            }
            payload["epoch"] = epoch
            payload["global_step"] = int(step_box[0])
            if fold_index is not None:
                payload["fold"] = int(fold_index)
            if repeat_index is not None:
                payload["repeat"] = int(repeat_index)
            if improved:
                payload[f"{split_name}/best_epoch"] = float(epoch)
            if per_class_f1 is not None:
                label_names = label_map.get("labels", [])
                for cls_id, f1 in enumerate(per_class_f1):
                    if not _finite_num(f1):
                        continue
                    name = label_names[cls_id] if cls_id < len(label_names) else str(cls_id)
                    payload[f"{split_name}/val_fine_f1__{name}"] = float(f1)
            if improved and confmat is not None:
                label_names = label_map.get("labels", [])
                cm_table = wandb.Table(columns=["true_label"] + list(label_names))
                for i, true_name in enumerate(label_names):
                    cm_table.add_data(true_name, *[int(x) for x in confmat[i]])
                payload[f"{split_name}/val_fine_confusion_matrix"] = cm_table
            if improved and reliability is not None:
                rel_table = wandb.Table(columns=["bin_lo", "bin_hi", "mean_pred", "mean_true", "count"])
                for b in reliability:
                    rel_table.add_data(b["bin_lo"], b["bin_hi"], b["mean_pred"], b["mean_true"], b["count"])
                payload[f"{split_name}/val_bin_reliability"] = rel_table
            wandb.log(payload, step=int(step_box[0]))

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            append_progress({"phase": "train", "fold": fold_index, "repeat": repeat_index, **{
                k: v for k, v in row.items() if k not in {"val_bin_threshold_table"}
            }})
        if (
            not args.no_early_stop
            and val_loader is not None
            and epoch >= int(args.early_stop_min_epochs)
            and patience_left <= 0
        ):
            append_progress({"phase": "early_stop", "fold": fold_index, "repeat": repeat_index, "epoch": epoch})
            break

    if best_row is None:
        torch.save(
            build_v03_ckpt_payload(
                model=model,
                cfg=cfg,
                epoch=args.epochs,
                train_cases=train_cases,
                val_cases=val_cases,
                metrics=history[-1] if history else {},
                normal_class_id=normal_class_id,
            ),
            best_path,
        )
        best_row = history[-1] if history else {}

    # write analysis artifacts for best val cases
    if val_loader is not None and isinstance(best_row, dict) and best_row.get("case_rows"):
        rows = best_row["case_rows"]
        pdf = pd.DataFrame([{k: v for k, v in r.items() if k not in ("z_proj", "h_binary")} for r in rows])
        tag = split_name
        pdf.to_csv(run_dir / f"{tag}_oof_head_consistency.csv", index=False)
        thr = best_row.get("val_bin_threshold_table")
        if thr:
            pd.DataFrame(thr).to_csv(run_dir / f"{tag}_oof_binary_thresholds.csv", index=False)
        # persist OOF representations (concept-space probing input; previously computed but discarded)
        np.savez(
            run_dir / f"{tag}_oof_representations.npz",
            case_id=np.array([r["case_id"] for r in rows]),
            y_abnormal=np.array([r["y_abnormal"] for r in rows], dtype=np.float64),
            z_proj=np.stack([r["z_proj"] for r in rows]),
            h_binary=np.stack([r["h_binary"] for r in rows]),
        )

    hist_path = run_dir / f"{split_name}_history.json"
    hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str))
    return {
        "split": split_name,
        "best_path": str(best_path),
        "best": {k: v for k, v in (best_row or {}).items() if k != "case_rows"},
        "normal_class_id": normal_class_id,
        "binary_pos_weight_effective": effective_pw,
        "finetune_version": FINETUNE_VERSION,
    }


def main() -> None:
    args = parse_args()
    # Under wandb agent, always enable logging even if flag omitted.
    if os.environ.get("WANDB_SWEEP_ID") or os.environ.get("WANDB_RUN_ID"):
        args.wandb = True
    if args.mode == "cv_final5":
        args.n_folds = 5
    elif args.mode in {"cv_repeated", "cv"} and args.mode == "cv_repeated":
        args.n_folds = int(args.n_folds) if args.n_folds else 3

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    labels_df = pd.read_csv(args.labels)
    label_map = load_label_map(args.label_map)
    normal_id = normal_class_id_from_map(label_map)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (args.root / DEFAULT_OUTPUT_ROOT / "runs" / f"{args.mode}_{stamp}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "finetune_version": FINETUNE_VERSION,
        "mode": args.mode,
        "plan": "DIAGNOSIS_FINETUNE_V03_PLAN.md",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False))
    append_progress({"phase": "start", "run_dir": str(run_dir), "mode": args.mode})

    wb = init_wandb(args, run_dir, config=meta)
    args = apply_wandb_config(args)
    global_step = [0]

    avail = [c for c in labels_df["case_id"].astype(str) if (Path(args.emb_dir) / f"{c}.parquet").exists()]
    labels_df = labels_df[labels_df["case_id"].astype(str).isin(avail)].copy()
    append_progress({"phase": "data", "n_cached": len(avail)})

    results = []
    try:
        if args.mode == "smoke":
            ids = list(avail)
            train_c, val_c = ids[:-4], ids[-4:]
            results.append(
                train_one_split(
                    train_cases=train_c,
                    val_cases=val_c,
                    labels_df=labels_df,
                    label_map=label_map,
                    normal_class_id=normal_id,
                    args=args,
                    run_dir=run_dir,
                    split_name="smoke",
                    fold_index=0,
                    global_step=global_step,
                )
            )
        elif args.mode == "cv_repeated":
            splits = make_repeated_group_folds(
                labels_df, n_folds=int(args.n_folds), seeds=args.cv_seeds, n_repeats=int(args.n_repeats)
            )
            (run_dir / "split_manifest.json").write_text(
                json.dumps(splits, indent=2, ensure_ascii=False)
            )
            for sp in splits:
                if args.fold is not None and int(sp["fold"]) != int(args.fold):
                    continue
                result_path = run_dir / f"{sp['split_id']}_result.json"
                if result_path.exists():
                    try:
                        prev = json.loads(result_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        raise RuntimeError(f"failed to load existing result {result_path}: {e}") from e
                    results.append(prev)
                    append_progress(
                        {
                            "phase": "fold_skip",
                            "repeat": sp["repeat"],
                            "fold": sp["fold"],
                            "split_id": sp["split_id"],
                            "reason": "result_exists",
                        }
                    )
                    continue
                append_progress(
                    {
                        "phase": "fold_start",
                        "repeat": sp["repeat"],
                        "fold": sp["fold"],
                        "n_train": len(sp["train"]),
                        "n_val": len(sp["val"]),
                    }
                )
                res = train_one_split(
                    train_cases=sp["train"],
                    val_cases=sp["val"],
                    labels_df=labels_df,
                    label_map=label_map,
                    normal_class_id=normal_id,
                    args=args,
                    run_dir=run_dir,
                    split_name=sp["split_id"],
                    fold_index=int(sp["fold"]),
                    repeat_index=int(sp["repeat"]),
                    global_step=global_step,
                )
                results.append(res)
                result_path.write_text(
                    json.dumps(res, indent=2, ensure_ascii=False, default=str)
                )
        else:
            # cv or cv_final5
            folds = make_stratified_farm_folds(labels_df, n_folds=int(args.n_folds), seed=args.seed)
            for i, fold in enumerate(folds):
                if args.fold is not None and i != int(args.fold):
                    continue
                res = train_one_split(
                    train_cases=fold["train"],
                    val_cases=fold["val"],
                    labels_df=labels_df,
                    label_map=label_map,
                    normal_class_id=normal_id,
                    args=args,
                    run_dir=run_dir,
                    split_name=f"fold{i}",
                    fold_index=i,
                    global_step=global_step,
                )
                results.append(res)

        summary = {
            "finetune_version": FINETUNE_VERSION,
            "run_dir": str(run_dir),
            "wandb_url": getattr(wb, "url", None) if wb else None,
            "results": results,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        if args.wandb and results:
            import wandb

            f1s = [
                r["best"].get("val_fine_macro_f1")
                for r in results
                if r.get("best") and _finite_num(r["best"].get("val_fine_macro_f1"))
            ]
            if f1s:
                wandb.summary.update(
                    {
                        "cv/mean_best_macro_f1": float(np.mean(f1s)),
                        "cv/std_best_macro_f1": float(np.std(f1s)),
                        "cv/n_splits": len(f1s),
                    }
                )
        append_progress({"phase": "done", "run_dir": str(run_dir), "n_results": len(results)})
    finally:
        if wb is not None:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
