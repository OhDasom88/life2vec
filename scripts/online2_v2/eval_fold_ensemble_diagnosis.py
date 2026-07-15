#!/usr/bin/env python3
"""Ensemble 5-fold best checkpoints and score all labeled cases.

Example:
  python scripts/online2_v2/eval_fold_ensemble_diagnosis.py \\
    --run-dir outputs/online2/v2_finetune/runs/cv_20260713_144719
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.diagnosis_dataset import (  # noqa: E402
    DiagnosisEventDataset,
    collate_diagnosis_batch,
    load_label_map,
)
from src.online2.v2.event_pooling_finetune import (  # noqa: E402
    EventPoolingConfig,
    EventPoolingDiagnosisModel,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/runs/cv_20260713_144719",
    )
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
    p.add_argument("--max-events", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=35)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path (default: <run-dir>/ensemble_all35.json)",
    )
    return p.parse_args()


def load_fold_model(ckpt_path: Path, device: torch.device) -> EventPoolingDiagnosisModel:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = EventPoolingConfig(**blob["cfg"])
    model = EventPoolingDiagnosisModel(cfg)
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_logits(
    model: EventPoolingDiagnosisModel, loader: DataLoader, device: torch.device
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for batch in loader:
        out = model(
            event_mean=batch["event_mean"].to(device),
            event_max=batch["event_max"].to(device),
            case_age_hours=batch["case_age_hours"].to(device),
            view_id=batch["view_id"].to(device),
            zone_id=batch["zone_id"].to(device),
            local_hour=batch["local_hour"].to(device),
            padding_mask=batch["padding_mask"].to(device),
        )
        chunks.append(out["logits"].float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    f1s = []
    for c in sorted(set(y_true)):
        tp = sum(p == c and y == c for p, y in zip(y_pred, y_true))
        fp = sum(p == c and y != c for p, y in zip(y_pred, y_true))
        fn = sum(p != c and y == c for p, y in zip(y_pred, y_true))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    label_map = load_label_map(args.label_map)
    id_to_name = {int(k): v for k, v in label_map["id_to_name"].items()}
    labels_df = pd.read_csv(args.labels)
    case_ids = [str(c) for c in labels_df["case_id"].tolist()]

    ckpts = sorted(run_dir.glob("fold*_best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no fold*_best.pt under {run_dir}")

    ds = DiagnosisEventDataset(
        case_ids,
        labels_df,
        label_map,
        args.emb_dir,
        max_events=args.max_events,
        require_complete=False,
    )
    if len(ds) == 0:
        raise RuntimeError("no cases with Stage A caches")
    loader = DataLoader(
        ds,
        batch_size=min(args.batch_size, len(ds)),
        shuffle=False,
        collate_fn=lambda xs: collate_diagnosis_batch(xs, max_events=args.max_events),
        num_workers=0,
    )
    y_true = [int(ds[i]["label"]) for i in range(len(ds))]
    device = torch.device(args.device)

    fold_logits: List[np.ndarray] = []
    fold_rows: List[Dict[str, Any]] = []
    for ckpt in ckpts:
        model = load_fold_model(ckpt, device)
        logits = predict_logits(model, loader, device)
        preds = logits.argmax(axis=-1).tolist()
        fold_name = ckpt.stem.replace("_best", "")
        row = {
            "fold": fold_name,
            "ckpt": str(ckpt),
            "acc": float(np.mean([p == y for p, y in zip(preds, y_true)])),
            "macro_f1": macro_f1(y_true, preds),
            "n_correct": int(sum(p == y for p, y in zip(preds, y_true))),
            "n": len(y_true),
        }
        fold_rows.append(row)
        fold_logits.append(logits)
        print(
            f"[{fold_name}] acc={row['acc']:.4f} macro_f1={row['macro_f1']:.4f} "
            f"correct={row['n_correct']}/{row['n']}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Softmax probability average across folds
    probs = np.stack([np.exp(x - x.max(axis=-1, keepdims=True)) for x in fold_logits], axis=0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    mean_prob = probs.mean(axis=0)
    ens_preds = mean_prob.argmax(axis=-1).tolist()
    ens_acc = float(np.mean([p == y for p, y in zip(ens_preds, y_true)]))
    ens_f1 = macro_f1(y_true, ens_preds)

    per_case = []
    for i, cid in enumerate(ds.case_ids):
        true_id = y_true[i]
        pred_id = ens_preds[i]
        per_case.append(
            {
                "case_id": cid,
                "label": id_to_name[true_id],
                "pred": id_to_name[pred_id],
                "correct": bool(pred_id == true_id),
                "prob_pred": float(mean_prob[i, pred_id]),
                "prob_label": float(mean_prob[i, true_id]),
                "fold_preds": [
                    id_to_name[int(fold_logits[f][i].argmax())] for f in range(len(fold_logits))
                ],
            }
        )

    errors = [r for r in per_case if not r["correct"]]
    summary = {
        "wandb_run": "7czhwn2y",
        "run_dir": str(run_dir),
        "n_cases": len(ds),
        "n_folds": len(ckpts),
        "ensemble": {
            "method": "mean_softmax_prob",
            "acc": ens_acc,
            "macro_f1": ens_f1,
            "n_correct": int(sum(r["correct"] for r in per_case)),
            "perfect": ens_f1 >= 1.0 - 1e-12 and ens_acc >= 1.0 - 1e-12,
        },
        "per_fold_all35": fold_rows,
        "n_errors": len(errors),
        "errors": errors,
        "per_case": per_case,
    }

    out = args.out or (run_dir / "ensemble_all35.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        {
            "ENSEMBLE": {
                "acc": ens_acc,
                "macro_f1": ens_f1,
                "correct": f"{summary['ensemble']['n_correct']}/{len(ds)}",
                "perfect": summary["ensemble"]["perfect"],
                "n_errors": len(errors),
            },
            "out": str(out),
        },
        flush=True,
    )
    if errors:
        print("errors:", flush=True)
        for e in errors:
            print(
                f"  {e['case_id']}: true={e['label']} pred={e['pred']} "
                f"folds={e['fold_preds']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
