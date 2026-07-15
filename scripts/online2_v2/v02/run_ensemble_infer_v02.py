#!/usr/bin/env python3
"""5-fold probability ensemble inference (finetune v0.2).

Loads <run-dir>/fold{i}_best.pt via shared checkpoint helper (same as eval pipeline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v02.checkpoint import load_five_fold_models  # noqa: E402
from src.online2.v2.finetune_v02.dataset import (  # noqa: E402
    DiagnosisEventDatasetV02,
    collate_diagnosis_batch_v02,
    load_label_map,
)
from src.online2.v2.finetune_v02.metrics import average_probability_ensemble  # noqa: E402
from src.online2.v2.finetune_v02.version import DEFAULT_OUTPUT_ROOT, FINETUNE_VERSION  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing fold0_best.pt … fold4_best.pt",
    )
    p.add_argument(
        "--emb-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/event_embeddings",
    )
    p.add_argument("--case-list", type=Path, required=True)
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
        "--out-dir",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_ROOT / "ensemble",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-events", type=int, default=4096)
    return p.parse_args()


@torch.no_grad()
def predict_proba(
    models: List[torch.nn.Module],
    case_ids: List[str],
    args: argparse.Namespace,
) -> np.ndarray:
    device = torch.device(args.device)
    labels_df = pd.read_csv(args.labels)
    label_map = load_label_map(args.label_map)
    have = set(labels_df["case_id"].astype(str))
    missing = [c for c in case_ids if c not in have]
    if missing:
        raise SystemExit(
            f"Labels missing for {len(missing)} cases (unlabeled problem infer TBD)."
        )

    ds = DiagnosisEventDatasetV02(
        case_ids,
        labels_df,
        label_map,
        args.emb_dir,
        max_events=args.max_events,
        require_complete=True,
    )
    order = []
    fold_probs = [[] for _ in models]
    for i in range(len(ds)):
        sample = ds[i]
        batch = collate_diagnosis_batch_v02([sample], max_events=args.max_events)
        order.append(sample["case_id"])
        for f, model in enumerate(models):
            out = model(
                event_mean=batch["event_mean"].to(device),
                event_max=batch["event_max"].to(device),
                case_age_hours=batch["case_age_hours"].to(device),
                view_id=batch["view_id"].to(device),
                zone_id=batch["zone_id"].to(device),
                local_hour=batch["local_hour"].to(device),
                padding_mask=batch["padding_mask"].to(device),
            )
            fold_probs[f].append(torch.softmax(out["logits"], dim=-1).cpu().numpy()[0])

    idx = {c: i for i, c in enumerate(order)}
    mats = []
    for fprobs in fold_probs:
        mats.append(np.stack([fprobs[idx[c]] for c in case_ids], axis=0))
    return mats  # list of [N,C]


def main() -> None:
    args = parse_args()
    cases = pd.read_csv(args.case_list)["case_id"].astype(str).tolist()
    device = torch.device(args.device)
    models, metas, ver = load_five_fold_models(args.run_dir, device)
    fold_mats = predict_proba(models, cases, args)
    ens = average_probability_ensemble(fold_mats)
    pred = ens.argmax(axis=1)
    top_stack = np.stack([m.argmax(axis=1) for m in fold_mats], axis=0)
    agree = (top_stack == pred).sum(axis=0)

    label_map = load_label_map(args.label_map)
    id_to_name = {int(v): k for k, v in label_map["name_to_id"].items()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, cid in enumerate(cases):
        rows.append(
            {
                "case_id": cid,
                "pred_id": int(pred[i]),
                "pred_name": id_to_name.get(int(pred[i]), str(pred[i])),
                "ensemble_probability": float(ens[i, pred[i]]),
                "model_agreement": int(agree[i]),
                **{f"p_{k}": float(ens[i, k]) for k in range(ens.shape[1])},
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "problem_predictions.parquet", index=False)
    df.to_csv(out_dir / "problem_predictions.csv", index=False)
    meta = {
        "finetune_version": FINETUNE_VERSION,
        "model_version": ver,
        "run_dir": str(args.run_dir),
        "n": len(df),
        "fold_epochs": [m.get("epoch") for m in metas],
    }
    (out_dir / "ensemble_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(meta, flush=True)


if __name__ == "__main__":
    main()
