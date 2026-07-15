#!/usr/bin/env python3
"""Hierarchical saliency + 5-fold consensus (finetune v0.2).

Loads fold{i}_best.pt from --run-dir (same convention as train + eval pipeline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v02.checkpoint import load_five_fold_models  # noqa: E402
from src.online2.v2.finetune_v02.config import SaliencyConfig  # noqa: E402
from src.online2.v2.finetune_v02.consensus import (  # noqa: E402
    consensus_table,
    select_report_events,
)
from src.online2.v2.finetune_v02.dataset import (  # noqa: E402
    DiagnosisEventDatasetV02,
    collate_diagnosis_batch_v02,
    load_label_map,
)
from src.online2.v2.finetune_v02.saliency import (  # noqa: E402
    event_input_x_gradient,
    hierarchical_plan,
)
from src.online2.v2.finetune_v02.version import DEFAULT_OUTPUT_ROOT, FINETUNE_VERSION  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--case-id", type=str, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
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
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--target",
        choices=["classification", "state", "cause"],
        default="classification",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (
        args.root / DEFAULT_OUTPUT_ROOT / "saliency" / "consensus" / args.case_id
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels)
    label_map = load_label_map(args.label_map)
    device = torch.device(args.device)
    models, metas, ver = load_five_fold_models(args.run_dir, device)

    ds = DiagnosisEventDatasetV02(
        [args.case_id], labels_df, label_map, args.emb_dir, require_complete=True
    )
    sample = ds[0]
    batch = collate_diagnosis_batch_v02([sample])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    pdf = pd.read_parquet(Path(args.emb_dir) / f"{args.case_id}.parquet")
    pdf = pdf.sort_values("event_order").reset_index(drop=True)
    t = int(batch["padding_mask"][0].sum().item())
    event_ids = pdf["event_id"].astype(str).tolist()[:t]

    fold_scores = []
    for i, model in enumerate(models):
        scores = event_input_x_gradient(model, batch, target=args.target)
        arr = scores.cpu().numpy()[:t]
        fold_scores.append(arr)
        per = args.root / DEFAULT_OUTPUT_ROOT / "saliency" / "per_model" / f"fold{i}"
        per.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"event_id": event_ids, "score": arr}).to_parquet(
            per / f"{args.case_id}_event_scores.parquet", index=False
        )

    cfg = SaliencyConfig()
    table = consensus_table(event_ids, fold_scores, cfg=cfg)
    for col in ("timestamp", "view", "zone", "case_age_hours", "local_hour"):
        if col in pdf.columns:
            table[col] = pdf[col].iloc[:t].tolist()
    table.to_parquet(out_dir / "event_consensus.parquet", index=False)
    splits = select_report_events(table, cfg=cfg)
    for name, df in splits.items():
        df.to_parquet(out_dir / f"event_{name}.parquet", index=False)

    oof = next((m["fold"] for m in metas if args.case_id in set(m["val_cases"])), None)
    plan = hierarchical_plan(cfg, len(event_ids))
    meta = {
        "finetune_version": FINETUNE_VERSION,
        "model_version": ver,
        "case_id": args.case_id,
        "plan": plan,
        "target": args.target,
        "oof_fold": oof,
        "run_dir": str(args.run_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(meta, flush=True)


if __name__ == "__main__":
    main()
