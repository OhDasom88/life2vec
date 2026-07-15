#!/usr/bin/env python3
"""P0 gate demo: fuse-aware event saliency + raw cell join on one case.

Writes under outputs/online2/v2_finetune_v03/p0/ — does not touch v0.1/v0.2.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.checkpoint import load_fold_model_v03  # noqa: E402
from src.online2.v2.finetune_v03.dataset import (  # noqa: E402
    DiagnosisEventDatasetV03,
    collate_diagnosis_batch_v03,
    load_label_map,
    normal_class_id_from_map,
)
from src.online2.v2.finetune_v03.evidence_raw import (  # noqa: E402
    enrich_events_with_raw,
    load_cells_for_farm,
    write_enriched_json,
)
from src.online2.v2.finetune_v03.risk_saliency import (  # noqa: E402
    event_ixg_abnormal_margin,
    risk_from_binary_logit,
    risk_margin_from_logits,
)
from src.online2.v2.finetune_v03.version import DEFAULT_OUTPUT_ROOT, FINETUNE_VERSION  # noqa: E402


def append_progress(msg: Dict[str, Any]) -> None:
    log = ROOT / DEFAULT_OUTPUT_ROOT / "logs" / "PROGRESS.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    msg = dict(msg)
    msg["ts"] = datetime.now(timezone.utc).isoformat()
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--case-id", type=str, default="F420458_2025-02-16_2025-03-01")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="fold*_best.pt from smoke/cv; if omit, randomly init model from batch dims",
    )
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
    p.add_argument(
        "--cells",
        type=Path,
        default=ROOT / "outputs/online2/build-v8-active80-r3/cell_occurrences.parquet",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--top-k", type=int, default=12)
    return p.parse_args()


def _events_frame_from_emb(emb_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(emb_path)
    cols = {}
    for c in ("event_id", "view", "zone", "timestamp", "event_order"):
        if c in df.columns:
            cols[c] = df[c]
    out = pd.DataFrame(cols)
    if "event_order" in out.columns:
        out = out.sort_values("event_order").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.root / DEFAULT_OUTPUT_ROOT / "p0" / f"demo_{args.case_id}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    append_progress({"phase": "p0_start", "case_id": args.case_id, "out_dir": str(out_dir)})

    labels_df = pd.read_csv(args.labels)
    label_map = load_label_map(args.label_map)
    normal_id = normal_class_id_from_map(label_map)
    device = torch.device(args.device)

    ds = DiagnosisEventDatasetV03(
        [args.case_id],
        labels_df=labels_df,
        label_map=label_map,
        emb_dir=args.emb_dir,
        interpretation_bank=None,
        require_complete=False,
        normal_class_id=normal_id,
    )
    if len(ds) == 0:
        raise SystemExit(f"no cache/sample for {args.case_id}")
    sample = ds[0]
    batch = collate_diagnosis_batch_v03([sample], max_events=sample["length"])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    if args.ckpt and Path(args.ckpt).exists():
        model, meta = load_fold_model_v03(Path(args.ckpt), device)
        ckpt_used = str(args.ckpt)
    else:
        from src.online2.v2.finetune_v03.config import FinetuneV03Config
        from src.online2.v2.finetune_v03.model import EventPoolingDiagnosisModelV03

        h = int(sample["hidden_size"])
        cfg = FinetuneV03Config(input_dim=h, num_classes=len(label_map["labels"]))
        model = EventPoolingDiagnosisModelV03(cfg).to(device)
        model.eval()
        meta = {"note": "random_init_no_ckpt"}
        ckpt_used = None

    with torch.no_grad():
        out = model(
            event_mean=batch["event_mean"],
            event_max=batch["event_max"],
            case_age_hours=batch["case_age_hours"],
            view_id=batch["view_id"],
            zone_id=batch["zone_id"],
            local_hour=batch["local_hour"],
            padding_mask=batch["padding_mask"],
            dino_vec=batch.get("dino_vec"),
            dino_mask=batch.get("dino_mask"),
        )
        r_margin = float(risk_margin_from_logits(out["logits"], normal_class_id=normal_id)[0].item())
        r_bin = float(risk_from_binary_logit(out["abnormal_logit"])[0].item())
        pred = int(out["logits"].argmax(dim=-1)[0].item())

    ixg = event_ixg_abnormal_margin(
        model, batch, normal_class_id=normal_id, use_binary=True
    )
    scores = ixg.detach().cpu().numpy()
    order = scores.argsort()[::-1]
    top: List[Dict[str, Any]] = []
    emb = _events_frame_from_emb(Path(args.emb_dir) / f"{args.case_id}.parquet")
    for rank, ti in enumerate(order[: args.top_k]):
        row: Dict[str, Any] = {"rank": rank + 1, "event_idx": int(ti), "ixg": float(scores[ti])}
        if ti < len(emb):
            for c in ("event_id", "view", "zone", "timestamp"):
                if c in emb.columns:
                    row[c] = str(emb.iloc[int(ti)][c])
        top.append(row)

    farm_id = str(args.case_id).split("_")[0]
    raw_pack: List[Dict[str, Any]] = []
    n_raw_quotes = 0
    if Path(args.cells).exists() and len(emb):
        cells = load_cells_for_farm(args.cells, farm_id)
        top_events = emb.iloc[[t["event_idx"] for t in top if t["event_idx"] < len(emb)]].copy()
        raw_pack = enrich_events_with_raw(top_events, cells, max_cells_per_event=16)
        n_raw_quotes = sum(len(r.get("raw_samples") or []) for r in raw_pack)

    evidence = {
        "finetune_version": FINETUNE_VERSION,
        "case_id": args.case_id,
        "ckpt": ckpt_used,
        "meta": meta,
        "pred_class_id": pred,
        "pred_class_name": label_map["labels"][pred],
        "risk_margin": r_margin,
        "risk_binary_p": r_bin,
        "top_events_ixg": top,
        "raw_on_top_events": raw_pack,
        "gate": {
            "n_raw_quotes": n_raw_quotes,
            "n_top_events": len(top),
            "pass_raw": n_raw_quotes > 0,
            "note_token_attr": "token IxG hook pending — event-level fuse path verified here",
        },
    }
    write_enriched_json(out_dir / "evidence_p0.json", evidence)
    (out_dir / "top_events_ixg.json").write_text(
        json.dumps(top, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    append_progress(
        {
            "phase": "p0_done",
            "case_id": args.case_id,
            "n_raw_quotes": n_raw_quotes,
            "pass_raw": n_raw_quotes > 0,
            "out": str(out_dir / "evidence_p0.json"),
        }
    )


if __name__ == "__main__":
    main()
