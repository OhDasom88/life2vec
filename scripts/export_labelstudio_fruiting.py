#!/usr/bin/env python3
"""Export predict_fruiting CSV to Label Studio import JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BUCKET_LABELS = {
    0: "0 (≤3)",
    1: "1 (4)",
    2: "2 (5)",
    3: "3 (6)",
    4: "4 (7)",
    5: "5 (≥8)",
}


def row_to_task(row: pd.Series, exp_id: str) -> dict:
    true_b = int(row["target_bucket"])
    pred_b = int(row["pred_bucket"])
    return {
        "data": {
            "sequence_id": str(row["sequence_id"]),
            "split": str(row["split"]),
            "true_count": float(row["target"]),
            "pred_count": float(row["prediction"]),
            "confidence": float(row["confidence"]),
            "true_bucket": BUCKET_LABELS.get(true_b, str(true_b)),
            "pred_bucket": BUCKET_LABELS.get(pred_b, str(pred_b)),
            "bucket_correct": bool(row["bucket_correct"]),
            "token_preview": str(row.get("input_summary", ""))[:2000],
            "top_tokens": str(row.get("top_tokens", ""))[:500],
            "class_probs": str(row.get("class_probs", "")),
        },
        "meta": {"exp_id": exp_id},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--exp-id", required=True, help="e.g. EXP-001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Include only bucket-incorrect predictions",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    if args.errors_only:
        df = df.loc[~df["bucket_correct"].astype(bool)]

    tasks = [row_to_task(row, args.exp_id) for _, row in df.iterrows()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks → {args.output}")


if __name__ == "__main__":
    main()
