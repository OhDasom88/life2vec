#!/usr/bin/env python3
"""Build Gemma multimodal report inputs / stub outputs (finetune v0.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v02.evidence import build_structured_evidence, write_jsonl
from src.online2.v2.finetune_v02.gemma_report import (
    GemmaReportInput,
    generate_report_stub,
    write_gemma_inputs,
)
from src.online2.v2.finetune_v02.version import DEFAULT_OUTPUT_ROOT, FINETUNE_VERSION


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_ROOT / "ensemble" / "problem_predictions.csv",
    )
    p.add_argument(
        "--consensus-root",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_ROOT / "saliency" / "consensus",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / DEFAULT_OUTPUT_ROOT / "reports",
    )
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    if args.limit:
        pred = pred.head(int(args.limit))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    evidences = []
    stub_reports = []
    for _, row in pred.iterrows():
        case_id = str(row["case_id"])
        cons = Path(args.consensus_root) / case_id
        state_path = cons / "event_primary.parquet"
        cause_path = cons / "event_primary.parquet"
        disputed_path = cons / "event_disputed.parquet"
        state_df = pd.read_parquet(state_path) if state_path.exists() else pd.DataFrame()
        cause_df = pd.read_parquet(cause_path) if cause_path.exists() else pd.DataFrame()
        disagreement = []
        if disputed_path.exists():
            ddf = pd.read_parquet(disputed_path)
            disagreement = ddf.head(20).to_dict(orient="records")

        probs = {
            c[2:]: float(row[c])
            for c in pred.columns
            if c.startswith("p_")
        }
        evidence = build_structured_evidence(
            case_id=case_id,
            diagnosis=str(row.get("pred_name", "")),
            ensemble_probability=float(row.get("ensemble_probability", 0.0)),
            model_agreement=int(row.get("model_agreement", 0)),
            state_events=state_df,
            cause_events=cause_df,
            disagreement=disagreement,
        )
        evidences.append(evidence)
        gin = GemmaReportInput(
            case_id=case_id,
            diagnosis=str(row.get("pred_name", "")),
            probabilities=probs,
            model_agreement=int(row.get("model_agreement", 0)),
            structured_evidence=evidence,
        )
        inputs.append(gin)
        stub_reports.append(
            {"case_id": case_id, "report": generate_report_stub(gin), "finetune_version": FINETUNE_VERSION}
        )

    write_gemma_inputs(out_dir / "gemma_inputs.jsonl", inputs)
    write_jsonl(out_dir / "structured_evidence.jsonl", evidences)
    write_jsonl(out_dir / "generated_reports.jsonl", stub_reports)
    print(
        {
            "finetune_version": FINETUNE_VERSION,
            "n": len(inputs),
            "out_dir": str(out_dir),
            "note": "generated_reports.jsonl uses stub until Gemma runtime is connected",
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
