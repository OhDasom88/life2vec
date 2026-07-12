#!/usr/bin/env python3
"""Inspect hackathon pretrain runs and refresh finetune sweep YAML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.wandb_pretrain import (  # noqa: E402
    HACKATHON_IMPLEMENTATION,
    HACKATHON_PRETRAIN_SWEEP_ID,
    list_pretrain_sweep_runs,
)

SWEEP_YAML = PROJECT_ROOT / "conf" / "sweep" / "hackathon_finetune.yaml"


def _print_runs(runs: list[dict]) -> None:
    print(f"Hackathon pretrain sweep: {HACKATHON_PRETRAIN_SWEEP_ID}")
    print(f"{'run_id':<12} {'score':>8}  {'weight':<6}  pretrain config")
    print("-" * 72)
    for row in runs:
        exists = "OK" if row["weight_exists"] else "MISSING"
        cfg = ", ".join(f"{k}={v}" for k, v in row["config"].items())
        print(f"{row['run_id']:<12} {row['score']:8.4f}  {exists:<6}  {cfg}")


def _write_sweep_yaml(run_ids: list[str]) -> None:
    with SWEEP_YAML.open() as f:
        doc = yaml.safe_load(f)

    values = ["none"] + run_ids
    doc["parameters"]["pretrained_run_id"]["values"] = values
    header = (
        "# W&B sweep: Hackathon binary CLS finetune.\n"
        "# Auto-updated by scripts/generate_hackathon_sweep_yaml.py\n"
        "#\n"
        "# Usage:\n"
        "#   source setup.sh\n"
        "#   wandb sweep conf/sweep/hackathon_finetune.yaml\n"
        "#   wandb agent dasom-oh/Berry2Vec/<sweep_id>\n\n"
    )
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    SWEEP_YAML.write_text(header + body)
    print(f"Updated {SWEEP_YAML} with pretrained_run_id values: {values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="dasom-oh")
    parser.add_argument("--project", default="Berry2Vec")
    parser.add_argument("--sweep-id", default=HACKATHON_PRETRAIN_SWEEP_ID)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--include-missing-weights",
        action="store_true",
        help="Include runs even if ~/weights/hackathon/.../sweep_<id>.pth is absent",
    )
    parser.add_argument("--list", action="store_true", help="Print pretrain runs and exit")
    parser.add_argument("--write", action="store_true", help="Write run ids into sweep yaml")
    args = parser.parse_args()

    runs = list_pretrain_sweep_runs(
        entity=args.entity,
        project=args.project,
        sweep_id=args.sweep_id,
        implementation=HACKATHON_IMPLEMENTATION,
        top_k=None,
        finished_only=True,
    )
    if not args.include_missing_weights:
        runs = [r for r in runs if r["weight_exists"]]
    runs = runs[: args.top_k]

    if not runs and not args.write:
        print("No finished hackathon pretrain runs with local weights found.")
        print("Finetune sweep still supports pretrained_run_id=none for scratch training.")
        if args.list:
            return
        raise SystemExit(1)

    if args.list or not args.write:
        _print_runs(runs)
    if args.write:
        _write_sweep_yaml([r["run_id"] for r in runs])


if __name__ == "__main__":
    main()
