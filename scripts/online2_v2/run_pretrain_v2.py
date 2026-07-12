#!/usr/bin/env python3
"""Launch Online2 V2 pretraining with OOM auto-recovery and resume support."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run_once(env: dict, extra_args: list[str]) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"experiment={env.get('ONLINE2_V2_EXPERIMENT', 'pretrain_online2_v2_smoke')}",
        *extra_args,
    ]
    print("RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "outputs/online2/v2_build")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/online2/v2_runs")
    parser.add_argument("--experiment", default="pretrain_online2_v2_smoke")
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--max-oom-retries", type=int, default=4)
    args = parser.parse_args()

    build = args.build_dir.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    parquet = build / "training_events_v2.parquet"
    registry = build / "life2vec_token_registry_v2.json"
    vocab = build / "vocab_v2.json"
    assert parquet.exists(), parquet
    assert registry.exists(), registry

    vocab_size = len(json.loads(vocab.read_text())["tokens"])
    batch_size = int(os.environ.get("ONLINE2_V2_BATCH_SIZE", "2"))
    grad_accum = int(os.environ.get("ONLINE2_V2_GRAD_ACCUM", "1"))

    base_env = os.environ.copy()
    base_env.update(
        {
            "ONLINE2_V2_EXPERIMENT": args.experiment,
            "ONLINE2_V2_BUILD": str(build),
            "ONLINE2_V2_RUN_DIR": str(run_dir),
            "ONLINE2_V2_VOCAB_SIZE": str(vocab_size),
            "ONLINE2_V2_VOCAB_PATH": str(vocab),
            "ONLINE2_PARQUET_PATH": str(parquet),
            "ONLINE2_BUILD_ID": "v2_transductive",
            "ONLINE2_REGISTRY_VERSION": "v2",
            "ONLINE2_TOKEN_REGISTRY_PATH": str(registry),
            "ONLINE2_REFERENCE_DATE": "2024-01-01",
            "ONLINE2_THRESHOLD": "2024-06-01",
            "ONLINE2_V2_BATCH_SIZE": str(batch_size),
            "ONLINE2_V2_GRAD_ACCUM": str(grad_accum),
        }
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_mode": "transductive_public_pretraining",
        "contains_problem_observations": True,
        "contains_problem_images": True,
        "contains_problem_hidden_targets": False,
        "build_dir": str(build),
        "run_dir": str(run_dir),
        "experiment": args.experiment,
        "attempts": [],
        "status": "running",
    }
    write_manifest(run_dir / "run_manifest_v2.json", manifest)

    extra = []
    if args.ckpt_path:
        extra.append(f"ckpt_path={args.ckpt_path}")

    rc = 1
    for attempt in range(1, args.max_oom_retries + 1):
        env = base_env.copy()
        env["ONLINE2_V2_BATCH_SIZE"] = str(batch_size)
        env["ONLINE2_V2_GRAD_ACCUM"] = str(grad_accum)
        # Hydra overrides for live batch size
        overrides = extra + [
            f"datamodule.batch_size={batch_size}",
            f"trainer.accumulate_grad_batches={grad_accum}",
        ]
        rc = run_once(env, overrides)
        attempt_info = {
            "attempt": attempt,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "returncode": rc,
        }
        manifest["attempts"].append(attempt_info)
        write_manifest(run_dir / "run_manifest_v2.json", manifest)
        if rc == 0:
            manifest["status"] = "PASS"
            write_manifest(run_dir / "run_manifest_v2.json", manifest)
            print("PRETRAIN_PASS", attempt_info)
            return
        # OOM recovery ladder
        if batch_size > 1:
            batch_size = max(1, batch_size // 2)
            grad_accum *= 2
            print(f"OOM_RECOVERY reduce batch_size->{batch_size} accum->{grad_accum}", flush=True)
            continue
        if grad_accum < 16:
            grad_accum *= 2
            print(f"OOM_RECOVERY increase accum->{grad_accum}", flush=True)
            continue
        # activation checkpointing hint via env for future model hooks
        env["ONLINE2_V2_ACTIVATION_CHECKPOINT"] = "1"
        print("OOM_RECOVERY set activation checkpoint flag", flush=True)
        break

    manifest["status"] = "FAIL"
    write_manifest(run_dir / "run_manifest_v2.json", manifest)
    print("PRETRAIN_FAIL", manifest["attempts"][-1] if manifest["attempts"] else {})
    raise SystemExit(rc if rc else 1)


if __name__ == "__main__":
    main()
