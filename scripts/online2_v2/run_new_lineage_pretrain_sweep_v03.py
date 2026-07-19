#!/usr/bin/env python3
"""Hierarchical Pretrain Sweep B for online2 new lineage (not a single-run retrain).

Runs a small ordered grid, records metrics, and writes champion report + lock.
Problem-20 tuning is forbidden (pretrain-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_one(cfg: Dict[str, Any], *, build_dir: Path, sweeps_root: Path) -> Dict[str, Any]:
    run_dir = sweeps_root / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pretrain.log"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/online2_v2/run_v2_pretrain_loop.py"),
        "--mode",
        "full",
        "--build-dir",
        str(build_dir),
        "--run-dir",
        str(run_dir),
        "--max-rows",
        "0",
        "--steps",
        str(cfg["steps"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--max-length",
        str(cfg["max_length"]),
        "--ckpt-every",
        str(cfg.get("ckpt_every", 200)),
        "--val-every",
        str(cfg.get("val_every", 100)),
        "--val-batches",
        str(cfg.get("val_batches", 32)),
        "--val-frac",
        "0.05",
        "--early-stop-patience",
        str(cfg.get("early_stop_patience", 15)),
        "--early-stop-min-delta",
        "1e-4",
        "--sop-reverse",
        "0.20",
        "--sop-shuffle",
        "0.20",
        "--target-vram-frac",
        "0.70",
        "--skip-vram-calibrate",
        "--no-wandb",
    ]
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    best = run_dir / "best.ckpt"
    manifest = run_dir / "run_manifest_v2.json"
    row: Dict[str, Any] = {
        "run_name": cfg["run_name"],
        "config": cfg,
        "exit_code": proc.returncode,
        "run_dir": str(run_dir),
        "best_ckpt": str(best) if best.exists() else None,
        "best_ckpt_sha256": sha256_file(best) if best.exists() else None,
        "manifest": str(manifest) if manifest.exists() else None,
    }
    if manifest.exists():
        try:
            row["manifest_summary"] = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            row["manifest_error"] = str(exc)
    return row


def _best_val_loss(row: Dict[str, Any]) -> float:
    man = row.get("manifest_summary") or {}
    best_val = man.get("best_val_loss")
    if best_val is not None:
        return float(best_val)
    hist_path = Path(row["run_dir"]) / "train_history.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        if isinstance(hist, dict) and hist.get("best_val_loss") is not None:
            return float(hist["best_val_loss"])
        if isinstance(hist, list):
            vals = [float(h["val_loss"]) for h in hist if isinstance(h, dict) and h.get("val_loss") is not None]
            if vals:
                return min(vals)
    return 1e9


def select_champion(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = []
    for r in rows:
        if r.get("exit_code") != 0 or not r.get("best_ckpt"):
            continue
        scored.append((_best_val_loss(r), r))
    if not scored:
        raise SystemExit("no successful pretrain sweep arms with best.ckpt")
    scored.sort(key=lambda x: x[0])
    champ = scored[0][1]
    return {
        "champion_run_name": champ["run_name"],
        "champion_best_val_loss": scored[0][0],
        "champion_ckpt": champ["best_ckpt"],
        "champion_ckpt_sha256": champ["best_ckpt_sha256"],
        "selection_rule": "min_best_val_loss_among_successful_arms",
        "n_successful_arms": len(scored),
        "n_arms": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=Path, default=ROOT / "outputs/online2/v2_build")
    ap.add_argument(
        "--sweeps-root",
        type=Path,
        default=ROOT / "outputs/online2/new_lineage_20260718/sweeps/pretrain_arms",
    )
    ap.add_argument(
        "--phase",
        choices=["probe", "champion_full"],
        default="probe",
        help="probe=short multi-arm sweep; champion_full=train selected arm to full early-stop budget",
    )
    ap.add_argument("--champion-from", type=Path, default=None, help="pretrain_champion_report.json for champion_full")
    args = ap.parse_args()

    args.sweeps_root.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "outputs/online2/new_lineage_20260718/sweeps"
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "probe":
        arms = [
            {
                "run_name": "probe_ml1024_bs56_s5000",
                "max_length": 1024,
                "batch_size": 56,
                "steps": 5000,
                "early_stop_patience": 8,
            },
            {
                "run_name": "probe_ml2048_bs28_s5000",
                "max_length": 2048,
                "batch_size": 28,
                "steps": 5000,
                "early_stop_patience": 8,
            },
            {
                "run_name": "probe_ml1024_bs40_s5000",
                "max_length": 1024,
                "batch_size": 40,
                "steps": 5000,
                "early_stop_patience": 8,
            },
        ]
        results = [run_one(a, build_dir=args.build_dir, sweeps_root=args.sweeps_root) for a in arms]
        champ = select_champion(results)
        payload = {
            "sweep_id": "pretrain_B_20260718_probe",
            "phase": "probe",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_dir": str(args.build_dir),
            "arms": results,
            "champion": champ,
            "status": "PROBE_COMPLETE_AWAITING_CHAMPION_FULL",
            "prohibited": ["problem20_tuning", "single_run_without_sweep"],
        }
        out = report_dir / "pretrain_champion_report.md"
        js = report_dir / "pretrain_sweep_results.json"
        js.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        out.write_text(
            "# Pretrain champion report (probe)\n\n"
            f"- selected: `{champ['champion_run_name']}`\n"
            f"- best_val_loss: {champ['champion_best_val_loss']}\n"
            f"- ckpt: `{champ['champion_ckpt']}`\n"
            f"- sha256: `{champ['champion_ckpt_sha256']}`\n"
            f"- next: run `--phase champion_full --champion-from {js}`\n",
            encoding="utf-8",
        )
        print(json.dumps(payload["champion"], indent=2))
        return 0

    # champion_full
    src = args.champion_from or (report_dir / "pretrain_sweep_results.json")
    prev = json.loads(src.read_text(encoding="utf-8"))
    base = prev["champion"]["champion_run_name"]
    # map probe arm hyperparams
    arm_cfg = None
    for a in prev["arms"]:
        if a["run_name"] == base:
            arm_cfg = dict(a["config"])
            break
    if arm_cfg is None:
        raise SystemExit(f"champion arm not found: {base}")
    arm_cfg.update(
        {
            "run_name": "champion_full_event_grain_earlystop",
            # Bound wall-clock: early-stop is the real stopper; 20k is a hard ceiling.
            "steps": 20000,
            "early_stop_patience": 12,
            "ckpt_every": 200,
            "val_every": 100,
        }
    )
    # also materialize into canonical path expected by cf config
    canonical = ROOT / "outputs/online2/v2_runs/full_event_grain_earlystop"
    row = run_one(arm_cfg, build_dir=args.build_dir, sweeps_root=args.sweeps_root)
    if row.get("best_ckpt"):
        canonical.mkdir(parents=True, exist_ok=True)
        best_src = Path(row["best_ckpt"])
        best_dst = canonical / "best.ckpt"
        best_dst.write_bytes(best_src.read_bytes())
        # copy sibling manifests if present
        for name in ("last.ckpt", "run_manifest_v2.json", "train_history.json"):
            p = best_src.parent / name
            if p.exists():
                (canonical / name).write_bytes(p.read_bytes())
        row["canonical_best_ckpt"] = str(best_dst)
        row["canonical_best_ckpt_sha256"] = sha256_file(best_dst)

    lock = {
        "sweep_id": "pretrain_B_20260718_champion_full",
        "phase": "champion_full",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe_source": str(src),
        "arm": row,
        "status": "CHAMPION_LOCKED" if row.get("exit_code") == 0 and row.get("best_ckpt") else "FAILED",
        "training_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
    }
    (report_dir / "pretrain_champion_lock.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (report_dir / "pretrain_champion_report.md").write_text(
        "# Pretrain champion lock\n\n"
        f"- status: {lock['status']}\n"
        f"- canonical: `{row.get('canonical_best_ckpt')}`\n"
        f"- sha256: `{row.get('canonical_best_ckpt_sha256')}`\n"
        f"- training_source_commit: `{lock['training_source_commit']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": lock["status"], "canonical": row.get("canonical_best_ckpt")}, indent=2))
    return 0 if lock["status"] == "CHAMPION_LOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
