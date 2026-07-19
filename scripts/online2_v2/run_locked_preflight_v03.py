#!/usr/bin/env python3
"""P1 Locked preflight: input checklist, archive prior outputs, dry-run static gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


REQUIRED_KEYS = [
    "stage_a_ckpt",
    "vocabulary_path",
    "tokenizer_path",
    "feature_schema_path",
    "binning_registry_path",
    "cells_path",
    "events_path",
    "abspos_reference_path",
    "embeddings_dir",
    "labels_path",
    "label_map_path",
    "run_dir",
]


def check_inputs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    missing = []
    for k in REQUIRED_KEYS:
        p = Path(cfg[k])
        ok = p.exists()
        if k == "embeddings_dir":
            n = len(list(p.glob("*.parquet"))) if ok else 0
            ok = ok and n >= 35
            detail = {"n_parquet": n}
        elif k == "run_dir":
            folds = sorted(p.glob("r0_fold*_best.pt")) if ok else []
            ok = ok and len(folds) >= 3
            detail = {"fold_ckpts": [str(x) for x in folds]}
        else:
            detail = {"bytes": p.stat().st_size if p.exists() else 0}
        row = {"key": k, "path": str(p), "ok": ok, **detail}
        if p.exists() and p.is_file() and p.stat().st_size < 500_000_000:
            row["sha256"] = sha256_file(p)
        rows.append(row)
        if not ok:
            missing.append(k)
    noop = Path((cfg.get("critic") or {}).get("noop_noise_artifact") or "outputs/cf_calibration/noop_noise.json")
    if not noop.is_absolute():
        noop = ROOT / noop
    noop_ok = noop.exists()
    rows.append({"key": "noop_noise", "path": str(noop), "ok": noop_ok})
    if not noop_ok:
        missing.append("noop_noise")
    return {"checks": rows, "missing": missing, "pass": len(missing) == 0}


def archive_prior_outputs(out_root: Path, archive_root: Path) -> Dict[str, Any]:
    archive_root.mkdir(parents=True, exist_ok=True)
    moved = []
    if out_root.exists() and any(out_root.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = archive_root / f"m1_m2_prereq_archive_{stamp}"
        shutil.move(str(out_root), str(dest))
        moved.append({"from": str(out_root), "to": str(dest)})
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "ARCHIVED_README.txt").write_text(
            f"Prior outputs moved to {dest} (read-only archive). New Locked attempt uses fresh namespace.\n",
            encoding="utf-8",
        )
    return {"moved": moved, "output_root": str(out_root)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "conf/m1/cf_m2_prereq_smoke.yaml")
    ap.add_argument("--archive", action="store_true", help="Move existing output_root to archive")
    ap.add_argument("--dry-run-orchestrator", action="store_true", help="Run orchestrator --dry-run-plan-only")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report_dir = ROOT / "reports/p1_rejection_recovery/locked_preflight"
    report_dir.mkdir(parents=True, exist_ok=True)

    checklist = check_inputs(cfg)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "evaluation_source_commit": git_head,
        "training_source_commit": git_head,
        "input_checklist": checklist,
        "recovery_decision": str(ROOT / "reports/p1_rejection_recovery/recovery_decision.json"),
    }

    if args.archive:
        payload["archive"] = archive_prior_outputs(
            Path(cfg["output_root"]),
            ROOT / "outputs/online2/new_lineage_20260718/archives",
        )

    orch = None
    if args.dry_run_orchestrator:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/online2_v2/v03/run_p1_acceptance_recovery_v03.py"),
            "--config",
            str(args.config),
            "--dry-run-plan-only",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT)}, capture_output=True, text=True)
        orch = {"exit_code": proc.returncode, "stdout_tail": proc.stdout[-4000:], "stderr_tail": proc.stderr[-2000:]}
        payload["orchestrator_dry_run"] = orch

    out = report_dir / "locked_preflight_checklist.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"pass": checklist["pass"], "missing": checklist["missing"], "report": str(out)}, indent=2))
    return 0 if checklist["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
