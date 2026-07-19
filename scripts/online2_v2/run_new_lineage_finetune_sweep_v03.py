#!/usr/bin/env python3
"""Hierarchical Finetune Sweep C for online2 new lineage.

Champion selection uses Example-35 group-aware repeated CV only.
Problem-20 is forbidden for hyperparameter selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_best_macro_f1(summary_path: Path) -> Optional[float]:
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    vals = []
    for r in payload.get("results") or []:
        best = r.get("best") or {}
        v = best.get("val_fine_macro_f1")
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv == fv:  # not NaN
            vals.append(fv)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def run_one(cfg: Dict[str, Any], *, emb_dir: Path, labels: Path, label_map: Path, sweeps_root: Path) -> Dict[str, Any]:
    run_dir = sweeps_root / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "finetune.log"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/online2_v2/v03/run_diagnosis_finetune_v03.py"),
        "--mode",
        "cv_repeated",
        "--n-folds",
        "3",
        "--n-repeats",
        "3",
        "--cv-seeds",
        "2023,2024,2025",
        "--epochs",
        str(cfg.get("epochs", 80)),
        "--lr",
        str(cfg["lr"]),
        "--dropout",
        str(cfg["dropout"]),
        "--lambda-binary",
        str(cfg["lambda_binary"]),
        "--lambda-consistency",
        str(cfg["lambda_consistency"]),
        "--early-stop-patience",
        str(cfg.get("early_stop_patience", 15)),
        "--early-stop-min-epochs",
        str(cfg.get("early_stop_min_epochs", 20)),
        "--monitor",
        "val_macro_f1",
        "--emb-dir",
        str(emb_dir),
        "--labels",
        str(labels),
        "--label-map",
        str(label_map),
        "--run-dir",
        str(run_dir),
        "--wandb",
        "false",
        "--device",
        "cuda",
    ]
    cmd.extend(["--pos-weight", str(cfg.get("pos_weight", "auto"))])
    env = {**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    summary = run_dir / "summary.json"
    row: Dict[str, Any] = {
        "run_name": cfg["run_name"],
        "config": cfg,
        "exit_code": proc.returncode,
        "run_dir": str(run_dir),
        "summary": str(summary) if summary.exists() else None,
        "mean_best_macro_f1": mean_best_macro_f1(summary) if summary.exists() else None,
        "fold_ckpts": sorted(str(p) for p in run_dir.glob("r0_fold*_best.pt")),
    }
    return row


def select_champion(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = []
    for r in rows:
        if r.get("exit_code") != 0:
            continue
        metric = r.get("mean_best_macro_f1")
        if metric is None:
            continue
        if not r.get("fold_ckpts"):
            continue
        scored.append((float(metric), r))
    if not scored:
        raise SystemExit("no successful finetune sweep arms")
    scored.sort(key=lambda x: x[0], reverse=True)
    champ = scored[0][1]
    return {
        "champion_run_name": champ["run_name"],
        "champion_mean_best_macro_f1": scored[0][0],
        "champion_run_dir": champ["run_dir"],
        "selection_rule": "max_mean_best_macro_f1_example35_cv_repeated_only",
        "selection_cohort": "example_35",
        "problem20_used_for_tuning": False,
        "n_successful_arms": len(scored),
        "n_arms": len(rows),
    }


def materialize_canonical(champ_run_dir: Path, canonical: Path) -> Dict[str, Any]:
    canonical.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(champ_run_dir.glob("r0_fold*_best.pt")):
        dst = canonical / src.name
        dst.write_bytes(src.read_bytes())
        copied.append({"src": str(src), "dst": str(dst), "sha256": sha256_file(dst)})
    for name in ("summary.json", "config.json", "split_manifest.json"):
        src = champ_run_dir / name
        if src.exists():
            dst = canonical / name
            dst.write_bytes(src.read_bytes())
            copied.append({"src": str(src), "dst": str(dst), "sha256": sha256_file(dst)})
    return {"canonical_run_dir": str(canonical), "copied": copied}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--emb-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune_v02/event_embeddings",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/labels_example_score90.csv",
        help="Example-35 only labels for champion selection (Problem-20 forbidden)",
    )
    ap.add_argument(
        "--label-map",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune/label_map.json",
    )
    ap.add_argument(
        "--sweeps-root",
        type=Path,
        default=ROOT / "outputs/online2/new_lineage_20260718/sweeps/finetune_arms",
    )
    ap.add_argument(
        "--canonical-run-dir",
        type=Path,
        default=ROOT / "outputs/online2/v2_finetune_v03/runs/cv_repeated_20260715_161945",
        help="Path expected by conf/m1/cf_m2_prereq_smoke.yaml",
    )
    args = ap.parse_args()

    # Ordered hierarchical probe (not full cartesian product).
    arms = [
        {
            "run_name": "ft_lr3e4_do02_lb1_lc005",
            "lr": 3e-4,
            "dropout": 0.2,
            "lambda_binary": 1.0,
            "lambda_consistency": 0.05,
            "pos_weight": "auto",
        },
        {
            "run_name": "ft_lr1e4_do02_lb1_lc005",
            "lr": 1e-4,
            "dropout": 0.2,
            "lambda_binary": 1.0,
            "lambda_consistency": 0.05,
            "pos_weight": "auto",
        },
        {
            "run_name": "ft_lr3e4_do03_lb1_lc01",
            "lr": 3e-4,
            "dropout": 0.3,
            "lambda_binary": 1.0,
            "lambda_consistency": 0.1,
            "pos_weight": "auto",
        },
        {
            "run_name": "ft_lr3e4_do01_lb05_lc0",
            "lr": 3e-4,
            "dropout": 0.1,
            "lambda_binary": 0.5,
            "lambda_consistency": 0.0,
            "pos_weight": "auto",
        },
    ]

    args.sweeps_root.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "outputs/online2/new_lineage_20260718/sweeps"
    report_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_one(a, emb_dir=args.emb_dir, labels=args.labels, label_map=args.label_map, sweeps_root=args.sweeps_root)
        for a in arms
    ]
    champ = select_champion(results)
    materialize = materialize_canonical(Path(champ["champion_run_dir"]), args.canonical_run_dir)

    payload = {
        "sweep_id": "finetune_C_20260718",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_cohort": "example_35_group_aware_cv_only",
        "holdout_forbidden_for_tuning": "problem_20",
        "arms": results,
        "champion": champ,
        "canonical": materialize,
        "status": "CHAMPION_LOCKED",
        "training_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
        "prohibited": ["problem20_tuning", "single_run_without_sweep", "p1_locked_feedback_tuning"],
    }
    js = report_dir / "finetune_sweep_results.json"
    js.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (report_dir / "finetune_champion_lock.json").write_text(
        json.dumps(
            {
                "status": "CHAMPION_LOCKED",
                "champion": champ,
                "canonical": materialize,
                "created_at": payload["created_at"],
                "training_source_commit": payload["training_source_commit"],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    (report_dir / "finetune_champion_report.md").write_text(
        "# Finetune champion lock\n\n"
        f"- selected: `{champ['champion_run_name']}`\n"
        f"- mean_best_macro_f1: {champ['champion_mean_best_macro_f1']}\n"
        f"- canonical_run_dir: `{materialize['canonical_run_dir']}`\n"
        f"- selection_cohort: example_35 only (Problem-20 forbidden)\n"
        f"- training_source_commit: `{payload['training_source_commit']}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "CHAMPION_LOCKED", "champion": champ, "canonical": materialize["canonical_run_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
