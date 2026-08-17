#!/usr/bin/env python3
"""Run the §9.7-style normal/abnormal concept-space separation probe on real
out-of-fold representations from a completed cv_repeated finetune run.

Consumes the `*_oof_representations.npz` files that
`run_diagnosis_finetune_v03.py::train_one_split` now persists per fold
(previously computed z_proj/h_binary was discarded after CSV export — this
was the only missing piece, no new training required).

Usage:
    python scripts/online2_v2/v03/probe_normal_abnormal_separation_v03.py \\
        --run-dir outputs/online2/v2_finetune_v03/runs/probe_oof_3fold
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.representation_explorer.label_separation_probe import (  # noqa: E402
    probe_label_separation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--out", type=Path, default=None, help="default: <run-dir>/normal_abnormal_separation_probe.json"
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_oof(run_dir: Path) -> dict[str, np.ndarray]:
    npz_paths = sorted(run_dir.glob("*_oof_representations.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"no *_oof_representations.npz under {run_dir}")
    case_ids: list[str] = []
    y_abnormal: list[float] = []
    z_proj: list[np.ndarray] = []
    h_binary: list[np.ndarray] = []
    seen: set[str] = set()
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        for cid, y, zp, hb in zip(data["case_id"], data["y_abnormal"], data["z_proj"], data["h_binary"]):
            cid = str(cid)
            if cid in seen:
                continue  # a case may appear in >1 repeat's val fold; keep first OOF rep
            seen.add(cid)
            case_ids.append(cid)
            y_abnormal.append(float(y))
            z_proj.append(zp)
            h_binary.append(hb)
    return {
        "case_ids": np.array(case_ids),
        "y_abnormal": np.array(y_abnormal),
        "z_proj": np.stack(z_proj),
        "h_binary": np.stack(h_binary),
        "n_source_files": len(npz_paths),
    }


def main() -> None:
    args = parse_args()
    out_path = args.out or (args.run_dir / "normal_abnormal_separation_probe.json")

    oof = load_oof(args.run_dir)
    n = len(oof["case_ids"])
    print(f"loaded {n} OOF cases from {oof['n_source_files']} fold file(s)")

    checkpoint_ids = sorted(p.stem for p in args.run_dir.glob("*_best.pt")) + sorted(
        p.stem for p in args.run_dir.glob("*fold*best*.pt")
    )
    checkpoint_ids = sorted(set(checkpoint_ids)) or ["unknown"]

    reports = {}
    for rep_name in ("h_binary", "z_proj"):
        report = probe_label_separation(
            oof[rep_name],
            oof["y_abnormal"],
            representation_name=rep_name,
            checkpoint_ids=checkpoint_ids,
            label_source="y_abnormal (label_map.json 정상_운영 vs abnormal, DiagnosisEventDatasetV03)",
            n_splits=args.n_splits,
            seed=args.seed,
        )
        reports[rep_name] = report
        print(
            f"[{rep_name}] n={report.metadata.n_samples} pos={report.metadata.n_positive} "
            f"AUROC={report.cv_auroc_mean:.3f}±{report.cv_auroc_std:.3f} "
            f"(folds={[round(a, 3) for a in report.cv_auroc_per_fold]}) "
            f"silhouette={report.silhouette:.3f}"
        )

    out = {
        rep_name: {
            "metadata": asdict(report.metadata),
            "cv_auroc_per_fold": list(report.cv_auroc_per_fold),
            "cv_auroc_mean": report.cv_auroc_mean,
            "cv_auroc_std": report.cv_auroc_std,
            "silhouette": report.silhouette,
            "disclaimer": report.disclaimer,
        }
        for rep_name, report in reports.items()
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
