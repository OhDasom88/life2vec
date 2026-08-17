#!/usr/bin/env python3
"""Run Online1 Phase 0→7 sequentially."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/dasom/life2vec")
SCRIPTS = ROOT / "scripts" / "online1"
PYTHON = sys.executable

PHASES = [
    "run_phase0_audit.py",
    "run_phase1_cube_preflight.py",
    "run_phase2_materialize.py",
    "run_phase3_4_smoke_preflight.py",
    "run_phase5_6_dev_eval.py",
    "run_phase7_report.py",
]


def main():
    for name in PHASES:
        path = SCRIPTS / name
        print("=" * 80)
        print("RUNNING", path)
        print("=" * 80)
        subprocess.check_call([PYTHON, str(path)], cwd=str(ROOT))
    print("ALL PHASES COMPLETE")


if __name__ == "__main__":
    main()
