#!/usr/bin/env python3
"""Package CF-1S Core run outputs with SHA manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[3]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--preflight", type=Path, default=None)
    parser.add_argument("--pytest-log", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"missing run dir: {run_dir}")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (ROOT / "outputs/cf1s_core" / f"CF1S_CORE_PACKAGE_{ts}")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(run_dir, out / "run", dirs_exist_ok=True)
    if args.preflight and Path(args.preflight).exists():
        shutil.copy2(args.preflight, out / "CF1S_CORE_PREFLIGHT.json")
    if args.pytest_log and Path(args.pytest_log).exists():
        shutil.copy2(args.pytest_log, out / "pytest.log")

    checksums: Dict[str, str] = {}
    for p in sorted(out.rglob("*")):
        if p.is_file():
            checksums[str(p.relative_to(out))] = _sha(p)
    (out / "CHECKSUMS.sha256").write_text(
        "\n".join(f"{h}  {n}" for n, h in sorted(checksums.items())) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifact": "CF1S_CORE_PACKAGE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "n_files": len(checksums),
        "package_sha256": _sha(out / "CHECKSUMS.sha256"),
    }
    (out / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PACKAGED", "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
