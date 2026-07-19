#!/usr/bin/env python3
"""Project an all-PASS CF1S FINAL verdict into an external completion artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-verdict", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/cf1s_core/completions",
    )
    args = parser.parse_args()

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_dumps,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_promotion import (
        fsync_dir,
        fsync_file,
        rename_noreplace,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_verifier import (
        project_completion,
        validate_completion_projection,
    )

    verdict_path = (
        args.final_verdict
        if args.final_verdict.is_absolute()
        else ROOT / args.final_verdict
    )
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    completion = project_completion(verdict)
    validate_completion_projection(completion, verdict)

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{completion['run_id']}.json"
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        validate_completion_projection(existing, verdict)
        print(json.dumps({"status": "EXISTING_VALID", "out": str(out)}, indent=2))
        return 0
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(canonical_json_dumps(completion) + "\n", encoding="utf-8")
    fsync_file(tmp)
    rename_noreplace(tmp, out)
    fsync_dir(out_dir)
    print(json.dumps({"status": "WRITTEN", "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
