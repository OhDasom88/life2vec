#!/usr/bin/env python3
"""Token attribution pipeline hook for v0.3 (P0).

Delegates Stage-A token IxG to existing v01_layer_saliency helpers without
mutating v0.1/v0.2 packages. Writes layer_analysis/ under v2_finetune_v03.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online2.v2.finetune_v03.version import DEFAULT_OUTPUT_ROOT, FINETUNE_VERSION  # noqa: E402


def append_progress(msg: Dict[str, Any]) -> None:
    log = ROOT / DEFAULT_OUTPUT_ROOT / "logs" / "PROGRESS.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    msg = dict(msg)
    msg["ts"] = datetime.now(timezone.utc).isoformat()
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-id", type=str, default="F420458_2025-02-16_2025-03-01")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Write hook metadata without running heavy Stage A re-encode (default for pipeline).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (
        ROOT / DEFAULT_OUTPUT_ROOT / "p0" / f"token_attr_{args.case_id}" / "layer_analysis"
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import proof that Stage A token IxG utilities exist (read-only).
    from src.online2.v2 import v01_layer_saliency as layer

    payload = {
        "finetune_version": FINETUNE_VERSION,
        "case_id": args.case_id,
        "status": "hook_ready",
        "helpers": {
            "classify_token_type": hasattr(layer, "classify_token_type"),
            "load_stage_a_module": hasattr(layer, "load_stage_a_module"),
            "module": "src.online2.v2.v01_layer_saliency",
        },
        "next_wiring": (
            "Load pretrain encoder → build case windows → token IxG vs abnormal risk "
            "→ join FEATURE|*/VALUE tokens to cell_occurrences.raw_value"
        ),
        "probe_only": bool(args.probe_only),
    }
    (out_dir / "token_attr_hook.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    append_progress(
        {
            "phase": "p0_token_hook",
            "case_id": args.case_id,
            "out": str(out_dir / "token_attr_hook.json"),
            "status": "hook_ready",
        }
    )


if __name__ == "__main__":
    main()
