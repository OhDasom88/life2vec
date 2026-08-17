#!/usr/bin/env python3
"""Audit UNK token rate for the finetune (CF1S) corpus.

No prior script measures this repeatably — the only known precedent was a
one-off, after-the-fact fix (`vocab.py`'s FARM_LOCAL slot 8->24) once UNK risk
was noticed by chance. This script mirrors the EXACT lookup finetune uses at
cache time (`cache_stage_a_event_embeddings.window_to_tensors`:
`vocab.token2index.get(t, unk)`) so a drift between build-time vocab and the
vocab currently on disk shows up here, not just in the pre-baked
`events_tokenized_v2.parquet.token_ids` column (which reflects whatever vocab
was current when build_v2.py ran, not now).

Usage:
    python scripts/online2_v2/audit_finetune_vocab_unk.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.online2_v2.cache_stage_a_event_embeddings import (  # noqa: E402
    case_events_from_frame,
    load_events_frame,
)
from src.data_new.vocabulary import RegistryVocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--manifest", type=Path, default=ROOT / "outputs/online2/v2_finetune/case_manifest.csv"
    )
    p.add_argument(
        "--events", type=Path, default=ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet"
    )
    p.add_argument(
        "--vocab",
        type=Path,
        default=ROOT / "outputs/online2/v2_build/life2vec_token_registry_v2.json",
    )
    p.add_argument(
        "--out", type=Path, default=ROOT / "outputs/online2/v2_finetune_v03/unk_audit_report.json"
    )
    p.add_argument("--fail-above", type=float, default=0.0, help="exit 1 if overall UNK rate exceeds this")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    manifest = pd.read_csv(args.manifest)
    farm_ids = set(manifest["farm_id"].astype(str))
    print(f"auditing {len(manifest)} finetune cases across {len(farm_ids)} farms...", flush=True)

    vocab = RegistryVocabulary(registry_path=str(args.vocab), registry_version="v2")
    unk_id = int(vocab.token2index["[UNK]"])

    events_df = load_events_frame(args.events, farm_ids)

    overall_total = 0
    overall_unk = 0
    per_farm: dict[str, dict] = {}
    per_case: list[dict] = []
    unk_token_counts: Counter = Counter()

    for _, case_row in manifest.iterrows():
        farm_id = str(case_row["farm_id"])
        case_id = str(case_row["case_id"])
        period_start = pd.Timestamp(case_row["period_start"])
        period_end = pd.Timestamp(case_row["period_end"])

        events = case_events_from_frame(
            events_df, farm_id, period_start, period_end, include_image=True
        )
        case_total = 0
        case_unk = 0
        for ev in events:
            for tok in ev.sentence_tokens:
                case_total += 1
                if tok not in vocab.token2index:
                    case_unk += 1
                    unk_token_counts[tok] += 1

        overall_total += case_total
        overall_unk += case_unk
        fstats = per_farm.setdefault(farm_id, {"total": 0, "unk": 0})
        fstats["total"] += case_total
        fstats["unk"] += case_unk
        per_case.append(
            {
                "case_id": case_id,
                "farm_id": farm_id,
                "n_tokens": case_total,
                "n_unk": case_unk,
                "unk_rate": (case_unk / case_total) if case_total else float("nan"),
            }
        )

    for f in per_farm.values():
        f["unk_rate"] = (f["unk"] / f["total"]) if f["total"] else float("nan")

    overall_rate = (overall_unk / overall_total) if overall_total else float("nan")
    report = {
        "vocab_path": str(args.vocab),
        "events_path": str(args.events),
        "manifest_path": str(args.manifest),
        "unk_id": unk_id,
        "n_cases": len(manifest),
        "n_farms": len(farm_ids),
        "overall_total_tokens": overall_total,
        "overall_unk_tokens": overall_unk,
        "overall_unk_rate": overall_rate,
        "per_farm": per_farm,
        "per_case": sorted(per_case, key=lambda r: -r["unk_rate"] if r["n_tokens"] else 0),
        "top_unk_tokens": unk_token_counts.most_common(30),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    print(f"overall UNK rate: {overall_rate:.6f} ({overall_unk}/{overall_total})")
    worst = sorted(per_farm.items(), key=lambda kv: -kv[1]["unk_rate"])[:5]
    for farm_id, stats in worst:
        print(f"  farm {farm_id}: unk_rate={stats['unk_rate']:.6f} ({stats['unk']}/{stats['total']})")
    if unk_token_counts:
        print("top UNK-falling tokens (never in current vocab):")
        for tok, n in unk_token_counts.most_common(10):
            print(f"  {tok!r}: {n}")
    print(f"wrote {args.out}")

    if overall_rate > args.fail_above:
        print(f"FAIL: overall UNK rate {overall_rate:.6f} exceeds threshold {args.fail_above}")
        sys.exit(1)


if __name__ == "__main__":
    main()
