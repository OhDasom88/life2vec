#!/usr/bin/env python3
"""Staged GPU smoke: one Development3 case, search fold 0, identity cold→critic path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    cfg = yaml.safe_load((ROOT / "conf/m1/cf1s_core_smoke.yaml").read_text()) or {}
    case_id = "F420458_2025-02-16_2025-03-01"

    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_canonical import (
        canonical_json_sha256,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_execution import (
        production_apply_edits_and_score,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_production import (
        make_production_fold_forward_fn,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_raw_transaction import (
        build_identity_transaction,
    )
    from src.online2.v2.finetune_v03.counterfactual.cf1s.core_runtime import (
        apply_exact_byte_deterministic_runtime,
        build_stage_a_reencoder,
        discover_critic_checkpoints,
        load_case_events,
    )
    from src.online2.v2.finetune_v03.counterfactual.evaluation.case_batch import (
        load_embedding_sidecar,
    )

    lock = apply_exact_byte_deterministic_runtime(before_cuda_init=not torch.cuda.is_initialized())
    events = load_case_events(cfg, case_id)
    for ev in events:
        if not getattr(ev, "sentence_tokens", None):
            ev.sentence_tokens = list(getattr(ev, "tokens", None) or [])
            ev.token_count = len(ev.sentence_tokens)

    side = load_embedding_sidecar(Path(cfg["embeddings_dir"]), case_id)
    sidecar = {}
    for row in side.itertuples(index=False):
        d = row._asdict() if hasattr(row, "_asdict") else dict(zip(side.columns, row))
        sidecar[str(d.get("event_id"))] = {
            "view": d.get("view", "UNKNOWN"),
            "zone": d.get("zone", 0),
            "case_age_hours": float(d.get("case_age_hours") or 0.0),
            "local_hour": int(float(d.get("local_hour") or 0)) % 24,
        }

    # Prefer a short-prefix smoke: only first N events to bound wall time, but still
    # exercise FULL_EVENT_COVERAGE_COLD_REBUILD API on that prefix.
    max_events = int(os.environ.get("CF1S_SMOKE_MAX_EVENTS", "32"))
    events = list(events)[:max_events]
    valid = [e for e in events if list(getattr(e, "sentence_tokens", None) or [])]
    if not valid:
        raise SystemExit("no valid events for smoke")
    payload = [
        {
            "event_id": str(getattr(e, "event_id", "")),
            "tokens": list(getattr(e, "sentence_tokens", None) or []),
        }
        for e in events
    ]
    identity_tx = build_identity_transaction(
        case_id=case_id,
        original_caseevents_sha=canonical_json_sha256(payload),
    )

    reencoder = build_stage_a_reencoder(cfg, case_id=case_id)
    ckpts = discover_critic_checkpoints(cfg)
    fold_ids = list(range(len(ckpts)))
    fold_forward = make_production_fold_forward_fn(
        gpu_fraction=float(cfg.get("gpu_memory_fraction") or 0.4)
    )

    out = production_apply_edits_and_score(
        events=events,
        validated_transaction=identity_tx,
        reencoder=reencoder,
        fold_forward_fn=fold_forward,
        checkpoint_paths=ckpts,
        fold_ids=fold_ids,
        requested_fold_ids=[0],
        sidecar_by_event_id=sidecar,
        scope="search",
        transaction_mode="FORCED_IDENTITY",
        batch_size=4,
    )
    risk = out["scores"]["by_fold"]["0"]["risk"]
    report = {
        "status": "GPU_SMOKE_PASS",
        "case_id": case_id,
        "reencode_mode": out["reencode_mode"],
        "valid_event_count": len(out["valid_event_ids"]),
        "fold0_risk": risk,
        "stage_a_output_sha": out.get("stage_a_output_sha"),
        "semantic_critic_input_sha": out.get("semantic_critic_input_sha"),
        "runtime_lock_sha256": lock.sha256(),
        "smoke_max_events": max_events,
        "interpretation_label": "Development3 selection-blind contract verification evidence",
    }
    out_dir = Path(cfg["output_root"]) / "gpu_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "CF1S_STAGE_GPU_SMOKE.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
