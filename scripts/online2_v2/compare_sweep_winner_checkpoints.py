#!/usr/bin/env python3
"""두 sweep(1479개 서사 corpus / 1517개 서사 corpus)에서 각각 찾은 최적
하이퍼파라미터 체크포인트를, SAE dead_feature_ratio + effective rank로 비교한다.

`compare_narrative_ablation_checkpoints.py`와 같은 계산 로직(effective_rank,
train_and_evaluate_sae, encode_all_positions)을 그대로 재사용하되, 이번엔
두 체크포인트가 서로 다른 코퍼스(따라서 서로 다른 vocab/vocab_size)에서
나왔으므로 각 체크포인트를 자기 자신의 corpus(vocab/events/training_events/
abspos_reference)로 평가해야 한다 -- 하나의 공유 vocab을 쓰는 기존 스크립트의
DEFAULT_ARMS 구조를 그대로는 못 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts" / "online2_v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "online2_v2"))

from cache_stage_a_event_embeddings import (  # noqa: E402
    construct_target_window,
    load_abspos_reference,
    load_events_frame,
    load_frozen_encoder,
    window_to_tensors,
)
from pilot_sae_layer_decoder_random_comparison import (  # noqa: E402
    effective_rank,
    encode_all_positions,
    train_and_evaluate_sae,
)
from pilot_sae_narrative_selected_activations import (  # noqa: E402
    build_farm_event_lists,
    find_narrative_targets,
    stratified_sample_targets,
)
from src.data_new.vocabulary import RegistryVocabulary  # noqa: E402

POSITIONS = ["layer5", "mlm_transform", "sop_transform"]

ARMS = {
    "expanded_full_1479_lr5e4": {
        "build_dir": ROOT / "outputs/online2/v2_build_expanded_full",
        "ckpt": ROOT / "outputs/online2/v2_runs/sweeps/expanded_full_sop_lr_sweep_kg5nv8zz/best.ckpt",
        "hp": {"lr": 0.0005, "sop_reverse": 0.2, "sop_shuffle": 0.2},
        "n_narratives": 1479,
    },
    "expanded_full1517_lr2e4": {
        "build_dir": ROOT / "outputs/online2/v2_build_expanded_full1517",
        "ckpt": ROOT / "outputs/online2/v2_runs/sweeps/expanded_full1517_sop_lr_sweep_arqqsci4/best.ckpt",
        "hp": {"lr": 0.0002, "sop_reverse": 0.2, "sop_shuffle": 0.2},
        "n_narratives": 1517,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-narrative-cap", type=int, default=150)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-targets", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dict-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/online2/sae_pilot/report_sweep_winners.json")
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    report: dict[str, dict] = {}

    for arm_name, spec in ARMS.items():
        build_dir = spec["build_dir"]
        ckpt_path = spec["ckpt"]
        if not ckpt_path.exists():
            print(f"[skip] {arm_name}: no checkpoint at {ckpt_path}")
            continue

        training_events = build_dir / "training_events_v2.parquet"
        events_path = build_dir / "events_tokenized_v2.parquet"
        vocab_path = build_dir / "life2vec_token_registry_v2.json"
        abspos_path = build_dir / "abspos_reference.json"

        print(f"[{arm_name}] selecting narrative-template target events from {training_events} ...")
        targets = find_narrative_targets(training_events)
        sampled = stratified_sample_targets(targets, per_narrative_cap=args.per_narrative_cap, seed=args.seed)
        print(f"  {len(sampled)} target events (cap={args.per_narrative_cap} x {spec['n_narratives']} narratives)")

        vocab = RegistryVocabulary(registry_path=str(vocab_path), registry_version="v2")
        abspos_reference = load_abspos_reference(abspos_path)

        farm_ids = {t.farm_id for t in sampled}
        events_df = load_events_frame(events_path, farm_ids)
        farm_events = build_farm_event_lists(events_df, farm_ids)
        event_id_to_local = {
            ev.event_id: (farm_id, idx) for farm_id, events in farm_events.items() for idx, ev in enumerate(events)
        }

        print(f"[{arm_name}] loading {ckpt_path} ...")
        model, hparams, _ = load_frozen_encoder(ckpt_path, vocab, device)

        rows: list[dict] = []
        pending_targets, pending_xs, pending_masks, pending_spans = [], [], [], []

        def flush() -> None:
            if not pending_targets:
                return
            per_position = encode_all_positions(model, pending_xs, pending_masks, pending_spans, device)
            for b, target in enumerate(pending_targets):
                for position in POSITIONS:
                    mean_v, _ = per_position[position][b]
                    rows.append({"event_id": target.event_id, "position": position, "event_mean": mean_v.tolist()})
            pending_targets.clear(); pending_xs.clear(); pending_masks.clear(); pending_spans.clear()

        skipped = 0
        for i, target in enumerate(sampled):
            located = event_id_to_local.get(target.event_id)
            if located is None:
                skipped += 1
                continue
            farm_id, local_idx = located
            events = farm_events[farm_id]
            window = construct_target_window(events, local_idx, max_length=args.max_length)
            case_t0 = events[0].timestamp
            x, mask, _ = window_to_tensors(
                events, window, vocab=vocab, abspos_reference=abspos_reference,
                case_t0=case_t0, max_length=args.max_length,
            )
            pending_targets.append(target)
            pending_xs.append(x)
            pending_masks.append(mask)
            pending_spans.append((window.target_token_start, window.target_token_end))
            if len(pending_targets) >= args.batch_targets:
                flush()
            if (i + 1) % 4000 == 0:
                print(f"  {arm_name}: encoded {i + 1}/{len(sampled)}")
        flush()
        print(f"  {arm_name}: done, {len(rows)} rows, skipped {skipped}")

        df = pd.DataFrame(rows)
        report[arm_name] = {"hp": spec["hp"], "n_narratives": spec["n_narratives"], "positions": {}}
        for position in POSITIONS:
            sub = df[df["position"] == position]
            means = np.array(sub["event_mean"].tolist(), dtype=np.float32)
            rank = effective_rank(means)
            sae_report = train_and_evaluate_sae(
                means, dict_size=args.dict_size, top_k=args.top_k, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            )
            report[arm_name]["positions"][position] = {
                "n": int(len(sub)), "effective_rank": rank, "sae": sae_report,
            }
            print(
                f"  {arm_name:28s} {position:14s} n={len(sub)} rank99={rank['pc_for_99pct']:3d} "
                f"EV={sae_report['explained_variance']:.3f} dead={sae_report['dead_feature_ratio']:.3f}"
            )

        del model
        torch.cuda.empty_cache()

    full_report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_narrative_cap": args.per_narrative_cap,
        "dict_size": args.dict_size,
        "top_k": args.top_k,
        "epochs": args.epochs,
        "positions": POSITIONS,
        "results": report,
    }
    args.out.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
