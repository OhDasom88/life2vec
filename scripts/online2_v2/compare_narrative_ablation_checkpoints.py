#!/usr/bin/env python3
"""SOP 재조정 + 서사 다양성 ablation으로 새로 학습한 6개 체크포인트를 비교한다.

`run_narrative_ablation_pretrain_sweep.sh`가 만든 6개 체크포인트
(control/sop_balanced/narrow_subset/single_table_only/multi_table_only/
dedup_reduced, 전부 5000 step·batch=40·max_length=1024로 동일 조건) 각각에
대해, 같은 14,544개 narrative-selected target으로 최종 encoder layer +
MLM/SOP 디코더 변환의 PCA 유효 차원과 SAE dead_feature_ratio를 측정해
한 표로 비교한다. `pilot_sae_layer_decoder_random_comparison.py`의
`encode_all_positions`/`train_and_evaluate_sae`/`effective_rank`를 그대로
재사용한다(계산 로직 중복 없음) — 이번엔 6개 layer 전부가 아니라
final layer + 두 디코더 변환 3개 지점만 본다(6개 체크포인트 x 6개 layer는
과함, 앞선 layer sweep에서 이미 "최종 layer가 최악"이라는 걸 확인했으므로
여기서는 그 최종 layer가 조건마다 어떻게 달라지는지에 집중).
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

# Use the step-5000 snapshot for every arm, not best.ckpt: best.ckpt tracks
# lowest val loss (checkpoint_every=100), which lands at a different step per
# arm (observed: control=4200, sop_balanced=5000, narrow_subset=5000,
# single_table_only=5500 (resumed with additive --steps, see below),
# multi_table_only=4900, dedup_reduced=4800). Comparing representational
# diversity across a different number of training steps would confound
# "narrative content" with "how much training happened" — checkpoint_step_5000.pt
# exists for all 6 arms and pins every comparison to an identical step budget.
# (single_table_only was interrupted mid-run and resumed; --steps is additive on
# resume in run_v2_pretrain_loop.py:687 `while step < start_step + steps`, so its
# best/last.ckpt reflect 6200 total steps — checkpoint_step_5000.pt sidesteps that.)
DEFAULT_ARMS = {
    "control": "outputs/online2/v2_runs/narrative_ablation/control/checkpoint_step_5000.pt",
    "sop_balanced": "outputs/online2/v2_runs/narrative_ablation/sop_balanced/checkpoint_step_5000.pt",
    "narrow_subset": "outputs/online2/v2_runs/narrative_ablation/narrow_subset/checkpoint_step_5000.pt",
    "single_table_only": "outputs/online2/v2_runs/narrative_ablation/single_table_only/checkpoint_step_5000.pt",
    "multi_table_only": "outputs/online2/v2_runs/narrative_ablation/multi_table_only/checkpoint_step_5000.pt",
    "dedup_reduced": "outputs/online2/v2_runs/narrative_ablation/dedup_reduced/checkpoint_step_5000.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-events", type=Path, default=ROOT / "outputs/online2/v2_build/training_events_v2.parquet"
    )
    parser.add_argument(
        "--events", type=Path, default=ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet"
    )
    parser.add_argument(
        "--vocab", type=Path, default=ROOT / "outputs/online2/v2_build/life2vec_token_registry_v2.json"
    )
    parser.add_argument("--per-narrative-cap", type=int, default=250)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-targets", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dict-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-last-ckpt-if-missing", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/online2/sae_pilot/report_narrative_ablation.json")
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("selecting narrative-template target events ...")
    targets = find_narrative_targets(args.training_events)
    sampled = stratified_sample_targets(targets, per_narrative_cap=args.per_narrative_cap, seed=args.seed)
    print(f"  {len(sampled)} target events")

    vocab = RegistryVocabulary(registry_path=str(args.vocab), registry_version="v2")
    device = torch.device(args.device)
    abspos_reference = load_abspos_reference(ROOT / "outputs/online2/v2_build/abspos_reference.json")

    farm_ids = {t.farm_id for t in sampled}
    events_df = load_events_frame(args.events, farm_ids)
    farm_events = build_farm_event_lists(events_df, farm_ids)
    event_id_to_local = {
        ev.event_id: (farm_id, idx) for farm_id, events in farm_events.items() for idx, ev in enumerate(events)
    }

    report: dict[str, dict] = {}
    for arm_name, ckpt_rel in DEFAULT_ARMS.items():
        ckpt_path = ROOT / ckpt_rel
        if not ckpt_path.exists():
            if args.use_last_ckpt_if_missing:
                ckpt_path = ckpt_path.with_name("last.ckpt")
            if not ckpt_path.exists():
                print(f"[skip] {arm_name}: no checkpoint at {ckpt_path}")
                continue

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
        report[arm_name] = {}
        for position in POSITIONS:
            sub = df[df["position"] == position]
            means = np.array(sub["event_mean"].tolist(), dtype=np.float32)
            rank = effective_rank(means)
            sae_report = train_and_evaluate_sae(
                means, dict_size=args.dict_size, top_k=args.top_k, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            )
            report[arm_name][position] = {"n": int(len(sub)), "effective_rank": rank, "sae": sae_report}
            print(
                f"  {arm_name:20s} {position:14s} n={len(sub)} rank99={rank['pc_for_99pct']:3d} "
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
