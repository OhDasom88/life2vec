#!/usr/bin/env python3
"""pipeline_explorer의 '개념/시퀀스 유사도' 탭이 매번 재학습하지 않고 바로 불러 쓸
수 있도록, 몇 개 (run, position) 조합에 대해 SAE를 학습해 state_dict로 저장한다.

이 세션 내내 SAE는 분석 스크립트 안에서 학습하고 버리는 방식으로만 써 왔다
(재현 가능한 실험을 위해서였다). 하지만 UI에서 사용자가 고른 임의의 두 시퀀스에
대해 "이 SAE 관점에서 활성 패턴이 얼마나 비슷한가"를 즉석에서 보여주려면, 그
SAE 자체가 디스크에 남아 있어야 한다 — 그래서 이번엔 예외적으로 모델을 저장한다.
"""

from __future__ import annotations

import argparse
import sys
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
from pilot_sae_layer_decoder_random_comparison import encode_all_positions  # noqa: E402
from pilot_sae_narrative_selected_activations import (  # noqa: E402
    build_farm_event_lists,
    find_narrative_targets,
    stratified_sample_targets,
)
from src.data_new.vocabulary import RegistryVocabulary  # noqa: E402
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.evaluation import (  # noqa: E402
    compute_activation_frequency,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.model import (  # noqa: E402
    SparseAutoencoder,
    sae_loss,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.training import (  # noqa: E402
    resample_dead_features,
)

sys.path.insert(0, str(ROOT / "src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/pipeline_explorer"))
from data_access import checkpoint_path_for_run  # noqa: E402

DEFAULT_COMBOS = [
    ("original", "layer5"),
    ("control", "layer5"),
    ("sop_balanced", "layer5"),
]


def train_sae(means: np.ndarray, *, dict_size: int, top_k: int, epochs: int, batch_size: int, lr: float, seed: int) -> SparseAutoencoder:
    torch.manual_seed(seed)
    x_all = torch.from_numpy(means)
    sae = SparseAutoencoder(input_dim=x_all.shape[1], dict_size=dict_size, sparsity_mode="TOPK", top_k=top_k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    n = x_all.shape[0]
    for epoch in range(epochs):
        permutation = torch.randperm(n)
        running_frequency = torch.zeros(dict_size)
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            batch = x_all[idx]
            optimizer.zero_grad()
            reconstruction, codes = sae(batch)
            loss = sae_loss(batch, reconstruction, codes, sparsity_mode="TOPK")
            loss.total.backward()
            optimizer.step()
            running_frequency += compute_activation_frequency(codes.detach())
            n_batches += 1
        running_frequency /= max(n_batches, 1)
        if epoch >= 2:
            resample_dead_features(sae, running_frequency, threshold=0.0)
    return sae


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
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs/online2/sae_pilot/pipeline_explorer_saes"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    for run_name, position in DEFAULT_COMBOS:
        ckpt_path = checkpoint_path_for_run(run_name)
        if not ckpt_path.exists():
            print(f"[skip] {run_name}: no checkpoint at {ckpt_path}")
            continue
        print(f"[{run_name}/{position}] loading {ckpt_path} ...")
        model, hparams, _ = load_frozen_encoder(ckpt_path, vocab, device)

        rows: list[np.ndarray] = []
        pending_xs, pending_masks, pending_spans = [], [], []

        def flush() -> None:
            if not pending_xs:
                return
            per_position = encode_all_positions(model, pending_xs, pending_masks, pending_spans, device)
            for mean_v, _ in per_position[position]:
                rows.append(mean_v)
            pending_xs.clear(); pending_masks.clear(); pending_spans.clear()

        for i, target in enumerate(sampled):
            located = event_id_to_local.get(target.event_id)
            if located is None:
                continue
            farm_id, local_idx = located
            events = farm_events[farm_id]
            window = construct_target_window(events, local_idx, max_length=args.max_length)
            case_t0 = events[0].timestamp
            x, mask, _ = window_to_tensors(
                events, window, vocab=vocab, abspos_reference=abspos_reference,
                case_t0=case_t0, max_length=args.max_length,
            )
            pending_xs.append(x)
            pending_masks.append(mask)
            pending_spans.append((window.target_token_start, window.target_token_end))
            if len(pending_xs) >= args.batch_targets:
                flush()
        flush()

        means = np.array(rows, dtype=np.float32)
        print(f"  training SAE on {means.shape} activations ...")
        sae = train_sae(
            means, dict_size=args.dict_size, top_k=args.top_k, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        )
        out_path = args.out_dir / f"{run_name}__{position}.pt"
        torch.save(
            {
                "state_dict": sae.state_dict(),
                "input_dim": sae.input_dim,
                "dict_size": sae.dict_size,
                "sparsity_mode": sae.sparsity_mode,
                "top_k": sae.top_k,
                "tied_weights": sae.tied_weights,
                "run_name": run_name,
                "position": position,
                "checkpoint_path": str(ckpt_path),
            },
            out_path,
        )
        print(f"  saved {out_path}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
