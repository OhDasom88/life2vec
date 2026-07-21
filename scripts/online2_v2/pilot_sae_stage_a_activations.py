#!/usr/bin/env python3
"""§9 SAE pilot을 실제 Stage-A 캐시 activation(event_mean, 384차원)으로 돌린다.

계획서 §9.2가 요구하는 3개 학습 지점 중 "사전학습/미세조정 encoder 최종층"에
해당한다 — ``scripts/online2_v2/cache_stage_a_event_embeddings.py``가 이미
캐시해 둔 ``outputs/online2/v2_finetune_v02/event_embeddings/*.parquet``의
``event_mean``을 그대로 쓴다(새로 activation을 계산하지 않는다).

실행: `conda run -n life2vec python scripts/online2_v2/pilot_sae_stage_a_activations.py`
GPU 불필요(384차원, 소규모 dictionary라 CPU로 충분).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.evaluation import (
    compute_activation_frequency,
    reconstruction_metrics,
    sparsity_metrics,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.model import (
    SparseAutoencoder,
    sae_loss,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.training import (
    resample_dead_features,
)


def load_activations(event_embeddings_dir: Path, *, max_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    files = sorted(event_embeddings_dir.glob("*.parquet"))
    per_file = max(1, max_rows // len(files))
    chunks = []
    for path in files:
        table = pq.read_table(path, columns=["event_mean"])
        vectors = np.array(table.column("event_mean").to_pylist(), dtype=np.float32)
        if len(vectors) > per_file:
            idx = rng.choice(len(vectors), size=per_file, replace=False)
            vectors = vectors[idx]
        chunks.append(vectors)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event-embeddings-dir",
        type=Path,
        default=Path("outputs/online2/v2_finetune_v02/event_embeddings"),
    )
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--dict-expansion-factor", type=int, default=8)
    parser.add_argument("--sparsity-mode", choices=["L1", "TOPK"], default="TOPK")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--l1-coefficient", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/online2/sae_pilot/report.json")
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] loading up to {args.max_rows} activations from {args.event_embeddings_dir} ...")
    activations = load_activations(args.event_embeddings_dir, max_rows=args.max_rows, seed=args.seed)
    print(f"  loaded {activations.shape[0]} vectors, dim={activations.shape[1]}")

    torch.manual_seed(args.seed)
    x_all = torch.from_numpy(activations)
    n_val = max(1, int(0.1 * len(x_all)))
    x_train, x_val = x_all[:-n_val], x_all[-n_val:]

    input_dim = x_all.shape[1]
    dict_size = input_dim * args.dict_expansion_factor
    model = SparseAutoencoder(
        input_dim=input_dim,
        dict_size=dict_size,
        sparsity_mode=args.sparsity_mode,
        top_k=args.top_k if args.sparsity_mode == "TOPK" else None,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"[2/3] training SAE: input_dim={input_dim} dict_size={dict_size} "
        f"sparsity_mode={args.sparsity_mode} epochs={args.epochs}"
    )
    n_train = x_train.shape[0]
    resampled_total = 0
    for epoch in range(args.epochs):
        permutation = torch.randperm(n_train)
        epoch_recon_loss = 0.0
        n_batches = 0
        running_frequency = torch.zeros(dict_size)
        for start in range(0, n_train, args.batch_size):
            batch_idx = permutation[start : start + args.batch_size]
            batch = x_train[batch_idx]
            optimizer.zero_grad()
            reconstruction, codes = model(batch)
            loss = sae_loss(
                batch, reconstruction, codes, sparsity_mode=args.sparsity_mode, l1_coefficient=args.l1_coefficient
            )
            loss.total.backward()
            optimizer.step()
            epoch_recon_loss += loss.reconstruction_loss
            n_batches += 1
            running_frequency += compute_activation_frequency(codes.detach())
        running_frequency /= n_batches
        if epoch >= 2:  # 처음 몇 epoch은 아직 안정화 전이라 resample을 미룸
            resampled = resample_dead_features(model, running_frequency, threshold=0.0)
            resampled_total += resampled
        print(f"  epoch {epoch+1}/{args.epochs} mean_recon_loss={epoch_recon_loss/n_batches:.5f}")

    print("[3/3] evaluating on held-out slice ...")
    with torch.no_grad():
        val_reconstruction, val_codes = model(x_val)
    recon_metrics = reconstruction_metrics(x_val, val_reconstruction)
    sparse_metrics = sparsity_metrics(val_codes)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dim": input_dim,
        "dict_size": dict_size,
        "sparsity_mode": args.sparsity_mode,
        "top_k": args.top_k if args.sparsity_mode == "TOPK" else None,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "epochs": args.epochs,
        "total_dead_features_resampled_across_training": resampled_total,
        "held_out_reconstruction_metrics": recon_metrics,
        "held_out_sparsity_metrics": sparse_metrics,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
