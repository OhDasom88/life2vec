#!/usr/bin/env python3
"""§9.2 SAE pilot을 여러 encoder layer(중간층 포함)의 activation으로 비교한다.

지금까지의 pilot은 전부 `forward_finetuning`의 **최종 layer** 출력만 썼다.
계획서 §9.2가 원래 요구한 3개 학습 지점 중 "사전학습 encoder 중간층"은 아직
시도한 적이 없었다 — PCA는 선형 구조만 보므로, pooling 이전이나 더 이른
layer에는 마지막 layer가 MLM/SOP 목적함수에 맞춰 눌러버린 것보다 더 풍부한
(혹은 최소한 다른) 구조가 남아있을 가능성이 있다.

`src/transformer/transformer.py`의 `Transformer.forward_finetuning`을 고치지
않고, 그 루프(embedding -> encoder layer 반복)를 그대로 복제해 **매 layer
직후의 pooled activation을 한 번의 순전파로 전부 캡처**한다(layer마다 다시
순전파하지 않음 - 효율적).

타깃 이벤트 선택은 `pilot_sae_narrative_selected_activations.py`의 narrative-
template 기반 선택을 그대로 재사용한다(이전 실험에서 dense와 다양성 차이가
없음을 이미 확인했으니, 여기서는 "어떤 이벤트를 뽑을까"가 아니라 "어느
layer를 볼까"만 바꾼다 - 통제된 비교를 위해 같은 타깃 집합을 씀).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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
from pilot_sae_narrative_selected_activations import (  # noqa: E402
    build_farm_event_lists,
    find_narrative_targets,
    stratified_sample_targets,
)
from src.data_new.vocabulary import RegistryVocabulary  # noqa: E402
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.evaluation import (  # noqa: E402
    compute_activation_frequency,
    reconstruction_metrics,
    sparsity_metrics,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.model import (  # noqa: E402
    SparseAutoencoder,
    sae_loss,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.sae.training import (  # noqa: E402
    resample_dead_features,
)
from src.transformer.models import TransformerEncoder  # noqa: E402


@torch.no_grad()
def encode_batch_pool_all_layers(
    model: TransformerEncoder,
    xs: list[torch.Tensor],
    masks: list[torch.Tensor],
    spans: list[tuple[int, int]],
    device: torch.device,
) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """`Transformer.forward_finetuning`과 같은 루프를 복제해, layer마다 직후의
    pooled activation을 전부 캡처한다(단 한 번의 순전파로 모든 layer를 얻음).
    """
    x = torch.stack(xs, dim=0).to(device)  # (B, 4, L)
    mask = torch.stack(masks, dim=0).to(device).long()  # (B, L)

    transformer = model.transformer
    hidden, _ = transformer.embedding(
        tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
    )

    per_layer: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for layer_idx, layer in enumerate(transformer.encoders):
        hidden = torch.einsum("bsh, bs -> bsh", hidden, mask)
        hidden = layer(hidden, mask)
        pooled: list[tuple[np.ndarray, np.ndarray]] = []
        for b, (s, e) in enumerate(spans):
            span = hidden[b, s:e, :]
            if span.numel() == 0:
                h = hidden.shape[-1]
                z = np.zeros(h, dtype=np.float32)
                pooled.append((z, z))
                continue
            mean_v = span.mean(dim=0).float().cpu().numpy().astype(np.float32)
            max_v = span.max(dim=0).values.float().cpu().numpy().astype(np.float32)
            pooled.append((mean_v, max_v))
        per_layer[layer_idx] = pooled
    return per_layer


def train_and_evaluate_sae(
    means: np.ndarray, *, dict_size: int, top_k: int, epochs: int, batch_size: int, lr: float, seed: int
) -> dict:
    torch.manual_seed(seed)
    x_all = torch.from_numpy(means)
    n_val = max(1, int(0.1 * len(x_all)))
    x_train, x_val = x_all[:-n_val], x_all[-n_val:]

    sae = SparseAutoencoder(input_dim=x_all.shape[1], dict_size=dict_size, sparsity_mode="TOPK", top_k=top_k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    n_train = x_train.shape[0]
    resampled_total = 0
    for epoch in range(epochs):
        permutation = torch.randperm(n_train)
        running_frequency = torch.zeros(dict_size)
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = permutation[start : start + batch_size]
            batch = x_train[idx]
            optimizer.zero_grad()
            reconstruction, codes = sae(batch)
            loss = sae_loss(batch, reconstruction, codes, sparsity_mode="TOPK")
            loss.total.backward()
            optimizer.step()
            running_frequency += compute_activation_frequency(codes.detach())
            n_batches += 1
        running_frequency /= max(n_batches, 1)
        if epoch >= 2:
            resampled_total += resample_dead_features(sae, running_frequency, threshold=0.0)

    with torch.no_grad():
        val_reconstruction, val_codes = sae(x_val)
    recon = reconstruction_metrics(x_val, val_reconstruction)
    sparse = sparsity_metrics(val_codes)
    return {
        "n_train": int(n_train),
        "n_val": int(n_val),
        "total_dead_features_resampled_across_training": resampled_total,
        **recon,
        **sparse,
    }


def effective_rank(vecs: np.ndarray, *, thresholds=(0.5, 0.8, 0.9, 0.95, 0.99)) -> dict[str, int]:
    centered = vecs - vecs.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    var_ratio = (s**2) / (s**2).sum()
    cum = np.cumsum(var_ratio)
    return {f"pc_for_{int(t*100)}pct": int(np.searchsorted(cum, t)) + 1 for t in thresholds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-events", type=Path, default=ROOT / "outputs/online2/v2_build/training_events_v2.parquet"
    )
    parser.add_argument(
        "--events", type=Path, default=ROOT / "outputs/online2/v2_build/events_tokenized_v2.parquet"
    )
    parser.add_argument(
        "--ckpt", type=Path, default=ROOT / "outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt"
    )
    parser.add_argument(
        "--vocab", type=Path, default=ROOT / "outputs/online2/v2_build/life2vec_token_registry_v2.json"
    )
    parser.add_argument(
        "--activation-cache",
        type=Path,
        default=ROOT / "outputs/online2/sae_pilot/layer_sweep_activations.parquet",
    )
    parser.add_argument("--reuse-cache", action="store_true")
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
    parser.add_argument("--out", type=Path, default=Path("outputs/online2/sae_pilot/report_layer_sweep.json"))
    args = parser.parse_args()
    args.activation_cache.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_cache and args.activation_cache.exists():
        print(f"[1/3] reusing cached per-layer activations at {args.activation_cache}")
        table = pq.read_table(args.activation_cache)
        df = table.to_pandas()
        n_layers = df["layer_idx"].nunique()
    else:
        print("[1/3] selecting narrative-template target events (reusing prior selection logic) ...")
        targets = find_narrative_targets(args.training_events)
        sampled = stratified_sample_targets(targets, per_narrative_cap=args.per_narrative_cap, seed=args.seed)
        print(f"  {len(sampled)} target events across up to 80 narrative templates")

        vocab = RegistryVocabulary(registry_path=str(args.vocab), registry_version="v2")
        device = torch.device(args.device)
        abspos_reference = load_abspos_reference(ROOT / "outputs/online2/v2_build/abspos_reference.json")
        model, hparams, _ = load_frozen_encoder(args.ckpt, vocab, device)
        n_layers = int(hparams["n_encoders"])
        print(f"  encoder has {n_layers} layers; capturing pooled activation after every layer")

        farm_ids = {t.farm_id for t in sampled}
        events_df = load_events_frame(args.events, farm_ids)
        farm_events = build_farm_event_lists(events_df, farm_ids)
        event_id_to_local = {
            ev.event_id: (farm_id, idx)
            for farm_id, events in farm_events.items()
            for idx, ev in enumerate(events)
        }

        print("[2/3] encoding target windows, capturing all layers per forward pass ...")
        rows: list[dict] = []
        pending_targets, pending_xs, pending_masks, pending_spans = [], [], [], []

        def flush() -> None:
            if not pending_targets:
                return
            per_layer = encode_batch_pool_all_layers(model, pending_xs, pending_masks, pending_spans, device)
            for b, target in enumerate(pending_targets):
                for layer_idx in range(n_layers):
                    mean_v, max_v = per_layer[layer_idx][b]
                    rows.append(
                        {
                            "event_id": target.event_id,
                            "farm_id": target.farm_id,
                            "narrative_id": target.narrative_id,
                            "layer_idx": layer_idx,
                            "event_mean": mean_v.tolist(),
                        }
                    )
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
            if (i + 1) % 2000 == 0:
                print(f"  encoded {i + 1}/{len(sampled)} (skipped so far: {skipped})")
        flush()
        print(f"  done. {len(rows)} rows ({len(rows)//n_layers} targets x {n_layers} layers), skipped {skipped}")

        df = pd.DataFrame(rows)
        df.to_parquet(args.activation_cache, index=False)
        print(f"  wrote {args.activation_cache}")

    print(f"[3/3] per-layer PCA effective rank + SAE (dict={args.dict_size}, top_k={args.top_k}) ...")
    per_layer_reports = {}
    for layer_idx in sorted(df["layer_idx"].unique()):
        sub = df[df["layer_idx"] == layer_idx]
        means = np.array(sub["event_mean"].tolist(), dtype=np.float32)
        rank = effective_rank(means)
        sae_report = train_and_evaluate_sae(
            means, dict_size=args.dict_size, top_k=args.top_k, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        )
        per_layer_reports[int(layer_idx)] = {"n": int(len(sub)), "effective_rank": rank, "sae": sae_report}
        print(
            f"  layer {layer_idx}: n={len(sub)} rank99={rank['pc_for_99pct']} "
            f"EV={sae_report['explained_variance']:.3f} dead={sae_report['dead_feature_ratio']:.3f}"
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_narrative_cap": args.per_narrative_cap,
        "dict_size": args.dict_size,
        "top_k": args.top_k,
        "epochs": args.epochs,
        "per_layer": per_layer_reports,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
