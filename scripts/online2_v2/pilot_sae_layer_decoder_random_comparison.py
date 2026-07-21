#!/usr/bin/env python3
"""§9.2 SAE pilot: (1) 학습된 인코더 vs 랜덤 초기화 인코더, (2) encoder 최종 layer
이후 MLM/SOP 디코더 헤드까지 표현 다양성이 계속 줄어드는지 비교한다.

지난 layer sweep에서 layer4->layer5(최종)로 갈 때 유효 차원이 줄고
dead_feature_ratio가 늘어드는 걸 확인했다. 이게 "MLM/SOP를 배우면서" 생긴
건지, 아니면 이 architecture/depth 자체의 특성인지 구분하려면 **학습 안 된
같은 구조의 인코더**와 비교해야 한다 — 같은 패턴이 랜덤 초기화에서도 나오면
학습과 무관한 것이고, 학습된 쪽에서만 나오면 MLM/SOP 학습이 원인이라는 뜻이다.

또한 "표현이 최종 encoder layer 이후 예측 헤드로 가면서 더 줄어드는지"를 보려면
실제 MLM 디코더(``MaskedLanguageModel``: V + tanh + l2_norm)와 SOP 디코더
(``CLS_Decoder``: in_layer + swish + ScaleNorm)의 최종 분류 직전 표현까지 같은
방식(PCA 유효 차원, SAE dead ratio)으로 측정한다. 모델 클래스는 그대로 두고
그 submodule(``model.mlm_decoder``, ``model.cls_decoder``)을 직접 호출한다.
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
from src.transformer.transformer_utils import l2_norm  # noqa: E402

POSITIONS = ["layer0", "layer1", "layer2", "layer3", "layer4", "layer5", "mlm_transform", "sop_transform"]


def load_untrained_encoder(ckpt_path: Path, vocab: RegistryVocabulary, device: torch.device):
    """load_frozen_encoder와 같은 hparams로 architecture만 만들고 가중치는 안 불러온다(랜덤 초기화)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = dict(ckpt["hparams"])
    raw.pop("vocabulary", None)
    hparams = {
        **raw,
        "vocabulary": vocab,
        "vocab_size": vocab.size(),
        "training_task": raw.get("training_task", "mlm"),
        "epsilon": float(raw.get("epsilon", 1e-6)),
    }
    model = TransformerEncoder(hparams)  # nn.Module 기본 랜덤 초기화, load_state_dict 생략
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return model, hparams


def _pool_spans(hidden: torch.Tensor, spans: list[tuple[int, int]]) -> list[tuple[np.ndarray, np.ndarray]]:
    pooled = []
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
    return pooled


@torch.no_grad()
def encode_all_positions(
    model: TransformerEncoder,
    xs: list[torch.Tensor],
    masks: list[torch.Tensor],
    spans: list[tuple[int, int]],
    device: torch.device,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """encoder 6개 layer + mlm_transform + sop_transform, 총 8개 지점의 pooled activation."""
    x = torch.stack(xs, dim=0).to(device)
    mask = torch.stack(masks, dim=0).to(device).long()

    transformer = model.transformer
    hidden, _ = transformer.embedding(tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3])

    result: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for layer_idx, layer in enumerate(transformer.encoders):
        hidden = torch.einsum("bsh, bs -> bsh", hidden, mask)
        hidden = layer(hidden, mask)
        result[f"layer{layer_idx}"] = _pool_spans(hidden, spans)

    final_hidden = hidden  # 마지막 encoder layer 출력, (B, S, H)

    # MLM 디코더: target span 토큰별로 V+tanh+l2_norm 적용(실제 forward와 동일한 변환) 후 풀링.
    mlm_module = model.mlm_decoder
    mlm_pooled: list[tuple[np.ndarray, np.ndarray]] = []
    for b, (s, e) in enumerate(spans):
        span = final_hidden[b, s:e, :]
        if span.numel() == 0:
            h = final_hidden.shape[-1]
            z = np.zeros(h, dtype=np.float32)
            mlm_pooled.append((z, z))
            continue
        transformed = l2_norm(mlm_module.act(mlm_module.V(span)))
        mean_v = transformed.mean(dim=0).float().cpu().numpy().astype(np.float32)
        max_v = transformed.max(dim=0).values.float().cpu().numpy().astype(np.float32)
        mlm_pooled.append((mean_v, max_v))
    result["mlm_transform"] = mlm_pooled

    # SOP/CLS 디코더: CLS 토큰(position 0)에 in_layer+swish+ScaleNorm 적용(최종 분류 직전).
    cls_module = model.cls_decoder
    cls_hidden = final_hidden[:, 0, :]
    cls_transformed = cls_module.norm(cls_module.act(cls_module.in_layer(cls_hidden)))
    result["sop_transform"] = [
        (
            v.float().cpu().numpy().astype(np.float32),
            v.float().cpu().numpy().astype(np.float32),
        )
        for v in cls_transformed
    ]

    return result


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
    return {"total_dead_features_resampled": resampled_total, **recon, **sparse}


def effective_rank(vecs: np.ndarray, *, thresholds=(0.5, 0.9, 0.99)) -> dict[str, int]:
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
        "--activation-cache-dir", type=Path, default=ROOT / "outputs/online2/sae_pilot/layer_decoder_random"
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
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/online2/sae_pilot/report_layer_decoder_random.json")
    )
    args = parser.parse_args()
    args.activation_cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    need_setup = any(
        not (args.reuse_cache and (args.activation_cache_dir / f"{mk}.parquet").exists())
        for mk in ["trained", "random_init"]
    )
    sampled = vocab = device = abspos_reference = farm_events = event_id_to_local = None
    if need_setup:
        print("selecting narrative-template target events ...")
        targets = find_narrative_targets(args.training_events)
        sampled = stratified_sample_targets(targets, per_narrative_cap=args.per_narrative_cap, seed=args.seed)
        print(f"  {len(sampled)} target events across up to 80 narrative templates")

        vocab = RegistryVocabulary(registry_path=str(args.vocab), registry_version="v2")
        device = torch.device(args.device)
        abspos_reference = load_abspos_reference(ROOT / "outputs/online2/v2_build/abspos_reference.json")

        farm_ids = {t.farm_id for t in sampled}
        events_df = load_events_frame(args.events, farm_ids)
        farm_events = build_farm_event_lists(events_df, farm_ids)
        event_id_to_local = {
            ev.event_id: (farm_id, idx)
            for farm_id, events in farm_events.items()
            for idx, ev in enumerate(events)
        }

    for model_kind in ["trained", "random_init"]:
        cache_path = args.activation_cache_dir / f"{model_kind}.parquet"
        if args.reuse_cache and cache_path.exists():
            print(f"[cache] reusing {cache_path}")
            dfs[model_kind] = pq.read_table(cache_path).to_pandas()
            continue

        print(f"[{model_kind}] loading encoder ...")
        if model_kind == "trained":
            model, hparams, _ = load_frozen_encoder(args.ckpt, vocab, device)
        else:
            model, hparams = load_untrained_encoder(args.ckpt, vocab, device)

        print(f"[{model_kind}] encoding {len(sampled)} targets across {len(POSITIONS)} positions ...")
        rows: list[dict] = []
        pending_targets, pending_xs, pending_masks, pending_spans = [], [], [], []

        def flush() -> None:
            if not pending_targets:
                return
            per_position = encode_all_positions(model, pending_xs, pending_masks, pending_spans, device)
            for b, target in enumerate(pending_targets):
                for position in POSITIONS:
                    mean_v, _ = per_position[position][b]
                    rows.append(
                        {
                            "event_id": target.event_id,
                            "farm_id": target.farm_id,
                            "narrative_id": target.narrative_id,
                            "position": position,
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
            if (i + 1) % 4000 == 0:
                print(f"  {model_kind}: encoded {i + 1}/{len(sampled)} (skipped {skipped})")
        flush()
        print(f"  {model_kind}: done, {len(rows)} rows, skipped {skipped}")

        df = pd.DataFrame(rows)
        df.to_parquet(cache_path, index=False)
        dfs[model_kind] = df

    print("comparing trained vs random_init across all positions ...")
    report: dict[str, dict] = {}
    for model_kind, df in dfs.items():
        report[model_kind] = {}
        for position in POSITIONS:
            sub = df[df["position"] == position]
            means = np.array(sub["event_mean"].tolist(), dtype=np.float32)
            rank = effective_rank(means)
            sae_report = train_and_evaluate_sae(
                means, dict_size=args.dict_size, top_k=args.top_k, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            )
            report[model_kind][position] = {"n": int(len(sub)), "effective_rank": rank, "sae": sae_report}
            print(
                f"  {model_kind:12s} {position:14s} n={len(sub)} rank99={rank['pc_for_99pct']:3d} "
                f"EV={sae_report['explained_variance']:.3f} dead={sae_report['dead_feature_ratio']:.3f}"
            )

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
