#!/usr/bin/env python3
"""§9 SAE pilot을 "서사(narrative) 템플릿이 실제로 선별한 이벤트"만으로 다시 돌린다.

이전 pilot(``pilot_sae_stage_a_activations.py``)은 CF1S 55-case가 밀집 캡처한
activation(그 농장 14일 전체, 평온한 구간까지 전부)을 썼다. 이번엔 같은 원시
이벤트 풀(55농장 x 14일, 222,309건 — 새 데이터 아님, 동일 원본)에서 **80개
narrative 템플릿이 실제로 매칭한 "타깃" 이벤트만** 골라 activation을 뽑는다.
같은 인코더·같은 원시 데이터, 표본 구성 방식만 다르게 해서 "서사 기반 선별이
SAE 다양성에 영향을 주는가"를 정확히 검증한다.

"타깃" 이벤트 정의: online2 builder.py 관례와 동일하게, 한 narrative 시퀀스의
마지막(=event_position이 가장 큰) 이벤트를 그 시퀀스가 실제로 "가리키는" 사건으로
본다(``sequence_v2 narrative_center`` 관례). ``training_events_v2.parquet``
(15,016,770행)에서 sequence_id별 최댓값을 청크 단위로 스트리밍 집계해
전체를 메모리에 올리지 않는다.

activation 계산 자체는 새로 만들지 않는다 — ``cache_stage_a_event_embeddings.py``의
``load_frozen_encoder``/``construct_target_window``/``window_to_tensors``/
``encode_batch_pool``을 그대로 import해서 쓴다. 그 다음 ``sae/`` 패키지의
``SparseAutoencoder``로 동일한 방식으로 학습·평가한다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
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
    CaseEvent,
    construct_target_window,
    encode_batch_pool,
    load_abspos_reference,
    load_events_frame,
    load_frozen_encoder,
    parse_view,
    sentence_to_tokens,
    window_to_tensors,
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


@dataclass(frozen=True)
class TargetRef:
    event_id: str
    farm_id: str
    narrative_id: str


def find_narrative_targets(training_events_path: Path, *, batch_size: int = 500_000) -> list[TargetRef]:
    """sequence_id별 event_position 최댓값 = 그 narrative 시퀀스의 "타깃" 이벤트.

    15M행을 한 번에 메모리에 올리지 않고 parquet batch 단위로 스트리밍
    집계한다(이전에 전체를 한 번에 읽다가 OOM으로 프로세스가 죽은 적 있음).
    """
    pf = pq.ParquetFile(training_events_path)
    best_position: dict[str, int] = {}
    best_ref: dict[str, TargetRef] = {}

    columns = ["sequence_id", "event_id", "event_position", "narrative_id", "farm_id"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        for row in df.itertuples(index=False):
            seq_id = row.sequence_id
            pos = int(row.event_position)
            if pos > best_position.get(seq_id, -1):
                best_position[seq_id] = pos
                best_ref[seq_id] = TargetRef(
                    event_id=row.event_id, farm_id=row.farm_id, narrative_id=row.narrative_id
                )

    return list(best_ref.values())


def stratified_sample_targets(
    targets: list[TargetRef], *, per_narrative_cap: int, seed: int
) -> list[TargetRef]:
    """narrative_id별로 최대 per_narrative_cap개, event_id 기준 중복 제거 후 균등 샘플."""
    by_event_id: dict[str, TargetRef] = {}
    for t in targets:
        by_event_id.setdefault(t.event_id, t)  # 같은 이벤트가 여러 시퀀스의 타깃이면 첫 것만

    by_narrative: dict[str, list[TargetRef]] = defaultdict(list)
    for t in by_event_id.values():
        by_narrative[t.narrative_id].append(t)

    rng = random.Random(seed)
    sampled: list[TargetRef] = []
    for narrative_id, group in by_narrative.items():
        rng.shuffle(group)
        sampled.extend(group[:per_narrative_cap])
    return sampled


def build_farm_event_lists(
    events_df: pd.DataFrame, farm_ids: set[str], *, include_image: bool = False
) -> dict[str, list[CaseEvent]]:
    """farm_id -> 그 농장 전체 기간(약 14일)을 시간순 정렬한 CaseEvent 리스트.

    cache_stage_a_event_embeddings.case_events_from_frame과 같은 원칙(서사/이미지
    제외, same_time_group/zone/view 순 정렬)을 따르되 날짜 범위 인자가 필요
    없다 - 이 저장소의 모든 농장은 events_tokenized_v2.parquet에 정확히 한
    번의 ~14일 구간만 갖고 있기 때문(실측 확인).
    """
    exclude = {"INTERPRETATION"}
    if not include_image:
        exclude.add("IMAGE")

    result: dict[str, list[CaseEvent]] = {}
    for farm_id in farm_ids:
        df = events_df[events_df["farm_id"].astype(str) == str(farm_id)].copy()
        df = df[~df["event_kind"].astype(str).isin(exclude)].copy()
        df["view_parsed"] = [parse_view(s, k) for s, k in zip(df["SENTENCE"], df["event_kind"])]
        df["zone_rank"] = pd.to_numeric(df["zone_id"], errors="coerce").fillna(999)
        view_order = {"ENVIRONMENT": 0, "ROOTZONE": 1, "ACTUATOR": 2, "GROWTH": 3}
        df["view_rank"] = df["view_parsed"].map(lambda v: view_order.get(v, 99))
        # itertuples는 언더스코어로 시작하는 컬럼명을 속성으로 못 읽으므로
        # (load_events_frame이 만든 "_ts") 정렬 전에 이름을 바꾼다.
        df = df.rename(columns={"_ts": "ts"})
        df = df.sort_values(
            ["ts", "same_time_group_id", "zone_rank", "view_rank", "event_id"]
        ).reset_index(drop=True)

        events: list[CaseEvent] = []
        for order, row in enumerate(df.itertuples(index=False)):
            toks = sentence_to_tokens(row.SENTENCE)
            events.append(
                CaseEvent(
                    event_id=row.event_id,
                    event_order=order,
                    same_time_group_id=row.same_time_group_id,
                    timestamp=row.ts,
                    view=row.view_parsed,
                    zone=str(row.zone_id),
                    event_kind=row.event_kind,
                    sentence_tokens=toks,
                    token_count=len(toks),
                )
            )
        result[farm_id] = events
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-events",
        type=Path,
        default=ROOT / "outputs/online2/v2_build/training_events_v2.parquet",
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
        default=ROOT / "outputs/online2/sae_pilot/narrative_selected_activations.parquet",
        help="한 번 뽑아두면 재사용(추출은 인코더 순전파가 필요해 SAE 실험보다 훨씬 느림).",
    )
    parser.add_argument("--reuse-cache", action="store_true", help="있으면 추출을 건너뛰고 캐시를 그대로 씀")
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
        "--out", type=Path, default=Path("outputs/online2/sae_pilot/report_narrative_selected.json")
    )
    args = parser.parse_args()
    args.activation_cache.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_cache and args.activation_cache.exists():
        print(f"[1/4] reusing cached activations at {args.activation_cache}")
        table = pq.read_table(args.activation_cache)
        means = np.array(table.column("event_mean").to_pylist(), dtype=np.float32)
        narrative_ids = table.column("narrative_id").to_pylist()
    else:
        print("[1/4] finding narrative-template target events (streamed over 15M rows) ...")
        targets = find_narrative_targets(args.training_events)
        print(f"  distinct sequences with a target: {len(targets)}")

        sampled = stratified_sample_targets(targets, per_narrative_cap=args.per_narrative_cap, seed=args.seed)
        print(f"  stratified sample: {len(sampled)} target events across up to 80 narrative templates")

        print("[2/4] loading frozen encoder + per-farm chronological event lists ...")
        vocab = RegistryVocabulary(registry_path=str(args.vocab), registry_version="v2")
        device = torch.device(args.device)
        abspos_reference = load_abspos_reference(
            ROOT / "outputs/online2/v2_build/abspos_reference.json"
        )
        model, hparams, _ = load_frozen_encoder(args.ckpt, vocab, device)

        farm_ids = {t.farm_id for t in sampled}
        events_df = load_events_frame(args.events, farm_ids)
        farm_events = build_farm_event_lists(events_df, farm_ids)

        event_id_to_local: dict[str, tuple[str, int]] = {}
        for farm_id, events in farm_events.items():
            for idx, ev in enumerate(events):
                event_id_to_local[ev.event_id] = (farm_id, idx)

        print("[3/4] encoding target windows through frozen encoder ...")
        rows: list[dict] = []
        pending_targets: list[TargetRef] = []
        pending_xs: list[torch.Tensor] = []
        pending_masks: list[torch.Tensor] = []
        pending_spans: list[tuple[int, int]] = []

        def flush() -> None:
            if not pending_targets:
                return
            pooled = encode_batch_pool(model, pending_xs, pending_masks, pending_spans, device)
            for target, (mean_v, max_v) in zip(pending_targets, pooled):
                rows.append(
                    {
                        "event_id": target.event_id,
                        "farm_id": target.farm_id,
                        "narrative_id": target.narrative_id,
                        "event_mean": mean_v.tolist(),
                        "event_max": max_v.tolist(),
                    }
                )
            pending_targets.clear()
            pending_xs.clear()
            pending_masks.clear()
            pending_spans.clear()

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
                events,
                window,
                vocab=vocab,
                abspos_reference=abspos_reference,
                case_t0=case_t0,
                max_length=args.max_length,
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
        print(f"  done. encoded {len(rows)} targets, skipped {skipped} (event_id not found in farm stream)")

        out_table = pd.DataFrame(rows)
        out_table.to_parquet(args.activation_cache, index=False)
        print(f"  wrote {args.activation_cache}")

        means = np.array([r["event_mean"] for r in rows], dtype=np.float32)
        narrative_ids = [r["narrative_id"] for r in rows]

    print(f"[4/4] training SAE on {means.shape[0]} narrative-selected activations, dim={means.shape[1]}")
    torch.manual_seed(args.seed)
    x_all = torch.from_numpy(means)
    n_val = max(1, int(0.1 * len(x_all)))
    x_train, x_val = x_all[:-n_val], x_all[-n_val:]

    input_dim = x_all.shape[1]
    sae = SparseAutoencoder(input_dim=input_dim, dict_size=args.dict_size, sparsity_mode="TOPK", top_k=args.top_k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    n_train = x_train.shape[0]
    resampled_total = 0
    for epoch in range(args.epochs):
        permutation = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        running_frequency = torch.zeros(args.dict_size)
        for start in range(0, n_train, args.batch_size):
            idx = permutation[start : start + args.batch_size]
            batch = x_train[idx]
            optimizer.zero_grad()
            reconstruction, codes = sae(batch)
            loss = sae_loss(batch, reconstruction, codes, sparsity_mode="TOPK")
            loss.total.backward()
            optimizer.step()
            epoch_loss += loss.reconstruction_loss
            n_batches += 1
            running_frequency += compute_activation_frequency(codes.detach())
        running_frequency /= max(n_batches, 1)
        if epoch >= 2:
            resampled_total += resample_dead_features(sae, running_frequency, threshold=0.0)
        print(f"  epoch {epoch+1}/{args.epochs} mean_recon_loss={epoch_loss/max(n_batches,1):.5f}")

    with torch.no_grad():
        val_reconstruction, val_codes = sae(x_val)
    recon = reconstruction_metrics(x_val, val_reconstruction)
    sparse = sparsity_metrics(val_codes)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_strategy": "narrative_template_target_events",
        "per_narrative_cap": args.per_narrative_cap,
        "n_total_activations": int(means.shape[0]),
        "n_distinct_narrative_ids": len(set(narrative_ids)),
        "input_dim": input_dim,
        "dict_size": args.dict_size,
        "top_k": args.top_k,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "epochs": args.epochs,
        "total_dead_features_resampled_across_training": resampled_total,
        "held_out_reconstruction_metrics": recon,
        "held_out_sparsity_metrics": sparse,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
