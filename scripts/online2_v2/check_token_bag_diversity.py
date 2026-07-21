#!/usr/bin/env python3
"""랜덤 초기화 인코더의 유효 차원 상한(rank99=118)이 임베딩 테이블이 아니라
토큰 co-occurrence 다양성 자체에서 온다는 걸 확인하는 대조 실험.

학습 가능한 파라미터를 전혀 쓰지 않고, target span에 등장한 토큰 id의
등장 빈도만으로 만든 bag-of-tokens(정규화된 one-hot 평균) 벡터의 PCA 유효
차원을 측정한다. 이게 랜덤 초기화 인코더의 유효 차원과 비슷하게 나오면,
"임베딩이 랜덤이라 다양성이 낮다"가 아니라 "애초에 어떤 토큰들이 같이
등장하는지의 다양성 자체가 낮다"는 뜻이다 — `Embeddings.forward`의
position/age/segment ReZero 게이트가 학습 전엔 정확히 0이라(위치/시간 정보가
아예 안 들어감) 랜덤 초기화 인코더가 보는 건 결국 이 co-occurrence 구조뿐이기
때문이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts" / "online2_v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "online2_v2"))

from cache_stage_a_event_embeddings import (  # noqa: E402
    construct_target_window,
    load_abspos_reference,
    load_events_frame,
    window_to_tensors,
)
from pilot_sae_narrative_selected_activations import (  # noqa: E402
    build_farm_event_lists,
    find_narrative_targets,
    stratified_sample_targets,
)
from src.data_new.vocabulary import RegistryVocabulary  # noqa: E402


def effective_rank(vecs: np.ndarray, *, thresholds=(0.5, 0.9, 0.95, 0.99)) -> dict[str, int]:
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
        "--vocab", type=Path, default=ROOT / "outputs/online2/v2_build/life2vec_token_registry_v2.json"
    )
    parser.add_argument("--per-narrative-cap", type=int, default=250)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vocab = RegistryVocabulary(registry_path=str(args.vocab), registry_version="v2")
    vocab_size = vocab.size()
    print("vocab_size:", vocab_size)

    targets = find_narrative_targets(args.training_events)
    sampled = stratified_sample_targets(targets, per_narrative_cap=args.per_narrative_cap, seed=args.seed)
    print("n targets:", len(sampled))

    abspos_reference = load_abspos_reference(ROOT / "outputs/online2/v2_build/abspos_reference.json")
    farm_ids = {t.farm_id for t in sampled}
    events_df = load_events_frame(args.events, farm_ids)
    farm_events = build_farm_event_lists(events_df, farm_ids)
    event_id_to_local = {
        ev.event_id: (farm_id, idx) for farm_id, events in farm_events.items() for idx, ev in enumerate(events)
    }

    bag_vectors: list[np.ndarray] = []
    all_token_ids_seen: set[int] = set()
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
        x, _mask, _ = window_to_tensors(
            events, window, vocab=vocab, abspos_reference=abspos_reference,
            case_t0=case_t0, max_length=args.max_length,
        )
        span_tokens = x[0, window.target_token_start : window.target_token_end].long().tolist()
        if not span_tokens:
            continue
        all_token_ids_seen.update(span_tokens)
        bag = np.zeros(vocab_size, dtype=np.float32)
        for t in span_tokens:
            bag[t] += 1.0
        bag_vectors.append(bag / bag.sum())
        if (i + 1) % 4000 == 0:
            print(f"  {i + 1}/{len(sampled)} processed")

    print("skipped:", skipped)
    print("distinct token ids used across all target spans:", len(all_token_ids_seen))

    bags = np.stack(bag_vectors, axis=0)
    print("bag matrix shape:", bags.shape)
    rank = effective_rank(bags)
    print("bag-of-tokens PCA effective rank:", rank)


if __name__ == "__main__":
    main()
