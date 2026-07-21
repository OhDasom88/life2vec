#!/usr/bin/env python3
"""§5.1 text_to_window의 임계값·가중치를 실제 Qwen3-Embedding-0.6B 임베딩으로 재보정.

사람이 라벨링한 정답 grounding pair는 아직 없다. 대신 자기참조(self-retrieval)
방식으로 실측한다: 실제 narrative 인스턴스의 observation 텍스트를 질의로 재사용해
"자신을 만든 템플릿을 다시 찾아내는가"를 측정한다. 이건 검증된 정답이 있는
유일한 신호이므로(관측문은 정확히 그 템플릿·그 farm/zone에서 나왔다),
threshold_and_weight 보정의 최소 기반으로 쓸 수 있다.

측정 세 그룹:
  - positive: 원래 관측문 그대로 (정답 템플릿 + 정답 farm/zone)
  - farm_mismatch: 관측문의 farm id를 다른 farm으로 치환(정답 템플릿, 오답 farm)
  - template_negative: 다른 템플릿에서 뽑은 관측문의 순위표에서 이 템플릿이 나오는 위치

세 그룹의 combined_score 분포가 얼마나 분리되는지로 AUTO_ACCEPT/REVIEW/QUARANTINE
임계값을 다시 정한다.

실행: `conda run -n life2vec python scripts/online2_v2/calibrate_narrative_grounding_thresholds.py`
GPU 필요, 최초 실행 시 이미 캐시된 Qwen/Qwen3-Embedding-0.6B를 로드한다(네트워크 재다운로드 없음).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.from_online2_corpus import (
    load_catalog,
    narrative_from_sequence_row,
)
from src.online2.v2.finetune_v03.narrative_sae_worldmodel.narrative_grounding.text_to_window import (
    Qwen3EmbeddingProvider,
    TemplateEmbeddingIndex,
    _structured_score,
    extract_structured_hints,
)

_FARM_RE = re.compile(r"F\d{5,6}")


def _sample_rows(build_dir: Path, per_template: int, seed: int) -> list[dict]:
    table = pq.read_table(
        build_dir / "sequences.parquet",
        columns=["sequence_id", "narrative_id", "narrative_center", "background_tokens",
                 "segment_ids", "event_views", "covered_time_span_hours", "quality_flags",
                 "op_eligible", "distinct_time_group_count"],
    )
    by_template: dict[str, list[int]] = {}
    for i, narrative_id in enumerate(table.column("narrative_id").to_pylist()):
        by_template.setdefault(narrative_id, []).append(i)
    rng = random.Random(seed)
    picked_indices: list[int] = []
    for narrative_id, indices in by_template.items():
        rng.shuffle(indices)
        picked_indices.extend(indices[:per_template])
    picked_indices.sort()
    return table.take(picked_indices).to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("outputs/online2/build-v8-active80-r3"))
    parser.add_argument("--per-template", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/online2/narrative_grounding_calibration"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(args.build_dir / "normalized_catalog.csv")
    provider = Qwen3EmbeddingProvider()
    print(f"[1/4] loading {provider.model_name} and embedding {len(catalog)} templates ...", flush=True)
    template_index = TemplateEmbeddingIndex(catalog, provider)

    print("[2/4] sampling narrative instances ...", flush=True)
    rows = _sample_rows(args.build_dir, args.per_template, args.seed)
    narratives = [narrative_from_sequence_row(row, catalog) for row in rows]
    print(f"  sampled {len(narratives)} instances across {len(catalog)} templates", flush=True)

    print("[3/4] embedding queries and scoring ...", flush=True)
    positive_records = []
    farm_mismatch_records = []
    other_farms = sorted({narrative.supporting_windows[0].farm_ids[0] for narrative in narratives if narrative.supporting_windows[0].farm_ids})

    for narrative in narratives:
        window = narrative.supporting_windows[0]
        true_template_id = window.narrative_template_id
        query = narrative.observation
        query_vector = provider.embed([query])[0]
        ranked = template_index.rank(query_vector)
        rank_of_true = next(i for i, (tid, _sim) in enumerate(ranked) if tid == true_template_id) + 1
        true_sim = next(sim for tid, sim in ranked if tid == true_template_id)
        top1_id, top1_sim = ranked[0]

        hints = extract_structured_hints(query)
        structured = _structured_score(window, hints)
        similarity_unit = (true_sim + 1.0) / 2.0
        combined = 0.7 * similarity_unit + 0.3 * structured
        positive_records.append(
            {
                "sequence_id": narrative.narrative_instance_id,
                "template_id": true_template_id,
                "rank_of_true_template": rank_of_true,
                "true_template_similarity": true_sim,
                "top1_template_id": top1_id,
                "top1_similarity": top1_sim,
                "structured_score": structured,
                "combined_score": combined,
            }
        )

        # farm_mismatch: 같은 질의문이지만 farm id를 다른 farm으로 바꿔치기.
        candidate_other_farms = [f for f in other_farms if f not in window.farm_ids]
        if candidate_other_farms and window.farm_ids:
            wrong_farm = rng_choice(candidate_other_farms, narrative.narrative_instance_id)
            mismatched_query = query.replace(window.farm_ids[0], wrong_farm)
            mismatched_hints = extract_structured_hints(mismatched_query)
            mismatched_structured = _structured_score(window, mismatched_hints)
            mismatched_combined = 0.7 * similarity_unit + 0.3 * mismatched_structured
            farm_mismatch_records.append(
                {
                    "sequence_id": narrative.narrative_instance_id,
                    "template_id": true_template_id,
                    "structured_score": mismatched_structured,
                    "combined_score": mismatched_combined,
                }
            )

    print("[4/4] aggregating ...", flush=True)

    def _stats(values: list[float]) -> dict:
        arr = np.array(values, dtype=np.float64)
        return {
            "n": len(arr),
            "mean": float(arr.mean()) if len(arr) else None,
            "p10": float(np.percentile(arr, 10)) if len(arr) else None,
            "p50": float(np.percentile(arr, 50)) if len(arr) else None,
            "p90": float(np.percentile(arr, 90)) if len(arr) else None,
            "min": float(arr.min()) if len(arr) else None,
            "max": float(arr.max()) if len(arr) else None,
        }

    top1_accuracy = sum(1 for r in positive_records if r["rank_of_true_template"] == 1) / len(positive_records)
    top3_accuracy = sum(1 for r in positive_records if r["rank_of_true_template"] <= 3) / len(positive_records)
    mrr = sum(1.0 / r["rank_of_true_template"] for r in positive_records) / len(positive_records)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": provider.model_name,
        "sample_size": len(positive_records),
        "templates_covered": len({r["template_id"] for r in positive_records}),
        "self_retrieval": {
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "mean_reciprocal_rank": mrr,
        },
        "combined_score_positive": _stats([r["combined_score"] for r in positive_records]),
        "combined_score_farm_mismatch": _stats([r["combined_score"] for r in farm_mismatch_records]),
        "raw_positive_records": positive_records,
        "raw_farm_mismatch_records": farm_mismatch_records,
    }

    out_path = args.out / "calibration_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"top1_accuracy={top1_accuracy:.3f} top3_accuracy={top3_accuracy:.3f} mrr={mrr:.3f}")
    print("positive combined_score:", report["combined_score_positive"])
    print("farm_mismatch combined_score:", report["combined_score_farm_mismatch"])


def rng_choice(items: list[str], seed_key: str) -> str:
    rng = random.Random(hash(seed_key) & 0xFFFFFFFF)
    return rng.choice(items)


if __name__ == "__main__":
    main()
