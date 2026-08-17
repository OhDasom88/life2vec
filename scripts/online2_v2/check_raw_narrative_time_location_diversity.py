#!/usr/bin/env python3
"""근본 원인이 "물리 데이터 자체의 낮은 차원(29/384)"인지, 아니면 "지금 모델이
안 쓰고 있는 정보(전체/농장 상대 bin으로 뭉개진 세분도, narrative_id, 정확한
시각/장소)를 빼서 생긴 인위적 손실"인지를, **모델을 전혀 거치지 않고** 직접
검증한다.

`check_token_bag_diversity.py`와 같은 계열의 대조 실험(학습 가능한 파라미터
없음)이지만, 그 스크립트는 기존 vocab의 bag-of-tokens(VALUE_GLOBAL_REL/
VALUE_FARM_REL 포함, narrative_id/정확한 시각 없음)을 썼다. 여기서는 반대로:

  1. VALUE_ABS만 쓰고(GLOBAL_REL/FARM_REL 제거), bin 개수를 늘려(기본
     10 -> 100 quantile bin) 원본 값에 더 가깝게 세분화한다.
  2. narrative_id, farm_id/zone_id, 정확한 관측 시각(시각-of-day sin/cos +
     day-from-start)을 raw 원소값 옆에 명시적 채널로 추가한다 — 지금
     `tokenize_events()`가 이벤트별 SENTENCE에 전혀 넣지 않는 정보들이다.

원본 raw 값은 `cell_occurrences.parquet`(build_v2.py의 tokenize_events()가
쓰는 것과 동일한 legacy 소스)에서 직접 읽는다 — 이미 만들어진 학습용 vocab을
전혀 거치지 않으므로 사전학습 재실행이 필요 없다(빠름, 분석 전용).

같은 14,544개 narrative-selected 타깃(다른 모든 pilot과 동일 샘플링, cap=250,
seed=0)에 대해 --include-narrative/--include-time/--include-location 조합별로
effective_rank를 따로 찍어, "무엇을 추가하면 유효 차원이 느는지" 하나씩
분리해서 볼 수 있게 한다.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts" / "online2_v2") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "online2_v2"))

from pilot_sae_layer_decoder_random_comparison import effective_rank  # noqa: E402
from pilot_sae_narrative_selected_activations import (  # noqa: E402
    find_narrative_targets,
    stratified_sample_targets,
)
from src.online2.v2.feature_schema import FeatureSchema  # noqa: E402

DEFAULT_LEGACY_BUILD = ROOT / "outputs/online2/build-v8-active80-r3"
N_FINE_BINS = 100


def _first_id(raw) -> str:
    if raw is None or raw == "":
        return ""
    values = json.loads(raw) if isinstance(raw, str) else list(raw)
    return str(values[0]) if values else ""


def load_event_join_keys(legacy_build: Path, event_ids: set[str]) -> pd.DataFrame:
    """event_id -> (farm, zone, ts, view) 조인 키. build_v2.tokenize_events와 동일한 파생."""
    events = pd.read_parquet(
        legacy_build / "events.parquet",
        columns=["event_id", "observation_timestamp", "event_view", "farm_ids", "zone_ids"],
    )
    events = events[events["event_id"].isin(event_ids)].copy()
    events["farm"] = events["farm_ids"].map(_first_id)
    events["zone"] = events["zone_ids"].map(_first_id)
    events["ts"] = events["observation_timestamp"].astype(str)
    events["view"] = events["event_view"].astype(str)
    return events[["event_id", "farm", "zone", "ts", "view", "observation_timestamp"]]


def load_raw_cells(legacy_build: Path, farms: set[str]) -> pd.DataFrame:
    cells = pd.read_parquet(
        legacy_build / "cell_occurrences.parquet",
        columns=["column_name", "raw_display", "is_null", "farm_id", "zone_id", "observation_timestamp", "modality"],
    )
    cells = cells[cells["farm_id"].astype(str).isin(farms)].copy()
    cells["farm"] = cells["farm_id"].astype(str)
    cells["zone"] = cells["zone_id"].astype(str)
    cells["ts"] = cells["observation_timestamp"].astype(str)
    cells["view"] = cells["modality"].astype(str)
    return cells


def fit_fine_bin_edges(cells: pd.DataFrame, continuous_features: set[str], n_bins: int) -> dict[str, np.ndarray]:
    """feature별 quantile bin edge(전역, n_bins개) — VALUE_ABS와 같은 "절대" 프레임,
    개수만 늘림(요청: 상대 bin 제거 + ABS bin 세분화)."""
    edges: dict[str, np.ndarray] = {}
    numeric = cells[~cells["is_null"].astype(bool)].copy()
    numeric["value"] = pd.to_numeric(numeric["raw_display"], errors="coerce")
    numeric = numeric.dropna(subset=["value"])
    for feature in continuous_features:
        vals = numeric.loc[numeric["column_name"] == feature, "value"].to_numpy()
        if len(vals) < n_bins * 2:
            continue
        qs = np.linspace(0, 1, n_bins + 1)
        e = np.unique(np.quantile(vals, qs))
        if len(e) >= 3:
            edges[feature] = e
    return edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-events", type=Path, default=ROOT / "outputs/online2/v2_build/training_events_v2.parquet"
    )
    parser.add_argument("--legacy-build", type=Path, default=DEFAULT_LEGACY_BUILD)
    parser.add_argument(
        "--feature-schema", type=Path, default=ROOT / "outputs/online2/v2_build/feature_schema_v2.yaml"
    )
    parser.add_argument("--per-narrative-cap", type=int, default=250)
    parser.add_argument("--n-bins", type=int, default=N_FINE_BINS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/online2/sae_pilot/report_raw_narrative_time_location.json")
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("selecting narrative-template target events (same sampling as other pilots) ...")
    targets = find_narrative_targets(args.training_events)
    sampled = stratified_sample_targets(targets, per_narrative_cap=args.per_narrative_cap, seed=args.seed)
    print(f"  {len(sampled)} target events")

    schema = FeatureSchema.load(args.feature_schema)
    continuous_features = {
        name for name, spec in schema.features.items() if spec.absolute_encoding
    }
    print(f"  {len(continuous_features)} continuous (absolute-encoded) features")

    event_ids = {t.event_id for t in sampled}
    join_keys = load_event_join_keys(args.legacy_build, event_ids)
    print(f"  resolved join keys for {len(join_keys)}/{len(event_ids)} target events")

    farms = set(join_keys["farm"].unique())
    print(f"loading raw cells for {len(farms)} farms ...")
    cells = load_raw_cells(args.legacy_build, farms)
    print(f"  {len(cells)} raw cell rows loaded")

    print(f"fitting {args.n_bins}-bin quantile edges per feature (global, absolute frame) ...")
    edges = fit_fine_bin_edges(cells, continuous_features, args.n_bins)
    print(f"  fit edges for {len(edges)}/{len(continuous_features)} features")

    cell_groups: dict[tuple[str, str, str, str], list[tuple]] = {}
    for rec in cells[["farm", "zone", "ts", "view", "column_name", "raw_display", "is_null"]].itertuples(
        index=False, name=None
    ):
        cell_groups.setdefault((rec[0], rec[1], rec[2], rec[3]), []).append(rec)

    narrative_ids = sorted({t.narrative_id for t in sampled})
    narrative_index = {n: i for i, n in enumerate(narrative_ids)}
    farm_index = {f: i for i, f in enumerate(sorted(farms))}
    zone_index = {z: i for i, z in enumerate(sorted(join_keys["zone"].unique()))}
    feature_index = {f: i for i, f in enumerate(sorted(edges.keys()))}
    n_bins = args.n_bins

    target_by_id = {t.event_id: t for t in sampled}
    rows = []
    skipped = 0
    for jk in join_keys.itertuples(index=False):
        target = target_by_id.get(jk.event_id)
        grp = cell_groups.get((jk.farm, jk.zone, jk.ts, jk.view))
        if target is None or grp is None:
            skipped += 1
            continue

        abs_block = np.zeros(len(feature_index) * n_bins, dtype=np.float32)
        n_measured = 0
        for _farm, _zone, _ts, _view, column_name, raw_display, is_null in grp:
            if bool(is_null) or column_name not in feature_index:
                continue
            try:
                value = float(raw_display)
            except (TypeError, ValueError):
                continue
            e = edges.get(column_name)
            if e is None:
                continue
            bin_idx = int(np.clip(np.searchsorted(e, value, side="right") - 1, 0, n_bins - 1))
            abs_block[feature_index[column_name] * n_bins + bin_idx] = 1.0
            n_measured += 1
        if n_measured == 0:
            skipped += 1
            continue

        narrative_vec = np.zeros(len(narrative_index), dtype=np.float32)
        narrative_vec[narrative_index[target.narrative_id]] = 1.0

        farm_vec = np.zeros(len(farm_index), dtype=np.float32)
        farm_vec[farm_index[jk.farm]] = 1.0
        zone_vec = np.zeros(len(zone_index), dtype=np.float32)
        zone_vec[zone_index[jk.zone]] = 1.0

        ts = pd.Timestamp(jk.observation_timestamp)
        hour_frac = (ts.hour + ts.minute / 60.0) / 24.0
        day_from_start = float(ts.toordinal())
        time_vec = np.array(
            [np.sin(2 * np.pi * hour_frac), np.cos(2 * np.pi * hour_frac)], dtype=np.float32
        )

        rows.append(
            {
                "event_id": jk.event_id,
                "abs_fine": abs_block,
                "narrative": narrative_vec,
                "farm": farm_vec,
                "zone": zone_vec,
                "time": time_vec,
                "day_from_start_raw": day_from_start,
            }
        )

    print(f"built {len(rows)} feature rows, skipped {skipped}")

    day_vals = np.array([r["day_from_start_raw"] for r in rows], dtype=np.float32)
    day_norm = (day_vals - day_vals.mean()) / (day_vals.std() + 1e-6)

    combos = {
        "abs_fine_only": ["abs_fine"],
        "abs_fine+narrative": ["abs_fine", "narrative"],
        "abs_fine+time": ["abs_fine", "time_norm"],
        "abs_fine+location": ["abs_fine", "farm", "zone"],
        "abs_fine+narrative+time+location": ["abs_fine", "narrative", "time_norm", "farm", "zone"],
    }

    report: dict[str, dict] = {}
    for combo_name, parts in combos.items():
        mats = []
        for r, dn in zip(rows, day_norm):
            pieces = []
            for p in parts:
                if p == "time_norm":
                    pieces.append(np.concatenate([r["time"], [dn]]))
                else:
                    pieces.append(r[p])
            mats.append(np.concatenate(pieces))
        mat = np.stack(mats, axis=0)
        # 연속값(sin/cos/day_norm)과 one-hot 컬럼이 스케일이 크게 다르면 PCA가
        # 스케일 큰 쪽에 편향된다(관측됨: 정규화 없이 time을 넣으면 rank99가
        # 오히려 떨어짐) — 컬럼별 표준화로 이 인위적 효과를 제거한다.
        std = mat.std(axis=0)
        std[std < 1e-8] = 1.0
        mat = (mat - mat.mean(axis=0)) / std
        rank = effective_rank(mat)
        report[combo_name] = {"dim": int(mat.shape[1]), "n": int(mat.shape[0]), "effective_rank": rank}
        print(f"  {combo_name:34s} dim={mat.shape[1]:5d} rank99={rank['pc_for_99pct']:4d} rank90={rank['pc_for_90pct']:4d}")

    full_report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_narrative_cap": args.per_narrative_cap,
        "n_bins": args.n_bins,
        "n_continuous_features_with_edges": len(edges),
        "results": report,
    }
    args.out.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
