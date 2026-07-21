"""pipeline_explorer의 순수 데이터 접근 함수 — Streamlit에 의존하지 않는다(단위 테스트 가능).

원시데이터(raw CSV) → 토큰(SENTENCE) → 시퀀스(사전학습 입력) → encoder 각 layer →
SAE feature까지, 이 세션에서 만든 모든 산출물(부분 코퍼스, 6-arm 체크포인트,
SAE 비교 리포트)을 하나의 화면에서 조회할 수 있도록 잇는 조회 전용 계층이다.
UI 화면 상태는 정본이 아니다 — 여기서 읽는 모든 파일은 다른 백엔드 스크립트가
이미 만들어 둔 것이고, 이 모듈은 아무것도 새로 쓰지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[7]
RAW_DATA_ROOT = ROOT / "datasets/agrichallenge/online2/data"
RAW_TABLES = ("E_environment", "R_rootzone", "A_actuator", "G_growth")

RUN_ROOT = ROOT / "outputs/online2/v2_runs"
ORIGINAL_RUN_DIR = RUN_ROOT / "full_event_grain_earlystop"
ABLATION_RUN_ROOT = RUN_ROOT / "narrative_ablation"
ABLATION_ARMS = (
    "control",
    "sop_balanced",
    "narrow_subset",
    "single_table_only",
    "multi_table_only",
    "dedup_reduced",
)

SAE_PILOT_DIR = ROOT / "outputs/online2/sae_pilot"


@dataclass(frozen=True)
class TokenTraceEntry:
    position: int
    token: str
    role: str
    group_id: str


def parse_token_trace(sentence: str, measurement_group_ids_json: str, token_roles_json: str) -> list[TokenTraceEntry]:
    """SENTENCE 문자열 + 토큰-정렬 side-channel 두 개를 하나의 토큰별 레코드 리스트로 합친다."""
    tokens = sentence.split()
    groups = json.loads(measurement_group_ids_json) if measurement_group_ids_json else []
    roles = json.loads(token_roles_json) if token_roles_json else []
    entries = []
    for i, tok in enumerate(tokens):
        role = roles[i] if i < len(roles) else "meta"
        group = groups[i] if i < len(groups) else "NONE"
        entries.append(TokenTraceEntry(position=i, token=tok, role=role, group_id=group))
    return entries


def _raw_table_dir(table: str) -> Path:
    return RAW_DATA_ROOT / table


@lru_cache(maxsize=64)
def _load_raw_zone_csv(table: str, farm_id: str, zone_id: str) -> pd.DataFrame | None:
    table_dir = _raw_table_dir(table)
    candidates = [table_dir / f"{farm_id}_z{zone_id}.csv", table_dir / f"{farm_id}_z{int(zone_id)}.csv"]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            return df
    return None


@lru_cache(maxsize=8)
def _load_raw_combined_csv(table: str) -> pd.DataFrame | None:
    table_dir = _raw_table_dir(table)
    for name in ("root_data.csv", "growth_data.csv", "actuator_data.csv", "environment_data.csv"):
        path = table_dir / name
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(path, nrows=0).columns else None)
            return df
    return None


def lookup_raw_rows(farm_id: str, zone_id: str, timestamp: str, *, window_hours: int = 0) -> dict[str, pd.DataFrame]:
    """farm_id/zone_id/timestamp로 4개 raw 테이블(E_environment/R_rootzone/A_actuator/G_growth)에서
    일치(또는 window_hours 이내) 행을 찾는다. 테이블이 없거나 해당 farm/zone이 없으면 그 테이블은 생략.
    """
    ts = pd.Timestamp(timestamp)
    results: dict[str, pd.DataFrame] = {}
    for table in RAW_TABLES:
        df = _load_raw_zone_csv(table, farm_id, zone_id)
        if df is None:
            df = _load_raw_combined_csv(table)
            if df is not None and "farm_id" in df.columns:
                df = df[df["farm_id"] == farm_id]
                if zone_id and "zone_id" in df.columns:
                    df = df[df["zone_id"].astype(str) == str(zone_id)]
        if df is None or df.empty or "timestamp" not in df.columns:
            continue
        if window_hours <= 0:
            matched = df[df["timestamp"] == ts]
        else:
            lo, hi = ts - pd.Timedelta(hours=window_hours), ts + pd.Timedelta(hours=window_hours)
            matched = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
        if not matched.empty:
            results[table] = matched.reset_index(drop=True)
    return results


def load_json_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def list_local_runs() -> list[dict]:
    """원본 체크포인트 + 6개 ablation arm의 run_manifest_v2.json을 한 목록으로 합친다."""
    runs = []
    if (ORIGINAL_RUN_DIR / "run_manifest_v2.json").exists():
        manifest = load_json_report(ORIGINAL_RUN_DIR / "run_manifest_v2.json")
        runs.append({"name": "original(full_event_grain_earlystop)", "run_dir": str(ORIGINAL_RUN_DIR), **manifest})
    for arm in ABLATION_ARMS:
        run_dir = ABLATION_RUN_ROOT / arm
        manifest_path = run_dir / "run_manifest_v2.json"
        if manifest_path.exists():
            manifest = load_json_report(manifest_path)
            runs.append({"name": arm, "run_dir": str(run_dir), **manifest})
    return runs


def checkpoint_path_for_run(run_name: str, *, step: int | None = 5000) -> Path:
    if run_name.startswith("original"):
        return ORIGINAL_RUN_DIR / "best.ckpt"
    run_dir = ABLATION_RUN_ROOT / run_name
    if step is not None and (run_dir / f"checkpoint_step_{step}.pt").exists():
        return run_dir / f"checkpoint_step_{step}.pt"
    return run_dir / "best.ckpt"


def load_ablation_report() -> dict | None:
    path = SAE_PILOT_DIR / "report_narrative_ablation.json"
    return load_json_report(path) if path.exists() else None


def load_layer_decoder_random_report() -> dict | None:
    path = SAE_PILOT_DIR / "report_layer_decoder_random.json"
    return load_json_report(path) if path.exists() else None
