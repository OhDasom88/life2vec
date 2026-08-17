"""pipeline_explorer의 순수 데이터 접근 함수 — Streamlit에 의존하지 않는다(단위 테스트 가능).

원시데이터(raw CSV) → 토큰(SENTENCE) → 시퀀스(사전학습 입력) → encoder 각 layer →
SAE feature까지, 이 세션에서 만든 모든 산출물(부분 코퍼스, 6-arm 체크포인트,
SAE 비교 리포트)을 하나의 화면에서 조회할 수 있도록 잇는 조회 전용 계층이다.
UI 화면 상태는 정본이 아니다 — 여기서 읽는 모든 파일은 다른 백엔드 스크립트가
이미 만들어 둔 것이고, 이 모듈은 아무것도 새로 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[7]
RAW_DATA_ROOT = ROOT / "datasets/agrichallenge/online2/data"
RAW_TABLES = ("E_environment", "R_rootzone", "A_actuator", "G_growth")
ONLINE2_ROOT = ROOT / "datasets/agrichallenge/online2"
LEGACY_BUILD_DIR = ROOT / "outputs/online2/build-v8-active80-r3"

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
    source_file_id: str = ""
    source_row_id: str = ""


def parse_token_trace(
    sentence: str,
    measurement_group_ids_json: str,
    token_roles_json: str,
    source_file_ids_json: str = "",
    source_row_ids_json: str = "",
) -> list[TokenTraceEntry]:
    """SENTENCE 문자열 + 토큰-정렬 side-channel들을 하나의 토큰별 레코드 리스트로 합친다.
    source_file_ids/source_row_ids는 2026-07-26부터 생기는 필드라 옵션 인자 -- 없는(오래된)
    코퍼스 행에서 호출해도 그냥 빈 문자열로 채워진다(하위호환)."""
    tokens = sentence.split()
    groups = json.loads(measurement_group_ids_json) if measurement_group_ids_json else []
    roles = json.loads(token_roles_json) if token_roles_json else []
    sfids = json.loads(source_file_ids_json) if source_file_ids_json else []
    srids = json.loads(source_row_ids_json) if source_row_ids_json else []
    entries = []
    for i, tok in enumerate(tokens):
        role = roles[i] if i < len(roles) else "meta"
        group = groups[i] if i < len(groups) else "NONE"
        sfid = sfids[i] if i < len(sfids) else ""
        srid = srids[i] if i < len(srids) else ""
        entries.append(
            TokenTraceEntry(
                position=i, token=tok, role=role, group_id=group,
                source_file_id=sfid, source_row_id=srid,
            )
        )
    return entries


def _raw_table_dir(table: str) -> Path:
    return RAW_DATA_ROOT / table


@lru_cache(maxsize=64)
def _load_raw_zone_csv(table: str, farm_id: str, zone_id: str) -> pd.DataFrame | None:
    table_dir = _raw_table_dir(table)
    candidates = [table_dir / f"{farm_id}_z{zone_id}.csv"]
    try:
        # zone_id can be "" (IMAGE events are farm-level, no zone) -- int("") raises,
        # so this candidate is only added when zone_id actually looks like a zone number.
        candidates.append(table_dir / f"{farm_id}_z{int(zone_id)}.csv")
    except (TypeError, ValueError):
        pass
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


def _normalize_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        # raw CSVs (_load_raw_zone_csv/_load_raw_combined_csv) are tz-naive; some callers
        # (e.g. events_tokenized_v2.parquet's observation_timestamp) pass tz-aware values
        # -- normalize here so `>=`/`<=` window comparisons don't raise, instead of
        # requiring every caller to remember to strip tz first.
        ts = ts.tz_localize(None)
    return ts


def _resolve_table_df(table: str, farm_id: str, zone_id: str) -> pd.DataFrame | None:
    df = _load_raw_zone_csv(table, farm_id, zone_id)
    if df is None:
        df = _load_raw_combined_csv(table)
        if df is not None and "farm_id" in df.columns:
            df = df[df["farm_id"] == farm_id]
            if zone_id and "zone_id" in df.columns:
                df = df[df["zone_id"].astype(str) == str(zone_id)]
    return df


@lru_cache(maxsize=1)
def _load_source_files() -> dict[str, dict]:
    """source_file_id -> {path, checksum, modality}. cell_occurrences.parquet 계보의
    두 번째 단계 -- build 시점에 실제 raw CSV 파일 경로/체크섬을 고정해 둔 표(551개 파일뿐,
    전체를 메모리에 캐시해도 무해함)."""
    path = LEGACY_BUILD_DIR / "source_files.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["source_file_id", "path", "checksum", "modality"])
    return {
        r.source_file_id: {"path": r.path, "checksum": r.checksum, "modality": r.modality}
        for r in df.itertuples(index=False)
    }


@lru_cache(maxsize=1)
def _load_source_rows() -> dict[str, dict]:
    """source_row_id -> {source_file_id, row_number}. row_number는 그 raw CSV 안에서
    1-indexed 데이터 행 번호(헤더 제외) -- farm/zone/timestamp로 재검색하지 않고
    바로 .iloc[row_number-1]로 접근하기 위한 키."""
    path = LEGACY_BUILD_DIR / "source_rows.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["source_row_id", "source_file_id", "row_number"])
    return {
        r.source_row_id: {"source_file_id": r.source_file_id, "row_number": int(r.row_number)}
        for r in df.itertuples(index=False)
    }


@lru_cache(maxsize=64)
def _read_full_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@dataclass(frozen=True)
class ExactRawRow:
    table: str
    path: str
    row_number: int
    row: dict
    drift: bool  # True면 build 시점 checksum과 현재 파일 checksum이 다름(raw 스냅샷이 바뀌었을 수 있음)


def resolve_exact_raw_row(source_file_id: str, source_row_id: str) -> ExactRawRow | None:
    """source_file_id/source_row_id(cell_occurrences.parquet 계보의 정확한 ID)로 raw CSV의
    행 하나를 직접 인덱싱한다 -- farm/zone/timestamp로 다시 찾는 게 아니라
    source_rows.parquet의 row_number로 바로 접근하므로 tz 변환·raw 스냅샷 시간차 문제가
    구조적으로 없다(찾는 게 아니라 지목하는 것이므로). 두 ID 중 하나라도 비어있으면
    이 토큰에 애초에 provenance가 없다는 뜻 -- None을 반환해 호출부가 fuzzy 검색으로
    넘어가게 한다.
    """
    if not source_file_id or not source_row_id:
        return None
    file_info = _load_source_files().get(source_file_id)
    row_info = _load_source_rows().get(source_row_id)
    if file_info is None or row_info is None:
        return None
    csv_path = ONLINE2_ROOT / file_info["path"]
    if not csv_path.exists():
        return None
    current_checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    drift = current_checksum != file_info["checksum"]
    df = _read_full_csv(str(csv_path))
    row_number = row_info["row_number"]
    idx = row_number - 1
    if idx < 0 or idx >= len(df):
        return None
    return ExactRawRow(
        table=file_info["modality"],
        path=file_info["path"],
        row_number=row_number,
        row=df.iloc[idx].to_dict(),
        drift=drift,
    )


def lookup_raw_rows_for_event(
    source_file_ids: list[str], source_row_ids: list[str]
) -> tuple[dict[str, pd.DataFrame], bool]:
    """이벤트 하나에 속한 토큰들의 (source_file_id, source_row_id) 쌍을 모아 정확한 raw 행을
    직접 조회한다 -- farm/zone/timestamp로 다시 찾는 게 아니라 build 시점에 기록된 행 번호로
    바로 지목하므로 tz 변환이나 raw 스냅샷 시간차 문제가 없다. provenance가 비어 있는 토큰은
    건너뛴다. 반환값의 두 번째 항목(drift)은 하나라도 raw 파일 checksum이 build 시점과
    달라졌으면 True -- 값은 그대로 반환하되 호출부가 "이 값은 스냅샷이 바뀐 뒤의 것일 수
    있다"고 경고할 수 있게 한다. 결과 dict가 비어 있으면(즉 이 이벤트에 provenance가 전혀
    없음 -- 오래된 코퍼스이거나 G_growth처럼 애초에 안 잡히는 테이블) 호출부는
    lookup_raw_rows/lookup_raw_rows_range로 fuzzy 검색해야 한다.
    """
    seen: set[tuple[str, str]] = set()
    by_table: dict[str, list[dict]] = {}
    any_drift = False
    for sfid, srid in zip(source_file_ids, source_row_ids):
        if not sfid or not srid or (sfid, srid) in seen:
            continue
        seen.add((sfid, srid))
        resolved = resolve_exact_raw_row(sfid, srid)
        if resolved is None:
            continue
        by_table.setdefault(resolved.table, []).append(resolved.row)
        any_drift = any_drift or resolved.drift
    return {table: pd.DataFrame(rows) for table, rows in by_table.items()}, any_drift


def lookup_raw_rows_range(farm_id: str, zone_id: str, start: str, end: str) -> dict[str, pd.DataFrame]:
    """farm_id/zone_id에 대해 [start, end] 구간의 모든 raw 행을 4개 테이블에서 가져온다.
    `lookup_raw_rows`가 '이벤트 하나'를 기준(± window)으로 찾는 것과 달리, 그 사이 raw 표본이
    실제로 몇 개나 있는지(연속 구간 전체)를 볼 때 쓴다 -- 예: 시퀀스 전체 속성 변화 그래프.
    """
    lo, hi = _normalize_ts(start), _normalize_ts(end)
    results: dict[str, pd.DataFrame] = {}
    for table in RAW_TABLES:
        df = _resolve_table_df(table, farm_id, zone_id)
        if df is None or df.empty or "timestamp" not in df.columns:
            continue
        matched = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
        if not matched.empty:
            results[table] = matched.reset_index(drop=True)
    return results


def lookup_raw_rows(farm_id: str, zone_id: str, timestamp: str, *, window_hours: int = 0) -> dict[str, pd.DataFrame]:
    """farm_id/zone_id/timestamp로 4개 raw 테이블(E_environment/R_rootzone/A_actuator/G_growth)에서
    일치(또는 window_hours 이내) 행을 찾는다. 테이블이 없거나 해당 farm/zone이 없으면 그 테이블은 생략.
    """
    ts = _normalize_ts(timestamp)
    delta = pd.Timedelta(hours=window_hours) if window_hours > 0 else pd.Timedelta(0)
    return lookup_raw_rows_range(farm_id, zone_id, ts - delta, ts + delta)


def load_case_events_full(
    events_path: Path,
    farm_id: str,
    period_start: str,
    period_end: str,
    *,
    include_image: bool = False,
) -> pd.DataFrame:
    """finetune 케이스 하나(`case_manifest.csv`의 farm_id+period)에 속하는 이벤트를
    Stage A 캐싱(`cache_stage_a_event_embeddings.py`)이 실제로 쓰는 것과 **동일한 이벤트
    목록/순서**로 가져오되, 그 함수가 `CaseEvent` dataclass로 축약하며 버리는
    measurement_group_ids/token_roles/source_file_ids/source_row_ids까지 유지한 DataFrame으로
    반환한다 -- 필터링/정렬 규칙을 여기서 다시 구현하지 않고 `case_events_from_frame`을
    그대로 호출한 뒤 event_id로 나머지 컬럼만 조인한다(로직 drift 방지)."""
    from cache_stage_a_event_embeddings import case_events_from_frame, load_events_frame
    import pyarrow.dataset as ds

    events_df = load_events_frame(Path(events_path), {str(farm_id)})
    case_events = case_events_from_frame(
        events_df, str(farm_id), pd.Timestamp(period_start), pd.Timestamp(period_end),
        include_image=include_image,
    )
    if not case_events:
        return pd.DataFrame()

    event_ids = [ce.event_id for ce in case_events]
    dataset = ds.dataset(str(events_path), format="parquet")
    available = set(dataset.schema.names)
    extra_cols = ["event_id", "measurement_group_ids", "token_roles"]
    has_provenance = "source_file_ids" in available and "source_row_ids" in available
    if has_provenance:
        extra_cols += ["source_file_ids", "source_row_ids"]
    extra_table = dataset.to_table(columns=extra_cols, filter=ds.field("event_id").isin(event_ids))
    extra_df = extra_table.to_pandas().set_index("event_id")

    rows = []
    for ce in case_events:
        extra = extra_df.loc[ce.event_id] if ce.event_id in extra_df.index else None
        rows.append(
            {
                "event_id": ce.event_id,
                "event_position": ce.event_order,
                "same_time_group_id": ce.same_time_group_id,
                "timestamp": ce.timestamp,
                "view": ce.view,
                "zone_id": ce.zone,
                "event_kind": ce.event_kind,
                "SENTENCE": " ".join(ce.sentence_tokens),
                "token_count": ce.token_count,
                "farm_id": str(farm_id),
                "measurement_group_ids": extra["measurement_group_ids"] if extra is not None else "",
                "token_roles": extra["token_roles"] if extra is not None else "",
                "source_file_ids": (
                    extra["source_file_ids"] if (extra is not None and has_provenance) else ""
                ),
                "source_row_ids": (
                    extra["source_row_ids"] if (extra is not None and has_provenance) else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def load_json_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_SKIP_RUN_DIR_NAMES = {ABLATION_RUN_ROOT.name, "sweeps", ORIGINAL_RUN_DIR.name}


def resolve_run_dir(run_name: str) -> Path:
    """run_name 하나를 실제 디렉토리로 푼다 -- 'original'(고정 위치),
    ABLATION_ARMS 중 하나(narrative_ablation/ 밑), 또는 `outputs/online2/v2_runs/`
    바로 아래의 임의 run 디렉토리(2026-07-26부터 새로 저장되는 실제 사전학습
    run, 예: full_event_grain_simplified_2026-07-26) 세 경우를 다 지원한다."""
    if run_name.startswith("original"):
        return ORIGINAL_RUN_DIR
    if run_name in ABLATION_ARMS:
        return ABLATION_RUN_ROOT / run_name
    return RUN_ROOT / run_name


_DEFAULT_VOCAB_BUILD_DIR = ROOT / "outputs/online2/v2_build"


def build_dir_for_run(run_name: str) -> Path:
    """이 run이 실제로 학습에 쓴 코퍼스 build 디렉토리 -- run_manifest_v2.json의
    `parquet` 필드(예: outputs/online2/v2_build_expanded_full1517/sequences_v2.parquet)
    의 부모 디렉토리로 판단한다. 2026-07-26 이전 run(original/ablation arms)은 전부
    v2_build를 썼으므로 그대로 v2_build가 나오고, 그 이후 정규화 로더로 학습한 run은
    자기 build_dir(예: v2_build_expanded_full1517)이 나온다 -- vocab/registry를 그
    run의 실제 vocab_size와 다른 걸 잘못 물리는 걸 막는다(2026-07-26 실제로 겪은
    index-out-of-range 버그와 같은 종류)."""
    manifest_path = resolve_run_dir(run_name) / "run_manifest_v2.json"
    if manifest_path.exists():
        try:
            manifest = load_json_report(manifest_path)
            parquet = manifest.get("parquet")
            if parquet:
                return Path(parquet).resolve().parent
        except (json.JSONDecodeError, OSError):
            pass
    return _DEFAULT_VOCAB_BUILD_DIR


def discover_all_run_names() -> list[str]:
    """체크포인트가 실제로 있는 모든 run 이름을 찾는다 -- 'original'+ABLATION_ARMS
    고정 목록이 아니라 `outputs/online2/v2_runs/` 아래 새로 생기는 run 디렉토리도
    자동으로 잡는다(2026-07-26: 이게 없어서 새로 학습한 모델이 UI 셀렉터에
    안 보이던 문제)."""
    names: list[str] = []
    if (ORIGINAL_RUN_DIR / "best.ckpt").exists():
        names.append("original")
    for arm in ABLATION_ARMS:
        run_dir = ABLATION_RUN_ROOT / arm
        if (run_dir / "best.ckpt").exists() or any(run_dir.glob("checkpoint_step_*.pt")):
            names.append(arm)
    if RUN_ROOT.exists():
        for d in sorted(RUN_ROOT.iterdir()):
            if not d.is_dir() or d.name in _SKIP_RUN_DIR_NAMES:
                continue
            if (d / "best.ckpt").exists() or (d / "last.ckpt").exists():
                names.append(d.name)
    return names


def list_local_runs() -> list[dict]:
    """체크포인트가 있는 모든 run의 run_manifest_v2.json을 한 목록으로 합친다."""
    runs = []
    for name in discover_all_run_names():
        run_dir = resolve_run_dir(name)
        manifest_path = run_dir / "run_manifest_v2.json"
        if manifest_path.exists():
            manifest = load_json_report(manifest_path)
            label = f"original({ORIGINAL_RUN_DIR.name})" if name == "original" else name
            runs.append({"name": label, "run_dir": str(run_dir), **manifest})
    return runs


def run_created_at(run_name: str) -> str | None:
    """`run_manifest_v2.json`의 `created_at_utc`를 조회 — 체크포인트 생성 선후 비교용."""
    manifest_path = resolve_run_dir(run_name) / "run_manifest_v2.json"
    if not manifest_path.exists():
        return None
    manifest = load_json_report(manifest_path)
    created = manifest.get("created_at_utc")
    if not created:
        return None
    return str(created).replace("T", " ").split(".")[0] + " UTC"


def checkpoint_path_for_run(run_name: str, *, step: int | None = 5000) -> Path:
    run_dir = resolve_run_dir(run_name)
    if run_name.startswith("original"):
        return run_dir / "best.ckpt"
    if step is not None and (run_dir / f"checkpoint_step_{step}.pt").exists():
        return run_dir / f"checkpoint_step_{step}.pt"
    if (run_dir / "best.ckpt").exists():
        return run_dir / "best.ckpt"
    return run_dir / "last.ckpt"


# finetune Stage A 캐시(event_embeddings 디렉토리 이름) -> 그걸 실제로 인코딩한
# 사전학습 run 이름. finetune run의 config.json에 기록된 emb_dir로부터 "이 케이스가
# 어느 vocab/encoder로 토큰화·인코딩됐는지"를 역추적할 때 쓴다 -- 하드코딩된
# v2_build/v2_finetune_v02 경로를 finetune run마다 다르게 고쳐야 했던 문제
# (2026-07-26, 예측/근거 탭이 새 run에서도 항상 구 코퍼스를 참조하던 버그) 방지.
EMB_DIR_TO_ENCODER_RUN = {
    "v2_finetune_v02": "original",
    "v2_finetune_v03_simplified": "full_event_grain_simplified_2026-07-26",
}


def encoder_run_for_emb_dir(emb_dir: Path) -> str:
    p = Path(emb_dir)
    # emb_dir경로가 ".../{corpus_tag}/event_embeddings" 형태이므로
    # p.name("event_embeddings")과 p.parent.name("v2_finetune_v03_simplified") 둘 다 확인
    return EMB_DIR_TO_ENCODER_RUN.get(p.name, EMB_DIR_TO_ENCODER_RUN.get(p.parent.name, "original"))



def finetune_run_emb_dir(run_dir: Path) -> Path:
    """finetune run(`run_diagnosis_finetune_v03.py --run-dir ...`)이 실제로 학습에 쓴
    `--emb-dir`을 그 run의 `config.json`에서 읽는다 -- 없으면(오래된 run) 원래
    스크립트 기본값(v2_finetune_v02)으로 폴백."""
    config_path = Path(run_dir) / "config.json"
    fallback = ROOT / "outputs/online2/v2_finetune_v02/event_embeddings"
    if not config_path.exists():
        return fallback
    try:
        config = load_json_report(config_path)
        emb_dir = config.get("args", {}).get("emb_dir")
        if not emb_dir:
            return fallback
        # args.emb_dir is saved relative-to-CWD-at-train-time (e.g.
        # "outputs/online2/v2_finetune_v03_simplified/event_embeddings", unlike
        # args.labels/label_map which happened to be saved absolute) -- resolve
        # against the repo root rather than trusting the Streamlit process's CWD.
        p = Path(emb_dir)
        return p if p.is_absolute() else ROOT / p
    except (json.JSONDecodeError, OSError):
        return fallback


def load_ablation_report() -> dict | None:
    path = SAE_PILOT_DIR / "report_narrative_ablation.json"
    return load_json_report(path) if path.exists() else None


def load_layer_decoder_random_report() -> dict | None:
    path = SAE_PILOT_DIR / "report_layer_decoder_random.json"
    return load_json_report(path) if path.exists() else None
