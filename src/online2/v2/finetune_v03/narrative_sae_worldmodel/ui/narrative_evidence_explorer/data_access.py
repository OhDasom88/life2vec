"""narrative_evidence_explorer의 순수 데이터 접근 함수 — Streamlit에 의존하지 않는다.

서사(narrative)는 "이 raw 구간을 이렇게 해석한다"는 사람이 쓴 주장(카탈로그의
`purpose`/`expected_pattern`/`agronomic_interpretation` 필드)이고, 실제로 모델이
보는 건 그 주장이 골라낸 원시데이터를 토큰화한 시퀀스일 뿐이다. 이 둘이 실제로
얼마나 대응하는지 — 서사가 "이런 패턴"이라고 주장하는 구간에서 진짜 raw 센서값이
그 패턴을 보이는지 — 사람이 직접 눈으로 확인할 수 있게 세 가지를 나란히 보여준다:
(1) 서사의 해석 텍스트, (2) 그 서사가 실제로 매칭한 시퀀스의 토큰화된 SENTENCE,
(3) 그 시퀀스가 가리키는 farm/zone/timestamp의 원본 raw 센서 CSV 값.

`pipeline_explorer.data_access`의 `parse_token_trace`/`lookup_raw_rows`를 그대로
재사용한다(같은 계산 로직을 두 번 만들지 않음) — 이 모듈은 그 위에 "서사 카탈로그
↔ 매칭된 시퀀스" 조회만 새로 얹는다.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[7]

# 두 디렉토리 모두 파일명이 data_access.py라 평범한 `import data_access`는
# 자기 자신과 충돌한다(먼저 sys.path에 오른 쪽이 이긴다) -- 파일 경로로 직접 로드.
_PIPELINE_EXPLORER_DATA_ACCESS = (
    ROOT / "src/online2/v2/finetune_v03/narrative_sae_worldmodel/ui/pipeline_explorer/data_access.py"
)
import sys as _sys  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_pipeline_explorer_data_access", _PIPELINE_EXPLORER_DATA_ACCESS
)
_pipeline_da = importlib.util.module_from_spec(_spec)
_sys.modules[_spec.name] = _pipeline_da  # dataclasses needs this registered before exec
_spec.loader.exec_module(_pipeline_da)
TokenTraceEntry = _pipeline_da.TokenTraceEntry
lookup_raw_rows = _pipeline_da.lookup_raw_rows
parse_token_trace = _pipeline_da.parse_token_trace

__all__ = [
    "TokenTraceEntry",
    "lookup_raw_rows",
    "parse_token_trace",
    "CorpusChoice",
    "CORPUS_CHOICES",
    "load_catalog",
    "narrative_row",
    "load_matched_instances",
    "instance_detail",
    "parse_matcher",
    "matched_keys",
    "fetch_evidence_raw",
    "available_numeric_columns",
    "build_evidence_long_frame",
]

CATALOG_PATH = ROOT / "datasets/agrichallenge/online2/online2_narrative_catalog.csv"
AUTO_EXPANSION_REPORT = ROOT / "outputs/online2/sae_pilot/narrative_auto_expansion_report.json"


@dataclass(frozen=True)
class CorpusChoice:
    label: str
    build_dir: Path

    @property
    def training_events_path(self) -> Path:
        return self.build_dir / "training_events_v2.parquet"


CORPUS_CHOICES = [
    CorpusChoice("expanded_full1517 (1517개 서사, 최신, lag_response 포함)", ROOT / "outputs/online2/v2_build_expanded_full1517"),
    CorpusChoice("expanded_full (1479개 서사)", ROOT / "outputs/online2/v2_build_expanded_full"),
    CorpusChoice("expanded88_phase2 (88개 서사)", ROOT / "outputs/online2/v2_build_expanded88_phase2"),
    CorpusChoice("control (원본 80개 서사)", ROOT / "outputs/online2/v2_build"),
]


def load_catalog() -> pd.DataFrame:
    """활성 서사 카탈로그(사람이 쓴 해석 텍스트 포함) 전체를 읽는다."""
    if not CATALOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CATALOG_PATH)
    return df[df["status"] == "ACTIVE"].reset_index(drop=True)


def load_auto_expansion_candidates() -> pd.DataFrame:
    """아직 실제 카탈로그에 반영 안 된 자동생성 후보(검토 전 단계)도 같이 볼 수 있게.

    이 함수가 읽는 리포트는 dry-run 검증 결과일 뿐 실제 매칭된 시퀀스가 없다 --
    UI에서는 "후보" 표시만 하고 시퀀스 조회는 하지 않는다.
    """
    if not AUTO_EXPANSION_REPORT.exists():
        return pd.DataFrame()
    payload = json.loads(AUTO_EXPANSION_REPORT.read_text(encoding="utf-8"))
    return pd.DataFrame(payload.get("validated_full", []))


def narrative_row(catalog: pd.DataFrame, narrative_id: str) -> dict | None:
    match = catalog[catalog["narrative_id"] == narrative_id]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def load_matched_instances(corpus: CorpusChoice, narrative_id: str, *, limit: int = 200) -> pd.DataFrame:
    """이 서사가 실제로 매칭한 시퀀스를 training_events_v2.parquet에서 조회한다.

    pyarrow filters로 narrative_id 술어를 밀어넣어(predicate pushdown) 1500만 행
    전체를 메모리에 올리지 않는다 -- row group 통계로 스킵 가능한 만큼 스킵.
    """
    path = corpus.training_events_path
    if not path.exists():
        return pd.DataFrame()
    columns = [
        "sequence_id", "event_id", "event_position", "farm_id", "zone_id",
        "narrative_id", "SENTENCE", "START_DATE", "AGE", "measurement_group_ids",
        "token_roles", "op_eligible", "order_semantics",
    ]
    table = pq.read_table(path, columns=columns, filters=[("narrative_id", "=", narrative_id)])
    df = table.to_pandas()
    if df.empty:
        return df
    # 시퀀스별 마지막 이벤트(가장 최근 event_position)를 대표 timestamp로 사용
    seq_ids = df["sequence_id"].unique()[:limit]
    return df[df["sequence_id"].isin(seq_ids)].sort_values(["sequence_id", "event_position"])


def parse_matcher(matcher: str) -> dict:
    """materialization_key(예: `run_ge:inside_temp_c:25.549:6`)를 사람이 읽을 수
    있는 형태로 분해한다 — "이 서사가 정확히 어떤 규칙으로 raw 데이터에서
    골라지는지"를 UI에 보여주기 위함(matcher 이름만으론 임계값·구간·비교 대상이
    안 보임). 반환하는 `columns`는 raw 데이터 조회 화면에서 관련 컬럼을 자동
    선택하는 데도 쓰인다.

    실제 판정 로직의 정본은 `src/online2/materializers.py`의 `_predicate`/
    `_named_predicate`/각 전용 매처 메서드다 — 여기 설명은 그 로직을 UI용으로
    재서술한 것이라 로직이 바뀌면 이 함수도 같이 고쳐야 한다.
    """
    parts = matcher.split(":")
    prefix = parts[0]
    columns: list[str] = []
    thresholds: list[dict] = []
    window_hours: int | None = None
    description = matcher

    def alias(col: str) -> str:
        return col

    if prefix in {"positive"}:
        columns = [parts[1]]
        description = f"{parts[1]} > 0 (가동/양수)인 단일 시점"
    elif prefix == "change":
        columns = [parts[1]]
        description = f"{parts[1]}이 직전 관측값과 다른 시점(상태 전환)"
    elif prefix in {"lt", "le", "gt"}:
        op = {"lt": "<", "le": "<=", "gt": ">"}[prefix]
        columns = [parts[1]]
        thresholds = [{"column": parts[1], "op": op, "value": float(parts[2])}]
        description = f"{parts[1]} {op} {float(parts[2]):.3f} (전체 데이터 절대 임계값)"
    elif prefix in {"quantile_high", "quantile_low"}:
        columns = [parts[1]]
        side = "상위 10%" if prefix == "quantile_high" else "하위 10%"
        description = f"{parts[1]}이 그 (farm,zone)의 {side} 분위에 속하는 시점"
    elif prefix == "jump":
        columns = [parts[1]]
        description = f"{parts[1]}이 직전 시점 대비 (farm,zone)별 평소 변동폭(중앙값+6×MAD)을 초과해 급변한 시점"
    elif prefix == "reset":
        columns = [parts[1]]
        description = f"{parts[1]}이 양수였다가 이번 시점에 0으로 떨어진 시점(리셋)"
    elif prefix == "fixed_zero":
        columns = [parts[1]]
        description = f"{parts[1]}이 관측기간 내내 0(2회 이상 연속)"
    elif prefix == "run_ge":
        columns = [parts[1]]
        val, run = float(parts[2]), int(parts[3])
        window_hours = run
        thresholds = [{"column": parts[1], "op": ">=", "value": val}]
        description = f"{parts[1]} >= {val:.3f}인 상태가 {run}시간 이상 연속"
    elif prefix in {"and", "and_gt"}:
        c1, v1, c2, v2 = parts[1], float(parts[2]), parts[3], float(parts[4])
        op2 = ">" if prefix == "and_gt" or c2 == "rain_detected" else "<"
        columns = [c1, c2]
        thresholds = [
            {"column": c1, "op": ">", "value": v1},
            {"column": c2, "op": op2, "value": v2},
        ]
        description = f"{c1} > {v1:.3f} 이면서 동시에 {c2} {op2} {v2:.3f}"
    elif prefix == "diff_positive":
        c1, c2 = parts[1], parts[2]
        columns = [c1, c2]
        description = f"{c1} - {c2} 값이 0이 아닌 시점(두 값이 서로 다름)"
    elif prefix == "joint_quantile":
        c1, c2 = parts[1], parts[2]
        columns = [c1, c2]
        description = f"{c1}와 {c2} 둘 다 그 (farm,zone)의 상위 10% 분위에 동시에 속하는 시점"
    elif prefix in {"tod_high", "tod_low"}:
        col, h0, h1 = parts[1], int(parts[2]), int(parts[3])
        columns = [col]
        side = "상위" if prefix == "tod_high" else "하위"
        description = f"지역시각이 {h0}시~{h1}시 구간이면서 {col}이 그 시간대 {side} 10% 분위"
    elif prefix in {"trend_intervention", "co_trend"}:
        c1, c2, hrs = parts[1], parts[2], int(parts[3])
        columns = [c1, c2]
        window_hours = hrs
        description = (
            f"최근 {hrs}시간 window 안에서 {c1}와 {c2} 둘 다 각자의 평소 변동폭"
            f"(중앙값+6×MAD 점프 크기)만큼 움직였는지(둘 다 range >= limit, 방향 무관 공존)"
        )
    elif prefix == "lag_response":
        trig, resp, lag = parts[1], parts[2], int(parts[3])
        columns = [trig, resp]
        window_hours = lag
        description = (
            f"{trig}가 0에서 양수로 켜진 시점 대비 정확히 {lag}시간 뒤에 {resp}가 "
            f"평소 변동폭(중앙값+6×MAD) 이상 움직였는지(방향 고정: {trig}→{resp}, 지연 인과)"
        )
    elif prefix == "multi_trend":
        cols_joined, hrs = parts[1], int(parts[2])
        columns = cols_joined.split("+")
        window_hours = hrs
        description = (
            f"최근 {hrs}시간 window 안에서 {', '.join(columns)} {len(columns)}개 전부가 "
            f"각자의 평소 변동폭(중앙값+6×MAD)만큼 움직였는지(AND 조건, 하나라도 안 움직이면 매칭 안 됨)"
        )
    elif prefix == "growth_any" or prefix == "mixed_growth":
        description = "생육 조사(observation_date) 2회차 간 변화 — 첫 조사와 둘째 조사의 차이"
    elif prefix == "crosszone_hourly":
        description = "동일 농장, 동일 절대 timestamp에서 4개 구역(zone)이 모두 관측된 순간을 구역별로 비교"
    elif prefix == "crossfarm_hourly":
        description = "동일 지역시각(0~23시, 절대 날짜 무시)에 20개 농장 이상 관측된 시각을 농장별로 비교"
    elif prefix == "crossfarm_growth":
        description = "동일 생육조사 순번(0=첫 조사, 1=둘째 조사)에 20개 농장 이상 있는 순번을 농장별로 비교"
    elif prefix == "zone_outlier_hourly":
        columns = parts[1:2] if len(parts) > 1 else []
        description = "동일 농장 동시각 4개 구역 중, 중앙값 대비 6×MAD 이상 벗어난 구역(이상치)을 그 구역의 트리거로 사용"
    elif prefix == "all_hourly":
        description = "임계값 조건 없이 하루(00시~23시) 전체를 그대로 담는 통계형 요약 -- 최저/최고/평균 등 분포 파악용"
    elif prefix == "window_summary":
        h0, h1 = int(parts[1]), int(parts[2])
        window_hours = (h1 - h0) if h1 > h0 else (24 - h0 + h1)
        description = f"임계값 조건 없이 하루 중 {h0}시~{h1}시 구간만 그대로 담는 통계형 요약 -- 하위 시간대 분포 파악용"
    elif prefix == "image_longitudinal_pair":
        description = "농장별 최초·최근 이미지 페어 + 각 시점 환경 맥락"

    return {
        "matcher": matcher,
        "prefix": prefix,
        "columns": [c for c in columns if c],
        "thresholds": thresholds,
        "window_hours": window_hours,
        "description": description,
    }


GROWTH_TIME_COL = "observation_date"
MAX_EVIDENCE_ENTITIES = 12


def _load_table_for_farm_zone(table: str, farm_id: str, zone_id: str) -> pd.DataFrame | None:
    """`pipeline_explorer`의 raw 로더를 재사용하되, G_growth처럼 `timestamp`가 아니라
    `observation_date`를 쓰는 테이블도 같은 시간축으로 다룰 수 있게 별칭을 붙인다
    (원래 `lookup_raw_rows`는 이 컬럼이 없으면 그냥 건너뛰어서 생육 관련 서사는
    raw 그래프에 전혀 안 나왔었다)."""
    df = _pipeline_da._load_raw_zone_csv(table, farm_id, zone_id)
    if df is None:
        df = _pipeline_da._load_raw_combined_csv(table)
        if df is not None and "farm_id" in df.columns:
            df = df[df["farm_id"] == farm_id]
            if zone_id and "zone_id" in df.columns:
                df = df[df["zone_id"].astype(str) == str(zone_id)]
    if df is None or df.empty:
        return None
    if "timestamp" not in df.columns and GROWTH_TIME_COL in df.columns:
        df = df.rename(columns={GROWTH_TIME_COL: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "timestamp" not in df.columns:
        return None
    df = df.copy()
    df["zone_id"] = df["zone_id"].astype(str)
    return df


def lookup_raw_window(
    farm_zone_pairs: list[tuple[str, str]], ts_lo: pd.Timestamp, ts_hi: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """`lookup_raw_rows`(단일 farm/zone, 단일 중심시각)의 확장판 -- 여러 (farm,zone) 쌍을
    각자의 [ts_lo, ts_hi] 구간으로 한 번에 모은다(crossfarm/crosszone 서사는 매칭된
    (farm,zone)이 여러 개라 단일 조회로는 부족했다)."""
    out: dict[str, list[pd.DataFrame]] = {t: [] for t in _pipeline_da.RAW_TABLES}
    for farm_id, zone_id in farm_zone_pairs:
        for table in _pipeline_da.RAW_TABLES:
            df = _load_table_for_farm_zone(table, farm_id, zone_id)
            if df is None:
                continue
            sub = df[(df["timestamp"] >= ts_lo) & (df["timestamp"] <= ts_hi)]
            if not sub.empty:
                out[table].append(sub)
    return {t: pd.concat(dfs, ignore_index=True) for t, dfs in out.items() if dfs}


def matched_keys(seq_events: pd.DataFrame) -> set[tuple[str, str, pd.Timestamp]]:
    """이 시퀀스 인스턴스에서 실제로 매칭에 쓰인 (farm_id, zone_id, timestamp) 집합.
    export 단계에서 이벤트 하나 = raw 관측 시각 하나로 만들어지므로(실측 확인:
    S03 같은 run_ge 서사는 이벤트 24개 = 서로 다른 시각 24개), 이 이벤트들의
    START_DATE 그대로가 "서사가 실제로 근거로 삼은" raw 시점 집합이다."""
    keys = set()
    for r in seq_events.itertuples():
        keys.add((str(r.farm_id), str(r.zone_id), pd.Timestamp(r.START_DATE)))
    return keys


def fetch_evidence_raw(
    seq_events: pd.DataFrame, *, context_hours: int, max_entities: int = MAX_EVIDENCE_ENTITIES,
) -> tuple[dict[str, pd.DataFrame], bool, dict[tuple[str, str], pd.Timestamp]]:
    """서사 인스턴스 하나의 raw 컨텍스트를 가져온다.

    단일 (farm,zone) 서사(대부분의 episodic 서사)는 매칭 시각들을 감싸는 하나의
    절대시간 구간 [min-context, max+context]로 가져온다.

    여러 (farm,zone)이 얽힌 서사(crossfarm_hourly/crossfarm_growth/crosszone_hourly/
    zone_outlier_hourly)는 각 farm/zone이 서로 완전히 다른 절대 날짜에 매칭되므로
    (실측: X13 20개 농장이 2024-10-14 ~ 2025-03-24까지 흩어져 있음) 절대시간 하나로
    묶으면 의미가 없다 -- 대신 각 farm/zone을 "자기 매칭 시각 기준 ±context_hours"로
    각자 따로 가져온다(차트에서는 상대시간으로 겹쳐 그려 비교한다).
    """
    m_keys = matched_keys(seq_events)
    # 등장 순서(event_position) 그대로 유지 -- 알파벳 정렬로 자르면 실제 시퀀스에
    # 없는 임의 부분집합이 뽑혀서(실측 확인: X13에서 첫 farm이 통째로 누락됨) 사람이
    # 위 "토큰화된 시퀀스" 절과 아래 그래프를 대조할 때 farm이 안 맞는 문제가 있었다.
    pairs_all = list(dict.fromkeys(
        (str(r.farm_id), str(r.zone_id)) for r in seq_events.itertuples()
    ))
    is_multi = len(pairs_all) > 1
    pairs = pairs_all[:max_entities]

    anchors: dict[tuple[str, str], pd.Timestamp] = {}
    for f, z, ts in m_keys:
        if (f, z) not in anchors or ts < anchors[(f, z)]:
            anchors[(f, z)] = ts

    if not is_multi:
        f, z = pairs[0]
        own_ts = [ts for (ff, zz, ts) in m_keys if ff == f and zz == z]
        lo = min(own_ts) - pd.Timedelta(hours=context_hours)
        hi = max(own_ts) + pd.Timedelta(hours=context_hours)
        raw = lookup_raw_window([(f, z)], lo, hi)
    else:
        parts: dict[str, list[pd.DataFrame]] = {t: [] for t in _pipeline_da.RAW_TABLES}
        for f, z in pairs:
            anchor = anchors.get((f, z))
            if anchor is None:
                continue
            lo, hi = anchor - pd.Timedelta(hours=context_hours), anchor + pd.Timedelta(hours=context_hours)
            for table, df in lookup_raw_window([(f, z)], lo, hi).items():
                parts[table].append(df)
        raw = {t: pd.concat(dfs, ignore_index=True) for t, dfs in parts.items() if dfs}

    return raw, is_multi, anchors


def available_numeric_columns(raw: dict[str, pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for df in raw.values():
        for c in df.columns:
            if c in {"farm_id", "zone_id", "timestamp"} or c in cols:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def build_evidence_long_frame(
    raw: dict[str, pd.DataFrame],
    columns: list[str],
    *,
    is_multi: bool,
    anchors: dict[tuple[str, str], pd.Timestamp],
    m_keys: set[tuple[str, str, pd.Timestamp]],
) -> pd.DataFrame:
    """raw 테이블 dict를 (feature별로 그래프 하나씩 그릴 수 있게) long-format으로 합친다.

    반환 컬럼: `x`(단일 farm/zone이면 절대시간, 여러 farm/zone이면 각자 매칭시각
    기준 상대시간[시간 단위] -- 이래야 서로 다른 절대 날짜의 농장들을 한 그래프에서
    "매칭 시점 기준 앞뒤"로 겹쳐 비교할 수 있다), `timestamp`(항상 절대시간, 툴팁용),
    `farm_zone`, `feature`, `value`, `matched`(이 정확한 시점이 서사가 실제로 쓴
    raw 시점인지 -- 앞뒤 문맥과 구분하는 플래그).
    """
    # G_growth는 실측 시각이 아니라 날짜 단위(observation_date)라 export 단계에서
    # 임의 시각(예: 15:00)이 붙는다 -- 그대로 정확 일치 비교를 하면 성장 서사는
    # 항상 matched=0이 나온다(실측 확인). 게다가 실측해보니 export된 START_DATE가
    # 실제 게시된 observation_date보다 일관되게 하루 이르다(예: START_DATE
    # 2025-03-23 15:00 vs 게시된 observation_date 2025-03-24 -- 정확히 9시간=KST
    # 오프셋만큼 자정을 건너뜀, growth_any/mixed_growth/crossfarm_growth 전부 동일
    # 패턴). 근본 원인(export 단계의 타임존 처리)은 이 UI 범위 밖이라 여기서는
    # 날짜 ±1일 허용치로 비교해 실제로는 맞는 매칭을 놓치지 않게 한다.
    m_dates = {(f, z, ts.normalize()) for f, z, ts in m_keys}

    frames = []
    for table, df in raw.items():
        value_cols = [c for c in columns if c in df.columns]
        if not value_cols:
            continue
        melted = df.melt(
            id_vars=["timestamp", "farm_id", "zone_id"], value_vars=value_cols,
            var_name="feature", value_name="value",
        )
        melted = melted.dropna(subset=["value"])
        if melted.empty:
            continue
        melted["farm_zone"] = melted["farm_id"].astype(str) + "/z" + melted["zone_id"].astype(str)
        if table == "G_growth":
            def _growth_matched(f: str, z: str, t: pd.Timestamp) -> bool:
                d = pd.Timestamp(t).normalize()
                return any(
                    fd == str(f) and zd == str(z) and abs((d - md).days) <= 1
                    for fd, zd, md in m_dates
                )

            melted["matched"] = [
                _growth_matched(f, z, t)
                for f, z, t in zip(melted["farm_id"], melted["zone_id"], melted["timestamp"])
            ]
        else:
            melted["matched"] = [
                (str(f), str(z), pd.Timestamp(t)) in m_keys
                for f, z, t in zip(melted["farm_id"], melted["zone_id"], melted["timestamp"])
            ]
        if is_multi:
            anchor_ts = [anchors.get((str(f), str(z))) for f, z in zip(melted["farm_id"], melted["zone_id"])]
            melted = melted[[a is not None for a in anchor_ts]]
            anchor_ts = [a for a in anchor_ts if a is not None]
            melted["x_hours"] = [
                (pd.Timestamp(t) - a).total_seconds() / 3600.0 for t, a in zip(melted["timestamp"], anchor_ts)
            ]
        else:
            melted["x_hours"] = None
        frames.append(melted)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def instance_detail(instance_events: pd.DataFrame) -> dict:
    """한 시퀀스의 이벤트 목록(event_position 오름차순)에서 farm/zone/대표 timestamp를 뽑는다."""
    last = instance_events.iloc[-1]
    return {
        "sequence_id": last["sequence_id"],
        "narrative_id": last["narrative_id"],
        "farm_id": str(last["farm_id"]),
        "zone_id": str(last["zone_id"]) if last["zone_id"] not in (None, "") else "",
        "target_timestamp": str(last["START_DATE"]),
        "n_events": len(instance_events),
        "op_eligible": bool(last.get("op_eligible", False)),
        "order_semantics": last.get("order_semantics", ""),
    }
