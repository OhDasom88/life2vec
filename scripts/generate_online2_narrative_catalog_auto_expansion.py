#!/usr/bin/env python3
"""서사 카탈로그를 88개에서 대폭 확장하는 조합적(combinatorial) 생성기 — 검토용 dry-run.

55농장×13일 센서데이터+사진에 비해 서사가 100개도 안 되는 게 이상하다는 지적에
대한 응답. 손으로 하나씩 한국어 설명을 쓰는 대신, 이미 검증된 조합 가능한
매처(`MaterializerRegistry.SIMPLE_PREFIXES`, `scripts/generate_online2_narrative_catalog.py`의
`occurrence()`가 실데이터로 바로 검증 가능)를 **feature × matcher × 파라미터
변형**으로 프로그래밍적으로 조합해서 만든다. 한국어 설명은 feature별 별칭
(`ALIASES`)을 템플릿에 채워 넣어 자동 생성한다.

세 가지 추가 요청 반영:
1. curated pair는 관계 유형별로 별도 카테고리(환경내부/근권내부/환경-근권/
   환경-관수/관수-근권/제어반응/제어개입/생육-제어)로 관리 — 이전엔 "이 pair의
   첫 컬럼이 어느 도메인이냐"로만 분류해서 관계 유형 정보가 카테고리에 안 남았다.
2. 페어(2자간)뿐 아니라 다자간(3개 이상) 관계 서사도 탐색 — `multi_trend:col1+col2+
   col3:hours` 매처 신규 추가(같은 구간 안에서 N개 컬럼 전부 유의미하게 변화).
3. SensorLM(arxiv 2506.09108)의 3단계 캡션 분류(통계형/구조형/의미형)를 라벨로
   붙인다. 통계형(순수 분포 요약, 임계값 조건 없음)이 기존에 하나도 없었다는 걸
   확인해서 신규로 추가(feature별 all_hourly 일별 요약).

이 스크립트는 **실제 카탈로그 CSV를 쓰지 않는다** — 후보를 전부 만든 뒤
기존 `occurrence()`로 실데이터 dry-run 검증만 하고, 결과를 카테고리/캡션계층별로
요약해 `outputs/online2/sae_pilot/narrative_auto_expansion_report.json`에 저장한다.
실제 반영(카탈로그에 추가 + 원시 빌드 재실행)은 이 리포트를 검토한 뒤 별도로 진행한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_online2_narrative_catalog import (  # noqa: E402
    ALIASES,
    load_data,
    occurrence,
    spec,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.online2.materializers import MaterializerRegistry  # noqa: E402

# ---- feature groups (물리적 의미로 분류, 임의 조합 방지) ----
ENV_CONTINUOUS = [
    "inside_temp_c", "inside_humidity_pct", "outside_temp_c", "outside_humidity_pct",
    "outside_solar_radiation", "wind_speed_m_s",
    # rain_detected는 0/1 이진이 아니라 시간당 강우 비율(0~1 연속값)이고,
    # wind_direction_deg는 이 데이터셋 범위(48~360도, 0도 부근 공백이라 실제로는
    # wrap-around 문제 없음)라 둘 다 일반 연속형 매처를 그대로 적용해도 된다 --
    # 이전엔 두 컬럼 다 주 서사 대상으로 전혀 안 쓰였다(rain_detected는 STRESS_AND_PAIRS
    # 안에서 보조 조건으로만 등장, wind_direction_deg는 아예 미사용).
    "rain_detected", "wind_direction_deg",
]
ENV_GAS = ["co2_ppm"]
ROOT_CONTINUOUS = [
    "substrate_temp_c", "substrate_water_content_pct", "substrate_ec_ds_m",
    "ec_sensor", "ph_sensor",
]
FLOW_CONTINUOUS = ["total_flow_rate", "line_flow_rate"]
ACTUATOR_BINARY = [
    "fcu_fan", "fcu_pump", "circulation_fan", "co2_supply", "tube_rail",
    "roof_vent_left", "roof_vent_right", "shade_screen", "thermal_curtain",
    "nutrient_solution_system",
]
GROWTH_METRICS = [
    "plant_height_cm", "leaf_length_cm", "leaf_width_cm", "petiole_length_cm",
    "leaf_count", "crown_diameter_mm", "flower_truss_order",
    "opened_flower_count", "unopened_flower_count",
]

CONTINUOUS_ALL = ENV_CONTINUOUS + ENV_GAS + ROOT_CONTINUOUS + FLOW_CONTINUOUS

# ---- 관계 유형별 curated pair (임의 전체조합 X, 물리적으로 의미 있는 것만) ----
ENV_ENV_DIFF_PAIRS = [
    ("outside_temp_c", "inside_temp_c"),
    ("outside_humidity_pct", "inside_humidity_pct"),
    ("wind_speed_m_s", "inside_temp_c"),
    ("outside_solar_radiation", "inside_humidity_pct"),
    ("outside_temp_c", "outside_solar_radiation"),
    ("rain_detected", "inside_humidity_pct"),
    ("rain_detected", "outside_humidity_pct"),
    ("wind_direction_deg", "wind_speed_m_s"),
    ("wind_direction_deg", "outside_temp_c"),
]
ENV_ENV_JOINT_PAIRS = [
    ("inside_temp_c", "inside_humidity_pct"),
    ("outside_solar_radiation", "co2_ppm"),
    ("wind_speed_m_s", "outside_temp_c"),
    ("co2_ppm", "inside_humidity_pct"),
    ("wind_speed_m_s", "inside_humidity_pct"),
    ("rain_detected", "wind_speed_m_s"),
    ("rain_detected", "outside_solar_radiation"),
    ("wind_direction_deg", "outside_humidity_pct"),
]
ROOT_ROOT_DIFF_PAIRS = [
    ("ec_sensor", "substrate_ec_ds_m"),
    ("ph_sensor", "substrate_ec_ds_m"),
    ("ph_sensor", "ec_sensor"),
]
ROOT_ROOT_JOINT_PAIRS = [
    ("substrate_water_content_pct", "substrate_ec_ds_m"),
    ("ec_sensor", "ph_sensor"),
    ("ec_sensor", "substrate_water_content_pct"),
    ("substrate_temp_c", "substrate_ec_ds_m"),
    ("ph_sensor", "substrate_water_content_pct"),
]
ENV_ROOT_DIFF_PAIRS = [
    ("substrate_temp_c", "inside_temp_c"),
    ("substrate_temp_c", "outside_temp_c"),
]
ENV_ROOT_JOINT_PAIRS = [
    ("substrate_water_content_pct", "inside_humidity_pct"),
    ("inside_temp_c", "substrate_temp_c"),
    ("rain_detected", "substrate_water_content_pct"),
    ("outside_temp_c", "substrate_temp_c"),
]
FLOW_FLOW_DIFF_PAIRS = [
    ("total_flow_rate", "line_flow_rate"),
]
ENV_FLOW_JOINT_PAIRS = [
    ("total_flow_rate", "substrate_water_content_pct"),  # 실제로는 관수-근권이지만 기존 joint 세트 유지
    ("rain_detected", "total_flow_rate"),
    ("outside_solar_radiation", "line_flow_rate"),
]

# co_trend(양쪽 연속값, 방향 무관 공변화) — 관계 유형별로 분리
ENV_ROOT_CO_TREND_PAIRS = [
    ("outside_temp_c", "substrate_temp_c", 12),
    ("inside_humidity_pct", "substrate_water_content_pct", 12),
    ("rain_detected", "substrate_water_content_pct", 6),
    ("outside_humidity_pct", "substrate_water_content_pct", 12),
]
ENV_FLOW_CO_TREND_PAIRS = [
    ("outside_solar_radiation", "total_flow_rate", 6),
    ("outside_solar_radiation", "line_flow_rate", 6),
    ("inside_temp_c", "total_flow_rate", 6),
    ("rain_detected", "total_flow_rate", 6),
]
FLOW_ROOT_CO_TREND_PAIRS = [
    ("total_flow_rate", "substrate_water_content_pct", 6),
    ("line_flow_rate", "substrate_ec_ds_m", 12),
]
ROOT_ROOT_CO_TREND_PAIRS = [
    ("substrate_water_content_pct", "substrate_ec_ds_m", 12),
    ("substrate_water_content_pct", "substrate_temp_c", 12),
]

# 다자간(N≥3) 관계 — 같은 구간 안에서 N개 컬럼 전부 유의미하게 변화.
# 3/4/5자간을 arity별로 분리 관리(카테고리도 "N자간관계[-제어]"로 세분화) —
# N이 커질수록 전부 동시에 유의미하게 변할 확률이 곱셈적으로 줄어드니
# 3/4/5를 같은 취급하면 안 된다(실측 통과율이 크게 다름).
MULTI_TREND_3WAY = [
    # (cols, hours) — 환경→관수→근권 3단 연쇄, 환경 복합, 근권 복합 등
    (("outside_solar_radiation", "total_flow_rate", "substrate_water_content_pct"), 6),
    (("inside_temp_c", "roof_vent_left", "co2_ppm"), 6),
    (("outside_temp_c", "inside_temp_c", "substrate_temp_c"), 12),
    (("total_flow_rate", "substrate_water_content_pct", "substrate_ec_ds_m"), 12),
    (("inside_humidity_pct", "substrate_water_content_pct", "substrate_temp_c"), 12),
    (("outside_solar_radiation", "inside_temp_c", "inside_humidity_pct"), 6),
]
MULTI_TREND_4WAY = [
    (("outside_solar_radiation", "outside_temp_c", "inside_temp_c", "inside_humidity_pct"), 6),
    (("outside_temp_c", "inside_temp_c", "substrate_temp_c", "substrate_water_content_pct"), 12),
    (("total_flow_rate", "substrate_water_content_pct", "substrate_ec_ds_m", "substrate_temp_c"), 12),
    (("wind_speed_m_s", "outside_temp_c", "inside_temp_c", "inside_humidity_pct"), 6),
]
MULTI_TREND_5WAY = [
    (("outside_solar_radiation", "outside_temp_c", "inside_temp_c", "inside_humidity_pct", "substrate_temp_c"), 12),
    (("outside_solar_radiation", "inside_temp_c", "total_flow_rate", "substrate_water_content_pct", "substrate_ec_ds_m"), 12),
]

# 다자간 + 제어 — 위 co_trend 페어(환경-근권/관수-근권/근권내부/환경내부)
# 대부분에 액추에이터가 안 묶여 있었다는 지적 반영. 물리적으로 개입 가능한
# 액추에이터가 있는 조합마다 그 액추에이터를 3번째 항으로 추가해 "센서 둘이
# 같이 변하는 구간에 제어도 같이 움직였는지"를 본다 — trend_intervention의
# 다자간 버전.
MULTI_TREND_3WAY_ACTUATOR = [
    # (cols, hours) — 어느 co_trend/diff/joint 페어를 확장한 것인지 주석에 표기
    (("outside_temp_c", "substrate_temp_c", "thermal_curtain"), 12),  # ENV_ROOT_CO_TREND 확장
    (("inside_humidity_pct", "substrate_water_content_pct", "nutrient_solution_system"), 12),  # ENV_ROOT_CO_TREND 확장
    (("substrate_water_content_pct", "substrate_ec_ds_m", "nutrient_solution_system"), 12),  # ROOT_ROOT_CO_TREND 확장
    (("substrate_water_content_pct", "substrate_temp_c", "thermal_curtain"), 12),  # ROOT_ROOT_CO_TREND 확장
    (("line_flow_rate", "substrate_ec_ds_m", "tube_rail"), 12),  # FLOW_ROOT_CO_TREND 확장
    (("total_flow_rate", "substrate_water_content_pct", "nutrient_solution_system"), 6),  # FLOW_ROOT_CO_TREND 확장
    (("inside_temp_c", "inside_humidity_pct", "roof_vent_left"), 6),  # ENV_ENV_JOINT 확장
    (("co2_ppm", "inside_humidity_pct", "roof_vent_left"), 6),  # ENV_ENV_JOINT 확장
    (("outside_solar_radiation", "inside_temp_c", "shade_screen"), 6),  # ENV_ENV_JOINT 확장
    (("ec_sensor", "substrate_ec_ds_m", "nutrient_solution_system"), 12),  # ROOT_ROOT_DIFF 확장
    (("wind_speed_m_s", "inside_temp_c", "roof_vent_left"), 6),  # ENV_ENV_JOINT 확장
]
MULTI_TREND_4WAY_ACTUATOR = [
    (("outside_solar_radiation", "inside_temp_c", "inside_humidity_pct", "shade_screen"), 6),
    (("outside_temp_c", "inside_temp_c", "substrate_temp_c", "thermal_curtain"), 12),
    (("co2_ppm", "inside_temp_c", "inside_humidity_pct", "roof_vent_left"), 6),
    (("total_flow_rate", "substrate_water_content_pct", "substrate_ec_ds_m", "nutrient_solution_system"), 12),
    (("wind_speed_m_s", "outside_temp_c", "inside_temp_c", "roof_vent_left"), 6),
]
MULTI_TREND_5WAY_ACTUATOR = [
    (("outside_solar_radiation", "outside_temp_c", "inside_temp_c", "inside_humidity_pct", "roof_vent_left"), 6),
    (("outside_temp_c", "inside_temp_c", "substrate_temp_c", "substrate_water_content_pct", "thermal_curtain"), 12),
]

# 제어반응(액추에이터가 단일 환경조건에 반응) — 미래정보 금지라 "켠 뒤 어떻게
# 변했나"는 못 만들고 "이 조건에서 반응해서 켜졌다"(과거만 사용)만 가능.
ACTUATOR_ENV_RESPONSE_PAIRS = [
    ("fcu_fan", "inside_temp_c", "high", 0.90),
    ("fcu_pump", "inside_temp_c", "high", 0.90),
    ("circulation_fan", "inside_temp_c", "high", 0.90),
    ("circulation_fan", "co2_ppm", "high", 0.90),
    ("co2_supply", "co2_ppm", "low", 0.10),
    ("roof_vent_left", "inside_temp_c", "high", 0.90),
    ("roof_vent_right", "inside_temp_c", "high", 0.90),
    ("roof_vent_left", "co2_ppm", "high", 0.90),
    ("shade_screen", "outside_solar_radiation", "high", 0.90),
    ("thermal_curtain", "inside_temp_c", "low", 0.10),
    ("nutrient_solution_system", "substrate_water_content_pct", "low", 0.10),
    ("tube_rail", "substrate_water_content_pct", "low", 0.10),
    ("roof_vent_left", "rain_detected", "low", 0.10),  # 강우 시 환기창을 닫는 경향(낮은 강우일 때만 가동)
    ("roof_vent_right", "rain_detected", "low", 0.10),
    ("shade_screen", "wind_speed_m_s", "high", 0.90),  # 강풍 시 차양막 관련 반응
]

# 제어개입(센서 변화 구간 + 액추에이터 상태 전환, trend_intervention) — 둘 다
# 트리거 시점까지의 과거에 속해 미래정보 금지 원칙을 안 어긴다.
TREND_INTERVENTION_PAIRS = [
    ("inside_temp_c", "fcu_fan", 6),
    ("inside_temp_c", "roof_vent_left", 6),
    ("inside_temp_c", "thermal_curtain", 6),
    ("co2_ppm", "co2_supply", 6),
    ("co2_ppm", "circulation_fan", 6),
    ("outside_solar_radiation", "shade_screen", 3),
    ("inside_humidity_pct", "roof_vent_right", 6),
    ("substrate_water_content_pct", "nutrient_solution_system", 6),
    ("substrate_water_content_pct", "tube_rail", 6),
    ("substrate_ec_ds_m", "nutrient_solution_system", 12),
    ("substrate_temp_c", "thermal_curtain", 12),
    ("total_flow_rate", "nutrient_solution_system", 6),
    ("line_flow_rate", "tube_rail", 6),
]

# 지연반응(lag_response) — trend_intervention/co_trend/multi_trend는 "같은 구간
# 안 어딘가"에서 둘 다 움직이면 매칭되는 방향 중립적 공존 주장이라 "제어기가
# 켜진 뒤 정확히 N시간 뒤에 센서가 반응했다"는 지연(lag)을 표현 못한다(사용자
# 지적, 실측 확인: multi_trend 인스턴스가 원인/결과 순서 없이 window를 통째로
# 이벤트화함). (트리거 액추에이터, 반응 센서) 순서 고정 — C06/C07처럼 손으로
# 만든 트리거+비대칭 window 서사의 일반화. C07은 105~155분(≈2시간) 근권온도
# 지연을 전문가 근거로 명시했었다.
LAG_RESPONSE_PAIRS = [
    ("fcu_pump", "inside_temp_c"),
    ("fcu_fan", "inside_temp_c"),
    ("fcu_pump", "substrate_temp_c"),
    ("roof_vent_left", "inside_temp_c"),
    ("roof_vent_left", "inside_humidity_pct"),
    ("roof_vent_right", "inside_humidity_pct"),
    ("thermal_curtain", "inside_temp_c"),
    ("circulation_fan", "inside_humidity_pct"),
    ("circulation_fan", "co2_ppm"),
    ("co2_supply", "co2_ppm"),
    ("line_flow_rate", "substrate_water_content_pct"),
    ("line_flow_rate", "substrate_ec_ds_m"),
    ("nutrient_solution_system", "substrate_ec_ds_m"),
]
LAG_RESPONSE_HOURS = [1, 2, 3]

# 생육-제어: 생육은 시간별이 아니라 조사(survey) 단위라 window 개념이 다르다 --
# 기존 mixed_growth 매처(생육 변화 + 그 이전 hourly 맥락을 한 시퀀스로 묶음,
# 이미 구현돼 있음)를 재사용.
GROWTH_ACTUATOR_PAIRS = [
    ("plant_height_cm", "fcu_fan"),
    ("plant_height_cm", "nutrient_solution_system"),
    ("leaf_count", "roof_vent_left"),
    ("crown_diameter_mm", "thermal_curtain"),
]

STRESS_AND_PAIRS = [
    # (primary continuous, primary level, secondary bool-like col, secondary op)
    ("inside_temp_c", 0.90, "rain_detected", "positive"),
    ("inside_humidity_pct", 0.90, "inside_temp_c", "gt_median"),
    ("outside_solar_radiation", 0.90, "inside_temp_c", "gt_median"),
    ("substrate_water_content_pct", 0.10, "substrate_temp_c", "lt_median"),
    ("co2_ppm", 0.90, "co2_supply", "zero"),
    ("wind_speed_m_s", 0.90, "outside_temp_c", "lt_median"),
    ("substrate_ec_ds_m", 0.90, "substrate_water_content_pct", "lt_median"),
    ("inside_temp_c", 0.10, "outside_temp_c", "lt_median"),
]

TOD_BUCKETS = [
    ("아침", 6, 11), ("낮", 11, 16), ("저녁", 16, 21), ("밤", 21, 6),
]
# 3시간 단위 8구간 -- 4구간보다 세밀하지만 시간별(24)만큼 인접 구간이
# 거의 안 겹치는 수준까지는 안 가는 절충점.
TOD_BUCKETS_FINE = [
    ("00-03시", 0, 3), ("03-06시", 3, 6), ("06-09시", 6, 9), ("09-12시", 9, 12),
    ("12-15시", 12, 15), ("15-18시", 15, 18), ("18-21시", 18, 21), ("21-24시", 21, 24),
]
RUN_LENGTHS = [1, 2, 3, 6, 12, 24, 48, 72]
RUN_QUANTILE_LEVELS = [0.75, 0.90]
QUANTILE_LEVELS_HIGH = [0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
QUANTILE_LEVELS_LOW = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]

CATEGORY_KO = {
    "env": "환경", "root": "근권", "flow": "관수", "actuator": "제어",
    "growth": "생육", "cross": "횡단비교", "cross_pair": "횡단비교-페어",
    "env_env": "환경내부", "root_root": "근권내부", "flow_flow": "관수내부",
    "env_root": "환경-근권", "env_flow": "환경-관수", "flow_root": "관수-근권",
    "actuator_response": "제어반응", "actuator_intervention": "제어개입",
    "lag_response": "지연반응",
    "growth_actuator": "생육-제어", "stat": "통계형요약",
    "multi3": "3자간관계", "multi3_actuator": "3자간관계-제어",
    "multi4": "4자간관계", "multi4_actuator": "4자간관계-제어",
    "multi5": "5자간관계", "multi5_actuator": "5자간관계-제어",
}

# SensorLM(arxiv 2506.09108) 3단계 캡션 분류: 통계형(순수 분포 요약, 조건 없음)/
# 구조형(추세·임계값·급변 탐지)/의미형(라벨·상태·비교관계). matcher prefix로 자동 분류.
STRUCTURAL_PREFIXES = {
    "quantile_high", "quantile_low", "jump", "run_ge", "gt", "lt", "tod_high",
    "tod_low", "trend_intervention", "co_trend", "multi_trend", "zone_outlier_hourly",
    "reset",
}
SEMANTIC_PREFIXES = {
    "positive", "change", "fixed_zero", "and", "and_gt", "diff_positive",
    "joint_quantile", "growth_any", "mixed_growth", "crosszone_hourly",
    "crossfarm_hourly", "crossfarm_growth", "image_longitudinal_pair",
    "lag_response",
}


def caption_tier(matcher: str) -> str:
    prefix = matcher.split(":")[0]
    if prefix in {"all_hourly", "window_summary"}:
        return "statistical"
    if prefix in STRUCTURAL_PREFIXES:
        return "structural"
    if prefix in SEMANTIC_PREFIXES:
        return "semantic"
    return "unclassified"


def q(hourly: pd.DataFrame, col: str, level: float) -> float:
    return float(hourly[col].quantile(level))


def build_auto_specs(hourly: pd.DataFrame, growth: pd.DataFrame) -> list[dict]:
    specs: list[dict] = []
    idx = 0

    def next_id(prefix: str) -> str:
        nonlocal idx
        idx += 1
        return f"{prefix}{idx:04d}"

    def add(category: str, req_cols: list[str], opt_cols: list[str], matcher: str,
             name: str, purpose: str, max_events: int, resolution: str = "1시간") -> None:
        specs.append(
            spec(
                next_id("AUTO"), CATEGORY_KO.get(category, category), name, purpose,
                ";".join(req_cols) if len(req_cols) <= 2 else req_cols[0],
                ";".join(["farm_id", "zone_id", "timestamp"] + req_cols),
                ";".join(opt_cols), resolution,
                f"{name} 단일/연속 구간", "조건 충족 시작", "조건 종료 또는 단일 관측",
                "none", "empirical", "조합 생성 규칙", f"{name} 패턴", "탐색 후보",
                "관찰", "자동 생성;재검토 필요", "자동 생성이라 사람 검토 전 low-confidence로 표시",
                max_events, "하", "낮음", matcher,
            )
        )

    # A. 연속형 feature 단일 임계값/변화 (windowed + instant 두 버전) — 구조형
    for col in CONTINUOUS_ALL:
        category = (
            "env" if col in ENV_CONTINUOUS + ENV_GAS
            else "root" if col in ROOT_CONTINUOUS else "flow"
        )
        alias = ALIASES.get(col, col.upper())
        for max_events, span_ko in [(1, "단일시점"), (4, "중간구간"), (12, "장구간")]:
            add(category, [col], [], f"quantile_high:{col}",
                f"{alias} 상위분위 {span_ko}", f"{alias} 농장내 상위 10분위 {span_ko} 포착",
                max_events)
            add(category, [col], [], f"quantile_low:{col}",
                f"{alias} 하위분위 {span_ko}", f"{alias} 농장내 하위 10분위 {span_ko} 포착",
                max_events)
        add(category, [col], [], f"jump:{col}",
            f"{alias} 급변 시점", f"{alias} 직전 대비 이상 변화 포착", 1)
        if col in FLOW_CONTINUOUS:
            add(category, [col], [], f"reset:{col}",
                f"{alias} 리셋 시점", f"{alias}가 양수에서 0으로 리셋", 1)
        for qlevel in RUN_QUANTILE_LEVELS:
            for run in RUN_LENGTHS:
                level = q(hourly, col, qlevel)
                add(category, [col], [], f"run_ge:{col}:{level:.3f}:{run}",
                    f"{alias} 상위{int(100-qlevel*100)}분위 {run}시간 지속",
                    f"{alias}가 상위 {int(100-qlevel*100)}분위를 {run}시간 이상 연속",
                    max(4, run // 2))
        for level in QUANTILE_LEVELS_HIGH:
            val = q(hourly, col, level)
            add(category, [col], [], f"gt:{col}:{val:.3f}",
                f"{alias} {int(level*100)}분위 초과", f"{alias}가 {int(level*100)}분위 값 초과",
                1)
        for level in QUANTILE_LEVELS_LOW:
            val = q(hourly, col, level)
            add(category, [col], [], f"lt:{col}:{val:.3f}",
                f"{alias} {int(level*100)}분위 미만", f"{alias}가 {int(level*100)}분위 값 미만",
                1)
        for tod_ko, h0, h1 in TOD_BUCKETS:
            add(category, [col], [], f"tod_high:{col}:{h0}:{h1}",
                f"{alias} {tod_ko} 상위분위", f"{tod_ko} 시간대의 {alias} 상위 10분위", 1)
            add(category, [col], [], f"tod_low:{col}:{h0}:{h1}",
                f"{alias} {tod_ko} 하위분위", f"{tod_ko} 시간대의 {alias} 하위 10분위", 1)
        for tod_ko, h0, h1 in TOD_BUCKETS_FINE:
            add(category, [col], [], f"tod_high:{col}:{h0}:{h1}",
                f"{alias} {tod_ko} 상위분위", f"{tod_ko} 시간대의 {alias} 상위 10분위", 1)
            add(category, [col], [], f"tod_low:{col}:{h0}:{h1}",
                f"{alias} {tod_ko} 하위분위", f"{tod_ko} 시간대의 {alias} 하위 10분위", 1)

    # A2. 통계형 요약 (SensorLM "statistical" tier) — 임계값 조건 없이 하루 전체를
    # 그대로 담는 all_hourly, feature별로 개별 적용(기존 D01/D04/D15는 여러
    # feature를 한 번에 묶은 것뿐이라 "이 feature 하나의 일별 분포"라는 개별
    # 통계형 서사가 없었다).
    for col in CONTINUOUS_ALL:
        alias = ALIASES.get(col, col.upper())
        add("stat", [col], [], "all_hourly",
            f"{alias} 일별 요약", f"{alias}의 하루 전체 관측(최저/최고/평균 등 분포 파악용)", 24)

    # A3. 통계형 요약 — 하루보다 짧은 window(6/12시간)도 추가. 실측 확인:
    # 24시간 통째와 부분 구간(예: 00-06시)은 서로 다른 raw 행을 선택한다
    # (all_hourly는 그날 전체, window_summary는 그중 일부 시간대만) — 임계값
    # 통과 검증(quantile_high/low, tod_high/low)과는 별개로 "이 구간의 분포
    # 자체"를 담는 통계형 서사가 window 크기별로 갖춰지지 않아서 추가.
    WINDOW_SUMMARY_BUCKETS = [
        ("0-6시", 0, 6), ("6-12시", 6, 12), ("12-18시", 12, 18), ("18-24시", 18, 24),
        ("0-12시(오전)", 0, 12), ("12-24시(오후)", 12, 24),
    ]
    for col in CONTINUOUS_ALL:
        alias = ALIASES.get(col, col.upper())
        for bucket_ko, h0, h1 in WINDOW_SUMMARY_BUCKETS:
            add("stat", [col], [], f"window_summary:{h0}:{h1}",
                f"{alias} {bucket_ko} 요약", f"{alias}의 {bucket_ko} 구간 전체 관측(분포 파악용)", 12)

    # B. 액추에이터/이진 feature — 의미형(상태 라벨)
    for col in ACTUATOR_BINARY:
        alias = ALIASES.get(col, col.upper())
        add("actuator", [col], [], f"positive:{col}",
            f"{alias} 가동 시점", f"{alias}가 양수(가동)인 단일 시점", 1)
        add("actuator", [col], [], f"change:{col}",
            f"{alias} 전환 시점", f"{alias} 직전 대비 상태 전환", 1)
        add("actuator", [col], [], f"fixed_zero:{col}",
            f"{alias} 미가동 지속", f"{alias}가 전체 관측기간 0(미가동)", 24)

    # B2. 제어반응 (액추에이터가 단일 환경조건에 반응) — 의미형
    for actuator, env_col, direction, level in ACTUATOR_ENV_RESPONSE_PAIRS:
        a_alias, e_alias = ALIASES.get(actuator, actuator), ALIASES.get(env_col, env_col)
        env_val = q(hourly, env_col, level)
        if direction == "high":
            matcher = f"and_gt:{actuator}:0:{env_col}:{env_val:.3f}"
            desc = f"{e_alias} 상위분위일 때 {a_alias} 가동"
        else:
            matcher = f"and:{actuator}:0:{env_col}:{env_val:.3f}"
            desc = f"{e_alias} 하위분위일 때 {a_alias} 가동"
        add("actuator_response", [actuator, env_col], [], matcher, f"{a_alias} 반응({e_alias})", desc, 4)

    # B3. 제어개입 (센서 변화 구간 + 액추에이터 상태 전환, trend_intervention) — 구조형
    for env_col, actuator_col, hrs in TREND_INTERVENTION_PAIRS:
        e_alias, a_alias = ALIASES.get(env_col, env_col), ALIASES.get(actuator_col, actuator_col)
        add("actuator_intervention", [env_col, actuator_col], [],
            f"trend_intervention:{env_col}:{actuator_col}:{hrs}",
            f"{e_alias} 변화중 {a_alias} 개입", f"{e_alias}가 {hrs}시간 동안 변화하는 구간에 {a_alias} 상태 전환 포함",
            max(6, hrs))

    # B4. co_trend — 방향 무관 연속값 공변화, 관계 유형별 분리 — 구조형
    for pairs, category in [
        (ENV_ROOT_CO_TREND_PAIRS, "env_root"),
        (ENV_FLOW_CO_TREND_PAIRS, "env_flow"),
        (FLOW_ROOT_CO_TREND_PAIRS, "flow_root"),
        (ROOT_ROOT_CO_TREND_PAIRS, "root_root"),
    ]:
        for col1, col2, hrs in pairs:
            a1, a2 = ALIASES.get(col1, col1), ALIASES.get(col2, col2)
            add(category, [col1, col2], [], f"co_trend:{col1}:{col2}:{hrs}",
                f"{a1}·{a2} 공변화", f"{a1}와 {a2}가 같은 {hrs}시간 구간에서 함께 유의미하게 변화", max(6, hrs))

    # B5. 다자간(N≥3) 공변화 — multi_trend, 구조형. arity(3/4/5)별로 카테고리
    # 분리 — N이 커질수록 전부 동시 매칭될 확률이 곱셈적으로 줄어서 3자간과
    # 5자간을 같은 카테고리로 묶으면 통과율 차이가 안 보인다.
    multi_groups = [
        (MULTI_TREND_3WAY, "multi3", False), (MULTI_TREND_3WAY_ACTUATOR, "multi3_actuator", True),
        (MULTI_TREND_4WAY, "multi4", False), (MULTI_TREND_4WAY_ACTUATOR, "multi4_actuator", True),
        (MULTI_TREND_5WAY, "multi5", False), (MULTI_TREND_5WAY_ACTUATOR, "multi5_actuator", True),
    ]
    for combos, category, has_actuator in multi_groups:
        for cols, hrs in combos:
            aliases = [ALIASES.get(c, c) for c in cols]
            key = "multi_trend:" + "+".join(cols) + f":{hrs}"
            suffix = "(제어포함)" if has_actuator else ""
            name = "·".join(aliases) + f" 동시변화{suffix}"
            add(category, list(cols), [], key, name,
                f"{'·'.join(aliases)} {len(cols)}개 지표가 같은 {hrs}시간 구간에서 모두 유의미하게 변화", max(8, hrs))

    # B6. 생육-제어 (기존 mixed_growth 매처 재사용) — 의미형
    for growth_col, actuator_col in GROWTH_ACTUATOR_PAIRS:
        g_alias, a_alias = ALIASES.get(growth_col, growth_col), ALIASES.get(actuator_col, actuator_col)
        add("growth_actuator", [growth_col, actuator_col], [], "mixed_growth",
            f"{g_alias} 변화기 {a_alias} 맥락", f"{g_alias} 조사간 변화 구간의 {a_alias} 가동 이력",
            8, resolution="period")

    # C. 관계 유형별 diff_positive / joint_quantile 페어 — 의미형
    diff_groups = [
        (ENV_ENV_DIFF_PAIRS, "env_env"), (ROOT_ROOT_DIFF_PAIRS, "root_root"),
        (ENV_ROOT_DIFF_PAIRS, "env_root"), (FLOW_FLOW_DIFF_PAIRS, "flow_flow"),
    ]
    for pairs, category in diff_groups:
        for c1, c2 in pairs:
            a1, a2 = ALIASES.get(c1, c1), ALIASES.get(c2, c2)
            add(category, [c1, c2], [], f"diff_positive:{c1}:{c2}",
                f"{a1}-{a2} 차이", f"{a1}와 {a2}의 차이가 0이 아닌 시점", 1)
    joint_groups = [
        (ENV_ENV_JOINT_PAIRS, "env_env"), (ROOT_ROOT_JOINT_PAIRS, "root_root"),
        (ENV_ROOT_JOINT_PAIRS, "env_root"), (ENV_FLOW_JOINT_PAIRS, "env_flow"),
    ]
    for pairs, category in joint_groups:
        for c1, c2 in pairs:
            a1, a2 = ALIASES.get(c1, c1), ALIASES.get(c2, c2)
            add(category, [c1, c2], [], f"joint_quantile:{c1}:{c2}",
                f"{a1}·{a2} 동시 상위", f"{a1}와 {a2}가 동시에 상위 10분위", 1)

    # C2. "and:" 결합 스트레스 패턴 (2개 조건 동시 충족) — 의미형, 환경내부로 분류
    for c1, level, c2, op in STRESS_AND_PAIRS:
        a1, a2 = ALIASES.get(c1, c1), ALIASES.get(c2, c2)
        v1 = q(hourly, c1, level)
        if op in {"positive", "zero"}:
            v2 = 0.0
        elif op in {"gt_median", "lt_median"}:
            v2 = q(hourly, c2, 0.5)
        else:
            continue
        add("env_env", [c1, c2], [], f"and:{c1}:{v1:.3f}:{c2}:{v2:.3f}",
            f"{a1}·{a2} 복합조건", f"{a1} 임계 초과와 {a2} 조건 동시 충족", 1)

    # C3. 전수 조합(상관계수 필터링) — "pair를 curated로만 골라서 빠진 조합이
    # 있는 게 아니냐"는 지적 반영. 실측: CONTINUOUS_ALL 16개 컬럼의 120개
    # 가능한 pair 중 |상관계수|>0.85인 건 total_flow_rate/line_flow_rate
    # 하나뿐(r=1.00, 사실상 중복 컬럼) — 나머지 119개는 raw 데이터 자체가
    # 서로 다른 행을 고르므로(약한 필요조건) 다른 패턴일 후보. 이미 위
    # curated 목록에 있는 pair는 건너뛰고 나머지만 추가.
    import itertools

    _corr = hourly[CONTINUOUS_ALL].corr().abs()
    _already_covered = set()
    for _pairs in [
        ENV_ENV_DIFF_PAIRS, ENV_ENV_JOINT_PAIRS, ROOT_ROOT_DIFF_PAIRS, ROOT_ROOT_JOINT_PAIRS,
        ENV_ROOT_DIFF_PAIRS, ENV_ROOT_JOINT_PAIRS, FLOW_FLOW_DIFF_PAIRS, ENV_FLOW_JOINT_PAIRS,
    ]:
        for _c1, _c2 in _pairs:
            _already_covered.add(frozenset((_c1, _c2)))

    def _domain(col: str) -> str:
        if col in ENV_CONTINUOUS + ENV_GAS:
            return "env"
        if col in ROOT_CONTINUOUS:
            return "root"
        return "flow"

    _domain_pair_category = {
        frozenset(("env",)): "env_env", frozenset(("root",)): "root_root",
        frozenset(("flow",)): "flow_flow", frozenset(("env", "root")): "env_root",
        frozenset(("env", "flow")): "env_flow", frozenset(("flow", "root")): "flow_root",
    }
    n_skipped_redundant = 0
    for c1, c2 in itertools.combinations(CONTINUOUS_ALL, 2):
        if frozenset((c1, c2)) in _already_covered:
            continue
        if _corr.loc[c1, c2] > 0.85:
            n_skipped_redundant += 1
            continue
        category = _domain_pair_category[frozenset((_domain(c1), _domain(c2)))]
        a1, a2 = ALIASES.get(c1, c1), ALIASES.get(c2, c2)
        add(category, [c1, c2], [], f"diff_positive:{c1}:{c2}",
            f"{a1}-{a2} 차이(전수)", f"{a1}와 {a2}의 차이가 0이 아닌 시점", 1)
        add(category, [c1, c2], [], f"joint_quantile:{c1}:{c2}",
            f"{a1}·{a2} 동시 상위(전수)", f"{a1}와 {a2}가 동시에 상위 10분위", 1)
    print(f"  [전수 pair] 상관계수 0.85 초과로 제외: {n_skipped_redundant}쌍")

    # D. 생육 개별 지표 (survey 1 vs 2) — 의미형
    for col in GROWTH_METRICS:
        alias = ALIASES.get(col, col.upper())
        add("growth", [col], [], "growth_any",
            f"{alias} 조사간 변화", f"{alias}의 첫/둘째 조사 변화", 3, resolution="period")
        add("cross", [col], [], "crossfarm_growth",
            f"{alias} 농장간 비교", f"동일 조사순번에서 {alias} 농장간 비교", 20,
            resolution="period")

    # E. 횡단비교(crosszone/crossfarm) 단일 feature — 의미형
    for col in ENV_CONTINUOUS + ENV_GAS + ROOT_CONTINUOUS + FLOW_CONTINUOUS:
        alias = ALIASES.get(col, col.upper())
        add("cross", [col], [], "crosszone_hourly",
            f"{alias} 구역간 비교", f"동일 농장 동시각 {alias} 구역간 비교", 4)
        add("cross", [col], [], "crossfarm_hourly",
            f"{alias} 농장간 비교", f"동일 지역시각 {alias} 농장간 비교", 20)

    # E2. 페어 기반 횡단비교 — 별도 카테고리(cross_pair)로 분리, 의미형
    all_pairs = (
        ENV_ENV_DIFF_PAIRS + ENV_ENV_JOINT_PAIRS + ROOT_ROOT_DIFF_PAIRS
        + ROOT_ROOT_JOINT_PAIRS + ENV_ROOT_DIFF_PAIRS + ENV_ROOT_JOINT_PAIRS
    )
    for c1, c2 in all_pairs:
        a1, a2 = ALIASES.get(c1, c1), ALIASES.get(c2, c2)
        add("cross_pair", [c1, c2], [], "crosszone_hourly",
            f"{a1}·{a2} 구역간 비교", f"동일 농장 동시각 {a1}·{a2} 구역간 비교", 4)
        add("cross_pair", [c1, c2], [], "crossfarm_hourly",
            f"{a1}·{a2} 농장간 비교", f"동일 지역시각 {a1}·{a2} 농장간 비교", 20)

    # G. 구역 이상치 — 구조형
    for col in ENV_CONTINUOUS + ENV_GAS + ROOT_CONTINUOUS:
        alias = ALIASES.get(col, col.upper())
        add("cross", [col], [], "zone_outlier_hourly",
            f"{alias} 구역 이상치", f"동일 농장 동시각 다른 구역 대비 {alias} 이상 구역", 1)

    # H. 지연반응(lag_response) — 트리거(액추에이터 ON) 후 정확히 N시간 뒤 반응
    # 컬럼이 유의미하게 움직였는지 요구하는 방향성 있는 지연 인과 주장.
    # trend_intervention/co_trend/multi_trend는 "같은 구간 안 어딘가"에서 둘 다
    # 움직이면 매칭되는 방향 중립적 공존 주장이라 이 claim을 표현 못한다(사용자
    # 지적, 실측 확인). 맨 뒤(H 섹션)에 추가 -- 중간에 넣으면 이후 섹션들의
    # AUTO#### id가 밀려서 이미 빌드된 코퍼스(v2_build_expanded_full)의
    # narrative_id 매핑과 어긋난다.
    for actuator, response_col in LAG_RESPONSE_PAIRS:
        a_alias, r_alias = ALIASES.get(actuator, actuator), ALIASES.get(response_col, response_col)
        for lag in LAG_RESPONSE_HOURS:
            add("lag_response", [actuator, response_col], [],
                f"lag_response:{actuator}:{response_col}:{lag}",
                f"{a_alias} 가동 {lag}시간 뒤 {r_alias} 반응",
                f"{a_alias}가 켜진 시점 대비 정확히 {lag}시간 뒤 {r_alias}가 평소 변동폭 이상 움직였는지",
                max(6, lag + 4))

    return specs


def main() -> None:
    print("loading real data ...")
    hourly, growth, actual_columns = load_data()
    print(f"  hourly={len(hourly)} rows, growth={len(growth)} rows")

    specs = build_auto_specs(hourly, growth)
    print(f"generated {len(specs)} raw candidates")

    validated = []
    failed = []
    for s in specs:
        if not MaterializerRegistry.supports(s["matcher"]):
            failed.append({"id": s["narrative_id"], "reason": "unsupported_matcher", "matcher": s["matcher"]})
            continue
        req = [c for c in (s["req"].split(";") if isinstance(s["req"], str) else s["req"])]
        missing = [c for c in req if c not in actual_columns]
        if missing:
            failed.append({"id": s["narrative_id"], "reason": "missing_columns", "missing": missing})
            continue
        try:
            count, unique_count, farms = occurrence(s, hourly, growth)
        except Exception as exc:  # noqa: BLE001
            failed.append({"id": s["narrative_id"], "reason": f"exception:{exc}"})
            continue
        # crossfarm_growth/crosszone_hourly/crossfarm_hourly류 SET_COMPARISON
        # 매처는 unique_count가 구조적으로 작다(예: crossfarm_growth는 survey_idx가
        # 0/1뿐이라 최대 2) -- 에피소드형과 같은 기준을 쓰면 부당하게 다 걸러짐.
        min_unique = 1 if s["matcher"] in {"crossfarm_growth", "crosszone_hourly", "crossfarm_hourly"} else 3
        if s["matcher"] == "all_hourly":
            min_unique = 1  # 통계형은 하루 단위 요약이라 인스턴스 자체가 적은 게 정상
        if count < 1 or unique_count < min_unique:
            failed.append({"id": s["narrative_id"], "reason": "too_few_matches", "count": count, "unique": unique_count})
            continue
        validated.append(
            {
                "narrative_id": s["narrative_id"], "category": s["category"], "name": s["name"],
                "matcher": s["matcher"], "trigger_row_count": count, "unique_instance_count": unique_count,
                "farms": len(farms), "max_events": s["max_events"], "caption_tier": caption_tier(s["matcher"]),
            }
        )

    print(f"validated (real matches): {len(validated)} / {len(specs)}")
    print(f"failed/filtered: {len(failed)}")

    by_category: dict[str, int] = {}
    by_matcher_prefix: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    for v in validated:
        by_category[v["category"]] = by_category.get(v["category"], 0) + 1
        prefix = v["matcher"].split(":")[0]
        by_matcher_prefix[prefix] = by_matcher_prefix.get(prefix, 0) + 1
        by_tier[v["caption_tier"]] = by_tier.get(v["caption_tier"], 0) + 1

    report = {
        "n_raw_candidates": len(specs),
        "n_validated": len(validated),
        "n_failed": len(failed),
        "by_category": by_category,
        "by_matcher_prefix": by_matcher_prefix,
        "by_caption_tier": by_tier,
        "validated_sample": validated[:50],
        "failed_reasons_sample": failed[:30],
        "validated_full": validated,
    }
    out_path = ROOT / "outputs/online2/sae_pilot/narrative_auto_expansion_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"\nby_category: {by_category}")
    print(f"by_matcher_prefix: {by_matcher_prefix}")
    print(f"by_caption_tier: {by_tier}")


if __name__ == "__main__":
    main()
