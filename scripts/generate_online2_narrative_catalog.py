from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets/agrichallenge/online2"
OUT = BASE / "online2_narrative_catalog.csv"
REGISTRY_DIR = BASE / "narratives/registries"
DISPOSITION_OUT = REGISTRY_DIR / "catalog_disposition_registry.csv"
GEMINI_CATALOG = BASE / "online2_narrative_catalog_ gemini - online2_narrative_catalog.csv"
CHATGPT_CATALOG = BASE / "online2_narrative_catalog chatgpt - online2_narrative_catalog.csv"

HEADERS = [
    "narrative_id", "category", "narrative_name_ko", "purpose", "data_sources",
    "required_columns", "optional_columns", "entity_scope", "time_resolution",
    "window_definition", "start_condition", "end_condition", "cell_atomization",
    "event_definition", "sequence_definition", "derived_features", "threshold_type",
    "threshold_or_rule", "expected_pattern", "agronomic_interpretation",
    "diagnosis_level", "confounders", "data_quality_checks", "token_example",
    "sequence_example", "max_events", "estimated_tokens", "applicable_farms",
    "expert_evidence", "implementation_priority", "confidence", "notes",
    "status", "materialization_key", "trigger_row_count", "unique_instance_count",
    "op_eligible", "order_semantics", "alignment_policy_id", "sampling_weight",
    "timestamp_precision",
]

KEYS = {"farm_id", "zone_id", "timestamp", "observation_date"}
ALIASES = {
    "outside_temp_c": "OUT_TEMP", "outside_humidity_pct": "OUT_RH",
    "outside_solar_radiation": "SOLAR", "wind_direction_deg": "WIND_DIR",
    "wind_speed_m_s": "WIND_SPEED", "rain_detected": "RAIN",
    "inside_temp_c": "IN_TEMP", "inside_humidity_pct": "IN_RH", "co2_ppm": "CO2",
    "fcu_fan": "FCU_FAN", "fcu_pump": "FCU_PUMP",
    "circulation_fan": "CIRC_FAN", "co2_supply": "CO2_SUPPLY",
    "tube_rail": "TUBE_RAIL", "roof_vent_left": "VENT_L",
    "roof_vent_right": "VENT_R", "shade_screen": "SHADE",
    "thermal_curtain": "THERMAL", "ec_sensor": "SOL_EC",
    "ph_sensor": "SOL_PH", "total_flow_rate": "TOTAL_FLOW",
    "line_flow_rate": "LINE_FLOW", "nutrient_solution_system": "NUTRIENT_SYS",
    "substrate_temp_c": "ROOT_TEMP", "substrate_water_content_pct": "ROOT_WC",
    "substrate_ec_ds_m": "ROOT_EC", "plant_height_cm": "PLANT_H",
    "leaf_length_cm": "LEAF_L", "leaf_width_cm": "LEAF_W",
    "petiole_length_cm": "PETIOLE_L", "leaf_count": "LEAF_N",
    "crown_diameter_mm": "CROWN_DIA", "flower_truss_order": "TRUSS",
    "opened_flower_count": "FLOWER_OPEN", "unopened_flower_count": "FLOWER_CLOSED",
}

EXPERT = {
    "cold": "동해관련 답변 p1; 6도 미만은 동해가 아니라 생육정지 또는 타발휴면 후보이며 동해는 0도 이하 노출",
    "gray": "동해관련 답변 p1; RH 90퍼센트 이상 약 4시간 연속은 잿빛곰팡이 방제 위험 기준",
    "growth": "동해관련 답변 p1; 2주 초장 증가 5.5cm는 영양생장형 후보이나 관부직경 감소 일반화는 부적합",
    "calcium": "동해관련 답변 pp1-2; 관수부족과 낮은 야간 함수율 및 높은 EC 지속이 칼슘 이동 저하 위험과 연관",
    "mite": "응애와 시들음병 답변 p1; 고온저습에서 생활사 가속; 25도 이하와 RH 65퍼센트 유지 권고",
    "wilt": "응애와 시들음병 답변 p2; 고온과 근권 과습은 시들음병 우호환경이나 확정은 배양 동정 필요",
    "condensation": "동해관련 답변 p2; 일출 시 과습과 온도급변 결로는 꽃곰팡이 및 수정불량 위험",
    "powdery": "동해관련 답변 p2; 오전 다습과 한낮 저습 반복이 우호환경; EC 단독 예측 금지",
    "anthracnose": "동해관련 답변 p2; 주간 25도 초과 지속과 강우 또는 두상관수 조건은 탄저병 가능성",
    "salt": "생리장해 및 병충해 진단 p2; F356651에서 pH 약 4와 EC 약 3 및 뿌리 염분피해 진단",
    "irrigation": "답변서 pp2-3; 첫 관수는 일출 후 약 1시간 또는 일사 100W 도달 시점; 1회 관수 후 함수율 약 2에서 3퍼센트포인트 상승; 예시 톱니는 약 8회",
    "rootlag": "답변서 p3; 근권온도의 실내기온 추종 지연은 확인 데이터에서 약 105분에서 155분",
    "sensorjump": "답변서 pp1-3; 자정 근권온도 2도에서 5도 순간 상승과 EC 0.66에서 1.65 단계 변화는 고장 보정 이동 또는 재삽입 가능성을 우선 검토",
    "rootec": "답변서 pp1-2; 생육단계별 EC 해석이 다르고 관수 직전 상승 및 관수 직후 공급 EC 접근; 공급값 없이 단정 금지",
    "none": "고정 전문가 임계값 없음; online2 농장 내 분포와 시간 패턴만 사용",
}


def spec(
    narrative_id: str, category: str, name: str, purpose: str, sources: str,
    req: str, opt: str, resolution: str, window: str, start: str, end: str,
    derived: str, threshold_type: str, rule: str, expected: str,
    interpretation: str, diagnosis: str, confounders: str, checks: str,
    max_events: int, priority: str, confidence: str, matcher: str,
    evidence: str = "none",
) -> dict:
    return locals()


SPECS = [
    # 일상 재배 15
    spec("D01", "일상재배", "24시간 온습도 일주기", "주야간 내부 환경 리듬 학습", "E_environment", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct", "outside_temp_c;outside_humidity_pct;outside_solar_radiation", "1시간", "현지 자정부터 24시간", "00시 이벤트", "23시 이벤트", "일최저;일최고;일교차;습도진폭;VPD일곡선", "none", "농장 구역 일별 패턴", "야간 냉각 후 주간 상승과 역상 습도", "기본 환경 리듬", "관찰", "계절;난방;환기", "24개 시각 완전성;범위;중복;VPD는 온습도에서 결정론적으로 계산", 24, "상", "높음", "all_hourly"),
    spec("D02", "일상재배", "일출 전후 환경 전환", "광 증가 전후 온습도 전환 학습", "E_environment", "farm_id;zone_id;timestamp;outside_solar_radiation;inside_temp_c;inside_humidity_pct", "outside_temp_c;co2_ppm", "1시간", "일사 최초 증가 전 3시간부터 후 4시간", "일사량이 야간 중앙값보다 상승", "시작 후 7시간", "일사변화량;온도변화량;습도변화량", "empirical", "농장 일별 일사 양의 변화 시점", "일사 상승 뒤 온도 상승과 습도 변화", "일출 반응 후보", "시간적 연관", "구름;인공조명 부재;계절", "일사 정체;경계일;시간순", 8, "상", "높음", "solar_up"),
    spec("D03", "일상재배", "일몰 전후 환경 전환", "광 소멸 전후 냉각과 고습화 학습", "E_environment;A_actuator", "farm_id;zone_id;timestamp;outside_solar_radiation;inside_temp_c;inside_humidity_pct", "thermal_curtain;outside_temp_c", "1시간", "일사 마지막 감소 전 3시간부터 후 4시간", "일사량 급감", "시작 후 7시간", "일사변화량;냉각률;습도상승률", "empirical", "농장 일별 일사 음의 변화 하위 10분위", "일사 감소 뒤 온도 하강과 습도 상승", "일몰 반응 후보", "시간적 연관", "난방;커튼;강우", "일사 정체;경계일", 8, "상", "높음", "solar_down"),
    spec("D04", "일상재배", "주야간 온도 차", "주간과 야간 환경 수준 비교", "E_environment;R_rootzone", "farm_id;zone_id;timestamp;inside_temp_c;outside_solar_radiation", "inside_humidity_pct;substrate_temp_c", "1시간", "하루 24시간", "00시", "23시", "일사 상위시간 주간평균;하위시간 야간평균;DIF", "empirical", "일사 농장 일별 중앙값으로 주야 구분", "주간 평균과 야간 평균의 차이", "주야간 관리 패턴", "관찰", "구름;계절", "주야 표본수;결측", 24, "상", "높음", "all_hourly"),
    spec("D05", "일상재배", "환기창 일중 운용", "좌우 천창 개도 패턴 학습", "A_actuator", "farm_id;zone_id;timestamp;roof_vent_left;roof_vent_right", "inside_temp_c;inside_humidity_pct;wind_speed_m_s", "1시간", "첫 개방 전 2시간부터 마지막 폐쇄 후 2시간", "좌우 중 하나가 0에서 증가", "양쪽 0 복귀 후 2시간", "개도변화량;좌우차;개방지속", "none", "0 초과 개방 구간", "개방 유지와 폐쇄 전환", "환기 운용", "제어 관찰", "자동제어 코드 의미;풍향", "0에서 100 범위;동시성", 24, "상", "높음", "positive:roof_vent_left"),
    spec("D06", "일상재배", "차광막 일중 운용", "차광막 위치 변화와 광환경 학습", "A_actuator;E_environment", "farm_id;zone_id;timestamp;shade_screen;outside_solar_radiation", "inside_temp_c", "1시간", "차광 변화 전후 각 3시간", "shade_screen 변화", "시작 후 6시간", "차광변화량;일사변화량", "none", "변화 시점 중심", "차광 위치 변화와 동시 광 변화", "차광 운용", "시간적 연관", "구름;표현 방향", "범위;고정값;시각정렬", 7, "상", "중간", "change:shade_screen"),
    spec("D07", "일상재배", "보온커튼 야간 운용", "야간 보온커튼 전개와 회수 패턴 학습", "A_actuator;E_environment", "farm_id;zone_id;timestamp;thermal_curtain;inside_temp_c", "outside_temp_c;substrate_temp_c", "1시간", "커튼 변화 전후 각 4시간", "thermal_curtain 변화", "시작 후 8시간", "커튼변화량;내외기온차", "none", "변화 시점 중심", "커튼 위치 변화와 내외기온차 유지", "보온 운용", "시간적 연관", "개도 방향;난방", "범위;변화 지속", 9, "상", "중간", "change:thermal_curtain"),
    spec("D08", "일상재배", "FCU 난방 운전", "팬과 펌프 운전 구간 학습", "A_actuator;E_environment", "farm_id;zone_id;timestamp;fcu_fan;fcu_pump;inside_temp_c", "substrate_temp_c;outside_temp_c", "1시간", "운전 시작 전 3시간부터 종료 후 4시간", "fcu_fan 또는 fcu_pump 비영", "연속 영 복귀 후 4시간", "운전상태;실내승온;지연", "none", "0과 비영 코드를 상태 토큰화", "운전 코드 뒤 실내온도 변화", "난방 반응 후보", "시간적 연관", "201 코드 의미;외기;커튼", "코드 사전;동시 운전", 24, "상", "중간", "positive:fcu_fan"),
    spec("D09", "일상재배", "순환팬 운전", "공기 순환 운전의 지속과 환경 균질화 학습", "A_actuator;E_environment", "farm_id;zone_id;timestamp;circulation_fan;inside_temp_c;inside_humidity_pct", "co2_ppm", "1시간", "운전 전 2시간부터 후 4시간", "circulation_fan 비영", "영 복귀 후 4시간", "운전지속;변동성전후", "none", "비영 상태 구간", "운전 중 환경 변동성 감소 후보", "순환 운용", "시간적 연관", "201 코드 의미;환기", "코드 사전;고정값", 12, "중", "중간", "positive:circulation_fan"),
    spec("D10", "일상재배", "CO2 공급 운전", "공급 명령과 농도 변화 학습", "A_actuator;E_environment", "farm_id;zone_id;timestamp;co2_supply;co2_ppm", "roof_vent_left;roof_vent_right", "1시간", "공급 전 2시간부터 후 5시간", "co2_supply 비영", "영 복귀 후 5시간", "공급지속;CO2변화량;최대지연", "none", "비영 공급 코드", "공급 뒤 농도 상승 또는 무반응", "CO2 반응 후보", "시간적 연관", "환기;작물흡수;201 코드", "CO2 범위;명령 코드", 12, "중", "중간", "positive:co2_supply"),
    spec("D11", "일상재배", "양액 공급 일중 패턴", "양액 시스템과 유량 시계열 학습", "A_actuator", "farm_id;zone_id;timestamp;nutrient_solution_system;total_flow_rate;line_flow_rate", "ec_sensor;ph_sensor", "1시간", "하루 24시간", "첫 비영 유량", "마지막 비영 유량 후 2시간", "공급횟수;유량합;첫관수;마지막관수", "none", "유량 비영 이벤트", "주간 펄스형 공급", "관수 운용", "제어 관찰", "유량이 누적값일 가능성;코드 이상", "단조성;리셋;음수", 24, "상", "중간", "positive:total_flow_rate"),
    spec("D12", "일상재배", "근권 함수율 톱니 패턴", "관수 후 상승과 건조 하강 반복 학습", "R_rootzone;A_actuator", "farm_id;zone_id;timestamp;substrate_water_content_pct;line_flow_rate", "substrate_ec_ds_m;outside_solar_radiation", "1시간", "24시간 이동창", "함수율 양의 변화", "다음 양의 변화 직전", "함수율차;상승폭;건조기울기", "hybrid", "농장 구역별 양의 변화와 후속 하강;전문가 1회 관수 상승폭은 참고만 사용", "급상승 뒤 완만한 하강", "관수 반응 후보", "시간적 연관", "센서 위치;배액;유량 의미", "점프;고정;시간정렬", 24, "상", "높음", "change:substrate_water_content_pct", "irrigation"),
    spec("D13", "일상재배", "배지 EC 일중 변화", "근권 염류 농도 리듬 학습", "R_rootzone;A_actuator", "farm_id;zone_id;timestamp;substrate_ec_ds_m;line_flow_rate", "ec_sensor;substrate_water_content_pct", "1시간", "하루 24시간", "00시", "23시", "EC일범위;관수전후차", "none", "고정 임계값 없이 변화량", "관수 전후 EC 희석 또는 농축", "근권 EC 반응 후보", "시간적 연관", "센서 보정;배액", "0.15에서 5 범위;점프", 24, "상", "높음", "change:substrate_ec_ds_m", "rootec"),
    spec("D14", "일상재배", "양액 EC와 pH 안정성", "공급 설정 센서의 일중 안정성 학습", "A_actuator", "farm_id;zone_id;timestamp;ec_sensor;ph_sensor", "nutrient_solution_system;line_flow_rate", "1시간", "하루 24시간", "00시", "23시", "일중 중앙값;MAD;범위", "empirical", "농장 구역별 MAD와 변화", "공급 시 설정값 유지 또는 변동", "양액 품질 관찰", "관찰", "센서값과 설정값 구분 불명", "pH 범위;EC 영값;고정값", 24, "중", "중간", "all_hourly"),
    spec("D15", "일상재배", "내외기 환경 차", "온실 완충 효과 학습", "E_environment", "farm_id;zone_id;timestamp;outside_temp_c;inside_temp_c;outside_humidity_pct;inside_humidity_pct", "roof_vent_left;thermal_curtain", "1시간", "하루 24시간", "00시", "23시", "내외온도차;내외습도차", "none", "시간별 차이", "외기 변동 대비 내부 완충", "온실 미기후", "관찰", "센서 위치;구역 공유 외기", "물리 범위;동시성", 24, "상", "높음", "all_hourly"),
    spec("D16", "일상재배", "양액 시스템 전환 순간", "양액 시스템 상태가 바뀐 단일 시각만 포착", "A_actuator", "farm_id;zone_id;timestamp;nutrient_solution_system", "line_flow_rate;total_flow_rate", "1시간", "전환 시각 단일 관측", "직전값과 다른 값 관측", "동일 관측", "전환여부", "none", "직전 시각 대비 값 변화", "단일 시각 상태 전환", "즉시 반응 후보", "관찰", "관수 스케줄;센서 지연", "직전값 보존;중복 제거", 1, "중", "중간", "change:nutrient_solution_system"),
    spec("D17", "일상재배", "순환팬 가동 순간", "순환팬이 켜진 단일 시각만 포착", "A_actuator", "farm_id;zone_id;timestamp;circulation_fan", "inside_temp_c;inside_humidity_pct", "1시간", "가동 시각 단일 관측", "circulation_fan 양수값 관측", "동일 관측", "가동여부", "none", "circulation_fan 양수", "단일 시각 가동 상태", "즉시 반응 후보", "관찰", "제어 로직;센서 지연", "직전값 보존;중복 제거", 1, "중", "중간", "positive:circulation_fan"),
    # 원인 반응 15
    spec("C01", "원인반응", "일사 상승 후 환기 반응", "일사와 온도 상승 뒤 천창 작동 후보 탐지", "E_environment;A_actuator", "farm_id;zone_id;timestamp;outside_solar_radiation;inside_temp_c;roof_vent_left;roof_vent_right", "inside_humidity_pct", "1시간", "일사 급증 전 2시간부터 후 6시간", "일사 양의 변화 상위 10분위", "시작 후 8시간", "일사변화;온도변화;환기지연", "empirical", "농장별 변화량 90분위", "일사와 온도 상승 후 환기 개도 증가", "제어 반응 후보", "시간적 연관", "시간대;자동설정;구름", "선후관계;미래 누수 금지", 9, "상", "중간", "solar_up"),
    spec("C02", "원인반응", "환기 후 온도 반응", "천창 개방 뒤 실내온도 변화 탐지", "A_actuator;E_environment", "farm_id;zone_id;timestamp;roof_vent_left;roof_vent_right;inside_temp_c;outside_temp_c", "wind_speed_m_s", "1시간", "개방 전 2시간부터 후 6시간", "천창 개도 증가", "시작 후 8시간", "개도변화;내외기온차;후속온도차", "none", "개도 증가 중심", "개방 뒤 실내온도 하강 또는 외기 접근", "환기 반응 후보", "시간적 연관", "난방;일사;풍속", "선후관계;동시 센서", 9, "상", "중간", "change:roof_vent_left"),
    spec("C03", "원인반응", "환기 후 습도 반응", "천창 개방 뒤 습도 회복 탐지", "A_actuator;E_environment", "farm_id;zone_id;timestamp;roof_vent_left;inside_humidity_pct;outside_humidity_pct", "wind_speed_m_s", "1시간", "개방 전 2시간부터 후 6시간", "천창 개도 증가", "시작 후 8시간", "개도변화;내외습도차;후속습도차", "none", "개도 증가 중심", "내부 고습 완화 또는 외기 고습 유입", "환기 반응 후보", "시간적 연관", "외기습도;안개", "선후관계;범위", 9, "상", "중간", "change:roof_vent_left"),
    spec("C04", "원인반응", "관수 후 함수율 반응", "유량 발생 뒤 근권 수분 상승 탐지", "A_actuator;R_rootzone", "farm_id;zone_id;timestamp;line_flow_rate;substrate_water_content_pct", "total_flow_rate;nutrient_solution_system", "1시간", "유량 발생 전 2시간부터 후 6시간", "line_flow_rate 양수", "시작 후 8시간", "유량;함수율상승;반응지연", "none", "양수 유량 중심", "관수 뒤 함수율 상승 또는 무반응", "관수 반응 후보", "시간적 연관", "유량 누적 여부;센서 위치", "리셋;동시성;선후관계", 9, "상", "중간", "positive:line_flow_rate"),
    spec("C05", "원인반응", "관수 후 배지 EC 반응", "유량 발생 뒤 배지 EC 희석 또는 농축 탐지", "A_actuator;R_rootzone", "farm_id;zone_id;timestamp;line_flow_rate;substrate_ec_ds_m", "substrate_water_content_pct;ec_sensor", "1시간", "유량 발생 전 2시간부터 후 6시간", "line_flow_rate 양수", "시작 후 8시간", "EC전후차;함수율전후차", "none", "양수 유량 중심", "관수 뒤 EC 하강 또는 양액 농도 반영", "근권 반응 후보", "시간적 연관", "배액;센서보정", "EC 점프;유량 의미", 9, "상", "중간", "positive:line_flow_rate"),
    spec("C06", "원인반응", "난방 후 실내 승온", "FCU 운전 뒤 실내온도 상승 탐지", "A_actuator;E_environment", "farm_id;zone_id;timestamp;fcu_pump;inside_temp_c;outside_temp_c", "fcu_fan;thermal_curtain", "1시간", "운전 전 3시간부터 후 6시간", "fcu_pump 비영", "시작 후 9시간", "운전상태;승온량;지연", "none", "비영 운전 중심", "외기 대비 실내온도 상승", "난방 반응 후보", "시간적 연관", "201 코드;일사", "선후관계;코드 사전", 10, "중", "중간", "positive:fcu_pump"),
    spec("C07", "원인반응", "난방 후 배지온도 지연", "난방 운전 뒤 근권온도 지연 반응 탐지", "A_actuator;E_environment;R_rootzone", "farm_id;zone_id;timestamp;fcu_pump;inside_temp_c;substrate_temp_c", "outside_temp_c", "1시간", "운전 전 3시간부터 후 8시간", "fcu_pump 비영", "시작 후 11시간", "실내승온지연;배지승온지연", "hybrid", "1시간 자료에서는 1에서 3시간 지연 후보;전문가 자료 105에서 155분 참고", "실내온도 후 배지온도 완만한 상승", "열전달 반응 후보", "시간적 연관", "일사;관수", "센서 점프;선후관계", 12, "중", "중간", "positive:fcu_pump", "rootlag"),
    spec("C08", "원인반응", "차광 후 일사 온도 변화", "차광 위치 변화 뒤 광과 온도 반응 탐지", "A_actuator;E_environment", "farm_id;zone_id;timestamp;shade_screen;outside_solar_radiation;inside_temp_c", "outside_temp_c", "1시간", "차광 변화 전 2시간부터 후 5시간", "shade_screen 변화", "시작 후 7시간", "차광변화;일사변화;온도변화", "none", "변화 시점 중심", "차광 변화와 광 및 온도 하강 후보", "차광 반응 후보", "시간적 연관", "구름;표현방향", "방향성 해석 보류", 8, "중", "중간", "change:shade_screen"),
    spec("C09", "원인반응", "강우 시 환기 제한과 고습", "강우 동시 제어와 내부 습도 변화 탐지", "E_environment;A_actuator", "farm_id;zone_id;timestamp;rain_detected;roof_vent_left;inside_humidity_pct", "outside_humidity_pct;wind_speed_m_s", "1시간", "강우 전 2시간부터 후 8시간", "rain_detected 양수", "강우 종료 후 4시간", "강우상태;개도변화;습도지속", "none", "강우 검출 구간", "환기 축소와 내부 고습 지속 후보", "기상 연관", "시간적 연관", "우적센서 노이즈;외기습도", "rain 범위;연속성", 16, "상", "중간", "positive:rain_detected"),
    spec("C10", "원인반응", "강풍 시 환기 조정", "풍속 상승과 천창 개도 조정 탐지", "E_environment;A_actuator", "farm_id;zone_id;timestamp;wind_speed_m_s;roof_vent_left;roof_vent_right", "wind_direction_deg", "1시간", "고풍속 전 2시간부터 후 5시간", "풍속 농장별 90분위 초과", "시작 후 7시간", "풍속분위;개도변화", "empirical", "농장별 풍속 90분위", "풍속 상승과 개도 축소 후보", "안전 제어 후보", "시간적 연관", "풍향;강우", "풍속 점프;구역 중복 외기", 8, "중", "중간", "quantile_high:wind_speed_m_s"),
    spec("C11", "원인반응", "CO2 공급 무반응", "공급 명령 뒤 농도 상승이 없는 구간 탐지", "A_actuator;E_environment", "farm_id;zone_id;timestamp;co2_supply;co2_ppm", "roof_vent_left;roof_vent_right", "1시간", "공급 전 1시간부터 후 4시간", "co2_supply 비영", "시작 후 5시간", "CO2후속변화;환기상태", "empirical", "공급 후 변화가 농장 MAD 이하", "명령은 있으나 농도 무반응", "운영 이상 후보", "반응 후보", "코드 의미;환기;흡수", "공급 선행;센서 고정", 6, "중", "낮음", "positive:co2_supply"),
    spec("C12", "원인반응", "관수 명령 무반응", "유량 뒤 함수율 상승이 없는 구간 탐지", "A_actuator;R_rootzone", "farm_id;zone_id;timestamp;line_flow_rate;substrate_water_content_pct", "nutrient_solution_system", "1시간", "유량 전 1시간부터 후 4시간", "line_flow_rate 양수", "시작 후 5시간", "후속함수율변화", "empirical", "후속 변화가 농장 MAD 이하", "유량은 있으나 함수율 무반응", "운영 이상 후보", "반응 후보", "누적유량;배액;센서위치", "유량 리셋;센서점프", 6, "중", "낮음", "positive:line_flow_rate"),
    spec("C13", "원인반응", "환경 선행 후 지연 환기", "실내 고온 또는 고습 뒤 늦은 환기 탐지", "E_environment;A_actuator", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct;roof_vent_left", "outside_temp_c", "1시간", "환경 급변 전 2시간부터 후 6시간", "온도 또는 습도 변화 상위 10분위", "시작 후 8시간", "환경변화시점;환기지연", "empirical", "농장별 변화량 90분위", "환경 변화가 먼저이고 환기가 후행", "지연 제어 후보", "시간적 연관", "설정값 미제공;외기", "미래 누수 금지;선후관계", 9, "중", "중간", "quantile_high:inside_temp_c"),
    spec("C14", "원인반응", "보온커튼 후 열보존", "커튼 변화 뒤 내외기온차 유지 탐지", "A_actuator;E_environment", "farm_id;zone_id;timestamp;thermal_curtain;inside_temp_c;outside_temp_c", "substrate_temp_c", "1시간", "커튼 변화 전 3시간부터 후 6시간", "thermal_curtain 변화", "시작 후 9시간", "내외기온차전후;냉각률", "none", "변화 시점 중심", "커튼 운용 뒤 냉각률 완화 후보", "보온 반응 후보", "시간적 연관", "표현 방향;난방", "방향성 해석 보류", 10, "중", "중간", "change:thermal_curtain"),
    spec("C15", "원인반응", "외기 급변의 내부 완충", "외기온 급변 대비 내부 반응 지연 탐지", "E_environment", "farm_id;zone_id;timestamp;outside_temp_c;inside_temp_c", "wind_speed_m_s;roof_vent_left", "1시간", "외기 급변 전 2시간부터 후 6시간", "외기온 변화량 상위 10분위", "시작 후 8시간", "외기변화;내기변화;감쇠비;지연", "empirical", "농장별 외기온 차분 90분위", "외기 급변보다 완만하고 지연된 내부 변화", "온실 완충 반응", "시간적 연관", "일사;환기", "센서 동시성;점프", 9, "상", "높음", "quantile_high:outside_temp_c"),
    # 장기 생육 10
    spec("G01", "장기생육", "2주 초장 변화", "조사 간 초장 변화 학습", "G_growth", "farm_id;zone_id;observation_date;plant_height_cm", "petiole_length_cm;leaf_count", "약 13일", "첫 조사부터 둘째 조사", "첫 observation_date", "둘째 observation_date", "초장차;일당변화", "none", "두 관측 차이", "초장 증가 또는 정체", "영양생장 후보", "관찰", "측정자;엽 자세", "정확히 두 시점;시간순", 3, "상", "높음", "growth_any", "growth"),
    spec("G02", "장기생육", "2주 엽장 변화", "잎 길이 변화 학습", "G_growth", "farm_id;zone_id;observation_date;leaf_length_cm", "leaf_width_cm;petiole_length_cm", "약 13일", "첫 조사부터 둘째 조사", "첫 조사", "둘째 조사", "엽장차;일당변화", "none", "두 관측 차이", "엽장 증가 또는 감소", "엽 생장 관찰", "관찰", "선정엽 차이", "시간순;범위", 3, "상", "높음", "growth_any"),
    spec("G03", "장기생육", "2주 엽폭 변화", "잎 폭 변화 학습", "G_growth", "farm_id;zone_id;observation_date;leaf_width_cm", "leaf_length_cm", "약 13일", "첫 조사부터 둘째 조사", "첫 조사", "둘째 조사", "엽폭차;엽장대엽폭비", "none", "두 관측 차이", "엽폭 변화와 잎 형태 변화", "엽 생장 관찰", "관찰", "선정엽 차이", "시간순;범위", 3, "상", "높음", "growth_any"),
    spec("G04", "장기생육", "2주 엽병장 변화", "엽병 신장 변화 학습", "G_growth", "farm_id;zone_id;observation_date;petiole_length_cm", "plant_height_cm", "약 13일", "첫 조사부터 둘째 조사", "첫 조사", "둘째 조사", "엽병장차;초장대비비율", "none", "두 관측 차이", "엽병 신장 또는 정체", "초형 변화 후보", "관찰", "측정 정의", "시간순;범위", 3, "상", "높음", "growth_any", "growth"),
    spec("G05", "장기생육", "엽수 증가와 신엽 전개", "새 잎 전개 여부 학습", "G_growth", "farm_id;zone_id;observation_date;leaf_count", "leaf_length_cm;leaf_width_cm", "약 13일", "첫 조사부터 둘째 조사", "첫 조사", "둘째 조사", "엽수차", "none", "두 관측 차이", "엽수 증가 또는 정체", "신엽 전개 관찰", "관찰", "적엽작업;조사 기준", "정수;시간순", 3, "상", "높음", "growth_any"),
    spec("G06", "장기생육", "관부직경 변화속도", "관부 비대 속도 학습", "G_growth", "farm_id;zone_id;observation_date;crown_diameter_mm", "plant_height_cm", "약 13일", "첫 조사부터 둘째 조사", "첫 조사", "둘째 조사", "관부직경차;일당변화", "none", "감소를 영양부족으로 일반화하지 않음", "비대 또는 측정 변동", "관부 비대 관찰", "관찰", "측정오차;위치", "감소도 측정변동으로 보존", 3, "상", "높음", "growth_any", "growth"),
    spec("G07", "장기생육", "화방 차수 진행", "화방 출현과 차수 변화 학습", "G_growth", "farm_id;zone_id;observation_date;flower_truss_order", "opened_flower_count;unopened_flower_count", "약 13일", "첫 조사부터 둘째 조사", "첫 조사", "둘째 조사", "화방차수차", "none", "두 관측 차이", "같은 차수 유지 또는 다음 화방 출현", "생식생장 진행", "관찰", "화방 제거;기록 기준", "정수;시간순", 3, "상", "높음", "growth_any"),
    spec("G08", "장기생육", "개화 진행", "미개화 꽃과 개화 꽃 변화 학습", "G_growth", "farm_id;zone_id;observation_date;opened_flower_count;unopened_flower_count", "flower_truss_order", "약 13일", "첫 조사부터 둘째 조사", "첫 조사", "둘째 조사", "개화수차;미개화수차;총꽃수차", "none", "두 관측 차이", "미개화 감소와 개화 증가 후보", "개화 진행", "관찰", "낙화;새 화방", "차수 동반 확인", 3, "상", "높음", "growth_any"),
    spec("G09", "장기생육", "구역 간 생육 불균일", "동일 농장 구역별 생육 차이 학습", "G_growth", "farm_id;zone_id;observation_date;plant_height_cm;leaf_count;crown_diameter_mm", "leaf_length_cm;petiole_length_cm", "조사일", "동일 농장 동일 조사일의 4개 구역", "조사일 첫 구역", "조사일 네 번째 구역", "농장내순위;구역간범위;robust_z", "empirical", "농장 조사일 내 상대 순위", "특정 구역의 반복적 상하위", "공간 불균일 후보", "상대 비교", "표본 식물 차이;관리 구역", "4개 구역 완전성", 4, "상", "높음", "crosszone_growth"),
    spec("G10", "장기생육", "영양 생식 균형 후보", "초장 엽병장과 화방 개화의 공동 변화 학습", "G_growth", "farm_id;zone_id;observation_date;plant_height_cm;petiole_length_cm;flower_truss_order;opened_flower_count", "leaf_count;crown_diameter_mm", "약 13일", "첫 조사부터 둘째 조사", "첫 조사", "둘째 조사", "생장변화벡터;개화변화벡터", "hybrid", "초장 5.5cm 증가는 후보 근거만 사용;다변량 상대 변화", "영양 지표와 생식 지표의 조합", "생장형 후보", "후보", "품종;생육단계;착과부담", "단일 지표 확정 금지", 3, "중", "중간", "growth_any", "growth"),
    # 스트레스 위험 12
    spec("S01", "스트레스위험", "야간 저온 생육정지 위험", "6도 미만 지속 노출 탐지", "E_environment", "farm_id;zone_id;timestamp;inside_temp_c", "outside_temp_c;fcu_pump", "1시간", "연속 저온 구간 최대 24시간", "inside_temp_c 6 미만", "6 이상 회복", "저온지속시간;최저온", "expert", "6도 미만은 생육정지 또는 휴면 위험 후보", "야간 저온 지속", "생육정지 또는 타발휴면 위험", "위험", "센서오차;짧은 노출", "동해 확정 표현 금지", 24, "상", "높음", "lt:inside_temp_c:6", "cold"),
    spec("S02", "스트레스위험", "영하 노출 동해 위험", "실제 0도 이하 내부 노출 탐지", "E_environment", "farm_id;zone_id;timestamp;inside_temp_c", "outside_temp_c;fcu_pump", "1시간", "영하 연속 구간 전후 각 3시간", "inside_temp_c 0 이하", "0 초과 회복 후 3시간", "영하시간;최저온;회복시간", "expert", "0도 이하 노출을 동해 위험으로 표시", "영하 노출", "동해 위험", "위험", "센서 보정;작물체 온도 차이", "영하 실측 확인;확진 금지", 24, "상", "높음", "le:inside_temp_c:0", "cold"),
    spec("S03", "스트레스위험", "고습 지속 잿빛곰팡이 위험", "RH 90퍼센트 이상 연속 구간 탐지", "E_environment", "farm_id;zone_id;timestamp;inside_humidity_pct", "inside_temp_c;roof_vent_left", "1시간", "연속 고습 구간 전후 각 2시간", "RH 90 이상", "90 미만 회복", "연속고습시간;VPD", "expert", "RH 90 이상 4시간 연속", "4시간 이상 고습", "잿빛곰팡이 우호환경", "위험", "엽면수분 미측정;센서포화", "연속성;확진 금지", 24, "상", "높음", "run_ge:inside_humidity_pct:90:4", "gray"),
    spec("S04", "스트레스위험", "고온 저습 응애 우호환경", "응애 생활사 가속 환경 탐지", "E_environment", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct", "outside_solar_radiation", "1시간", "주간 연속 조건 최대 12시간", "온도 25 초과와 RH 65 미만", "조건 해제", "지속시간;VPD", "expert", "25도 초과와 RH 65 미만을 우호환경으로 사용", "고온 저습 동시 지속", "응애 발생 우호환경", "우호환경", "실제 해충 밀도 미측정", "이미지 또는 예찰 없이는 확진 금지", 12, "상", "높음", "and:inside_temp_c:25:inside_humidity_pct:65", "mite"),
    spec("S05", "스트레스위험", "고온 과습 근권 시들음병 우호환경", "고온과 높은 근권 함수율 동시 구간 탐지", "E_environment;R_rootzone", "farm_id;zone_id;timestamp;inside_temp_c;substrate_water_content_pct;substrate_temp_c", "line_flow_rate", "1시간", "연속 조건 최대 48시간;40개 이벤트로 절단", "온도와 함수율이 농장별 상위 10분위", "둘 중 하나 해제", "공동상위지속;근권온도", "empirical", "농장별 온도와 함수율 90분위 동시 초과", "고온 과습 동시 지속", "시들음병 우호환경", "우호환경", "병원균 존재 미측정;센서 위치", "확진 금지;농장내 분위", 40, "중", "중간", "joint_quantile:inside_temp_c:substrate_water_content_pct", "wilt"),
    spec("S06", "스트레스위험", "저EC 장기 지속 후보", "배지 EC의 농장 내 저수준 지속 탐지", "R_rootzone", "farm_id;zone_id;timestamp;substrate_ec_ds_m", "substrate_water_content_pct;ec_sensor", "1시간", "최대 72시간 이동창;48개 이벤트로 절단", "농장별 EC 10분위 미만", "10분위 이상 회복", "저EC지속;농장내분위", "empirical", "농장별 10분위;특정 병 진단에 사용하지 않음", "상대적 저EC 지속", "약한 양분 수준 후보", "후보", "생육단계;관수 직후 희석", "흰가루병 예측 금지", 48, "중", "중간", "quantile_low:substrate_ec_ds_m", "powdery"),
    spec("S07", "스트레스위험", "고EC 근권 염류 스트레스 후보", "배지 EC 1.8 이상 지속 탐지", "R_rootzone", "farm_id;zone_id;timestamp;substrate_ec_ds_m", "substrate_water_content_pct;ph_sensor", "1시간", "연속 고EC 최대 72시간;48개 이벤트로 절단", "EC 1.8 이상", "1.8 미만 회복", "고EC지속;최대EC", "expert", "EC 1.8 이상이 지속될 때 높은 수준 후보", "고EC 지속", "근권 염류 스트레스 후보", "위험", "센서 보정;품종;배액", "단일 순간값 확정 금지", 48, "상", "높음", "run_ge:substrate_ec_ds_m:1.8:2", "calcium"),
    spec("S08", "스트레스위험", "관수 부족과 칼슘 이동 저하 위험", "높은 수요 조건에서 낮은 관수와 함수율 탐지", "E_environment;A_actuator;R_rootzone", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct;line_flow_rate;substrate_water_content_pct", "substrate_ec_ds_m", "1시간", "주간 12시간 이동창", "VPD 상위와 유량 및 함수율 하위", "조건 해제", "VPD;유량분위;함수율분위;마지막관수", "hybrid", "관수 부족을 핵심으로 농장별 분위 결합", "증산수요 대비 낮은 수분공급", "칼슘 이동 저하 위험", "위험", "VPD 근사;유량 누적 여부", "칼슘 결핍 확진 금지", 12, "중", "중간", "le:line_flow_rate:0", "calcium"),
    spec("S09", "스트레스위험", "일출 결로 수정불량 위험", "일출 전후 포화 접근과 온도 급변 탐지", "E_environment", "farm_id;zone_id;timestamp;outside_solar_radiation;inside_temp_c;inside_humidity_pct", "", "1시간", "일출 전 3시간부터 후 4시간", "일사 상승 시 RH 90 이상", "일출 후 4시간", "이슬점차근사;온도변화;고습지속", "hybrid", "일출 시 RH 90 이상과 온도 변화", "일출 전후 포화 접근", "꽃 결로 및 수정불량 위험", "위험", "엽온과 꽃온도 미측정", "결로 실측 아님;확진 금지", 8, "상", "중간", "and:inside_humidity_pct:90:inside_temp_c:40", "condensation"),
    spec("S10", "스트레스위험", "오전 고습 수정벌 활동 저하 가능성", "오전 장시간 고습 구간 탐지", "E_environment", "farm_id;zone_id;timestamp;inside_humidity_pct;outside_solar_radiation", "inside_temp_c", "1시간", "06시부터 12시", "오전 RH 90 이상", "12시 또는 회복", "오전고습시간", "expert", "오전 RH 90 장시간 지속", "오전 고습 지속", "수정벌 활동 저하 가능성", "가능성", "벌통 상태;농약;계절", "벌 활동 직접 측정 아님", 7, "중", "중간", "run_ge:inside_humidity_pct:90:2", "condensation"),
    spec("S11", "스트레스위험", "주야 습도 반복 흰가루병 우호환경", "오전 다습과 한낮 저습 반복 탐지", "E_environment", "farm_id;zone_id;timestamp;inside_humidity_pct;inside_temp_c", "outside_solar_radiation", "1시간", "연속 2일", "오전 고습 후 주간 RH 50 이하", "둘째 날 종료", "습도진폭;반복횟수", "expert", "오전 다습과 한낮 RH 50 이하 반복", "큰 습도 진폭 반복", "흰가루병 우호환경", "우호환경", "포자 존재;초세", "배지 EC로 예측 금지;확진 금지", 48, "중", "중간", "le:inside_humidity_pct:50", "powdery"),
    spec("S12", "스트레스위험", "고온 강우 탄저병 위험환경", "25도 초과 지속과 강우 동시 조건 탐지", "E_environment", "farm_id;zone_id;timestamp;inside_temp_c;rain_detected", "outside_temp_c;inside_humidity_pct", "1시간", "주간 조건 전후 각 6시간", "온도 25 초과와 강우 검출", "조건 해제 후 6시간", "고온시간;강우시간", "expert", "주간 25도 초과 지속과 강우를 위험환경으로 사용", "고온 강우 동시 또는 인접", "탄저병 우호환경", "위험", "두상관수 미제공;육묘기 여부", "확진 금지;계절 확인", 24, "중", "중간", "and:inside_temp_c:25:rain_detected:0", "anthracnose"),
    spec("S13", "스트레스위험", "강풍 순간", "농장 내 풍속 상위 10분위 단일 시각 포착", "E_environment", "farm_id;zone_id;timestamp;wind_speed_m_s", "wind_direction_deg;outside_temp_c", "1시간", "단일 시각", "농장 상위 10분위 풍속", "동일 관측", "풍속상대순위", "empirical", "농장별 상위 10분위", "단일 시각 강풍 후보", "환기·낙과 위험 후보", "관찰", "돌풍 순간성;센서 위치", "농장 내 상대 분포만 사용", 1, "중", "중간", "quantile_high:wind_speed_m_s"),
    spec("S14", "스트레스위험", "배양액 pH 이상 순간", "농장 내 pH 상위 10분위 단일 시각 포착", "R_rootzone", "farm_id;zone_id;timestamp;ph_sensor", "ec_sensor;substrate_ec_ds_m", "1시간", "단일 시각", "농장 상위 10분위 pH", "동일 관측", "pH상대순위", "empirical", "농장별 상위 10분위", "단일 시각 pH 이상 후보", "양분흡수 저해 위험 후보", "위험", "센서 보정 주기;확진 불가", "농장 내 상대 분포만 사용;확진 금지", 1, "중", "중간", "quantile_high:ph_sensor"),
    # 센서 운영 이상 12
    spec("A01", "센서운영이상", "실내온도 급격한 점프", "작물 반응과 센서 이상 분리", "E_environment", "farm_id;zone_id;timestamp;inside_temp_c", "outside_temp_c;substrate_temp_c", "1시간", "점프 전후 각 3시간", "차분 robust z 6 초과", "시작 후 3시간", "차분;농장MAD;동반센서차분", "empirical", "농장 구역별 차분 median과 MAD 기반 robust z", "단일 시점 점프 후 복귀", "sensor_anomaly", "이상 후보", "실제 급변;환기", "원시값 보존;동반 변화 확인", 7, "상", "높음", "jump:inside_temp_c"),
    spec("A02", "센서운영이상", "근권온도 급격한 순간 변화", "비현실적 근권온도 점프 탐지", "R_rootzone;E_environment", "farm_id;zone_id;timestamp;substrate_temp_c", "inside_temp_c", "1시간", "점프 전후 각 3시간", "차분 robust z 6 초과", "시작 후 3시간", "차분;robust_z", "hybrid", "농장 구역별 차분 MAD;자정 점프 우선 점검", "순간 점프 또는 단계 변화", "sensor_anomaly", "이상 후보", "관수 온도 영향", "실내온도 동반 여부", 7, "상", "높음", "jump:substrate_temp_c", "sensorjump"),
    spec("A03", "센서운영이상", "배지 EC 비정상 단계 변화", "EC 센서 이동 또는 보정 의심 탐지", "R_rootzone;A_actuator", "farm_id;zone_id;timestamp;substrate_ec_ds_m", "substrate_water_content_pct;line_flow_rate", "1시간", "점프 전후 각 6시간", "차분 robust z 6 초과", "새 수준 안정 또는 복귀", "차분;전후중앙값;함수율동반", "hybrid", "농장 구역별 차분 MAD;전문가 자료의 갑작스러운 2배 이상 변화는 센서 이동 우선 검토", "EC만 단계적으로 변함", "sensor_anomaly", "이상 후보", "실제 관수;농축", "함수율과 유량 동반 확인", 13, "상", "높음", "jump:substrate_ec_ds_m", "sensorjump"),
    spec("A04", "센서운영이상", "함수율 비정상 점프", "센서 재삽입 또는 이동 의심 탐지", "R_rootzone;A_actuator", "farm_id;zone_id;timestamp;substrate_water_content_pct", "line_flow_rate;substrate_ec_ds_m", "1시간", "점프 전후 각 6시간", "차분 robust z 6 초과", "새 수준 안정 또는 복귀", "차분;유량동반;EC동반", "empirical", "농장 구역별 차분 MAD", "유량 없이 함수율 급변", "sensor_anomaly", "이상 후보", "실제 관수", "유량 동반 확인", 13, "상", "높음", "jump:substrate_water_content_pct"),
    spec("A05", "센서운영이상", "순환팬 장기 무가동 후보", "14일 내내 순환팬 OFF인 구역을 운영 점검 대상으로 분리", "A_actuator;E_environment", "farm_id;zone_id;timestamp;circulation_fan", "inside_temp_c;inside_humidity_pct", "1시간", "케이스 전체 14일", "circulation_fan이 전 기간 0", "period_end", "ON비율;최장OFF시간", "none", "0은 OFF;201은 ON으로 정규화;전 기간 OFF를 점검 후보로 표시", "14일 동안 ON 이벤트 없음", "operation_anomaly 또는 의도적 미사용 후보", "이상 후보", "의도적 미사용;설비 미설치", "201을 비정상값으로 처리하지 않음;장비 구성 확인", 48, "상", "높음", "fixed_zero:circulation_fan"),
    spec("A06", "센서운영이상", "양액 시스템 코드 범위 이상", "nutrient_solution_system의 다값 코드 탐지", "A_actuator", "farm_id;zone_id;timestamp;nutrient_solution_system", "total_flow_rate;line_flow_rate", "1시간", "비정상 값 전후 각 2시간", "값 1 초과", "1 이하 복귀", "코드고유값;유량동반", "empirical", "관측 범위 0에서 402;1 초과를 미해독 코드로 분리", "희소한 큰 코드", "operation_anomaly 또는 미해독 상태", "이상 후보", "장비 상태 코드일 가능성", "코드북 없이는 고장 확정 금지", 8, "상", "높음", "gt:nutrient_solution_system:1"),
    spec("A07", "센서운영이상", "누적 유량 리셋 후보", "유량 급감과 리셋 패턴 탐지", "A_actuator", "farm_id;zone_id;timestamp;total_flow_rate;line_flow_rate", "nutrient_solution_system", "1시간", "감소 전후 각 3시간", "양수에서 0 또는 큰 음의 차분", "새 누적 시작 후 3시간", "음의차분;리셋전값", "none", "양수에서 영으로 감소", "높은 값 뒤 영으로 복귀", "operation_anomaly 또는 카운터 리셋", "이상 후보", "시간당 유량이면 정상 무관수", "누적성 먼저 판정", 7, "중", "중간", "reset:total_flow_rate"),
    spec("A08", "센서운영이상", "연관 센서 동시 점프", "환경과 근권 다변량 점프 분리", "E_environment;R_rootzone", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct;substrate_temp_c;substrate_water_content_pct", "substrate_ec_ds_m", "1시간", "점프 전후 각 3시간", "두 개 이상 차분 robust z 6 초과", "시작 후 3시간", "동시점프수;변화방향", "empirical", "개별 농장 구역 MAD 기반", "여러 센서가 같은 시각 불연속", "sensor_anomaly 또는 전원 사건", "이상 후보", "실제 관수와 환기 동시", "시각 동기화 확인", 7, "중", "중간", "jump:inside_humidity_pct"),
    spec("A09", "센서운영이상", "장기간 완전 고정 센서", "stuck sensor 구간 탐지", "A_actuator;E_environment;R_rootzone", "farm_id;zone_id;timestamp;tube_rail", "ec_sensor;ph_sensor;co2_ppm", "1시간", "연속 24시간 이상", "24시간 동일 값", "값 변화", "run_length;고유값수", "none", "연속 24개 동일", "완전 고정", "sensor_anomaly 또는 미사용 채널", "이상 후보", "실제 미사용 액추에이터", "센서와 제어 채널 구분", 48, "상", "높음", "fixed:tube_rail"),
    spec("A10", "센서운영이상", "좌우 천창 불연속", "좌우 개도 차이의 비정상 지속 탐지", "A_actuator;E_environment", "farm_id;zone_id;timestamp;roof_vent_left;roof_vent_right", "wind_direction_deg", "1시간", "연속 차이 최대 24시간", "좌우차 농장별 99분위", "차이 회복", "좌우절대차;지속시간", "empirical", "농장별 좌우차 99분위", "한쪽만 큰 개도", "operation_anomaly 후보", "이상 후보", "의도적 풍상풍하 제어", "풍향과 동반 확인", 24, "중", "중간", "diff_positive:roof_vent_left:roof_vent_right"),
    spec("A11", "센서운영이상", "구역 간 물리적 불연속", "동일 농장 동시각 구역 센서 차이 탐지", "E_environment;R_rootzone", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct;substrate_temp_c", "substrate_water_content_pct", "1시간", "동일 농장 동일 시각 4개 구역", "구역 범위 농장별 99분위", "다음 시각", "구역범위;이상구역", "empirical", "농장 내 동시각 상대 범위", "한 구역만 극단적", "sensor_anomaly 또는 공간 불균일", "이상 후보", "실제 구역 미기후", "4개 구역 완전성", 4, "상", "높음", "crosszone_hourly"),
    spec("A12", "센서운영이상", "제어값과 환경 반응 불일치", "환기 변화 뒤 온습도 모두 무반응인 구간 탐지", "A_actuator;E_environment", "farm_id;zone_id;timestamp;roof_vent_left;inside_temp_c;inside_humidity_pct", "outside_temp_c;wind_speed_m_s", "1시간", "제어 변화 전 2시간부터 후 4시간", "천창 변화", "시작 후 6시간", "제어변화;후속온습도MAD", "empirical", "후속 변화가 농장 MAD 이하", "제어 변화에도 환경 정체", "operation_anomaly 후보", "반응 후보", "외기와 내기 유사;작은 개도", "인과 확정 금지", 7, "중", "낮음", "change:roof_vent_left"),
    # 멀티모달 6
    spec("M01", "멀티모달", "14일 환경 누적과 초장 변화", "과거 환경 요약과 이후 생육 변화 연결", "E_environment;G_growth", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct;observation_date;plant_height_cm", "outside_solar_radiation", "1시간과 약 13일", "첫 조사 초과 둘째 조사 이하의 과거 전용 창", "첫 조사 직후", "둘째 조사 시점", "과거온도평균;고습시간;초장차", "none", "둘째 조사 이전 센서만 집계", "누적 환경과 초장 변화 조합", "환경 생육 연관", "시간적 연관", "처리;품종;초기생육", "둘째 조사 이후 데이터 금지", 32, "상", "높음", "mixed_growth"),
    spec("M02", "멀티모달", "14일 근권 누적과 관부 변화", "과거 근권 상태와 관부 비대 연결", "R_rootzone;G_growth", "farm_id;zone_id;timestamp;substrate_temp_c;substrate_water_content_pct;substrate_ec_ds_m;observation_date;crown_diameter_mm", "line_flow_rate", "1시간과 약 13일", "첫 조사 초과 둘째 조사 이하", "첫 조사 직후", "둘째 조사", "근권평균;극단시간;관부차", "none", "과거 근권만 집계", "근권 조건과 관부 변화 조합", "근권 생육 연관", "시간적 연관", "측정오차;초기크기", "관부 감소를 부족으로 확정 금지;누수 금지", 32, "상", "높음", "mixed_growth", "growth"),
    spec("M03", "멀티모달", "관수 패턴과 엽수 변화", "과거 관수 운용과 신엽 전개 연결", "A_actuator;R_rootzone;G_growth", "farm_id;zone_id;timestamp;line_flow_rate;substrate_water_content_pct;observation_date;leaf_count", "inside_temp_c", "1시간과 약 13일", "첫 조사 초과 둘째 조사 이하", "첫 조사 직후", "둘째 조사", "관수빈도;유량합;함수율하위시간;엽수차", "none", "과거 데이터만 집계", "관수 리듬과 엽수 변화 조합", "운영 생육 연관", "시간적 연관", "적엽작업;유량 의미", "미래 누수 금지", 32, "중", "중간", "mixed_growth"),
    spec("M04", "멀티모달", "구역별 환경과 생육 불균일 비교", "동일 농장 4개 구역의 환경과 생육 순위 비교", "E_environment;R_rootzone;G_growth", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct;substrate_water_content_pct;observation_date;plant_height_cm", "crown_diameter_mm", "1시간과 조사일", "둘째 조사 이전 13일", "첫 조사 직후", "둘째 조사", "구역별과거평균순위;생육변화순위", "empirical", "농장 내 4개 구역 상대 순위", "환경 하위 구역과 생육 하위 구역 비교", "공간 연관", "상대 비교", "구역별 표본 식물", "4개 구역 완전성;누수 금지", 16, "상", "높음", "mixed_growth"),
    spec("M05", "멀티모달", "고습 위험 이력과 개화 변화", "과거 고습 지속시간과 개화 진행 연결", "E_environment;G_growth", "farm_id;zone_id;timestamp;inside_humidity_pct;observation_date;opened_flower_count;unopened_flower_count", "inside_temp_c", "1시간과 약 13일", "첫 조사 초과 둘째 조사 이하", "첫 조사 직후", "둘째 조사", "RH90시간;최장연속;개화수차", "expert", "RH 90 이상 4시간 위험 이력;진단 아님", "고습 이력과 개화 변화 조합", "수정 위험 연관 후보", "시간적 연관", "벌 활동;약해;꽃곰팡이 미관찰", "질병 또는 수정불량 확정 금지", 32, "중", "중간", "mixed_growth", "gray"),
    spec("M06", "멀티모달", "score90 terminal interpretation relation", "example score90 공개 관측을 시간순 입력으로 두고 terminal interpretation과의 관계를 학습", "answers/reference_answers/score90;E_environment;R_rootzone;A_actuator", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct;substrate_temp_c;substrate_water_content_pct;substrate_ec_ds_m", "line_flow_rate;rain_detected", "period", "example case의 period_start부터 period_end까지", "period_start", "score90 interpretation terminal event", "위험시간;농장구역분위;terminal_interpretation", "hybrid", "example score90 interpretation은 마지막 관계 노드로만 사용하고 공개 관측이 선행", "공개 관측 요약 뒤 score90 terminal interpretation", "example terminal relation", "후향적 관계", "답변 등급;구역 선택", "score90 interpretation을 입력 원자로 사용 금지;problem hidden 또는 corrected interpretation 사용 금지;누수 차단", 32, "중", "중간", "expert_examples", "none"),
]

# A09의 tube_rail 전시각 반복은 정보량이 없는 품질검사이므로 ACTIVE에서 제외한다.
SPECS = [s for s in SPECS if s["narrative_id"] != "A09"]

SPECS.extend([
    spec("X01", "센서운영이상", "FCU 팬 펌프 불일치", "팬과 펌프 상태코드의 비동기 구간 식별", "A_actuator", "farm_id;zone_id;timestamp;fcu_fan;fcu_pump", "inside_temp_c", "1시간", "불일치 전 1시간부터 복귀 후 1시간", "fcu_fan과 fcu_pump 불일치", "두 값 일치 또는 8시간", "불일치지속;선행장치", "none", "0과 201 상태코드 직접 비교", "팬 또는 펌프만 가동", "operation_anomaly 후보", "이상 후보", "순차 기동;시간집계", "단일 시각과 지속 불일치 분리", 8, "중", "중간", "fan_pump_mismatch"),
    spec("X02", "일상재배", "CO2 고농도 공급 OFF", "공급 코드 없이 CO2가 높은 예외 구간 보존", "A_actuator;E_environment", "farm_id;zone_id;timestamp;co2_supply;co2_ppm", "roof_vent_left;inside_temp_c", "1시간", "고농도 전후 각 2시간", "CO2 농장구역 p99 이상이며 공급 0", "2시간 후", "CO2분위;감쇠;환기상태", "empirical", "entity p99와 co2_supply 0", "고CO2와 공급 OFF 동시", "예외 패턴", "관찰", "이전 공급;환기;센서", "이전 공급 이력과 점프 확인", 5, "중", "중간", "co2_high_off"),
    spec("X03", "일상재배", "공급 EC 근권 EC divergence", "공급계 추정 EC와 근권 EC 차이 학습", "A_actuator;R_rootzone", "farm_id;zone_id;timestamp;ec_sensor;substrate_ec_ds_m", "line_flow_rate;substrate_water_content_pct", "1시간", "연속 12시간", "entity 절대 EC gap 상위 5퍼센트", "12시간", "근권minus공급EC;gap분위", "empirical", "entity 절대 gap p95", "공급계와 근권 EC 괴리", "관계 서사", "시간적 연관", "ec_sensor 의미;센서보정", "두 컬럼 의미를 합치지 않음", 12, "상", "중간", "ec_divergence"),
    spec("X04", "일상재배", "동시간 구역별 제어 정책 비교", "동일 시각 네 구역의 제어 정책 병렬 비교", "A_actuator", "farm_id;zone_id;timestamp;roof_vent_left;shade_screen;thermal_curtain;fcu_fan;line_flow_rate", "co2_supply;circulation_fan;nutrient_solution_system", "1시간", "동일 farm timestamp 네 구역", "네 구역 모두 존재", "zone 4 수집", "구역정책거리;가동비율", "none", "구역 주소 순서 관계", "동일 외기에서 구역별 제어 차이", "구역 비교", "상대 비교", "설비 구성 차이", "tube_rail 제외;시간 OP 비활성화", 4, "상", "높음", "crosszone_hourly"),
    spec("X05", "스트레스위험", "배지온 12도 미만 4시간 지속", "저온 근권의 뿌리활력 저하 위험환경 기록", "R_rootzone;E_environment", "farm_id;zone_id;timestamp;substrate_temp_c", "inside_temp_c;substrate_water_content_pct", "1시간", "저온 전후 최대 16시간", "substrate_temp_c 12 미만 4시간", "12 이상 회복", "저온지속;최저배지온", "expert", "12도 미만 4시간 연속 screening", "배지 저온 지속", "뿌리활력 저하 위험 후보", "위험", "배지종류;센서깊이", "근권온 점프 제외;확진 금지", 16, "상", "중간", "run_lt:substrate_temp_c:12:4"),
    spec("X06", "센서운영이상", "근권 세 센서 동시 점프", "온도 함수율 EC 동시 급변 탐지", "R_rootzone", "farm_id;zone_id;timestamp;substrate_temp_c;substrate_water_content_pct;substrate_ec_ds_m", "", "1시간", "점프 전후 각 3시간", "세 컬럼 abs delta 각각 p95 이상", "3시간 후", "공동점프점수;방향벡터", "empirical", "각 근권 컬럼 entity p95", "세 근권 센서 동시 불연속", "sensor_anomaly", "이상 후보", "대량관수;급격가온", "유량과 실내온 동반 확인", 7, "상", "높음", "root_joint_jump"),
    spec("X07", "일상재배", "정오 일사 구간", "1시간 자료의 정오 광환경 반응 학습", "E_environment", "farm_id;zone_id;timestamp;outside_solar_radiation;inside_temp_c", "inside_humidity_pct", "1시간", "10시부터 14시까지 5개 이벤트", "시각 10시", "시각 14시", "일사피크;온도피크", "none", "1시간 정오 구간", "정오 일사와 내부온 변화", "기본 환경 리듬", "관찰", "구름;계절", "5개 시각 완전성", 5, "중", "높음", "noon_solar"),
    spec("X08", "멀티모달", "이미지 이전 환경 기간 정렬", "farm image와 같은 case period의 환경을 탐색적으로 연결", "I_images;E_environment", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct;outside_solar_radiation", "co2_ppm;rain_detected", "period", "14일 daily summary와 farm image", "period_start", "period image event", "일별환경요약;이미지수", "none", "farm-period 일치만 허용", "기간 환경 뒤 이미지 관측", "multimodal_candidate", "후보", "촬영시각과 구역 불명", "farm-period alignment만 허용", 15, "하", "낮음", "period_images"),
    spec("X09", "멀티모달", "이미지 근권 상태 기간 정렬", "farm image와 같은 case period의 근권 분포 연결", "I_images;R_rootzone", "farm_id;zone_id;timestamp;substrate_temp_c;substrate_water_content_pct;substrate_ec_ds_m", "", "period", "14일 근권 summary와 farm image", "period_start", "period image event", "근권분포;이상횟수", "none", "farm-period 일치만 허용", "기간 근권 뒤 이미지 관측", "multimodal_candidate", "후보", "촬영시각과 구역 불명", "farm-period alignment만 허용", 15, "하", "낮음", "period_images"),
    spec("X10", "멀티모달", "이미지 생육조사 기간 정렬", "farm image와 두 생육조사 변화를 기간 수준으로 연결", "I_images;G_growth", "farm_id;zone_id;observation_date;plant_height_cm;leaf_length_cm;leaf_width_cm;leaf_count", "crown_diameter_mm", "period", "두 조사와 farm image period event", "첫 조사", "period image event", "생육변화;이미지수", "none", "farm-period 일치만 허용", "조사 변화 뒤 이미지 관측", "multimodal_candidate", "후보", "사진 개체와 측정 개체 다름", "특정 zone image로 연결 금지", 10, "하", "낮음", "period_images"),
    spec("X11", "멀티모달", "example problem 공개관측 유사성 관계", "공개 센서 생육 요약 간 example problem 유사성 관계 학습", "E_environment;R_rootzone;G_growth;example_set;problem_set", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct;substrate_temp_c;substrate_water_content_pct;substrate_ec_ds_m", "observation_date;plant_height_cm;leaf_count", "period", "각 case period 공개 관측 요약", "period_start", "period_end", "환경근권요약거리;생육요약거리", "empirical", "공개 관측 특징 거리만 사용", "example에서 problem으로 유사 관계", "관계 서사", "관계", "농장별 기간과 구역 차이", "problem interpretation과 corrected label 사용 금지", 16, "중", "낮음", "public_relation"),
    spec("X12", "멀티모달", "농장 이미지 최초-최근 페어", "한 농장의 촬영기간 전체(최대 약 13일)를 두 이미지와 각 시점 환경 맥락으로 연결", "I_images;E_environment", "farm_id;zone_id;timestamp", "inside_temp_c;inside_humidity_pct", "period", "농장 관측기간 전체", "최초 이미지 직전 환경 맥락", "최근 이미지 직전 환경 맥락", "촬영간격", "none", "farm 내 이미지 2장 이상만 허용", "촬영기간 전체를 아우르는 두 이미지 쌍과 각 시점 환경", "multimodal_candidate", "후보", "촬영시각과 구역 불명;중간 이미지 미사용", "farm-period alignment만 허용;이미지 2장 미만 제외", 8, "하", "낮음", "image_longitudinal_pair"),
    spec("X13", "환경비교", "동시각 농장 간 온습도 비교", "같은 지역시각의 여러 농장 내부 온습도를 병렬 비교", "E_environment", "farm_id;zone_id;timestamp;inside_temp_c;inside_humidity_pct", "outside_temp_c;outside_humidity_pct", "1시간", "동일 지역시각 다수 농장", "지역시각당 20개 농장 이상 관측", "동일 지역시각 관측 종료", "농장간범위;농장간표준편차", "empirical", "지역시각 단위로 농장을 병렬 비교", "농장별 관리·기후차로 인한 온습도 산포", "농장간 비교", "상대 비교", "농장별 캘린더 비동기;계절차", "20개 농장 이상 완전성", 20, "중", "중간", "crossfarm_hourly"),
    spec("X14", "환경비교", "동시각 농장 간 CO2 일사 비교", "같은 지역시각의 여러 농장 CO2·일사를 병렬 비교", "E_environment", "farm_id;zone_id;timestamp;co2_ppm;outside_solar_radiation", "co2_supply", "1시간", "동일 지역시각 다수 농장", "지역시각당 20개 농장 이상 관측", "동일 지역시각 관측 종료", "농장간범위;농장간표준편차", "empirical", "지역시각 단위로 농장을 병렬 비교", "농장별 CO2 시비·환기 정책 차이", "농장간 비교", "상대 비교", "농장별 캘린더 비동기;계절차", "20개 농장 이상 완전성", 20, "중", "중간", "crossfarm_hourly"),
    spec("X15", "환경비교", "동일 조사순번 농장 간 생육 비교", "같은 생육조사 순번(첫/둘째 조사)의 여러 농장 초장·엽수를 병렬 비교", "G_growth", "farm_id;zone_id;observation_date;plant_height_cm;leaf_count", "crown_diameter_mm", "1시간", "동일 조사 순번 다수 농장", "조사순번당 20개 농장 이상 관측", "동일 조사 순번 관측 종료", "농장간범위;농장간표준편차", "empirical", "조사 순번 단위로 농장을 병렬 비교", "농장별 재배환경차로 인한 생육 산포", "농장간 비교", "상대 비교", "농장별 정식일 비동기;품종차", "20개 농장 이상 완전성", 20, "중", "중간", "crossfarm_growth"),
])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_FILES = sorted(
    p for p in BASE.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)
if not IMAGE_FILES:
    SPECS = [s for s in SPECS if s["narrative_id"] not in {"X08", "X09", "X10", "X12"}]

MATERIALIZATION_KEYS = {
    "X01": "fan_pump_mismatch", "X02": "co2_high_off",
    "X03": "ec_sensor_root_divergence", "X04": "cross_zone_actuation",
    "X05": "rootcold12", "X06": "root_simultaneous_jump",
    "X07": "noon_solar_hourly", "X08": "image_preceding_env",
    "X09": "image_root_context", "X10": "image_growth_alignment",
    "X11": "public_observation_similarity", "M06": "score90_terminal_relation",
}
NON_OP_IDS = {"X04", "X08", "X09", "X10", "X11", "M06"}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    act = pd.concat(
        [pd.read_csv(p) for p in sorted((BASE / "data/A_actuator").glob("*.csv"))],
        ignore_index=True,
    )
    env = pd.concat(
        [pd.read_csv(p) for p in sorted((BASE / "data/E_environment").glob("*.csv"))],
        ignore_index=True,
    )
    root = pd.read_csv(BASE / "data/R_rootzone/root_data.csv")
    growth = pd.read_csv(BASE / "data/G_growth/growth_data.csv")
    hourly = env.merge(act, on=["farm_id", "zone_id", "timestamp"], validate="one_to_one")
    hourly = hourly.merge(root, on=["farm_id", "zone_id", "timestamp"], validate="one_to_one")
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    growth["observation_date"] = pd.to_datetime(growth["observation_date"])
    columns = set(hourly.columns) | set(growth.columns)
    columns |= {"period_start", "period_end", "set"}
    return hourly.sort_values(["farm_id", "zone_id", "timestamp"]), growth, columns


def _episode_count(mask: pd.Series, hourly: pd.DataFrame) -> int:
    mask = mask.fillna(False)
    starts = mask & ~mask.groupby([hourly.farm_id, hourly.zone_id]).shift(fill_value=False)
    return int(starts.sum())


def occurrence(
    spec_row: dict, hourly: pd.DataFrame, growth: pd.DataFrame
) -> tuple[int, int, list[str]]:
    m = spec_row["matcher"]
    groups = ["farm_id", "zone_id"]
    mask = pd.Series(False, index=hourly.index)
    if m == "all_hourly":
        mask[:] = True
    elif m.startswith("window_summary:"):
        _, h0, h1 = m.split(":")
        h0, h1 = int(h0), int(h1)
        hour = hourly["timestamp"].dt.hour
        mask = (hour >= h0) & (hour < h1) if h0 < h1 else (hour >= h0) | (hour < h1)
    elif m == "solar_up":
        mask = hourly["outside_solar_radiation"].groupby([hourly.farm_id, hourly.zone_id]).diff() > 0
    elif m == "solar_down":
        mask = hourly["outside_solar_radiation"].groupby([hourly.farm_id, hourly.zone_id]).diff() < 0
    elif m.startswith("positive:"):
        mask = hourly[m.split(":")[1]] > 0
    elif m.startswith("change:"):
        col = m.split(":")[1]
        mask = hourly.groupby(groups)[col].diff().fillna(0).ne(0)
    elif m.startswith("quantile_high:"):
        col = m.split(":")[1]
        q = hourly.groupby(groups)[col].transform(lambda x: x.quantile(.9))
        mask = hourly[col] > q
    elif m.startswith("quantile_low:"):
        col = m.split(":")[1]
        q = hourly.groupby(groups)[col].transform(lambda x: x.quantile(.1))
        mask = hourly[col] < q
    elif m.startswith(("lt:", "le:", "gt:")):
        op, col, val = m.split(":")
        val = float(val)
        mask = {"lt": hourly[col] < val, "le": hourly[col] <= val, "gt": hourly[col] > val}[op]
    elif m.startswith("and:"):
        _, c1, v1, c2, v2 = m.split(":")
        mask = (hourly[c1] > float(v1)) & (hourly[c2] < float(v2))
        if c2 == "rain_detected":
            mask = (hourly[c1] > float(v1)) & (hourly[c2] > float(v2))
    elif m.startswith("and_gt:"):
        _, c1, v1, c2, v2 = m.split(":")
        mask = (hourly[c1] > float(v1)) & (hourly[c2] > float(v2))
    elif m.startswith(("trend_intervention:", "co_trend:")):
        _, col1, col2, hrs = m.split(":")
        hrs = int(hrs)
        g = hourly.sort_values(["farm_id", "zone_id", "timestamp"])

        def _window_range(col: str) -> pd.Series:
            gb = g.groupby(groups)[col]
            return gb.transform(lambda x: x.rolling(hrs, min_periods=2).max() - x.rolling(hrs, min_periods=2).min())

        def _jump_limit_series(col: str) -> pd.Series:
            d = hourly.groupby(groups)[col].diff().abs()
            med = d.groupby([hourly.farm_id, hourly.zone_id]).transform("median")
            mad = (d - med).abs().groupby([hourly.farm_id, hourly.zone_id]).transform("median")
            return med + 6 * mad.replace(0, np.nan)

        range1, range2 = _window_range(col1), _window_range(col2)
        limit1, limit2 = _jump_limit_series(col1).reindex(range1.index), _jump_limit_series(col2).reindex(range2.index)
        mask = (range1 >= limit1) & (range2 >= limit2)
    elif m.startswith("multi_trend:"):
        _, cols_joined, hrs = m.split(":")
        cols = cols_joined.split("+")
        hrs = int(hrs)
        g = hourly.sort_values(["farm_id", "zone_id", "timestamp"])

        def _window_range2(col: str) -> pd.Series:
            gb = g.groupby(groups)[col]
            return gb.transform(lambda x: x.rolling(hrs, min_periods=2).max() - x.rolling(hrs, min_periods=2).min())

        def _jump_limit_series2(col: str) -> pd.Series:
            d = hourly.groupby(groups)[col].diff().abs()
            med = d.groupby([hourly.farm_id, hourly.zone_id]).transform("median")
            mad = (d - med).abs().groupby([hourly.farm_id, hourly.zone_id]).transform("median")
            return med + 6 * mad.replace(0, np.nan)

        mask = pd.Series(True, index=hourly.index)
        for col in cols:
            r = _window_range2(col)
            lim = _jump_limit_series2(col).reindex(r.index)
            mask = mask & (r >= lim)
    elif m.startswith("lag_response:"):
        # trend_intervention/multi_trend는 "같은 구간 안 어딘가"에서 둘 다
        # 움직이면 매칭되는 방향 중립적 공존 주장이라 "트리거 후 정확히
        # lag_hours 뒤에 반응했다"는 지연을 표현 못한다(사용자 지적) --
        # 트리거 컬럼이 0에서 양수로 전환된 시점 대비 lag_hours 뒤 시점에서
        # 반응 컬럼이 자기 평소 점프폭 이상 움직였는지를 요구한다.
        _, trig_col, resp_col, lag = m.split(":")
        lag = int(lag)
        g = hourly.sort_values(["farm_id", "zone_id", "timestamp"])
        trig_prev = g.groupby(groups)[trig_col].shift()
        trig_on = (trig_prev <= 0) & (g[trig_col] > 0)
        resp_lag = g.groupby(groups)[resp_col].shift(-lag)
        d = g.groupby(groups)[resp_col].diff().abs()
        med = d.groupby([g.farm_id, g.zone_id]).transform("median")
        mad = (d - med).abs().groupby([g.farm_id, g.zone_id]).transform("median")
        limit = med + 6 * mad.replace(0, np.nan)
        diff = (resp_lag - g[resp_col]).abs()
        mask = trig_on & diff.notna() & (diff >= limit)
    elif m.startswith("run_ge:"):
        _, col, val, run = m.split(":")
        raw = hourly[col] >= float(val)
        run_id = raw.ne(raw.groupby([hourly.farm_id, hourly.zone_id]).shift()).groupby(
            [hourly.farm_id, hourly.zone_id]
        ).cumsum()
        lengths = raw.groupby([hourly.farm_id, hourly.zone_id, run_id]).transform("sum")
        mask = raw & (lengths >= int(run))
    elif m.startswith("run_lt:"):
        _, col, val, run = m.split(":")
        raw = hourly[col] < float(val)
        run_id = raw.ne(raw.groupby([hourly.farm_id, hourly.zone_id]).shift()).groupby(
            [hourly.farm_id, hourly.zone_id]
        ).cumsum()
        lengths = raw.groupby([hourly.farm_id, hourly.zone_id, run_id]).transform("sum")
        mask = raw & (lengths >= int(run))
    elif m.startswith("joint_quantile:"):
        _, c1, c2 = m.split(":")
        q1 = hourly.groupby(groups)[c1].transform(lambda x: x.quantile(.9))
        q2 = hourly.groupby(groups)[c2].transform(lambda x: x.quantile(.9))
        mask = (hourly[c1] > q1) & (hourly[c2] > q2)
    elif m.startswith(("tod_high:", "tod_low:")):
        kind, col, h0, h1 = m.split(":")
        h0, h1 = int(h0), int(h1)
        hour = hourly["timestamp"].dt.hour
        in_window = (hour >= h0) & (hour < h1) if h0 < h1 else (hour >= h0) | (hour < h1)
        if kind == "tod_high":
            q = hourly.groupby(groups)[col].transform(lambda x: x.quantile(.9))
            mask = in_window & (hourly[col] >= q)
        else:
            q = hourly.groupby(groups)[col].transform(lambda x: x.quantile(.1))
            mask = in_window & (hourly[col] <= q)
    elif m.startswith("jump:"):
        col = m.split(":")[1]
        d = hourly.groupby(groups)[col].diff().abs()
        med = d.groupby([hourly.farm_id, hourly.zone_id]).transform("median")
        mad = (d - med).abs().groupby([hourly.farm_id, hourly.zone_id]).transform("median")
        mask = d > med + 6 * mad.replace(0, np.nan)
    elif m.startswith("reset:"):
        col = m.split(":")[1]
        prev = hourly.groupby(groups)[col].shift()
        mask = (prev > 0) & (hourly[col] == 0)
    elif m.startswith("fixed:"):
        col = m.split(":")[1]
        mask = hourly.groupby(groups)[col].transform("nunique") == 1
    elif m.startswith("fixed_zero:"):
        col = m.split(":")[1]
        mask = hourly.groupby(groups)[col].transform("max") == 0
    elif m.startswith("diff_positive:"):
        _, c1, c2 = m.split(":")
        mask = (hourly[c1] - hourly[c2]).abs() > 0
    elif m == "fan_pump_mismatch":
        mask = hourly["fcu_fan"] != hourly["fcu_pump"]
    elif m == "co2_high_off":
        q = hourly.groupby(groups)["co2_ppm"].transform(lambda x: x.quantile(.99))
        mask = (hourly["co2_ppm"] >= q) & (hourly["co2_supply"] == 0)
    elif m == "ec_divergence":
        gap = (hourly["ec_sensor"] - hourly["substrate_ec_ds_m"]).abs()
        q = gap.groupby([hourly.farm_id, hourly.zone_id]).transform(lambda x: x.quantile(.95))
        mask = gap >= q
    elif m == "root_joint_jump":
        cols = ["substrate_temp_c", "substrate_water_content_pct", "substrate_ec_ds_m"]
        flags = []
        for col in cols:
            delta = hourly.groupby(groups)[col].diff().abs()
            q = delta.groupby([hourly.farm_id, hourly.zone_id]).transform(
                lambda x: x.quantile(.95)
            )
            flags.append(delta >= q)
        mask = flags[0] & flags[1] & flags[2]
    elif m == "noon_solar":
        mask = hourly["timestamp"].dt.hour.between(10, 14)
    elif m == "crosszone_hourly":
        counts = hourly.groupby(["farm_id", "timestamp"]).zone_id.nunique()
        farms = sorted(counts[counts == 4].index.get_level_values(0).unique())
        count = int((counts == 4).sum())
        return count * 4, count, farms
    elif m in {"growth_any", "mixed_growth"}:
        counts = growth.groupby(groups).size()
        farms = sorted(counts[counts >= 2].index.get_level_values(0).unique())
        count = int((counts >= 2).sum())
        return int(counts[counts >= 2].sum()), count, farms
    elif m == "crosszone_growth":
        counts = growth.groupby(["farm_id", "observation_date"]).zone_id.nunique()
        farms = sorted(counts[counts == 4].index.get_level_values(0).unique())
        count = int((counts == 4).sum())
        return count * 4, count, farms
    elif m == "expert_examples":
        files = list((BASE / "answers/reference_answers/score90").glob("*_answer.txt"))
        farms = sorted(p.stem.split("_")[0] for p in files)
        return len(files), len(files), farms
    elif m == "period_images":
        farms = sorted({re.search(r"F\d{6}", str(p)).group(0) for p in IMAGE_FILES
                        if re.search(r"F\d{6}", str(p))})
        return len(IMAGE_FILES), len(IMAGE_FILES), farms
    elif m == "public_relation":
        example = pd.read_csv(BASE / "example_set/case_list.csv")
        problem = pd.read_csv(BASE / "problem_set/case_list.csv")
        farms = sorted(set(example.farm_id) | set(problem.farm_id))
        # 각 problem case에 대해 적어도 하나의 example 이웃 관계를 생성한다.
        return len(example) * len(problem), len(problem), farms
    elif m == "crossfarm_hourly":
        # 농장 간 관측 캘린더가 거의 안 겹친다(실측: 같은 절대 timestamp를
        # 공유하는 농장이 최대 9개). 절대 시각 대신 지역시각(hour-of-day)으로
        # 묶는다 — 이 기준으로는 55개 농장 전부가 매 시각에 데이터를 가짐.
        hour = hourly["timestamp"].dt.hour
        counts = hourly.groupby(hour).farm_id.nunique()
        ok_hours = counts[counts >= 20].index
        farms = sorted(hourly.loc[hour.isin(set(ok_hours)), "farm_id"].unique())
        return int(counts.loc[ok_hours].sum()), int(len(ok_hours)), farms
    elif m == "crossfarm_growth":
        # 생육 조사도 농장마다 조사일이 다르므로 절대 날짜 대신 조사 순번
        # (survey_idx: 0=첫 조사, 1=둘째 조사)으로 묶는다.
        g = growth.sort_values(["farm_id", "zone_id", "observation_date"])
        survey_idx = g.groupby(["farm_id", "zone_id"]).cumcount()
        counts = g.groupby(survey_idx).farm_id.nunique()
        ok_idx = counts[counts >= 20].index
        farms = sorted(g.loc[survey_idx.isin(set(ok_idx)), "farm_id"].unique())
        return int(counts.loc[ok_idx].sum()), int(len(ok_idx)), farms
    elif m == "zone_outlier_hourly":
        col = spec_row["req"].split(";")[-1] if isinstance(spec_row["req"], str) else spec_row["req"][-1]
        counts = hourly.groupby(["farm_id", "timestamp"]).zone_id.nunique()
        valid_keys = set(counts[counts == 4].index)
        sub = hourly[hourly.set_index(["farm_id", "timestamp"]).index.isin(valid_keys)]

        def _outlier_flags(g: pd.DataFrame) -> pd.Series:
            vals = g[col]
            med = vals.median()
            mad = (vals - med).abs().median() or 1e-9
            return (vals - med).abs() >= 6 * mad

        flags = sub.groupby(["farm_id", "timestamp"], group_keys=False).apply(_outlier_flags)
        count = int(flags.sum())
        farms = sorted(sub.loc[flags.reindex(sub.index, fill_value=False)].farm_id.unique()) if count else []
        return count, count, farms
    elif m == "image_longitudinal_pair":
        counts_by_farm: Counter = Counter()
        for p in IMAGE_FILES:
            match = re.search(r"F\d{6}", str(p))
            if match:
                counts_by_farm[match.group(0)] += 1
        farms = sorted(f for f, c in counts_by_farm.items() if c >= 2)
        return len(IMAGE_FILES), len(farms), farms
    else:
        raise ValueError(m)
    mask = mask.fillna(False)
    row_count = int(mask.sum())
    if m == "all_hourly" or m.startswith("window_summary:"):
        unique_count = int(
            hourly.loc[mask].assign(date=hourly.loc[mask, "timestamp"].dt.date)
            .drop_duplicates(["farm_id", "zone_id", "date"]).shape[0]
        )
    elif m == "noon_solar":
        unique_count = int(
            hourly.loc[mask].assign(date=hourly.loc[mask, "timestamp"].dt.date)
            .drop_duplicates(["farm_id", "zone_id", "date"]).shape[0]
        )
    else:
        unique_count = _episode_count(mask, hourly)
    return row_count, unique_count, sorted(hourly.loc[mask, "farm_id"].unique())


def atomization(req_cols: list[str]) -> str:
    parts = ["farm_id→FARM|value", "zone_id→ZONE|value", "timestamp 또는 observation_date→TIME|ISO"]
    for col in req_cols:
        if col in KEYS:
            continue
        alias = ALIASES[col]
        if col in {"fcu_fan", "fcu_pump", "circulation_fan", "co2_supply", "tube_rail"}:
            parts.append(f"{col}→{alias}|OFF_ON;0→OFF;201→ON")
        else:
            parts.append(f"{col}→{alias}|farm_qbin_00_09")
    return ";".join(parts)


def examples(req_cols: list[str], mixed: bool) -> tuple[str, str]:
    values = [c for c in req_cols if c not in KEYS]
    toks = ["FARM|F130230", "ZONE|1", "TIME|H09"]
    binary_controls = {"fcu_fan", "fcu_pump", "circulation_fan", "co2_supply", "tube_rail"}
    toks += [f"{ALIASES[c]}|{'ON' if c in binary_controls else 'b05'}" for c in values[:4]]
    token = ";".join(toks)
    if mixed:
        seq = "[T-2|FARM|F130230|ZONE|1|HIST_ENV|b04]→[T-1|HIST_ROOT|b06]→[T0|GROWTH_DELTA|b07]"
    else:
        lead = ALIASES[values[0]] if values else "VALUE"
        seq = f"[T0|FARM|F130230|ZONE|1|{lead}|b03]→[T1|{lead}|b05]→[T2|{lead}|b06]"
    return token, seq


def build_rows(hourly: pd.DataFrame, growth: pd.DataFrame, actual_columns: set[str]) -> list[dict]:
    rows = []
    for s in SPECS:
        req = s["req"].split(";")
        missing = [c for c in req if c not in actual_columns]
        if missing:
            raise AssertionError(f"{s['narrative_id']} missing columns {missing}")
        count, unique_count, farms = occurrence(s, hourly, growth)
        if count < 1 or unique_count < 1:
            raise AssertionError(f"{s['narrative_id']} cannot materialize")
        mixed = "observation_date" in req and "timestamp" in req
        if s["matcher"] == "period_images":
            event_def = "기간별 센서 또는 생육 summary event와 farm-level IMAGE_EMBED_SLOT event를 분리하고 image zone_id는 null로 유지"
            seq_def = "기간 summary event 오름차순 뒤 LOGICAL_ANCHOR image event;특정 zone 귀속 금지;OP 비활성화;미래 정보 사용 금지"
        elif s["matcher"] == "image_longitudinal_pair":
            event_def = "농장별 최초 이미지 event와 최근 이미지 event 두 개만 유지;중간 이미지는 사용하지 않음"
            seq_def = "촬영시각 오름차순 최초→최근 이미지 두 노드;PERIOD_ALIGNED;LONGITUDINAL_PAIR;OP 비활성화;미래 정보 사용 금지"
        elif s["matcher"] == "crossfarm_hourly":
            event_def = "동일 지역시각(hour-of-day)의 서로 다른 farm_id 토큰을 농장별 하위 이벤트로 유지;절대 날짜는 무시"
            seq_def = "동일 시각 안에서 farm_id 주소 오름차순;시간 SOP와 OP 비활성화;미래 정보 사용 금지;PERMUTATION_INVARIANT_SET"
        elif s["matcher"] == "crossfarm_growth":
            event_def = "동일 생육조사 순번(첫 조사 또는 둘째 조사)의 서로 다른 farm_id 토큰을 농장별 하위 이벤트로 유지;절대 조사일은 무시"
            seq_def = "동일 조사 순번 안에서 farm_id 주소 오름차순;시간 SOP와 OP 비활성화;미래 정보 사용 금지;PERMUTATION_INVARIANT_SET"
        elif mixed:
            event_def = "시간자료는 farm_id+zone_id+timestamp;생육은 farm_id+zone_id+observation_date;원시 이벤트를 임의 병합하지 않고 둘째 조사시점에 과거 전용 집계 토큰만 결합"
            seq_def = "farm_id+zone_id별 시간 오름차순;첫 조사 초과 둘째 조사 이하만 집계;센서 요약 이벤트 2개와 둘째 생육 이벤트;좌측 패딩;오래된 센서부터 절단"
        elif "observation_date" in req:
            event_def = "동일 farm_id+zone_id+observation_date의 생육 cell 토큰을 하나의 조사 이벤트로 묶음"
            seq_def = "farm_id+zone_id별 observation_date 오름차순;두 조사 이벤트와 변화량 이벤트;부족 시 PAD;미래 조사 사용 금지"
        elif s["matcher"] == "crosszone_hourly":
            event_def = "동일 farm_id+timestamp의 zone_id 1에서 4 토큰을 구역별 하위 이벤트로 유지"
            seq_def = "동일 timestamp 안에서 zone_id 주소 오름차순;시간 SOP와 OP 비활성화;미래 정보 사용 금지"
        elif s["matcher"] in {"expert_examples", "public_relation"}:
            event_def = "공개 관측 요약 노드와 terminal 관계 노드를 분리하며 problem interpretation은 생성하지 않음"
            seq_def = "공개 관측 노드 오름차순 뒤 terminal 또는 상대 case 노드;RELATION_ORDERED;OP 비활성화;미래 정보 사용 금지"
        else:
            event_def = "동일 farm_id+zone_id+timestamp의 원자 토큰을 하나의 시간 이벤트로 묶음"
            seq_def = "farm_id+zone_id별 timestamp 오름차순;start와 end 경계 포함;최대 이벤트 초과 시 오래된 이벤트 절단;부족 시 왼쪽 PAD;미래값 금지"
        token, sequence = examples(req, mixed)
        if s["matcher"] == "period_images":
            token = token + ";IMAGE_EMBED_SLOT"
            sequence = "[PERIOD_START|SUMMARY]→[PERIOD_END|SUMMARY]→[LOGICAL_ANCHOR|IMAGE_EMBED_SLOT]"
        elif s["matcher"] == "expert_examples":
            token = token + ";TEXT_EMBED_SLOT"
            sequence = "[PERIOD_START|SENSOR_SUMMARY]→[PERIOD_END|SENSOR_SUMMARY]→[TERMINAL|TEXT_EMBED_SLOT]"
        elif s["matcher"] == "public_relation":
            token = token + ";RELATION|EXAMPLE_PROBLEM_SIMILARITY"
            sequence = "[EXAMPLE|PUBLIC_SUMMARY]→[RELATION|SIMILARITY]→[PROBLEM|PUBLIC_SUMMARY]"
        n_values = max(1, len([c for c in req if c not in KEYS]))
        est = max(16, int(s["max_events"]) * (n_values + 3))
        evidence = EXPERT[s["evidence"]]
        narrative_id = s["narrative_id"]
        if narrative_id in {"X08", "X09", "X10", "X12"}:
            order_semantics = "PERIOD_LOGICAL_ALIGNMENT"
            alignment_policy = "farm_period_only"
        elif narrative_id == "X11":
            order_semantics = "RETRIEVAL_PAIR"
            alignment_policy = "public_observation_only"
        elif narrative_id == "M06":
            order_semantics = "RELATION_ORDERED"
            alignment_policy = "public_observation_terminal_only"
        elif narrative_id == "X04" or s["matcher"] in {
            "crosszone_hourly", "crosszone_growth", "crossfarm_hourly", "crossfarm_growth",
        }:
            order_semantics = "SET_COMPARISON"
            alignment_policy = (
                "same_hour_of_day_farm_address" if s["matcher"] in {"crossfarm_hourly", "crossfarm_growth"}
                else "same_farm_timestamp_zone_address"
            )
        else:
            order_semantics = "STRICT_CHRONOLOGICAL"
            alignment_policy = "entity_time_strict"
        if s["matcher"] == "period_images":
            entity_scope = "farm_id+period;image_zone_id=null"
        elif s["matcher"] == "public_relation":
            entity_scope = "example_case+problem_case"
        elif s["matcher"] == "expert_examples":
            entity_scope = "example_farm+period"
        else:
            entity_scope = "farm_id+zone_id"
        if s["matcher"] == "period_images":
            notes = (
                f"online2 실측 materialization 후보 {count}건;발생 농장 {len(farms)}개;"
                "farm-period 정렬만 허용;image zone_id=null;alignment confidence=low"
            )
        elif s["matcher"] == "expert_examples":
            notes = (
                f"online2 실측 materialization 후보 {count}건;발생 농장 {len(farms)}개;"
                "example score90 terminal interpretation만 허용;problem interpretation 사용 금지"
            )
        else:
            notes = (
                f"online2 실측 materialization 후보 {count}건;발생 농장 {len(farms)}개;"
                "원본값과 파생값 분리"
            )
        fields = {
            "narrative_id": narrative_id, "category": s["category"],
            "narrative_name_ko": s["name"], "purpose": s["purpose"],
            "data_sources": s["sources"], "required_columns": s["req"],
            "optional_columns": s["opt"], "entity_scope": entity_scope,
            "time_resolution": "period" if s["resolution"] == "period" or "observation_date" in req else "1시간",
            "window_definition": s["window"],
            "start_condition": s["start"], "end_condition": s["end"],
            "cell_atomization": atomization(req), "event_definition": event_def,
            "sequence_definition": seq_def, "derived_features": s["derived"],
            "threshold_type": s["threshold_type"], "threshold_or_rule": s["rule"],
            "expected_pattern": s["expected"],
            "agronomic_interpretation": s["interpretation"],
            "diagnosis_level": s["diagnosis"], "confounders": s["confounders"],
            "data_quality_checks": s["checks"] + ";farm_id zone_id timestamp 보존;중복키 검사;미래 정보 혼입 검사",
            "token_example": token, "sequence_example": sequence,
            "max_events": int(s["max_events"]), "estimated_tokens": est,
            "applicable_farms": f"{len(farms)}개 농장;" + ";".join(farms),
            "expert_evidence": evidence, "implementation_priority": s["priority"],
            "confidence": s["confidence"],
            "notes": notes,
            "status": "ACTIVE",
            "materialization_key": MATERIALIZATION_KEYS.get(narrative_id, s["matcher"]),
            "trigger_row_count": count,
            "unique_instance_count": unique_count,
            "op_eligible": "false" if (
                narrative_id in NON_OP_IDS
                or order_semantics not in {
                    "STRICT_CHRONOLOGICAL",
                    "SPARSE_CHRONOLOGICAL",
                    "CHRONOLOGICAL_WITH_TIES",
                }
                or int(s["max_events"]) < 2
            ) else "true",
            "order_semantics": order_semantics,
            "alignment_policy_id": alignment_policy,
            "sampling_weight": "0.20" if narrative_id in {"X08", "X09", "X10", "X12"} else (
                "0.50" if narrative_id in {"X11", "M06"} else "1.00"
            ),
            "timestamp_precision": "period" if (
                narrative_id in {"X08", "X09", "X10", "X11", "M06", "X12"}
                or "observation_date" in req
            ) else "1h",
        }
        for key, value in fields.items():
            if isinstance(value, str) and ("," in value or "\n" in value or "\r" in value):
                raise AssertionError(f"{s['narrative_id']} {key} contains forbidden punctuation")
        rows.append(fields)
    return rows


def validate(rows: list[dict], actual_columns: set[str]) -> None:
    assert len(HEADERS) == 41
    assert len(rows) >= 50
    assert len({r["narrative_id"] for r in rows}) == len(rows)
    assert all(list(r) == HEADERS for r in rows)
    required_fields = ["narrative_id", "category", "narrative_name_ko", "purpose", "data_sources",
                       "required_columns", "event_definition", "sequence_definition",
                       "token_example", "sequence_example", "max_events", "estimated_tokens"]
    assert all(all(r[f] not in ("", None) for f in required_fields) for r in rows)
    for r in rows:
        assert set(r["required_columns"].split(";")) <= actual_columns
        assert isinstance(r["max_events"], int) and isinstance(r["estimated_tokens"], int)
        assert 16 <= r["estimated_tokens"] <= 5120
        assert r["sequence_example"].count("→") >= 2
        assert r["status"] == "ACTIVE"
        assert r["time_resolution"] in {"1시간", "period"}
        assert r["timestamp_precision"] in {"1h", "period"}
        if r["order_semantics"] == "PERIOD_LOGICAL_ALIGNMENT":
            assert r["timestamp_precision"] == "period"
        assert isinstance(r["trigger_row_count"], int) and r["trigger_row_count"] >= 1
        assert isinstance(r["unique_instance_count"], int) and r["unique_instance_count"] >= 1
        assert r["unique_instance_count"] <= r["trigger_row_count"]
        assert r["op_eligible"] in {"true", "false"}
        assert r["order_semantics"] in {
            "STRICT_CHRONOLOGICAL", "SPARSE_CHRONOLOGICAL",
            "CHRONOLOGICAL_WITH_TIES", "SET_COMPARISON",
            "PERIOD_LOGICAL_ALIGNMENT", "RELATION_ORDERED", "RETRIEVAL_PAIR",
        }
        if r["order_semantics"] not in {
            "STRICT_CHRONOLOGICAL",
            "SPARSE_CHRONOLOGICAL",
            "CHRONOLOGICAL_WITH_TIES",
        }:
            assert r["op_eligible"] == "false"
        assert 0 < float(r["sampling_weight"]) <= 1
        assert int(re.search(r"materialization 후보 (\d+)건", r["notes"]).group(1)) > 0
        if r["category"] == "스트레스위험":
            assert r["diagnosis_level"] not in {"확진", "진단"}
        if r["category"] == "센서운영이상":
            assert "anomaly" in r["agronomic_interpretation"]
        assert "오름차순" in r["sequence_definition"]
        assert "미래" in r["sequence_definition"] or "미래" in r["data_quality_checks"]


BASELINE_IDS = (
    [f"D{i:02d}" for i in range(1, 16)]
    + [f"C{i:02d}" for i in range(1, 16)]
    + [f"G{i:02d}" for i in range(1, 11)]
    + [f"S{i:02d}" for i in range(1, 13)]
    + [f"A{i:02d}" for i in range(1, 13)]
    + [f"M{i:02d}" for i in range(1, 7)]
)

CHATGPT_MERGES = {
    "N001": "D01", "N002": "D01", "N003": "D02", "N004": "D03",
    "N005": "D04", "N006": "D01", "N007": "D15", "N008": "C01",
    "N009": "C02", "N010": "A10", "N011": "D06", "N012": "D07",
    "N013": "D08", "N014": "D08", "N016": "D09", "N017": "D10",
    "N019": "C09", "N020": "C10", "N021": "D11", "N022": "C04",
    "N023": "D12", "N024": "C05", "N025": "C07", "N026": "D12",
    "N027": "D13", "N029": "D14", "N030": "A07", "N031": "C13",
    "N032": "C06", "N033": "A11", "N035": "G01", "N036": "G02",
    "N037": "G04", "N038": "G05", "N039": "G06", "N040": "G07",
    "N041": "G08", "N042": "G10", "N043": "M01", "N044": "M02",
    "N045": "G09", "N046": "M04", "N047": "S01", "N048": "S02",
    "N050": "S03", "N051": "S04", "N052": "S05", "N053": "S06",
    "N054": "S07", "N055": "S08", "N056": "S09", "N057": "S10",
    "N058": "S11", "N059": "S12", "N060": "S09", "N061": "A02",
    "N063": "A03", "N064": "A01", "N065": "A07", "N066": "A07",
    "N067": "A06", "N071": "A12", "N072": "C06", "N075": "M06",
    "N076": "X11", "N077": "X11",
}
CHATGPT_KEEPS = {
    "N015": "X01", "N018": "X02", "N028": "X03", "N034": "X04",
    "N049": "X05", "N062": "X06",
}
IMAGE_SOURCE_MAP = {"N073": "X08", "N074": "X09", "N080": "X10"}


def disposition_rows(active_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    for narrative_id in BASELINE_IDS:
        if narrative_id == "A09":
            disposition, canonical, reason = (
                "VALIDATION", "", "tube_rail 전시각 상수 채널은 학습 서사가 아닌 품질검사"
            )
        else:
            disposition, canonical, reason = "KEEP", narrative_id, "기준 canonical ACTIVE 유지"
        rows.append({
            "source_catalog": "baseline70", "source_narrative_id": narrative_id,
            "disposition": disposition, "canonical_narrative_id": canonical, "reason": reason,
        })

    gemini_ids = pd.read_csv(GEMINI_CATALOG, dtype=str)["narrative_id"].tolist()
    for narrative_id in gemini_ids:
        if narrative_id == "N002":
            disposition, canonical, reason = "KEEP", "X07", "5분 정오 제안을 실제 1시간 구간으로 변환"
        elif narrative_id == "N005":
            disposition, canonical, reason = "MERGE", "D01", "VPD derived evidence를 D01 일주기에 흡수"
        else:
            disposition, canonical, reason = (
                "DROP", "", "미선택 Gemini 제안은 online2 schema 또는 시간 해상도와 불일치"
            )
        rows.append({
            "source_catalog": "gemini50", "source_narrative_id": narrative_id,
            "disposition": disposition, "canonical_narrative_id": canonical, "reason": reason,
        })

    chatgpt_ids = pd.read_csv(CHATGPT_CATALOG, dtype=str)["narrative_id"].tolist()
    for narrative_id in chatgpt_ids:
        if narrative_id in CHATGPT_KEEPS:
            disposition, canonical, reason = (
                "KEEP", CHATGPT_KEEPS[narrative_id], "실제 online2 컬럼과 1시간 해상도로 ACTIVE 구현"
            )
        elif narrative_id in CHATGPT_MERGES:
            disposition, canonical, reason = (
                "MERGE", CHATGPT_MERGES[narrative_id], "기존 또는 재정의 canonical과 의미 중복"
            )
        elif narrative_id in {"N068", "N069", "N070", "N078"}:
            disposition, canonical, reason = "VALIDATION", "", "학습 서사가 아닌 데이터 및 답변 정합성 검사"
        elif narrative_id == "N079":
            disposition, canonical, reason = (
                "AUXILIARY", "", "problem corrected interpretation은 ACTIVE 입력과 terminal label에서 금지"
            )
        elif narrative_id in IMAGE_SOURCE_MAP:
            canonical = IMAGE_SOURCE_MAP[narrative_id]
            if canonical in active_ids:
                disposition, reason = "KEEP", "실제 이미지 파일과 farm-period link가 존재"
            else:
                disposition, canonical, reason = (
                    "AUXILIARY", "", "실제 이미지 파일이 없어 farm-period 정렬을 materialize할 수 없음"
                )
        else:
            disposition, canonical, reason = (
                "DROP", "", "명확한 비중복 materialization 근거 또는 필수 컬럼이 부족"
            )
        rows.append({
            "source_catalog": "chatgpt80", "source_narrative_id": narrative_id,
            "disposition": disposition, "canonical_narrative_id": canonical, "reason": reason,
        })
    return rows


REGISTRY_HEADERS = ["registry_id", "source_catalog", "source_narrative_id", "purpose", "reason"]


def registry_rows(active_ids: set[str]) -> dict[str, list[dict]]:
    validation = [
        ("V001", "baseline70", "A09", "constant_channel", "tube_rail 전시각 상수 검증"),
        ("V002", "chatgpt80", "N068", "constant_channel", "튜브레일 상수 채널 검사"),
        ("V003", "chatgpt80", "N069", "cross_zone_weather", "구역 간 외기센서 정합성 검사"),
        ("V004", "chatgpt80", "N070", "time_axis", "336개 hourly timestamp 순서와 중복 검사"),
        ("V005", "chatgpt80", "N078", "answer_alignment", "score90 답변 수치 재현 감사"),
        ("V006", "generated", "TIME_AXIS", "time_axis", "1시간 간격과 시간 역전 검사"),
        ("V007", "generated", "MISSINGNESS", "missingness", "필수 컬럼 결측 검사"),
        ("V008", "generated", "FILE_COMPLETENESS", "file_completeness", "소스별 파일과 336행 완전성 검사"),
        ("V009", "generated", "CATALOG", "catalog_integrity", "ID 헤더 상태 및 materialization 검증"),
    ]
    auxiliary = [
        ("AUX001", "chatgpt80", "N079", "forbidden_interpretation", "problem corrected interpretation 격리"),
        ("AUX002", "generated", "UNALIGNED_IMAGES", "image_holding", "farm 또는 period 정렬 근거가 없는 이미지 격리"),
    ]
    for source_id in IMAGE_SOURCE_MAP:
        if IMAGE_SOURCE_MAP[source_id] not in active_ids:
            auxiliary.append((
                f"AUX_{source_id}", "chatgpt80", source_id, "image_holding",
                "실제 이미지 파일이 없어 ACTIVE farm-period link 생성 불가",
            ))
    rag = [
        ("RAG001", "generated", "RETRIEVAL_INDEX", "retrieval_index", "순수 문서 검색 인덱스"),
        ("RAG002", "generated", "RETRIEVAL_QUALITY", "retrieval_quality", "검색 재현율과 근거 적합성 평가"),
    ]
    return {
        "validation_registry.csv": [dict(zip(REGISTRY_HEADERS, r)) for r in validation],
        "auxiliary_registry.csv": [dict(zip(REGISTRY_HEADERS, r)) for r in auxiliary],
        "rag_registry.csv": [dict(zip(REGISTRY_HEADERS, r)) for r in rag],
    }


def write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    hourly, growth, actual_columns = load_data()
    rows = build_rows(hourly, growth, actual_columns)
    validate(rows, actual_columns)
    active_ids = {row["narrative_id"] for row in rows}
    dispositions = disposition_rows(active_ids)
    assert len(dispositions) == 200
    assert len({
        (r["source_catalog"], r["source_narrative_id"]) for r in dispositions
    }) == 200
    assert {r["disposition"] for r in dispositions} <= {
        "KEEP", "MERGE", "DROP", "AUXILIARY", "VALIDATION", "RAG"
    }
    assert all(
        not r["canonical_narrative_id"] or r["canonical_narrative_id"] in active_ids
        for r in dispositions
    )

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT, HEADERS, rows)
    disposition_headers = [
        "source_catalog", "source_narrative_id", "disposition",
        "canonical_narrative_id", "reason",
    ]
    write_csv(DISPOSITION_OUT, disposition_headers, dispositions)
    generated_registries = registry_rows(active_ids)
    for filename, registry in generated_registries.items():
        write_csv(REGISTRY_DIR / filename, REGISTRY_HEADERS, registry)

    parsed = pd.read_csv(OUT, encoding="utf-8-sig")
    assert list(parsed.columns) == HEADERS and len(parsed) == len(rows)
    assert OUT.read_bytes().startswith(b"\xef\xbb\xbf")
    print(f"wrote={OUT}")
    print(f"narratives={len(rows)}")
    print("categories=" + ";".join(f"{k}:{v}" for k, v in Counter(r["category"] for r in rows).items()))
    print(f"materializable={sum('materialization 후보 0건' not in r['notes'] for r in rows)}")
    print(f"generated_files={2 + len(generated_registries)}")
    print(f"disposition_rows={len(dispositions)}")
    print("registries=" + ";".join(
        f"{name}:{len(registry)}" for name, registry in generated_registries.items()
    ))
    print(f"image_files={len(IMAGE_FILES)}")
    print("validation=PASS")


if __name__ == "__main__":
    main()
