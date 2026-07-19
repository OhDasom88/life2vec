# 온실 제어 반사실 추론(CF) M1 아키텍처 검토 및 확정 수정안

- 상태: **조건부 승인 / Blocking Revisions Required**
- 범위: Attribution, Intervention Locus Selection, Path A/B M1
- 목적: 토큰·이벤트 기여도에서 실제 편집 후보 위치와 B0 개입 연산 후보를 생성하고, 재인코딩 후 모델 공간 위험도 변화로 검증하는 M1 정본
- 주의: M1은 물리 반응 모델, B1 장비 정밀 제원, PLC 자동 제어를 포함하지 않는다.

---

## 1. 종합 판정

Attribution 및 Intervention Locus Selection 계층을 공식 아키텍처에 반영하는 방향은 승인한다.

다음 설계는 적절하다.

1. 정적 단계 번호가 아닌 의존성 DAG로 실행 순서를 제어한다.
2. Attribution은 편집 위치와 탐색 우선순위를 제공하고, 최종 효과는 실제 Perturbation과 Forward Pass의 위험도 변화로 검증한다.
3. 토큰 → measurement group → event → actuator span의 계층적 집계를 도입한다.
4. Path B B0 연산자를 `NO_OP`, `truncate_start`, `truncate_end`, `clear_span`으로 제한한다.
5. M1을 설명·전문가 검토 수준으로 제한하고 B1, response model, PLC 제어를 제외한다.

다만 현재 초안의 일부 수식과 적격성 표현은 그대로 정본화할 수 없다. 다음 수정 사항을 반영한 뒤 M1 구현 정본으로 동결한다.

---

## 2. Blocking Revision 1 — Attribution 부호를 물리 증감 방향으로 직접 해석하지 않는다

### 2.1 문제

다음 규칙은 일반적으로 성립하지 않는다.

```text
S > 0 → 물리값 decrease
S < 0 → 물리값 increase
```

Input×Gradient 또는 embedding-level attribution의 부호는 현재 표현이 위험도 출력을 지지하거나 억제하는지를 나타낸다. 이것이 곧 원시 온도·습도·EC를 어느 방향으로 바꿔야 하는지를 의미하지는 않는다.

### 2.2 확정 규칙

```text
Attribution
→ intervention locus의 위험 기여 우선순위 선정

인접 원시값 probe
→ direction_hint 생성

전체 후보 perturbation + 재인코딩
→ 최종 방향과 후보 선택
```

원시값 감소·증가 probe를 다음처럼 평가한다.

\[
\Delta R_- = R(x_{\mathrm{lower}})-R(x)
\]

\[
\Delta R_+ = R(x_{\mathrm{upper}})-R(x)
\]

`direction_hint`는 더 작은 위험도를 만드는 probe 방향으로 설정하되 `proposal_only` 상태를 유지한다.

```json
{
  "direction_hint": {
    "value": "decrease",
    "source": "adjacent_raw_perturbation",
    "status": "proposal_only",
    "delta_r_lower": -0.08,
    "delta_r_upper": 0.03
  }
}
```

최종 방향은 Gate 1~4, 재토큰화, Stage A 재인코딩, 동결 진단 critic의 Forward Pass를 거친 `delta_r`로 확정한다.

---

## 3. Blocking Revision 2 — 실행 DAG 의존성 수정

Feature 및 actuator의 의미를 알기 전에 temporal recovery와 locus selection을 실행할 수 없다.

### 3.1 확정 DAG

```text
                 [P0-A 진단 Artifact 동결]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   [P1 Feature/Actuator 의미 감사]   [P0-B Attribution 계산]
              │                           │
              ▼                           │
     [Gate 1 권한 정책 준비]              │
              │                           │
              ▼                           │
      [P4 Temporal Recovery]              │
              │                           │
              └─────────────┬─────────────┘
                            ▼
             [P0-C Intervention Locus Selection]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      [Path A Locus]              [Path B Span Locus]
              │                           │
      [P3-A 내용 후보]             [P5-A Op 후보]
              │                           │
              └─────────────┬─────────────┘
                            ▼
             [Gate 2·3·4 실행 및 재토큰화]
                            │
                            ▼
              [Stage A + 동결 진단 Critic]
                            │
                            ▼
                  [M1 결과 및 적격성]
```

### 3.2 역할 분리

- P1: Feature semantics, temporal semantics, editability, actuator raw 의미를 정의하는 공통 registry
- Gate 1: 후보 생성 전 권한·editability 사전 필터
- P4: actuator 로그를 span 또는 interval quantity로 복원
- P0-C: 복원된 여러 event/span 중 실제 CF 후보 locus를 선택
- Gate 2~4: 후보별 runtime 검증

---

## 4. Attribution 집계 계층

### 4.1 Fold 처리 원칙

각 fold에서 token → measurement group → event → span 집계를 먼저 완료한 뒤 fold 간 median, IQR, 방향 합의를 계산한다.

```text
권장:
fold별 token → MG → event → span
→ fold 간 median / IQR / agreement

금지:
fold attribution을 먼저 합친 뒤 한 번만 span 집계
```

### 4.2 Token Attribution

각 token에는 다음 정보를 기록한다.

```text
case_id
fold_id
event_id
measurement_group_id
token_index
token_string
token_role
signed_attribution
absolute_attribution
normalization_method
```

편집 점수에서 제외하거나 낮은 가중치를 적용할 token:

- feature identity
- unit
- source
- farm/zone/time context
- immutable metadata

원시값을 직접 표현하는 ABS token을 가장 높은 비중으로 두고, GLOBAL/FARM_REL은 보조 검증에 사용한다.

### 4.3 Measurement Group Score

단순 합계는 token 개수 편향을 만들기 때문에 가중 평균을 기본으로 한다.

\[
S_{MG}
=
\frac{\sum_{t\in MG}\alpha_t\widetilde S_t}
{\sum_{t\in MG}\alpha_t}
\]

함께 기록할 값:

```text
token_count
value_token_count
signed_mean
absolute_mean
signed_sum
absolute_sum
```

### 4.4 Event Score

\[
S_{event}
=
\frac{\sum_{MG\in event}w_{MG}S_{MG}}
{\sum_{MG\in event}w_{MG}}
\]

same-time group은 편집 단위가 아니라 ENV/ROOT/ACT 동시성 설명과 context join에 사용한다.

### 4.5 Actuator Span Score

불규칙 timestamp를 고려하여 유효 지속시간으로 가중한다.

\[
S_{span}
=
\frac{
\sum_{e\in span}\Delta t_e S_e
}{
\sum_{e\in span}\Delta t_e
}
\]

반드시 다음 지표를 병렬 보관한다.

```text
signed_total_attribution
absolute_total_attribution
duration_weighted_mean
duration_weighted_positive_ratio
max_positive_attribution
max_absolute_attribution
attribution_mass_center
start_mass_ratio
middle_mass_ratio
end_mass_ratio
peak_locus_index
fold_agreement_rate
```

Positive ratio도 event 개수 대신 시간 비중으로 계산한다.

\[
positive\_ratio
=
\frac{
\sum_e\Delta t_e\mathbf{1}(S_e>0)
}{
\sum_e\Delta t_e
}
\]

---

## 5. Intervention Locus 계약

### 5.1 Path A Locus

Path A의 편집 단위는 event 내부 measurement group이다.

```json
{
  "locus_id": "locus_a_001",
  "case_id": "case_001",
  "path": "A",
  "level": "measurement_group",
  "event_ids": ["event_104"],
  "measurement_group_ids": ["mg_104_temp"],
  "same_time_group_id": "stg_104",
  "feature": "inside_temp_c",
  "timestamp_start": "2026-06-10T14:00:00+09:00",
  "timestamp_end": "2026-06-10T14:00:00+09:00",
  "attribution": {
    "signed_median": 0.12,
    "absolute_median": 0.12,
    "iqr": 0.03,
    "positive_folds": 4,
    "total_folds": 5
  },
  "editability": {
    "path_a": true,
    "path_b": false,
    "feature_type": "OBSERVED_STATE"
  },
  "provenance": {
    "model_artifact_id": "...",
    "tokenizer_hash": "...",
    "registry_hash": "...",
    "raw_occurrence_ids": ["cell_104_temp"]
  }
}
```

### 5.2 Path B Locus

Path B의 유일한 1급 시간 편집 객체는 actuator span이다.

```json
{
  "locus_id": "locus_b_001",
  "case_id": "case_001",
  "path": "B",
  "level": "actuator_span",
  "event_ids": ["e10", "e11", "e12"],
  "span_id": "fan_span_03",
  "feature": "circulation_fan",
  "timestamp_start": "2026-06-10T10:00:00+09:00",
  "timestamp_end": "2026-06-10T13:00:00+09:00",
  "temporal_evidence": {
    "temporal_semantics": "state_until_next_sample",
    "sampling_interval_min": 60,
    "missing_gap_count": 0,
    "span_confidence": "ZOH_ASSUMED"
  }
}
```

### 5.3 P3/P5 진입 게이트

다음 조건을 충족하지 못하면 CF search를 시작하지 않는다.

```text
진단과 동일한 encode_events 경로
raw occurrence join 완료
feature semantics 확정
editability 및 Gate 1 통과
fold consensus 기준 통과
artifact provenance 기록
```

실패 결과:

```text
NO_VALID_INTERVENTION_LOCUS
```

---

## 6. Path A 내용 후보 생성

Attribution은 locus를 고르고, 방향은 인접 원시값 probe로 제안한다.

### 6.1 후보 구간

1. 인접 ABS bin
2. 동일 농장·생육단계 정상 reference 구간
3. 정상 quantile
4. 관측된 정상 bundle
5. constrained MLM 후보

### 6.2 원시 목표값

기본 정책:

```text
nearest_feasible_interior
```

목표 구간에 진입하는 현재값과 가장 가까운 내부값을 선택하고 센서 해상도 또는 bin 폭 기반 margin을 적용한다.

### 6.3 검증

```text
raw target
→ Gate 2 safe intersection
→ Gate 3 derived feature recompute
→ Gate 4 full retokenization
→ Stage A re-encoding
→ diagnosis critic
→ delta_r
```

---

## 7. Path B B0 Operation Candidate Generator

### 7.1 M1 허용 연산자

```text
NO_OP
truncate_start
truncate_end
clear_span
```

`pause`, `punch_hole`, `shift`, `extend_span`은 M1 제외다.

### 7.2 Attribution Mass 기반 휴리스틱

단일 peak가 아니라 attribution mass 분포를 사용한다.

#### truncate_end

```text
end_mass_ratio >= configured threshold
AND duration_weighted_positive_ratio >= configured threshold
AND fold agreement 통과
```

#### truncate_start

```text
start_mass_ratio >= configured threshold
AND duration_weighted_positive_ratio >= configured threshold
AND fold agreement 통과
```

#### clear_span

```text
span 전체 positive ratio가 높음
AND duration-weighted mean attribution이 critical threshold 이상
AND 장비 제거가 whitelist상 허용됨
```

#### NO_OP

모든 span 후보에 의무 삽입한다.

### 7.3 Threshold

초기값 0.7, 0.9 등의 값은 정본 상수가 아니라 offline calibration 대상이다.

```yaml
positive_ratio_truncate: [0.6, 0.7, 0.8]
positive_ratio_clear: [0.8, 0.9, 1.0]
tail_mass_ratio: [0.5, 0.6, 0.7]
```

재학습하지 않고 기존 attribution 및 perturbation 결과를 재사용해 선택한다.

선정 기준:

```text
perturbation_confirmation_rate
NO_OP 대비 delta_r
direction_hint 오류율
Gate 기각률
clear_span 과생성률
```

### 7.4 절단량 후보

연산자명뿐 아니라 실제 cut boundary 후보를 생성한다.

M1에서는 event boundary 또는 1시간 단위 경계를 사용한다.

```json
{
  "operation": "truncate_end",
  "source_span_id": "span_001",
  "from_start": "2026-06-10T10:00:00+09:00",
  "from_end": "2026-06-10T13:00:00+09:00",
  "target_end_candidates": [
    "2026-06-10T12:00:00+09:00",
    "2026-06-10T11:00:00+09:00"
  ],
  "intensity_policy": "preserve_observed"
}
```

최소 ON/OFF 시간, source event boundary, missing gap, switch 제약을 위반하는 cut은 후보 생성 시 제거한다.

---

## 8. M1의 Path B 위험도 의미

M1에는 B1 장비 인스턴스 정밀 스펙과 response model이 없다.

따라서 Path B에서 계산하는 값은 다음으로 명명한다.

```text
model_space_delta_r
```

정의:

```text
actuator token/span 편집
→ 가능한 경우 ENV/ROOT mask 및 MLM infill 가설
→ 전체 재토큰화·재인코딩
→ 동결 진단 critic의 위험도 변화
```

다음 의미는 갖지 않는다.

```text
실제 온실 물리 반응
인과적 개입 효과
실행 성공 보장
정밀 용량 처방
```

위험도 계산은 MLM decoder가 아니라 다음 경로를 사용한다.

```text
최종 raw/token sequence
→ Stage A encoder
→ sequence encoder
→ 동결 Binary/Fine diagnosis critic
→ R(x')
```

---

## 9. Gate 2의 M1 처리

M1은 B1을 제외하므로 Path B의 operational bounds가 없을 수 있다.

```json
{
  "gate2": {
    "status": "PARTIAL",
    "hard_bounds": "PASSED",
    "agronomic_bounds": "PASSED",
    "operational_bounds": "UNKNOWN",
    "reason_codes": [
      "DEVICE_INSTANCE_SPEC_MISSING"
    ]
  },
  "operational_eligibility": "EXPERT_REVIEW_REQUIRED"
}
```

M1에서는 Path B에 대해 `OPERATIONAL_CANDIDATE`를 출력하지 않는다.

---

## 10. Artifact Registry

```yaml
- outputs/m1/artifacts/token_attribution.parquet
- outputs/m1/artifacts/measurement_group_attribution.parquet
- outputs/m1/artifacts/event_attribution.parquet
- outputs/m1/artifacts/actuator_span_attribution.parquet
- outputs/m1/artifacts/same_time_group_context.parquet

- outputs/m1/artifacts/intervention_loci_path_a.jsonl
- outputs/m1/artifacts/intervention_loci_path_b.jsonl

- outputs/m1/artifacts/path_a_content_candidates.jsonl
- outputs/m1/artifacts/path_b_operation_candidates.jsonl

- outputs/m1/results/path_a_cf_results.jsonl
- outputs/m1/results/path_b_b0_results.jsonl
- outputs/m1/reports/locus_selection_report.md
- outputs/m1/reports/m1_acceptance_report.md
```

---

## 11. W&B Logging

### 11.1 Locus Selection

```text
locus/event_candidate_count
locus/mg_candidate_count
locus/span_candidate_count
locus/fold_agreement_rate
locus/raw_join_success_rate
locus/editable_candidate_rate
locus/no_valid_locus_rate
locus/candidate_reduction_ratio
locus/pre_filter_candidate_count
locus/post_filter_candidate_count
locus/attribution_fold_iqr
locus/top_k_stability
locus/span_reconstruction_confidence
```

### 11.2 Attribution → Perturbation

```text
locus/perturbation_confirmation_rate
locus/direction_hint_flip_rate
locus/attribution_positive_but_delta_r_worse_rate
```

### 11.3 Gate

```text
gate/gate1_reject_rate
gate/gate2_reject_rate
gate/gate2_partial_rate
gate/gate3_reject_rate
gate/gate4_reject_rate
```

### 11.4 Path B

```text
path_b/operation_candidate_count
path_b/no_op_win_rate
path_b/truncate_end_win_rate
path_b/truncate_start_win_rate
path_b/clear_span_win_rate
path_b/model_space_delta_r
path_b/no_op_margin
path_b/operation_tie_rate
path_b/no_valid_operation_rate
```

Win rate의 분모는 최소 하나의 유효 operation 후보가 존재한 span이다. 동률은 별도 집계하고 기본 tie-break는 `NO_OP` 우선으로 한다.

---

## 12. M1 완료 범위

### 12.1 In Scope

1. 진단 경로와 동일한 attribution 계산
2. fold별 token → MG → event → span 집계
3. Feature semantics 및 temporal semantics 기반 locus 생성
4. Path A raw target 후보 생성
5. Path B B0 `edit_scope` 후보 생성
6. Gate 1, Gate 3, Gate 4 완전 검증
7. Gate 2 hard/agronomic 검증 및 operational 상태 표시
8. 전체 재토큰화 및 Stage A 재인코딩
9. 동결 진단 critic을 이용한 `delta_r` 또는 `model_space_delta_r`
10. Attribution 힌트와 Perturbation 결과 일치율 기록
11. `EXPLANATORY_ONLY` 또는 `EXPERT_REVIEW_REQUIRED` 적격성 출력

### 12.2 Out of Scope

1. B1 장비 인스턴스 정밀 스펙
2. 장비 유량·개도·출력의 정밀 용량 계산
3. 환경 및 작물 반응 모델
4. 인과적 개입 효과 보장
5. PLC 또는 Closed-loop 자동 제어
6. `OPERATIONAL_CANDIDATE` 출력

---

## 13. 최종 확정 문구

Attribution 및 Locus Selection 계층은 M1 공식 아키텍처에 반영한다.

다만 Attribution은 물리적 증감 방향을 직접 결정하지 않는다. Attribution은 편집 위치와 탐색 우선순위를 제공하며, 물리 방향과 최종 개입 내용은 인접 원시값 probe, Gate 검증, 재토큰화, Stage A 재인코딩 및 동결 진단 critic의 실제 위험도 변화로 확정한다.

M1 Path B 결과는 `model_space_delta_r`로 제한하며, 실제 물리 반응·인과 효과·정밀 용량 처방으로 해석하지 않는다. B1 장비 인스턴스 정보와 response validation이 없는 상태에서는 모든 Path B 결과를 최대 `EXPERT_REVIEW_REQUIRED`로 제한한다.
