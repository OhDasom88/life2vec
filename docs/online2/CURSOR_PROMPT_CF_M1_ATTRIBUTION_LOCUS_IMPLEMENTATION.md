# Cursor 실행 프롬프트 — CF M1 Attribution, Locus Selection 및 B0 Path B 구현

당신은 현재 저장소의 기존 v0.3 진단 및 Counterfactual 설계를 보존하면서, **Milestone 1(M1)의 Attribution → Intervention Locus Selection → Path A/Path B B0 탐색 → 4단계 Gate → 재인코딩 → 동결 진단 Critic 검증**을 구현하는 수석 ML/시스템 엔지니어다.

아래 요구사항은 설계 권고가 아니라 구현 계약이다. 먼저 저장소를 탐색하고 실제 코드·artifact·문서 경로를 확인한 뒤, 존재하는 구현을 최대한 재사용하라. 경로가 다르면 임의로 병렬 구현하지 말고 현재 정본 구조에 맞게 조정하라.

---

## 0. 작업 목표

기존 파이프라인에는 다음 기능이 있거나 설계되어 있다.

- 정상/비정상 Binary 및 Fine diagnosis
- Known / Unknown / Reject routing
- Stage A event encoding
- tokenization 및 binning registry
- 일부 saliency/evidence raw join
- Counterfactual Path A/B 설계
- Action Grounding 및 Gate 설계

현재 M1에서 닫아야 할 공백은 다음이다.

```text
어느 시점의 어떤 event / measurement group / actuator span을
Counterfactual 탐색 대상으로 선택할 것인가?

선택된 locus에서 어떤 Path A 값 후보 또는 Path B B0 operation 후보를
만들고, Gate와 실제 Forward Pass로 어떻게 검증할 것인가?
```

최종 구현 흐름은 다음과 같아야 한다.

```text
진단 Artifact 동결
→ Feature/Actuator Semantics 확인
→ 진단 경로와 동일한 Attribution 계산
→ Temporal Semantics Recovery
→ token → MG → event → span 집계
→ Intervention Locus Selection
→ Path A 값 후보 / Path B B0 operation 후보
→ Gate 1~4
→ raw/derived/token 전체 재생성
→ Stage A 재인코딩
→ 동결 진단 Critic
→ delta_r / model_space_delta_r
→ M1 적격성 출력
```

---

## 1. 작업 시작 전 필수 조사

코드를 수정하기 전에 다음을 조사하고 결과를 짧은 구현 메모로 남겨라.

1. 현재 진단 모델 진입점과 config
2. 최종 Binary/Fine head와 risk 계산 함수
3. Stage A의 실제 `encode_events` 또는 동등 함수
4. token hidden, event hidden, fusion 직전/직후 representation을 얻는 경로
5. 현재 saliency 구현과 signed Input×Gradient 구현 여부
6. token index → event → measurement group → raw occurrence join 경로
7. `cell_occurrences.parquet` 또는 동등 raw occurrence 자산
8. actuator 데이터의 timestamp, farm, zone, feature, raw value 구조
9. binning registry와 encode/decode API
10. 기존 Counterfactual package, search, diff, grounding, validator 파일
11. W&B logging wrapper
12. existing tests, smoke scripts, acceptance report 형식

반드시 검색할 이름:

```text
encode_events
Stage A
Input×Gradient
saliency
evidence_raw
cell_occurrences
same_time_group
measurement_group
literal_state
ZERO
POSITIVE
binning_registry
counterfactual
action_grounding
intervention
risk
diagnosis critic
wandb
```

조사 결과 실제 경로가 설계 문서와 다르면 실제 구조를 우선하되, 최종 보고서에 차이를 기록하라.

---

## 2. 절대 금지사항

1. 진단 모델의 가중치 또는 routing threshold를 M1 구현 중 변경하지 말 것.
2. 별도의 단순화된 encoder로 attribution을 계산하지 말 것.
3. 진단에 사용한 실제 fusion/encoding 경로와 다른 경로를 attribution에 사용하지 말 것.
4. embedding-level attribution의 부호를 물리값의 `increase/decrease`로 직접 매핑하지 말 것.
5. VPD, trend, rolling mean, 누적량 등 derived feature를 직접 편집하지 말 것.
6. token 하나만 바꾸고 동일 raw measurement에서 파생된 다른 token을 그대로 두지 말 것.
7. Path A target을 actuator action 또는 setpoint로 표현하지 말 것.
8. B0 actuator raw code를 capacity 또는 실제 유량으로 표현하지 말 것.
9. B1 및 response model 없이 `OPERATIONAL_CANDIDATE`를 출력하지 말 것.
10. M1 Path B 위험도 변화를 실제 물리 반응 또는 인과효과로 표현하지 말 것.
11. 유효 후보가 없을 때 억지로 개입안을 생성하지 말 것.
12. 새로운 대규모 학습, RL, physical simulator, PLC integration을 추가하지 말 것.

---

## 3. 실행 DAG

의존성은 다음 DAG를 따라야 한다.

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

번호 순서대로 무조건 실행하는 runner를 만들지 말고, 의존 artifact가 준비됐는지 확인하는 명시적 preflight를 구현하라.

---

## 4. 권장 패키지 구조

현재 저장소에 유사 패키지가 있으면 그 안에 배치한다. 없다면 아래를 기준으로 생성한다.

```text
src/online2/v2/finetune_v03/counterfactual/
├── attribution/
│   ├── token_attribution.py
│   ├── aggregation.py
│   ├── fold_consensus.py
│   └── schemas.py
├── locus/
│   ├── selector.py
│   ├── path_a_locus.py
│   ├── path_b_span_locus.py
│   └── schemas.py
├── temporal/
│   ├── semantics.py
│   ├── span_builder.py
│   └── schemas.py
├── candidates/
│   ├── path_a_generator.py
│   ├── path_b_b0_generator.py
│   └── schemas.py
├── gates/
│   ├── context_abac.py
│   ├── safety_bounds.py
│   ├── causal_consistency.py
│   ├── local_rules.py
│   └── schemas.py
├── grounding/
│   ├── raw_target.py
│   ├── actuator_span.py
│   └── retokenize.py
├── evaluation/
│   ├── diagnosis_critic.py
│   ├── perturbation.py
│   └── metrics.py
├── pipeline.py
└── cli.py
```

기존 package가 `src/online2/v2/counterfactual/`라면 중복 패키지를 만들지 말고 그 경로를 사용하라.

---

## 5. P0-A — 진단 Artifact 동결

다음 정보를 manifest로 저장하거나 기존 manifest를 검증하라.

```json
{
  "diagnosis_model_artifact": "...",
  "checkpoint_hashes": ["..."],
  "tokenizer_artifact": "...",
  "tokenizer_hash": "...",
  "binning_registry_artifact": "...",
  "binning_registry_hash": "...",
  "routing_config": "...",
  "risk_definition": "...",
  "stage_a_encoder_version": "...",
  "fold_ids": [0, 1, 2, 3, 4]
}
```

결과 경로:

```text
outputs/m1/manifests/diagnosis_model_manifest.json
```

다음 중 하나라도 없으면 M1 실행을 중단하라.

```text
model checkpoint
tokenizer
binning registry
risk function
Stage A encoding path
raw occurrence mapping
```

---

## 6. P1 — Feature/Actuator Semantics Registry

각 feature를 다음 중 하나로 분류하라.

```text
OBSERVED_STATE
DERIVED_STATE
ACTUATOR_STATE
ACTUATOR_COMMAND
INTERVAL_QUANTITY
IMMUTABLE_CONTEXT
NON_INVERTIBLE
UNKNOWN_SEMANTICS
```

필수 필드:

```yaml
inside_temp_c:
  feature_type: OBSERVED_STATE
  unit: degC
  editable_path_a: true
  editable_path_b: false
  temporal_semantics: point_observation
  directly_controllable: false

vpd:
  feature_type: DERIVED_STATE
  editable_path_a: false
  editable_path_b: false
  parents:
    - inside_temp_c
    - inside_humidity_pct
  recompute_fn: calculate_vpd

circulation_fan:
  feature_type: ACTUATOR_STATE
  editable_path_a: false
  editable_path_b: true
  temporal_semantics: state_until_next_sample
  raw_semantics: state_code
```

UNKNOWN_SEMANTICS는 자동으로 보수적으로 처리하라.

```text
Path A editable = false
Path B editable = false
operational eligibility = BLOCKED
```

결과:

```text
outputs/m1/registry/feature_semantics_registry.yaml
outputs/m1/reports/feature_semantics_audit.csv
outputs/m1/reports/actuator_semantics_audit.csv
```

---

## 7. P0-B — Attribution 계산

### 7.1 동일 경로 강제

Attribution은 진단 inference와 동일한 다음 경로를 사용해야 한다.

```text
raw/tokens
→ actual Stage A event encoding
→ actual fusion
→ actual sequence encoder
→ frozen diagnosis risk
```

간이 surrogate model이나 별도 classifier를 사용하지 말라.

### 7.2 Risk

기존 정본 risk를 재사용하라. 새로운 임의 weighted sum을 만들지 말라.

가능하면 primary risk는 calibrated Binary abnormal probability를 사용하고 Fine, prototype, energy는 auxiliary validator로 유지하라.

### 7.3 Token Attribution

기존 signed Input×Gradient가 있으면 재사용한다.

각 fold, case, token마다 다음을 저장한다.

```text
case_id
fold_id
event_id
measurement_group_id
same_time_group_id
token_index
token_string
token_role
signed_attribution
absolute_attribution
normalized_signed_attribution
normalized_absolute_attribution
```

fold 내부 normalization은 percentile 또는 robust scale을 사용하되, 설정을 artifact에 기록하라.

출력:

```text
outputs/m1/artifacts/token_attribution.parquet
```

### 7.4 Attribution 부호 해석

Attribution의 부호는 다음 의미로만 사용하라.

```text
현재 표현이 위험도를 지지하는지 또는 억제하는지
```

다음 매핑은 금지한다.

```text
S > 0 → 물리값 decrease
S < 0 → 물리값 increase
```

물리 방향은 별도의 adjacent raw perturbation으로 결정한다.

---

## 8. Attribution 집계

### 8.1 Fold 내부 집계 우선

각 fold에서 다음 집계를 먼저 수행한다.

```text
token → measurement group → event → actuator span
```

그 뒤 fold 간 median, IQR, agreement를 계산한다.

### 8.2 Measurement Group

가중 평균:

\[
S_{MG}
=
\frac{\sum_{t\in MG}\alpha_t\widetilde S_t}
{\sum_{t\in MG}\alpha_t}
\]

권장 token role weight를 config로 분리하라.

```yaml
token_role_weights:
  abs_value: 1.0
  global_value: 0.5
  farm_relative_value: 0.5
  trend: 0.25
  feature_identity: 0.0
  unit: 0.0
  source: 0.0
  time_context: 0.0
  farm_context: 0.0
```

Feature identity가 모델상 중요하더라도 실제 편집 locus 점수에는 직접 사용하지 않는다. 설명 지표로는 별도 보관할 수 있다.

출력:

```text
outputs/m1/artifacts/measurement_group_attribution.parquet
```

### 8.3 Event

\[
S_{event}
=
\frac{\sum_{MG\in event}w_{MG}S_{MG}}
{\sum_{MG\in event}w_{MG}}
\]

출력:

```text
outputs/m1/artifacts/event_attribution.parquet
```

### 8.4 Span

P4에서 복원된 span에 event attribution을 결합하라.

\[
S_{span}
=
\frac{
\sum_{e\in span}\Delta t_e S_e
}{
\sum_{e\in span}\Delta t_e
}
\]

반드시 다음을 계산하라.

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
```

Positive ratio는 event 수가 아니라 duration으로 계산한다.

출력:

```text
outputs/m1/artifacts/actuator_span_attribution.parquet
```

---

## 9. P4 — Temporal Semantics Recovery

### 9.1 의미

P4는 개입시점을 선택하지 않는다. actuator 로그가 상태인지, 구간 누적량인지 해석하고 span을 생성할 뿐이다.

### 9.2 `state_until_next_sample`

Zero-Order Hold를 사용하되 다음을 기록한다.

```text
start_time
end_time
estimate_duration_min
lower_bound_min
upper_bound_min
sampling_interval_min
missing_gap_count
span_confidence
source_event_ids
```

결측 gap이 있으면 span을 연결하지 말라.

### 9.3 `interval_quantity`

누적량을 해당 구간 평균 flux로 변환할 수 있으나, 순간 유량으로 표현하지 말라.

\[
average\_flux = \frac{interval\_quantity}{\Delta t}
\]

출력:

```text
outputs/m1/artifacts/actuator_spans.parquet
outputs/m1/artifacts/interval_quantities.parquet
```

---

## 10. P0-C — Intervention Locus Selection

### 10.1 Path A Locus

필수 조건:

```text
measurement group 존재
raw occurrence join 성공
OBSERVED_STATE
editable_path_a = true
derived/immutable 아님
fold consensus 통과
```

### 10.2 Path B Locus

필수 조건:

```text
actuator span 존재
source_event_ids와 span 경계 일치
missing_gap_count = 0
ACTUATOR_STATE 또는 ACTUATOR_COMMAND
editable_path_b = true
whitelist 통과
fold consensus 통과
```

### 10.3 출력

```text
outputs/m1/artifacts/intervention_loci_path_a.jsonl
outputs/m1/artifacts/intervention_loci_path_b.jsonl
```

유효 locus가 없으면 다음을 정상 결과로 반환하라.

```text
NO_VALID_INTERVENTION_LOCUS
```

전수 탐색으로 fallback하지 말라.

---

## 11. Path A 후보 생성

### 11.1 방향 Probe

각 locus의 원시값에 대해 가능한 경우 lower/upper 인접 후보를 생성한다.

```text
lower adjacent bin interior
upper adjacent bin interior
conditional normal reference
```

각 probe를 전체 재토큰화·재인코딩하여 다음을 계산한다.

```text
delta_r_lower
delta_r_upper
```

더 나은 방향을 `direction_hint`로 기록하되 `proposal_only`로 유지한다.

### 11.2 원시값 후보

우선순위:

```text
adjacent ABS bin
same farm/stage normal interval
global stage normal interval
observed normal bundle
constrained MLM
```

기본 target 정책:

```text
nearest_feasible_interior
```

### 11.3 출력

```text
outputs/m1/artifacts/path_a_content_candidates.jsonl
```

---

## 12. Path B B0 Operation Candidate Generator

### 12.1 허용 연산자

```text
NO_OP
truncate_start
truncate_end
clear_span
```

다른 연산자는 M1에서 생성하지 말라.

### 12.2 Attribution Mass

Operation 후보는 단일 peak가 아니라 다음 값으로 생성한다.

```text
start_mass_ratio
end_mass_ratio
duration_weighted_positive_ratio
duration_weighted_mean
fold agreement
```

### 12.3 Candidate Boundary

truncate operation은 실제 cut boundary 후보를 포함해야 한다.

1시간 데이터라면 source event boundary 또는 1시간 단위로 후보를 생성한다.

```text
truncate 1 event
truncate 2 events
...
최소 허용 ON duration까지
```

### 12.4 `clear_span`

다음 조건을 모두 만족할 때만 후보로 생성한다.

```text
높은 positive ratio
높은 attribution mean
clear 허용 whitelist
Gate 1 통과
```

최종 채택은 Gate 및 perturbation 결과로 결정하므로 heuristic만으로 선택하지 말라.

### 12.5 Threshold Config

초기 threshold는 config로 둔다.

```yaml
positive_ratio_truncate: 0.7
positive_ratio_clear: 0.9
tail_mass_ratio: 0.6
critical_attribution_percentile: 90
```

후속 offline calibration이 가능하도록 하드코딩하지 말라.

### 12.6 출력

```text
outputs/m1/artifacts/path_b_operation_candidates.jsonl
```

---

## 13. Gate 구현

### Gate 1 — Context ABAC

후보 생성 전에 적용한다.

```text
Path A: observed state만
Path B: whitelist actuator span만
derived/immutable/non-invertible 차단
```

### Gate 2 — Safety Bounds

```text
hard bounds
agronomic bounds
operational bounds
```

M1 Path B는 B1이 없으므로 operational bounds가 `UNKNOWN`일 수 있다.

이 경우:

```text
Gate 2 status = PARTIAL
operational eligibility = EXPERT_REVIEW_REQUIRED
```

`PASSED`로 과장하지 말라.

### Gate 3 — Causal Consistency

직접 편집한 raw 변수에서 derived feature를 단방향 재계산한다.

```text
temperature/humidity → VPD
flow × duration → interval quantity
raw changes → trend/rolling feature
```

충돌:

```text
ERR_CAUSAL_CONFLICT
```

### Gate 4 — Local Rules

최종 raw state에서 전체 token bundle을 재생성하고 최초 제안 token 방향과 일치하는지 검사한다.

충돌:

```text
ERR_TOKEN_BUNDLE_CONFLICT
```

`token_edits`는 Gate 4 통과 후의 원본-최종 시퀀스 diff로만 생성하라.

---

## 14. Perturbation 및 Diagnosis Critic

### 14.1 최종 평가 경로

```text
최종 raw/token sequence
→ actual Stage A encoder
→ actual sequence encoder
→ frozen Binary/Fine diagnosis critic
→ risk
```

위험도 계산에 MLM decoder 출력을 직접 사용하지 말라.

### 14.2 Path A

출력:

```text
risk_before
risk_after
delta_r
folds_improved
route_before
route_after
uncertainty_before
uncertainty_after
```

### 14.3 Path B B0

출력 명칭:

```text
model_space_delta_r
```

문서와 schema에 다음 한계를 명시한다.

```text
classifier-oriented hypothesis evaluation
not physical response validation
not causal intervention effect
not executable prescription
```

### 14.4 선택 규칙

- 항상 NO_OP과 비교
- tie는 NO_OP 우선
- route_after가 REJECT면 기각
- uncertainty가 의미 있게 증가하면 기각 또는 낮은 적격성
- Gate에서 모두 기각되면 `NO_VALID_OPERATION`

---

## 15. 출력 적격성

M1에서는 다음만 허용한다.

```text
EXPLANATORY_ONLY
EXPERT_REVIEW_REQUIRED
BLOCKED
```

금지:

```text
OPERATIONAL_CANDIDATE
```

Path A 기본:

```text
EXPLANATORY_ONLY
```

Path B B0 기본:

```text
EXPERT_REVIEW_REQUIRED
```

B1 또는 response validation이 없으면 승격하지 말라.

---

## 16. Artifact 출력

반드시 생성할 파일:

```text
outputs/m1/manifests/diagnosis_model_manifest.json

outputs/m1/registry/feature_semantics_registry.yaml

outputs/m1/artifacts/token_attribution.parquet
outputs/m1/artifacts/measurement_group_attribution.parquet
outputs/m1/artifacts/event_attribution.parquet
outputs/m1/artifacts/actuator_spans.parquet
outputs/m1/artifacts/actuator_span_attribution.parquet
outputs/m1/artifacts/same_time_group_context.parquet

outputs/m1/artifacts/intervention_loci_path_a.jsonl
outputs/m1/artifacts/intervention_loci_path_b.jsonl

outputs/m1/artifacts/path_a_content_candidates.jsonl
outputs/m1/artifacts/path_b_operation_candidates.jsonl

outputs/m1/results/path_a_cf_results.jsonl
outputs/m1/results/path_b_b0_results.jsonl

outputs/m1/reports/locus_selection_report.md
outputs/m1/reports/m1_acceptance_report.md
```

각 artifact에 schema version, model hash, tokenizer hash, registry hash, 생성 시간, config path를 포함하라.

---

## 17. W&B Logging

다음 metric을 가능한 한 summary와 table로 기록하라.

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

locus/perturbation_confirmation_rate
locus/direction_hint_flip_rate
locus/attribution_positive_but_delta_r_worse_rate

gate/gate1_reject_rate
gate/gate2_reject_rate
gate/gate2_partial_rate
gate/gate3_reject_rate
gate/gate4_reject_rate

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

Win rate 분모:

```text
최소 하나의 유효 operation 후보가 존재한 span
```

동률은 별도 기록하고 tie-break는 NO_OP 우선이다.

---

## 18. 테스트

### 18.1 단위 테스트

1. Token attribution schema
2. Fold별 MG aggregation
3. Token 수가 다른 MG의 편향 방지
4. Event aggregation
5. Irregular timestamp duration weighting
6. Positive ratio duration weighting
7. ZOH span reconstruction
8. Missing gap에서 span 분리
9. Path A raw join
10. Derived feature 편집 차단
11. Gate 1 권한 차단
12. Gate 2 PARTIAL 처리
13. Gate 3 VPD 및 interval quantity 재계산
14. Gate 4 full retokenization
15. NO_OP 항상 포함
16. truncate boundary 생성
17. tie에서 NO_OP 선택
18. B0 capacity unknown 유지
19. Path B eligibility가 OPERATIONAL_CANDIDATE가 되지 않음

### 18.2 통합 테스트

최소 한 개 Known abnormal case에 대해:

```text
진단
→ attribution
→ locus
→ Path A candidates
→ Gate
→ retokenization
→ delta_r
```

최소 한 개 actuator span case에 대해:

```text
temporal recovery
→ span attribution
→ B0 operations
→ Gate
→ token diff
→ model_space_delta_r
```

### 18.3 No-op 테스트

원본 시퀀스를 NO_OP으로 평가했을 때:

```text
raw diff = 0
token diff = 0
delta_r ≈ 0
route unchanged
```

### 18.4 Determinism

동일 checkpoint, config, case로 두 번 실행했을 때:

```text
top-K locus
candidate list
Gate result
delta_r
```

가 허용 오차 내에서 동일해야 한다.

---

## 19. M1 Acceptance Criteria

다음 항목을 `m1_acceptance_report.md`에 PASS/FAIL로 기록하라.

```text
[ ] 진단과 동일 encoding/fusion 경로 사용
[ ] 모든 artifact hash 기록
[ ] token → MG → event → span fold별 집계
[ ] raw join 성공률 기록
[ ] intervention locus artifact 생성
[ ] 유효 locus 없을 때 안전 종료
[ ] attribution 부호를 물리 방향으로 직접 사용하지 않음
[ ] adjacent perturbation direction probe 구현
[ ] Path A nearest feasible raw target 구현
[ ] Path B NO_OP 포함
[ ] B0 4개 operation 이외 생성 안 함
[ ] Gate 1~4 결과 및 reason code 기록
[ ] full retokenization 수행
[ ] Stage A 재인코딩 수행
[ ] Path A delta_r 산출
[ ] Path B model_space_delta_r 명칭 사용
[ ] Gate 2 operational unknown을 PARTIAL로 처리
[ ] OPERATIONAL_CANDIDATE 출력 0건
[ ] W&B 핵심 metric 기록
[ ] NO_OP 및 determinism 테스트 PASS
```

하나라도 blocking 항목이 실패하면 M1 완료로 보고하지 말라.

---

## 20. 실행 CLI

기존 CLI 스타일을 따르되 다음과 같은 단계 실행을 지원하라.

```bash
python -m ...cf_m1 audit-semantics --config ...
python -m ...cf_m1 compute-attribution --config ...
python -m ...cf_m1 recover-temporal --config ...
python -m ...cf_m1 select-loci --config ...
python -m ...cf_m1 run-path-a --config ...
python -m ...cf_m1 run-path-b-b0 --config ...
python -m ...cf_m1 validate --config ...
python -m ...cf_m1 build-report --config ...
```

가능하면 `run-all`을 제공하되 artifact dependency preflight를 반드시 수행하라.

---

## 21. 최종 보고 형식

작업 완료 후 다음을 보고하라.

1. 실제 수정 파일 목록
2. 재사용한 기존 코드와 새로 만든 코드
3. 설계 문서와 실제 저장소 경로 차이
4. 실행한 명령
5. 단위·통합 테스트 결과
6. 생성한 artifact
7. Acceptance PASS/FAIL
8. 아직 구현하지 않은 M1 외 범위
9. 데이터 또는 장비 의미가 불명확해 BLOCKED 처리한 항목
10. known limitation

중요: 구현이 일부만 완료됐으면 완료됐다고 표현하지 말고, 정확히 어느 단계까지 작동하는지 기록하라.
