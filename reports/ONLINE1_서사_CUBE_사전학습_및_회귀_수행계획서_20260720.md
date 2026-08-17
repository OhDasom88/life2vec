# Online1 서사·Cube 사전학습 및 회귀 계획

## 0. 판정과 단계별 승인

현재 문서는 실행 가능한 상세 수행계획서이며 실험 결과 보고서가 아니다.

### 판정

- 문서 설계: **승인**
- Phase 0~3 audit·materialization·smoke: **승인 가능**
- Full corpus build 및 전체 Sweep: **조건부 승인**
  - 정량 acceptance·anchor/정렬 계약 고정
  - Phase 0~4 PASS 후 승인

### 승인 사다리

```text
문서 설계 승인
→ Phase 0 원시 데이터 감사 승인
→ Phase 1 Cube/sequence smoke 승인
→ Phase 2 Split/materialization 승인
→ Phase 3 Training smoke 승인
→ Phase 4 계산량 preflight 승인
→ Phase 5 Development Sweep 승인
→ Phase 6 Final validation 승인
→ Phase 7 Transductive run 승인
```

Full build 전 필수 P0 계약:

1. 정량 PASS/FAIL 기준
2. `train_y` regression anchor 결합
3. last-known target baseline 제외
4. Cube-env 정렬 허용 오차

## 1. 원시 데이터, scope, lineage 계약

- 허용 원본은 Online1 `train_X.csv`, `test_X.csv`, train/test cube `.hdr/.raw`, 회귀 결합 단계의 `train_y.csv`뿐이다.
- Online2 dataset·vocab·bin·embedding·checkpoint·산출물은 사용하지 않고 코드 구조만 참고한다.
- 모든 feature/token/artifact에 lineage를 저장한다.

```yaml
feature_lineage:
  source_columns: []
  transform_chain: []
  target_dependency: false
  future_dependency: false
```

- encoder 입력은 `target_dependency=false AND future_dependency=false`만 허용한다.
- alias, rolling target, root-zone/deficit/anomaly flag, target 결측·quality proxy까지 lineage graph traversal로 차단한다.
- test label 파일 접근 0건과 `contains_hidden_targets=false`를 file-access audit로 검증한다.

### Corpus scope

```yaml
corpus_scope:
  inductive:
    sources: [train_X, train_cube]
  transductive_public:
    sources: [train_X, test_X, train_cube, test_cube]
    hidden_labels_used: false
```

결과 namespace:

- `INDUCTIVE_CV`
- `INDUCTIVE_FULL_TRAIN`
- `TRANSDUCTIVE_PUBLIC_CV`
- `TRANSDUCTIVE_PUBLIC_TEST_SUBMISSION`

`INDUCTIVE_CV`는 outer-train raw block만으로 fold-specific preprocessing·pretraining을 수행한다. 비용 운영은 §9의 개발/최종 2단계로 완화한다. test_X/cube 포함 결과를 미관측 일반화 성능으로 표현하지 않는다.

## 2. Split-first temporal 평가와 embargo

금지:

```text
전체 narrative/sequence 생성
→ sequence_id 임의 분할
```

강제:

```text
원시 DAT/time block inventory
→ outer temporal split + embargo
→ split별 preprocessing fit
→ split별 Cube image/embedding 생성
→ split별 narrative materialization
→ sequence/chunk/export/regression example
```

- 최종 평가는 DAT block forward holdout을 기본으로 한다.
- 보조 평가는 DAT GroupKFold 또는 `DAT × source_window_block` blocked CV로 한다.

```text
embargo >= max(
  regression_lookback,
  narrative_maximum_lookback,
  cube_event_window,
  derived_feature_dependency_horizon
)
```

- validation 시작 직전 embargo 구간에 걸리는 train anchor는 생성하지 않는다.
- `raw_row_id`, `cube_id`, `source_window_id`, context timestamp interval, 원시 event-derived narrative/chunk의 train/validation 교집합은 정확히 0이어야 한다.
- 모든 파생물은 원시 block의 immutable `split_id`를 상속한다.

## 3. Regression anchor와 `train_y` 결합 계약 (P0)

행 단위는 반드시 하나의 y anchor다.

```text
regression_example_id =
  hash(dataset_scope, DAT, zone, target_timestamp)

X context:
  [target_timestamp - lookback, target_timestamp]  # past_only

Cube:
  capture_end_timestamp < target_timestamp         # STRICT_PAST_ONLY

Target:
  train_y matched by exact DAT + zone + timestamp
```

### 고정 정책

- y 고유키: `DAT + zone + target_timestamp`
  - 복원 실패 시 fail 또는 skip 정책을 manifest에 사전 고정한다.
- 동일 anchor 중복:
  - dedup 후 단일 행
  - target 충돌 시 fail
- X context가 없는 y:
  - skip하고 사유를 기록
- target 결측:
  - target 3종 중 일부 결측은 target별 loss mask
  - 전부 결측이면 표본 제외
- Cube 없는 anchor:
  - C0 경로로 유지
  - Cube를 sparse modality로 취급하며 강제 제외하지 않음
- 과거 Cube가 여러 개:
  - `nearest_past`
  - `max_cube_events` 및 lookback 한도 적용
  - 미래 Cube 금지
- event별 target 복제를 금지한다.

이 계약은 full build 이전 P0로 lock한다.

## 4. Cube-env 정렬 계약 (P0)

```yaml
cube_env_alignment:
  exact_match_preferred: true
  max_backward_distance_minutes: preflight_locked
  future_match_allowed: false
  multiple_match_policy: nearest_past
  unmatched_policy: quality_token_or_exclude
  same_minute_tie_break: ENV_THEN_CUBE
```

- 회귀는 항상 과거 방향 nearest matching만 허용한다.
- 사전학습에서만 bidirectional alignment를 허용할 수 있다.
- 정렬 거리·방향·exact/approximate 여부·unmatched 비율을 기록한다.
- unmatched 처리는 exclude 또는 quality token 중 하나로 사전 고정하고 표본 수를 보고한다.

## 5. 서사 카탈로그, 결측, 표현 경계

### Catalog 필수 필드

```yaml
narrative:
  narrative_id:
  version:
  source_types:
  anchor_definition:
  temporal_direction:
  lookback:
  horizon:
  stride:
  grouping_keys:
  required_fields:
  optional_fields:
  event_grain:
  order_semantics:
  sop_eligible:
  op_eligible:
  allowed_token_families:
  forbidden_lineage_tags:
  cube_policy:
  max_cube_events:
  chunk_policy:
  regression_eligible:
  pretrain_only:
  minimum_coverage:
  missingness_policy:
  dedup_key:
```

### 서사군

- 단기:
  - `ACTUATOR_RESPONSE_CANDIDATE`
  - `TEMPORAL_ASSOCIATION_WINDOW`
  - 환풍팬·FCU·포깅·커튼·환기 전후
  - 인과 확정 명칭 금지
- 중기:
  - snapshot
  - channel trace
  - delta/hold
  - day/night
- 장기:
  - 외기·일사·강우의 반일~DAT 궤적
- 병렬:
  - 동일 weather slot의 4구역 비교
- multimodal:
  - cube/env 서사

### 서사 정책

- 사전학습은 `bidirectional_pretrain` 가능
- 회귀는 `past_only`만 허용
- 동일 materializer라도 policy 분리
- 중립 view/boundary token은 허용 가능
- target/진단 narrative token 금지
- `narrative_id`는 metadata-only
- neutral view token ablation 유지

### 결측·불규칙 시간축

- 원시값 임의 보간 최소화
- 결측은 missingness/quality event로 보존
- forward fill 사용 시 최대 허용 길이와 `future_dependency` 여부 기록
- actuator와 sensor 보간 정책 분리
- 큰 gap에서 narrative 분리
- 결측 비율이 `minimum_coverage` 미만이면 제외
- Cube 부재는 정상 sparse modality로 취급

family-balanced 또는 temperature sampling으로 단기 과대표집을 막고 family/token/mask/cube/length 비율을 기록한다. 보고서는 표현학습·회귀 모델임을 명시하고 인과추론 주장을 금지한다.

## 6. 서사 길이, chunk, 계산량

- 서사 원형 길이는 제한하지 않는다.
- 모델 지원 상한은 4096 token이다.
- 실제 기본 길이는 preflight를 통해 1024/2048/4096 중 lock한다.
- 상한 초과는 event-grain deterministic chunking을 적용한다.
- event 중간 절단을 금지한다.
- `narrative_id`, event boundary, chunk offset, anchor-relative position을 보존한다.
- 회귀는 마지막 chunk가 anchor 직전까지 포함되어야 한다.
- 필요 시 `HEAD_CHUNK + TAIL_CHUNK` 또는 계층적 pooling을 사용한다.
- length bucket과 `max_tokens_per_batch`를 적용한다.

```text
effective_tokens_per_update =
  micro_batch × sequence_length × gradient_accum
```

### 전체 Cube store 생성 전 gate

```text
Cube 1개 변환
→ Cube 10~100개 재현성/처리량 검사
→ 저장공간·GPU-hour 예상치 보고
→ 전체 store 생성 승인
```

필수 예상치:

- 전체 Cube 수
- band 이미지 수 = Cube 수 × 10
- 이미지 저장공간
- embedding 저장공간
- image forward 총시간
- sequence 수와 총 token 수
- epoch당 예상시간
- Sweep 총 GPU-hour

## 7. Cube 밴드 이미지·표현 (후보 표현)

- zone `0..3 ↔ A..D` 미검증 Cube는 제외하거나 build fail한다.
- `unverified_zone_mapping` PASS 기준은 0건이다.
- 보수적인 `capture_end_timestamp` 규칙을 고정한다.
- 밴드 10장을 각각 2D 이미지로 변환한다.

```yaml
band_input_contract:
  source_dtype: uint16
  invalid_pixel_policy: header_nodata_and_dead_pixel_mask
  percentile_scope: scope_specific_global
  percentile_low: preflight_locked
  percentile_high: preflight_locked
  channel_mapping: replicate_1_to_3
  resize_method: bicubic
  input_size: [224, 224]
  normalization: model_specific
  interpolation_antialias: true
```

- 이미지별 percentile fit 금지
- outer-train/scope 허용 원본에서만 fit
- 이미지 변환 전에 원 단위 밴드 통계 보존:
  - median
  - p10
  - p90
  - mean
  - std
  - saturation_ratio
  - valid_ratio
- ROI·배경·invalid·orientation·registration 검사와 quality metadata 저장
- RGB image encoder + 1→3 replicate는 후보 표현이며 채택 전제는 아님

### Cube Arm

- C0: Cube 미사용
- C1: frozen band image embedding + 고정 pooling
- C2: frozen embedding + wavelength positional encoding + trainable lightweight aggregator
- 참고 ablation: 단순 밴드 원시 통계

C1/C2가 밴드 통계보다 낮거나 비용 대비 개선이 미미하면 image embedding 채택을 중단한다. image encoder는 항상 frozen이며 trainable 영역은 Life2vec projection과 C2 aggregator뿐이다.

## 8. Cube 회귀 시간 경계

- 기본 `STRICT_PAST_ONLY`:

```text
cube_capture_end_timestamp < target_anchor_timestamp
```

- `PAST_OR_SIMULTANEOUS`는 별도 ablation으로만 허용한다.
- 미래 Cube nearest matching 금지
- centered window 금지
- 미래값 보간 금지
- DAT 종료 후 통계 금지
- target 기반 normalization/narrative selection 금지

## 9. Vocabulary, objective, champion score

- Online1 scope 데이터만으로 frozen vocabulary와 서사별 view manifest를 만든다.
- Objective:
  - grouped MLM
  - hard-negative SOP
  - masked cube latent: `1 - cos(P(h), sg(e))`
  - cross-modal alignment
- Cube가 없는 sequence에는 Cube loss를 적용하지 않는다.
- loss denominator는 실제 eligible sample 수로 계산한다.
- SOP hard negative는 같은 zone·유사 시간·같은 family·유사 길이/token family 분포에서 구성한다.
- zone/낮밤/Cube 존재/padding shortcut을 금지한다.
- unordered set은 SOP에서 제외한다.

### Champion 선택

- 정책 A: 순수 사전학습 champion
- 회귀 label로 전체 pretraining trial을 선택하지 않는다.
- Sweep 시작 전 score 식을 lock한다.

```text
S =
  w1 * Z(-L_MLM)
  + w2 * Z(A_SOP_hard)
  + w3 * Z(-L_cube)
  + w4 * Z(R@K_alignment)
```

- C0에는 Cube/alignment 지표가 없으므로 C0/C1/C2 내부에서 각각 champion을 선택한다.
- lock된 arm champion만 downstream frozen evaluation으로 비교한다.
- 개발 단계:
  - 고정 development DAT split
  - 작은 corpus
  - 512/1024 token
  - architecture/objective 오류 확인
- 최종 검증:
  - 소수 설정만 outer fold별 재학습
  - 전체 Sweep의 fold별 반복 금지
  - seed 3회는 champion에 한정

### W&B Sweep

- `online1_pretrain_sweep`
- `online1_regression_life2vec_sweep`
- `online1_regression_table_sweep`

단계:

1. Stage A: 하이퍼파라미터
2. Stage B: architecture·길이·C0/C1/C2
3. Stage C: seed 확인

C0/C1/C2 비교는 동일 processed token을 주 기준으로 하고 동일 GPU-hour 결과도 보고한다.

## 10. Regression 비교와 공정성

- 공통 입력은 lock된 encoder의 frozen pooled sequence representation이다.
- 원시 CSV를 비교 모델에 직접 입력하지 않는다.
- 차원 축소·scaling은 outer-train에서만 fit한다.

### 기본 비교군

1. Train-fold global mean 및 fold-local zone mean
   - previous soil target(last-known target)은 기본 비교군에서 제외
   - test/배포에서 과거 y가 제공되지 않으면 사용 금지
   - 별도 sequential protocol에서만 분리 실험
2. Ridge/ElasticNet
3. LightGBM/CatBoost
4. 작은 MLP
5. Life2vec `Flat_Decoder`/`AttentionDecoderL`
6. TabPFN 등 table foundation
   - anchor당 1행
   - feasibility gate 통과 시에만
7. 별도: encoder fine-tuning + Life2vec decoder
   - frozen head 비교에서 분리

TabPFN이 실행 불가능하면 실패 근거를 기록하고 유리한 강제 축소 없이 standard table baseline을 유지한다.

### Multi-target 평가

- outer-train scale로 정규화한 MSE 평균을 학습 loss로 사용한다.
- 원 단위 지표:
  - target별 RMSE
  - target별 MAE
  - target별 R²
  - normalized RMSE
  - macro average
  - fold 평균·분산
  - zone별/DAT별/Cube 유무별 오류

## 11. 정량 Acceptance (P0)

“검사했다”와 “통과했다”를 구분하기 위해 PASS/FAIL을 수치로 고정한다.

```yaml
acceptance:
  leakage:
    target_lineage_violations: 0
    future_dependency_violations: 0
    cross_fold_raw_overlap: 0
    cross_fold_cube_overlap: 0
    cross_fold_source_window_overlap: 0
    cross_fold_timestamp_interval_overlap: 0
    strict_past_violations: 0
    test_label_accesses: 0
    fold_external_fit_violations: 0

  cube:
    unverified_zone_mapping: 0
    malformed_cube_count: 0
    frozen_weight_sha_changed: false
    embedding_reproducibility_min_cosine: 0.9999
    invalid_pixel_ratio_max: preflight_locked
    cube_env_alignment_max_backward_minutes: preflight_locked
    future_alignment_in_regression: 0

  training:
    nan_inf_steps: 0
    resume_parity_required: true
    wandb_required_fields_missing: 0
    oov_required_families: 0
    padding_loss_applied: false
    cube_missing_cube_loss_applied: false
    frozen_encoder_grad_norm_max: 0.0

  compute:
    forward_backward_resume_smoke: PASS
    default_max_length_locked: true
    full_store_cost_estimate_reported: true

  champion:
    score_formula_locked_before_sweep: true
    arm_internal_selection_only: true
    tie_break_rule: lower_compute_then_earlier_trial
```

### 추가 검증

- SOP shortcut probe
- C0의 Cube artifact 접근 0건
- band permutation과 wavelength positional encoding 동시 변경
- pretrained vs random-init/no-pretrain
- C0/C1/C2 vs 밴드 통계
- seed/fold variance
- 모델 크기·GPU-hour·tokens/sec·추론시간 동시 보고
- W&B 필수 필드 누락 0건
- 원시 Cube 전체 W&B 업로드 금지
- 경로·checksum·manifest만 등록
- W&B offline smoke 지원

## 12. Phase별 승인 기준과 실행 순서

### Phase 승인 기준

- Phase 0 — Raw audit
  - schema·target lineage·timestamp·zone 확인
  - acceptance YAML 및 y-anchor 계약 lock
- Phase 1 — Cube preflight
  - shape·dtype·10 bands·ROI·embedding 재현성·비용 추정 PASS
- Phase 2 — Split/materialization
  - overlap·future·target 누수 0건
  - split_id 전파 PASS
- Phase 3 — Training smoke
  - C0/C1 forward/backward/resume PASS
- Phase 4 — Compute preflight
  - 기본 max length·token budget lock
- Phase 5 — Development Sweep
  - champion 식·budget·split 고정
  - arm별 champion
- Phase 6 — Final validation
  - 소수 설정 temporal fold·seed·ablation 완료
- Phase 7 — Transductive run
  - 대회 규칙 확인
  - 별도 namespace

### 실행 순서

1. Raw inventory/checksum, target-lineage deny graph, acceptance YAML, y-anchor·정렬 계약 고정
2. inductive/transductive claim manifest 분리
3. 원시 DAT temporal split·embargo 선행
4. Cube metadata 검증
5. Cube 1개 → 10~100개 변환·재현성·처리량 → 비용 추정 → store 승인
6. split별 sensor-only materialization
7. 512/1024 C0/C1 smoke
8. 1024/2048/4096 preflight 및 기본 길이 lock
9. Development Sweep 및 arm 내부 champion 선택
10. frozen representation export 후 regression baseline/sweep
11. temporal CV·forward holdout acceptance
12. 규칙 확인 후 Phase 7 transductive artifact
13. 모든 manifest/checksum·W&B reference·acceptance report lock

## 최종 산출물

- raw/split/lineage/acceptance manifests
- preprocessing registries
- narrative catalog
- band image·embedding stores
- event/sequence/chunk parquet
- scope별 vocabulary
- W&B sweep/champion manifests
- frozen representation tensors
- regression comparison report
- leakage/performance/compute acceptance report

실제 결과 판정에는 W&B run, split/leakage report, Cube preflight, smoke 로그, regression metric이 추가로 필요하다.
