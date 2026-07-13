# Online2 진단명 예측 Finetune 계획 초안

상태: draft  
작성일: 2026-07-13  
관련 pretrain: `outputs/online2/v2_runs/full_event_grain_earlystop/`  
관련 계약: `docs/online2/contracts/online2_corpus_build_plan_v7.md`, `docs/online2/contracts/online2_pretraining_master_prompt_v8.md`

---

## 1. 목표

현재 V2 pretrain encoder 표현을 이용해 **example_set 진단명 예측**을 학습하고, 같은 모델로 **problem_set 진단명**을 추론한다.  
최종적으로는 **예측 진단명 + saliency score**를 바탕으로 위험 요인(관측 토큰/피처 패턴)을 정리한다.

| 단계 | 입력 | 출력 | 정답 사용 |
|------|------|------|-----------|
| Pretrain (진행 중) | example+problem 관측 | encoder 표현 | problem 해석 금지 |
| Downstream finetune | example 시퀀스 + 진단명 | 매칭/분류 헤드 | score90 진단명만 |
| Problem 추론 | problem 시퀀스 | 진단명 확률 + saliency | 정답 사용 금지 |

---

## 2. 데이터·계약

### 2.1 Example / Problem

| Set | Case list | n | 공개 해석 |
|-----|-----------|---|-----------|
| example_set | `datasets/agrichallenge/online2/example_set/case_list.csv` | 35 | score90 등 공개 |
| problem_set | `datasets/agrichallenge/online2/problem_set/case_list.csv` | 20 | 비공개 (forbidden) |

- 차이는 정상/이상이 아니라 **최종 해석 공개 여부**.
- farm overlap 없음.
- case 단위: `farm_id` + `period_start` ~ `period_end` (약 14일).

### 2.2 라벨 원천

- 경로: `datasets/agrichallenge/online2/answers/reference_answers/score90/F*_answer.txt`
- **짧은 진단명만** 사용. score90 전문(구역별 수치·원인 서술)은 라벨/입력에 넣지 않음.

추출 규칙(초안):

```text
종합 진단은 '([^']+)'     → 진단명
종합 판단은 ([^.]+).      → 정상 운영 등
```

### 2.3 정규화 진단명 후보 (현재 파싱 결과, n=35)

| n | 진단명 (raw) | 정규화 제안 |
|---|--------------|-------------|
| 5 | 칼슘부족 / 잎끝마름 | `칼슘부족_잎끝마름` |
| 4 | 정상 운영입니다 | `정상_운영` |
| 4 | 기형과 생리장해 | `기형과_생리장해` |
| 4 | 탄저병 발생 위험 | `탄저병_발생_위험` |
| 3 | 동해 (난방 실패) | `동해_난방실패` |
| 3 | 잿빛곰팡이병 발생 위험 | `잿빛곰팡이병_발생_위험` |
| 3 | 영양생장 과다 | `영양생장_과다` |
| 3 | 영양생장 부족 | `영양생장_부족` |
| 3 | 흰가루병 발생 위험 | `흰가루병_발생_위험` |
| 3 | 뿌리 발육 부진 | `뿌리_발육_부진` |

→ **감독 학습 라벨 집합 = 위 10종** (example score90 기준).  
→ 제출 코드(`submission_online2_v2`)는 여기에 **확장 4종**을 더하고, 미발화 시 **정상 운영 폴백**을 쓴다 (§14).

### 2.4 누수 금지

Finetune 입력에서 제외:

- problem hidden / corrected interpretation
- score90 전문, `[원인 분석]`, `[관리 제안]` 본문
- `NARRATIVE` / `CATEGORY` / `DATASET` metadata 토큰 (V2 policy)
- `DISEASE|*` 토큰 신설·주입 (vocab 정책 유지)

허용:

- case window 관측 시퀀스 (환경/근권/액추에이터/생육/이미지 슬롯)
- example score90에서 추출한 **짧은 진단명 문자열** (라벨 쪽만)

**대회/제출 코드와 동일한 금지:** `farm_id`로 진단명·판독문을 조회하는 매핑을 두지 않는다.  
finetune 입력·피처에도 farm_id 원-핫/임베딩을 **진단 단서로 쓰지 않는다** (데이터 로드 키만).

---

## 3. 학습 과제 정의

### 3.1 기본 아이디어: 시퀀스–진단명 매칭

각 example 시퀀스는 하나의 정답 진단명에만 대응한다.

- **Positive**: `(seq_i, y_i)` → match = 1  
- **Negative**: `(seq_i, y_j)` where `y_j ≠ y_i` → match = 0  

동일 진단명을 공유하는 다른 농장 라벨은 negative가 아니다.

효과:

- unique 시퀀스는 35개로 동일
- 대비 학습 신호는 시퀀스당 최대 9 negatives까지 확장 가능
- problem 추론은 후보 10개 스코어링으로 일관

### 3.2 권장 목적함수

닫힌 10클래스이므로, 단순 이진 pair 무한 증강보다 아래를 1차 기본안으로 둔다.

1. 시퀀스 임베딩 `h = Enc_θ(seq)` (V2 encoder + pooling)
2. 진단명 표현  
   - **A안 (우선)**: 학습 가능한 클래스 벡터 `C ∈ R^{10×d}`  
   - **B안**: 짧은 진단명 텍스트를 별도 임베딩 후 `c_k`
3. 시퀀스마다  
   `p = softmax(sim(h, c_k) / τ)`  
   `L = CE(p, y_true)`  
   (동등하게 in-batch InfoNCE / multi-negative ranking 가능)

Neg 샘플링:

- 기본: 시퀀스당 **전 후보 softmax** (K=9, 소규모라 가능)
- 보조: hard negative 가중 (예: 흰가루↔잿빛, 영양과다↔부족)

### 3.3 대안 (비교용)

- 표준 multi-class CLS head (`num_targets=10`) — fruiting finetune 템플릿과 가장 유사
- 1:K binary BCE — 실험 브랜치용. pos:neg 비율·threshold 관리 필요

1차 실험은 **A안(클래스 벡터 + softmax CE)** 로 시작하고, 필요 시 B안·binary를 ablation한다.

---

## 4. 시퀀스 구성

### 4.1 소스

- V2 event-grain artifacts:  
  `outputs/online2/v2_build/training_events_v2.parquet`  
  `outputs/online2/v2_build/sequences_v2.parquet`  
  `outputs/online2/v2_build/vocab_v2.json` (size 1309)
- case window로 farm/period 필터링해 **case 단위 시퀀스** 재구성

### 4.2 규칙 (초안)

> **갱신 (2026-07-13):** 이벤트 pooling 계층 구조.  
> Finetune 고정: **`max_length=4096`**, **`batch_size=1`**.  
> 상세: `outputs/online2/v2_finetune/EVENT_POOLING_FINETUNE_ARCH.md`

- 단위: 1 case = 1 라벨; 1차 학습 샘플 = case 이벤트 시퀀스 (`T≈3930≤4096`), split은 case/farm
- Stage A: 이벤트 단위 Enc + pool
- Stage B: 이벤트 시퀀스 **`max_length=4096`**, **`batch_size=1`** (필요 시 window MIL)
- interpretation / score90 본문 제외; IMAGE_SLOT은 1차 포함·ablation
- 동일 전처리를 example / problem에 적용

산출물(예정):

```text
outputs/online2/v2_finetune/
  labels_example_score90.csv      # farm_id, period_*, diagnosis_id, diagnosis_name
  case_sequences_example.parquet
  case_sequences_problem.parquet
  label_map.json                  # id ↔ 정규화 진단명
```

### 4.3 Finetune 시퀀스 예시 (토큰 ↔ 원시값)

아래는 `training_events_v2_smoke.parquet`에서 뽑은 **example case** 조각이다.  
본학습은 full `training_events_v2.parquet`를 case window로 자르며, 형식은 동일하다.

#### 4.3.1 케이스 카드

| 항목 | 값 |
|------|-----|
| farm_id | `F814133` |
| set | example_set |
| period | 2025-03-24 ~ 2025-04-06 |
| score90 진단 | `기형과 생리장해` |
| 예시 event 시각 | 2025-04-04 13:00:00 (Z4) |
| event_kind | `LOCAL_OBSERVATION` |
| sequence_id (smoke) | `sequence_00000127f873964d…912ad` |

Finetune 라벨: **짧은 진단명만** (`기형과_생리장해`). score90 본문·narrative는 시퀀스에 넣지 않음.

#### 4.3.2 동일 시각 원시 CSV (비교용)

출처: `datasets/agrichallenge/online2/data/`  
타임스탬프 `2025-04-04 13:00:00`, zone **Z4** (환경·근권·구동기 시각 일치, Δ=0s).

**환경** (`E_environment/F814133_z4.csv`)

| 필드 | 원시값 |
|------|--------|
| outside_temp_c | 27.937 |
| inside_temp_c | 29.569 |
| inside_humidity_pct | 66.483 |
| co2_ppm | 430.836 |
| rain_detected | 0 |

**근권** (`R_rootzone/root_data.csv`, 같은 farm/zone/시각)

| 필드 | 원시값 |
|------|--------|
| substrate_temp_c | 12.55 |
| substrate_water_content_pct | 38.52 |
| substrate_ec_ds_m | 1.305 |

**구동기** (`A_actuator/F814133_z4.csv`)

| 필드 | 원시값 |
|------|--------|
| fcu_fan / fcu_pump / circulation_fan / co2_supply | 0 |
| shade_screen | 0.0 |
| thermal_curtain | 24.174 |
| ec_sensor (공급 EC) | 1.2 |
| ph_sensor | 5.6 |
| total_flow_rate | 51.0 |

**생육** (case 기간 양끝, zone별)

| zone | 2025-03-24 초장/엽수/관부 | 2025-04-06 초장/엽수/관부 | Δ초장 | Δ관부 |
|------|---------------------------|---------------------------|-------|-------|
| Z1 | 23.95 / 7 / 16.20 | 25.57 / 8 / 16.47 | +1.62 | +0.27 |
| Z2 | 23.11 / 7 / 15.49 | 25.02 / 8 / 16.21 | +1.91 | +0.72 |
| Z3 | 22.23 / 7 / 15.65 | 25.91 / 8 / 16.51 | +3.68 | +0.86 |
| Z4 | 24.61 / 7 / 16.43 | 23.66 / 8 / 16.93 | −0.95 | +0.50 |

#### 4.3.3 토큰화 event (ENVIRONMENT, 같은 시각·Z4)

V2 SENTENCE (smoke, 1 event = 동시각 다피처 묶음):

```text
[EVENT_SEP] VIEW|ENVIRONMENT EVENT_KIND|OBSERVATION
[MEAS_SEP] FEATURE|CO2_PPM
  VALUE_ABS|ABS_B18 CO2_PPM|ABS_B18
  VALUE_GLOBAL_REL|GLOBAL_REL_B18 CO2_PPM|GLOBAL_REL_B18
  VALUE_FARM_REL|FARM_REL_B06 CO2_PPM|FARM_REL_B06 QUALITY|OK
[MEAS_SEP] FEATURE|INSIDE_HUMIDITY_PCT
  VALUE_ABS|ABS_B32 … VALUE_FARM_REL|FARM_REL_B09 … QUALITY|OK
[MEAS_SEP] FEATURE|INSIDE_TEMP_C
  VALUE_ABS|ABS_B28 … VALUE_FARM_REL|FARM_REL_B08 … QUALITY|OK
[MEAS_SEP] FEATURE|OUTSIDE_TEMP_C …
[MEAS_SEP] FEATURE|RAIN_DETECTED OBSERVED_VALUE|ZERO …
…
```

**원시값 ↔ 토큰 대응 (요지)**

| 원시 필드 | 원시값 | 시퀀스에서의 표현 |
|-----------|--------|-------------------|
| (구조) | — | `VIEW\|ENVIRONMENT` + `[EVENT_SEP]` / `[MEAS_SEP]` |
| inside_temp_c | 29.569 | `FEATURE\|INSIDE_TEMP_C` + `VALUE_ABS\|ABS_B*` + global/farm rel bins |
| inside_humidity_pct | 66.483 | `FEATURE\|INSIDE_HUMIDITY_PCT` + ABS / GLOBAL_REL / FARM_REL |
| co2_ppm | 430.836 | `FEATURE\|CO2_PPM` + … (`ABS_B18` 등) |
| rain_detected | 0 | `FEATURE\|RAIN_DETECTED OBSERVED_VALUE\|ZERO` |
| substrate_* / actuator | (별도 event) | 보통 `VIEW\|ROOTZONE` / `VIEW\|ACTUATOR` event로 같은 `same_time_group`에  сосед |

참고:

- 연속값은 **bin 토큰**으로만 들어가며, 시퀀스에 `29.569` 같은 float 리터럴은 없다.
- bin 경계는 `outputs/online2/v2_build/binning_registry_v2_transductive.json` (`kind=ABS` 등). build 버전과 smoke가 어긋나면 `ABS_Bn` 번호가 재현과 다를 수 있으니, 본 finetune 빌드에서는 **동일 registry로 재토큰화한 case 시퀀스**를 쓴다.
- 현재 registry 기준 예: `inside_temp_c=29.569` → 대략 `(29.08, 30]` 구간, `co2_ppm=430.836` → 대략 `(429.3, 435.6]`.

같은 시각 **ACTUATOR** event 스니펫 (smoke):

```text
[EVENT_SEP] VIEW|ACTUATOR EVENT_KIND|OBSERVATION
[MEAS_SEP] FEATURE|CIRCULATION_FAN OBSERVED_VALUE|ZERO …
[MEAS_SEP] FEATURE|CO2_SUPPLY OBSERVED_VALUE|ZERO …
```

↔ 원시 `circulation_fan=0`, `co2_supply=0`.

#### 4.3.4 Case 집약 원시 요약 (규칙 진단기 비교용)

제출 `diagnose.features`와 같은 방식으로 period·zone 집약한 값 (F814133):

| zone | 주간온 | 야간온 | RH | CO2 | 배지온 | 수분 | 배지EC |
|------|--------|--------|-----|-----|--------|------|--------|
| Z1 | 24.50 | 19.90 | 82.07 | 400 | 15.30 | 39.28 | 0.50 |
| Z2 | 25.14 | 20.53 | 80.87 | 457 | 14.52 | 37.10 | 1.20 |
| Z3 | 24.07 | 20.08 | 81.60 | 429 | 13.05 | 31.28 | **2.90** |
| Z4 | 25.80 | 20.09 | 78.79 | 459 | 14.49 | 36.93 | 1.21 |

규칙 BASE `기형과 생리장해`: 배지EC&gt;0.85 AND 주간&gt;23.71 → 이 케이스는 집약 피처로도 설명 가능.  
Finetune은 위 표 하나가 아니라 **시간순 토큰 시퀀스**로 같은 정보를 본다.

#### 4.3.5 저온(동해) 대비 예시 — 원시 vs 토큰

| 항목 | 값 |
|------|-----|
| farm | `F257178` (score90: `동해 (난방 실패)`) |
| event | 2024-12-30 18:00:00 Z3 ENVIRONMENT |
| 원시 inside_temp_c | **8.759** |
| 원시 RH / CO2 | 75.367 / 329.169 |
| 시퀀스 | `FEATURE\|INSIDE_TEMP_C VALUE_ABS\|ABS_B* …` (저온 bin) |

→ 동일 FEATURE 스키마에서 원시 수준만 달라지고, finetune/saliency는 이런 bin 토큰 차이에 민감해진다.

#### 4.3.6 Finetune 샘플로 쓸 때 형태

```text
입력 시퀀스  = case period 내 LOCAL_OBSERVATION (+ GROWTH, IMAGE_SLOT)
               − interpretation / narrative / score90 본문
타깃         = 기형과_생리장해   (id ∈ 0..9)
비교 베이스라인 = diagnose.features(원시 집약) → 규칙 진단명
```

문서화·디버깅 시 권장 병기 컬럼: `farm_id, zone_id, timestamp, SENTENCE, raw_json(주요 센서)`.

---

## 5. 모델·체크포인트

### 5.1 Encoder

| 항목 | 값 |
|------|-----|
| Run | `outputs/online2/v2_runs/full_event_grain_earlystop/` |
| 후보 ckpt | `best.ckpt` (우선), 필요 시 `last.ckpt` / step ckpt |
| hidden / layers / heads | 384 / 6 / 8 |
| vocab | 1309 |
| attention | performer |

기존 `finetune_agri_fruiting` (hidden 128, vocab 357)와 **호환되지 않음** → Online2 V2용 CLS/matching 모듈 신규.

### 5.2 Finetune 헤드

- pooling: sequence pooled embedding (기존 `pooled` 경로 재사용 검토)
- head: 10-way similarity 또는 linear CLS
- freeze 전략 (소표본 기본안):
  1. embedding + encoder **동결**, head만 학습
  2. 상위 1–2 encoder layer unfreeze (low LR)
  3. 전층 unfreeze는 overfit 시에만

참고 코드:

- `src/transformer/cls_model.py`, `src/transformer/agri_fruiting_model.py`
- `scripts/predict_fruiting.py` (Captum InputXGradient)

### 5.3 하이퍼파라미터 초안

| 항목 | 초안 |
|------|------|
| max_length | **4096** (이벤트 시퀀스; pretrain 1024와 별도) |
| batch_size | **1** (grad accum으로 effective batch 보완) |
| epochs / early stop | held-out farm metric 기준, patience 짧게 |
| LR (head) | 1e-3 ~ 5e-4 |
| LR (unfrozen encoder) | head의 1/10 |
| weight decay | 0.01 |
| seed | 고정 + 3-seed 평균 권장 |
| loss | CE (class-weighted optional) |

---

## 6. 평가 프로토콜

Problem에는 공개 정답이 없으므로 **example 내부 평가만**으로 모델 선택.

### 6.1 Split

- **Leave-one-farm-out** 또는 **stratified group K-fold (farm)**  
  - 같은 `diagnosis_id`가 train/val에  alike 나뉘도록 계층화
  - pair를 늘려도 split 단위는 **farm/case**

### 6.2 Metrics

| Metric | 용도 |
|--------|------|
| Macro-F1 | 주 선택 지표 |
| Balanced accuracy | 보조 |
| Per-class recall | 질병/위험 누락 점검 |
| Confusion matrix | 유사 진단 혼동 분석 |
| Top-2 accuracy | 운영상 후보 제시용 |

Pair accuracy는 참고만. 보고·early stop에는 사용하지 않음.

### 6.3 Pretrain transductive 고지

Problem **관측**은 pretrain에 포함됨. 정답 유출은 아니나, 평가 문서에 “관측 분포는 encoder가 이미 본 상태”를 명시한다.

---

## 7. Problem 추론 · Saliency

### 7.1 추론

1. problem case 시퀀스 인코딩
2. 10 진단명 후보 스코어 → softmax
3. top-1 / top-2 진단명 + 확률 저장

산출물(예정):

```text
outputs/online2/v2_finetune/predictions/
  problem_diagnosis_probs.csv
  problem_diagnosis_topk.json
```

### 7.2 Saliency

- 방법: Captum **InputXGradient** on token embeddings (fruiting 경로 이식)
- 대상 로짓: **예측(또는 관심) 진단명** 점수
- 집약:
  - token → `FEATURE|*` / `VALUE_*` / 시간대 / zone
  - same_time_group / measurement group 단위 합산
- 해석: saliency는 **모델 민감 관측**이지 인과 확정이 아님
- 안정화: CV fold·seed 합의 top features만 리포트

산출물(예정):

```text
outputs/online2/v2_finetune/saliency/
  {farm_id}_token_saliency.csv
  {farm_id}_feature_agg.csv
  problem_risk_factor_summary.md
```

위험 요인 매핑(후속): narrative catalog의 우호환경·스트레스 서사와 feature를 느슨히 연결 (확진 문구 사용 금지).

---

## 8. 구현 작업 분해

| # | 작업 | 산출 | 의존 |
|---|------|------|------|
| 1 | score90 → 정규화 라벨 테이블 | `labels_example_score90.csv`, `label_map.json` | — |
| 2 | case 시퀀스 빌더 (example/problem) | parquet | V2 events |
| 3 | V2 matching/CLS 모델 + Hydra config | `conf/experiment/finetune_online2_diagnosis.yaml` | V2 ckpt |
| 4 | datamodule / collate (case + label) | `src/...` | #1–2 |
| 5 | example CV 학습 루프 | run dir + metrics | #3–4 |
| 6 | problem predict 스크립트 | probs/topk | #5 |
| 7 | saliency export + feature 집약 | csv/md | #6 |
| 8 | 누수 체크 (입력에 interpretation 부재 assert) | test | #2 |

재사용:

- fruiting finetune config / CLS / predict+saliency
- V2 vocab·tokenizer·abspos 설정 (`run_manifest_v2.json`)

신규 필요:

- Online2 diagnosis 전용 experiment / datamodule / label schema
- V2(384, 1309) 가중치 로드 어댑터

---

## 9. 실험 로드맵

### Phase 0 — 라벨·시퀀스 고정

- [ ] 진단명 파서 + 수동 검수 (35건)
- [ ] label_map 확정
- [ ] case 시퀀스 빌드 + interpretation 미포함 assert
- [ ] 시퀀스↔원시값 대조 샘플 export (§4.3 형식, 클래스당 1건 권장)

### Phase 1 — Sanity

- [ ] encoder freeze + 10-way CE, example 전체 overfit sanity
- [ ] 동일 설정 farm-held-out 1-fold smoke

### Phase 2 — 본학습

- [ ] stratified group CV
- [ ] freeze vs partial unfreeze 비교
- [ ] class weight on/off
- [ ] IMAGE_SLOT on/off
- [ ] best ckpt 선정 (macro-F1)

### Phase 3 — Problem + Saliency

- [ ] problem 확률·top-k export
- [ ] 규칙 진단기(`diagnose.py`)와 교차표 (일치/불일치 분석; 답 복사가 아님)
- [ ] 저신뢰 → 확장 규칙 / 유보 게이트
- [ ] 예측 클래스 기준 saliency (+ PRIORITY_RULE 축 집약)
- [ ] farm별 위험 요인 요약 초안
- [ ] (선택) KB 템플릿으로 3단 답안 초안 생성·400단어 점검

### Phase 4 — Ablation (여유 시)

- [ ] 진단명 텍스트 인코더(B안)
- [ ] hard-negative 가중 (유사 병해·BASE↔EXT)
- [ ] binary 1:K pair loss
- [ ] 집약 피처 MLP vs 시퀀스 encoder 비교 (규칙 피처 teacher)

---

## 10. 리스크와 완화

| 리스크 | 완화 |
|--------|------|
| n=35 과적합 | freeze, 강한 WD, farm CV, 다중 seed |
| score90 전문 누수 | 짧은 진단명만 파싱; 전문 입력 금지 테스트 |
| “정상 운영입니다” 표기 불일치 | 정규화 맵에서 `정상_운영`으로 고정 |
| 유사 병해 혼동 | confusion 분석 + hard negative |
| saliency 불안정 | fold/seed 교집합 feature만 채택 |
| pretrain 미완료 | finetune에 쓸 ckpt step/metric을 manifest에 고정 |
| 기존 fruiting config 차원 불일치 | V2 전용 config로 분리 |
| example에 없는 확장 진단 | 규칙 EXT + 저신뢰 유보; 14클래스 강제 학습 지양 |
| farm_id 암기 | ID를 진단 피처로 금지; held-out farm CV |
| 배지온 위상 아티팩트 | 수준(평균) 중심; 야간 근권−공기 **동시각 격차**만 예외 |
| 이미지가 진단 과결정 | IMAGE_SLOT ablation; 이미지는 보조 근거 |

---

## 11. 성공 기준 (초안)

1. Example CV macro-F1이 chance(~0.10) 대비 유의미히 상승하고, 혼동이 임상적으로 납득 가능
2. Example에서 규칙 BASE 9종(+정상)과 **대체로 정합**하되, farm_id 조회 없이도 재현
3. Problem 20건에 대해 재현 가능한 top-1/top-2 + (필요 시) 확장/유보 플래그
4. Farm별 saliency 집약이 예측 진단·PRIORITY_RULE 축과 모순되지 않는 상위 요인 3–10개
5. 입력 파이프라인에서 interpretation/problem 정답·farm→진단 매핑 누수 0건

---

## 12. 참고 경로

```text
# 데이터
datasets/agrichallenge/online2/example_set/case_list.csv
datasets/agrichallenge/online2/problem_set/case_list.csv
datasets/agrichallenge/online2/answers/reference_answers/score90/

# 규칙 기반 제출 레퍼런스 (ML 대체 대상 = diagnose 단계)
datasets/agrichallenge/online2/submission_online2_v2/
  README.md
  src/diagnose.py          # features + BASE/EXT/정상
  src/image_features.py    # RGB 보조 관찰
  src/generator.py         # 3단 보고서
  assets/knowledge_base.py # 기전·관리·PRIORITY_RULE
  assets/mechanisms.json

# Pretrain
outputs/online2/v2_runs/full_event_grain_earlystop/
outputs/online2/v2_build/training_events_v2.parquet
outputs/online2/v2_build/vocab_v2.json

# 템플릿
conf/experiment/finetune_agri_fruiting.yaml
src/transformer/agri_fruiting_model.py
scripts/predict_fruiting.py

# 계약·리뷰
docs/online2/contracts/online2_pretraining_master_prompt_v8.md
outputs/online2/v2_build/VOCAB_DISEASE_CAUSAL_FIT_REVIEW.md
```

---

## 13. 미결 사항

1. Zone별 예측 후 집약 vs farm 단일 시퀀스 — 1차는 farm 단일, zone은 후속  
   (제출 코드는 zone 집약 통계 → 진단 후 PRIORITY_RULE로 구역 선택)
2. score70/50을 soft label로 쓸지 — 1차는 score90만
3. 진단명 텍스트 인코더(B안) 도입 시점
4. Finetune에 고정할 pretrain ckpt (`best` vs 특정 step)
5. Hybrid에서 ML vs 규칙 충돌 시 우선순위 (권장: §14.5)
6. EXT 4종을 리포트 후보에만 둘지, 별도 게이트로 둘지
7. 최종 제출 시 finetune 진단만 쓸지, 기존 generator/KB 파이프라인에 꽂을지

---

## 14. 제출 코드(`submission_online2_v2`) 분석 → 계획 반영

### 14.1 파이프라인 역할 분리

제출물은 **ML이 아니라 규칙+템플릿**이다.

| 단계 | 모듈 | 역할 | Finetune과의 관계 |
|------|------|------|-------------------|
| 집약 피처 | `diagnose.features` | 기간·zone 통계 (VPD, 주야온, EC, Δ초장·Δ관부, EC비, …) | teacher / 해석 축 / 보조 베이스라인 |
| **진단명** | `diagnose.diagnose` | BASE→EXT→정상 | **대체·병행 대상 (본 finetune 핵심)** |
| 이미지 | `image_features` | RGB 관찰 (진단 주장 안 함) | 보조 근거·saliency 대조; 주분류기로 쓰지 않음 |
| 보고서 | `generator` / `custom_answers` + KB | 3단·400단어·5채널 | 진단 이후 단계; 모델 밖으로 유지 가능 |

즉 finetune은 제출 파이프라인 전체를 다시 만드는 것이 아니라,  
**`데이터 → 진단명` 단계를 시퀀스 표현 기반으로 교체/보강**하는 것이 목표와 맞다.

### 14.2 진단명 체계 (중요)

```text
우선순위:  BASE 9종  →  EXT 4종  →  정상 운영(폴백)
```

| 층 | 진단 | example 감독 라벨 |
|----|------|-------------------|
| BASE | 동해, 잿빛, 영양과다/부족, 뿌리부진, 칼슘/잎끝마름, 기형과, 탄저, 흰가루 | 있음 (정상 제외 9 + 정상 4) |
| EXT | 염분장해, 응애, 시들음병, 야간고온장해(근권 과열) | **없음** (example 오탐 0 설계) |
| KB만 | 수정불량·착색불량(광부족) | diagnose 규칙 없음 |
| 폴백 | 정상 운영 | 있음 (4건) = “규칙 미발화” |

제출본 problem 답안 분포(참고, 규칙 산출):  
정상 5, 칼슘 4, 응애 2, 야간고온 2, 기형과 2, 그 외 BASE·염분 등.  
→ **problem에서 EXT가 실제 발화**하므로, example 10클래스만으로 닫으면 확장 진단을 구조적으로 못 낸다.

### 14.3 정상 클래스

제출 코드의 정상은 **양성 정상 프로토타입이 아니라 폴백**이다.  
계획 반영:

- 학습 라벨에 `정상_운영` **유지** (score90 4건)
- 추론 시에도 “뚜렷한 BASE 없음” ≈ 정상 또는 유보와 정렬
- 정상을 “센서 완전 무이상”으로 해석하지 않음 (제출 generator도 구역 편차·5채널을 서술)

### 14.4 미등록 진단 대처 (제출 코드가 주는 답)

| 상황 | 제출 코드 | Finetune 반영 |
|------|-----------|---------------|
| BASE로 설명 | 임계값 매칭 | 10-way (9+정상) 주력 학습 |
| BASE 밖·기전 명확 | EXT 규칙 | **규칙 게이트** 또는 저신뢰 시 EXT 후보 |
| 둘 다 아님 | 정상 폴백 | `정상` 또는 `max(p)<τ` 유보 |
| 더 새로운 병명 | 없음 | top-k + 유보 + saliency (억지 신설 클래스 X) |

권장 hybrid (1차):

1. Encoder로 BASE9+정상 확률  
2. `max(p_base) ≥ τ` → 해당 진단  
3. 아니면 `diagnose.EXT` 평가 → 발화 시 확장 진단  
4. 아니면 `정상_운영` (또는 유보 플래그)

### 14.5 피처·도메인 제약 (학습/해석에 이식)

제출 `features()`가 쓰는 축은 saliency·리포트 집약의 **표준 vocabulary**로 삼는다.

- 주간/야간/최고온, RH, **VPD**, 배지온·수분·EC, Δ초장·Δ관부  
- EC비(배지EC/공급EC), 야간 근권−공기 격차, 고온저습구역 수, 과습구역 수, 차광  
- **배지온은 수준(평균) 위주** — 농가 간 일주기 위상 무작위(README 주의). 야간고온장해만 농가 내 동시각 격차 사용  
- 이미지: 갈변·퇴색·백색·엽면적·착색과 (example 분위수 임계). **병명 확정에 쓰지 않음**

KB `PRIORITY_RULE` (ec / temp / moist / rootT / night / shade)는  
saliency를 zone·피처 축으로 묶을 때 진단별 기본 포커스로 재사용한다.

### 14.6 규정·출력 형식

- farm_id → 진단 조회 **금지** (코드·KB에 F###### 없음) — finetune도 동일  
- 답안: `[상태 진단]/[원인 분석]/[관리 제안]`, **400단어**, NaN/None/inf 금지  
- 채점 5채널: 환경·근권·제어·생육·이미지  
- finetune 산출은 우선 **진단명+확률+saliency**; 문장 생성은 기존 KB/generator 재사용이 효율적

### 14.7 계획에 넣지 않을 것 / 주의

- 제출 규칙 임계값을 **정답 라벨로 복제**해 problem을 학습하지 않음 (leak·규정 위배 소지)
- example 35/35 재현 규칙을 “ML이 외울 타깃 테이블”로 쓰지 않음 — **held-out 일반화**가 목적
- 규칙과의 일치율은 **참고 메트릭**이지 학습 손실이 아님 (일치 강제 시 규칙 복사기화)

### 14.8 반영 체크리스트

- [x] 정상 = 폴백 클래스 유지  
- [x] 미등록/확장 = EXT 규칙 + 유보 (닫힌 10만으로 끝내지 않음)  
- [x] finetune 범위 = 진단명 단계; 보고서는 KB 파이프라인  
- [x] farm_id 비사용, 이미지 비주진단  
- [x] 집약 피처·PRIORITY_RULE·배지온 위상 제약을 해석 설계에 포함  
- [ ] Hybrid 게이트 τ·우선순위를 CV로 확정 (Phase 3)  
- [ ] 규칙 교차표를 실험 로그에 상시 기록

---

*이 문서는 초안이다. Phase 0 라벨 검수 및 §14 hybrid 게이트 확정 후 확정본으로 올린다.*
