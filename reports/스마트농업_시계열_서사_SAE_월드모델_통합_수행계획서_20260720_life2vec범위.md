# 스마트농업 시계열–서사–SAE–월드모델 통합 수행계획서 (life2vec 범위판)

- 문서 버전: `V1.1_LIFE2VEC_SCOPED`
- 원본: `/data/datasets/agrichallenge/스마트농업_시계열_서사_SAE_월드모델_통합_수행계획서_20260720.md` (`V1.0`)
- 작성일: `2026-07-20`
- 문서 상태: `PROPOSED_FOR_APPROVAL`
- 적용 저장소: `/home/dasom/life2vec` (`src/online2` 모듈)
- 기본 원칙: `versioned`, `reproducible`, `fail-closed`, `holdout-blind`

---

## 0. 구현 범위와 저장소 원칙 (신규)

본 계획의 **모든 코드 구현은 `life2vec` 저장소의 `src/online2` 모듈 내에서 수행**한다. 구체적으로:

- 사전학습/미세조정/Stage-A 재인코딩: `src/online2/v2/finetune_v03/pipeline_m2.py`, `stage_a_reencoder.py` 확장
- 반사실적 이벤트 편집·거버넌스: `src/online2/v2/finetune_v03/counterfactual/cf1s/*` 확장 (`core_raw_transaction.py`, `core_execution.py`, `core_candidates.py`, `disposition_profiles.py`, `core_verifier.py`, `core_locks.py` 등 이미 존재하는 모듈)
- 신규 서브시스템(서사 grounding, SAE, 월드모델, RL 정책)은 위 기존 모듈과 같은 `src/online2/v2/finetune_v03/` 트리 밑에 새 하위 패키지로 추가한다.

`/home/dasom/agrichallenge`의 기존 코드(`src/`, `online1_tabpfn/`, `scripts/` — TabPFN·LightGBM·Ridge 등 테이블형 베이스라인)는 **설계·비교 기준 참고용으로만 사용**하며, 본 계획의 구현체로 직접 import·재사용하지 않는다. 이는 `reports/ONLINE1_서사_CUBE_사전학습_및_회귀_수행계획서_20260720.md`가 이미 채택한 원칙("Online2 dataset·vocab·bin·embedding·checkpoint·산출물은 사용하지 않고 코드 구조만 참고한다")의 역방향 적용이다 — 그쪽은 life2vec 산출물을 참고용으로 격리했고, 이 문서는 agrichallenge 산출물을 참고용으로 격리한다.

**이미 존재하여 재사용 가능한 life2vec 인프라** (아래 각 섹션에서 구체적으로 연결):

| 계획서 요구사항 | life2vec 기존 구현 |
|---|---|
| dataset/vocab/ontology/model hash 기록 (A2) | `core_locks.py`의 `REQUIRED_STABLE_LOCK_SHA_FIELDS`, `tree_manifest_sha`, `CF1S_STABLE_LOCK_MANIFEST_V2` |
| TRAIN/DEVELOPMENT/HOLDOUT 접근 로그 (A3) | `core_fold_provenance()`의 `training_seen`/`holdout_role`/`scientifically_independent_holdout` 계약 |
| 학습 case 35개 / 미사용 20개 | 이미 `CF1S_TRAIN35_REFERENCE_AGGREGATE_V1` (`build_train35_reference_report`, Development3 3건 + Primary32 32건)과 Validation20(20건, holdout)으로 구현·검증 완료 — §14 참조 |
| Constraint verifier (물리·시간·ontology·권한) | `disposition_profiles.py`의 5종 disposition + `core_verifier.py`의 G1–G12 게이트 |
| soft score 대비 hard constraint 즉시 REJECT | `CoreContractError` 기반 fail-closed 패턴 (모든 core_*.py 모듈에서 이미 일관 적용) |
| 편집 전후 raw/token/output diff (G4) | `core_execution.py`의 `production_apply_edits_and_score` 반환값(`stage_a_output_sha`, `semantic_critic_input_sha`, per-fold risk/logit) |
| 데이터→서사 생성 (일부) | `scripts/online2_v2/generate_gemma_reports.py`, 오늘 작성한 `scripts/online2_v2/v03/generate_cf1s_case_analysis_v03.py` (경량 버전) |

이 표에 있는 항목은 **새로 설계하지 않고 확장**한다. 나머지(서사→시계열 검색, SAE, 월드모델, RL 정책)는 신규 구현이다.

---

## 1. 목적

본 계획의 목적은 원시 시계열·Cube·이미지·서사를 이벤트 기반 시퀀스로 통합하고, 사전학습과 미세조정 과정에서 형성되는 모델 표현을 희소 오토인코더(Sparse Autoencoder, SAE)로 분석하며, 검증된 표현을 예측과 이벤트 편집까지 연결하는 통합 시스템을 `life2vec` 저장소 안에 구축하는 것이다.

최종적으로 다음 계보를 재현 가능하게 연결한다.

```text
원시 데이터 포인트
→ 데이터 window
→ 이벤트 및 서사
→ 토큰과 시퀀스
→ 사전학습 활성화
→ SAE 희소 특징
→ 미세조정 활성화와 예측
→ 이벤트 편집 행동
→ 편집 후 상태·예측·특징 변화
```

본 계획에서 `Monosemantic`은 모든 개별 가중치가 하나의 의미를 가진다는 뜻으로 사용하지 않는다. 모델의 dense activation에서 SAE로 추출한 희소 특징 중 의미적 순도와 기능적 개입 효과가 검증된 특징을 해석 단위로 사용한다.

---

## 2. 핵심 산출물

1. 시계열–서사 양방향 Grounding 데이터셋과 검토 UI (신규)
2. 원시 데이터부터 모델 예측까지 연결되는 lineage registry (`core_locks.py` 확장)
3. 이벤트 단위 Sequence Composer 및 편집 transaction (`core_raw_transaction.py`/`core_candidates.py` 확장)
4. MLM·SOP·시계열 복원·Time–Text 대조학습을 포함한 사전학습 모델 (`pipeline_m2.py` 확장)
5. 회귀 미세조정 모델과 예측 설명 패키지 (기존 finetune_v03 회귀 head 확장)
6. 사전학습·미세조정 활성화에 대한 SAE pilot 및 Feature Dashboard (신규)
7. 개념 분할·병합과 ontology/vocab 변경을 관리하는 Concept Governance UI (신규, `core_locks.py`의 vocab/policy hash 잠금 원칙 재사용)
8. 상태전이·개연성·불확실성 모델 및 World Model Validator (신규)
9. bounded search, preference/imitation 및 조건부 offline RL 기반 이벤트 편집 정책 (`core_candidates.py`의 후보 구성 로직을 정책의 action space 정의로 재사용)
10. W&B run/sweep/artifact와 연결된 통합 검증 보고서

---

## 3. 범위와 단계 구분

### 3.1 현재 필수 범위

- 원시 데이터–이벤트–토큰–시퀀스 계보 고정 (기존 `core_canonical.py` 해시 체인 확장)
- 시계열→서사 및 서사→시계열 검색 (신규)
- 이벤트 단위 시퀀스 생성·절단·제외·마스킹
- 멀티목적 사전학습 및 회귀 미세조정
- 선택 layer 대상 SAE pilot
- 3D 참고 시각화와 SAE 분석 UI 분리
- 단일 및 bounded multi-event 편집 (기존 CF1S `TWO_EVENT_ONLY` 범위를 시작점으로 확장)
- 편집 전후 예측·SAE activation delta 분석
- W&B 기반 데이터·모델·SAE artifact 관리

### 3.2 조건부 확장 범위

다음 항목은 본 계획의 acceptance를 통과한 경우에만 수행한다.

- action-conditioned dynamics/world model
- multi-step rollout 기반 개연성 평가
- preference/ranking 또는 behavior cloning
- conservative model-based offline RL
- cross-layer circuit tracing
- 현장 제어 또는 실행 권고

### 3.3 제외 범위

- 모든 base weight에 대한 단일 의미 보장
- 3D projection 결과만으로 개념 의미 확정
- holdout 결과를 사용한 데이터·ontology·정책 튜닝
- 물리·시간·권한·누수 조건의 학습 보상 대체
- 검증되지 않은 RL 편집안의 실제 농장 자동 제어
- 단순 회귀 모델을 별도 검증 없이 월드모델로 선언
- **agrichallenge 저장소 코드를 life2vec 구현체로 직접 import** (신규 제외 항목)

---

## 4. 기본 데이터 계약

### 4.1 정본 객체

| 객체 | 필수 내용 | life2vec 대응 |
|---|---|---|
| `RawPoint` | 원본 파일, 행·열, timestamp, farm/zone, feature, raw value | `cells_path` parquet (기존) |
| `DataWindow` | 시작·종료, 포함 point, 집계·결측·변화점 정보 | 신규 |
| `Event` | event type, feature, 상태·변화·지속시간, 공간 범위 | `CaseEvent` (기존, `pipeline_m2._load_case_events`) |
| `Narrative` | 관측, 파생 사실, 해석, 인과 상태, 추천 상태, 근거 window | 신규 (§5) |
| `TokenSpan` | vocab version, token ID, event ID, sequence position | `RegistryVocabulary` v2 (기존, 고정) |
| `Sequence` | event ordering, 포함·제외 이력, 절단·padding·modality 정보 | 기존 `sentence_tokens` 시퀀스 확장 |
| `ActivationRef` | checkpoint, layer, position, pooling, activation artifact | 신규 (Stage-A/critic activation 캡처) |
| `SAEFeatureRef` | SAE version, latent ID, activation, validation state | 신규 |
| `Prediction` | model/head, output, uncertainty, split, checkpoint | 기존 critic fold 출력 확장 |
| `EditTransaction` | before/after, action, scope, dependency, constraint 결과 | `ValidatedRawTransaction` (기존, `core_raw_transaction.py`) |
| `Concept` | ontology concept, 정의, 계층, event·narrative·SAE 연결 | 신규 |
| `ArtifactLineage` | 상위 artifact, hash, version, 생성 run | `CF1S_STABLE_LOCK_MANIFEST_V2` (기존) |

### 4.2 필수 추적 키

```text
raw_point_id
→ window_id
→ event_id
→ narrative_id
→ token_span_id
→ sequence_id
→ checkpoint_id/layer/position
→ sae_id/feature_id
→ prediction_id
→ edit_transaction_id
```

`event_id` 이후 사슬(`sequence_id`, `checkpoint_id`, `edit_transaction_id`)은 이미 CF1S의 `canonical_json_sha256` 체인(`transaction_sha`, `evidence_root_sha`, `stage_a_output_sha`)으로 구현돼 있다. 신규로 필요한 건 `narrative_id`, `sae_id/feature_id` 두 축의 연결뿐이다.

### 4.3 데이터 split 및 권한

- `TRAIN`: 사전학습·미세조정·정책 학습 허용 범위 명시
- `DEVELOPMENT`: threshold·hyperparameter·ontology 후보 평가
- `HOLDOUT`: 최종 blind 평가만 허용 — **이미 Validation20이 이 역할을 수행 중** (`holdout_role: SELECTION_BLIND_REEVALUATION`, `case_fold_provenance` 계약으로 강제)
- transductive 사전학습을 사용할 경우 unlabeled test 원시 데이터 접근과 target/정답 접근을 분리한다.
- 회귀 target, 전문가 정답 또는 미세조정 예측에서 파생된 test 서사는 사전학습 입력으로 금지한다.
- 모든 artifact에 `split_origin`, `label_access`, `allowed_objectives`를 기록한다.

---

## 5. 시계열–서사 양방향 Grounding (신규 서브시스템)

### 5.1 서사→시계열

1. 입력 서사를 관측·파생·해석·인과·추천 문장으로 분해한다.
2. 관련 feature, 생육 단계, farm/zone, 시간 조건을 구조화한다.
3. 원시값, 변화점, 지속 구간, 다변량 동시 변화 window를 검색한다.
4. ontology 규칙, Time–Text 임베딩, SAE feature, 전문가 mapping을 결합해 후보를 순위화한다.
5. 근거 window뿐 아니라 반대 근거와 결측·불확실성을 함께 표시한다.
6. 시퀀스 반영 여부를 `AUTO_ACCEPT_CANDIDATE`, `REVIEW`, `QUARANTINE`, `REJECT`로 분기한다.

### 5.2 시계열→서사

서사 생성의 기본 단위는 단일 point가 아니라 다음 중 하나로 한다.

- 변화점 중심 window
- 임계값 또는 정상범위 이탈 지속 구간
- 생육 단계별 상태 구간
- 다변량 동시 변화 구간
- 회복·악화·안정 패턴
- 이미지/Cube와 시간적으로 연결된 관측 구간

생성 서사는 다음 구조를 가진다.

```json
{
  "observation": "데이터에서 직접 확인된 사실",
  "derived_state": "공식 또는 집계로 계산된 상태",
  "interpretation": "가능한 농업적 해석",
  "causal_status": "NOT_ESTABLISHED",
  "recommendation_status": "NOT_GENERATED",
  "supporting_windows": [],
  "contradicting_windows": [],
  "confidence": {},
  "sequence_recommendation": "INCLUDE|EXCLUDE|REVIEW"
}
```

오늘 만든 `scripts/online2_v2/v03/generate_cf1s_case_analysis_v03.py`는 CF1S 검증 결과(disposition, delta risk)만 입력으로 받아 요약/진단/원인/제안 4단 구조를 생성하는 **훨씬 좁은 범위의 프로토타입**이다. 위 스키마는 이를 관측 window 단위로 일반화하고 `causal_status`/`sequence_recommendation` 필드를 추가한 상위 호환이며, 기존 스크립트의 프롬프트 구조를 그대로 확장 출발점으로 쓸 수 있다.

### 5.3 자동 큐레이션과 사람 검토

optimizer step 중 사람이 학습 데이터를 변경하지 않는다. 사람 검토는 dataset version을 고정하기 전에 수행한다.

검토 대상으로 우선 배정할 사례는 다음과 같다.

- 모델·규칙·전문가 판정 불일치
- 신규 또는 저빈도 개념
- 인과·추천 문장이 포함된 서사
- 근거와 반대 근거가 동시에 강한 사례
- 예측 영향도가 큰 사례
- concept split/merge 후보
- holdout과 유사하나 학습 권한이 불명확한 사례

### 5.4 Grounding 평가

- Narrative→Window Recall@K
- Window→Narrative Precision@K
- temporal IoU
- feature-set overlap
- farm/zone scope accuracy
- cycle consistency
- unsupported claim rate
- expert acceptance rate
- 자동 승인 오류율 및 사람 검토율

---

## 6. 이벤트·토큰·시퀀스 생성

### 6.1 이벤트 계약

이벤트는 최소한 다음 정보를 가진다.

- feature와 raw source
- 절대값·변화량·지속시간
- 시간 및 공간 범위
- 생성 규칙과 버전
- 서사 연결과 근거 수준
- modality availability
- dependency/conflict 관계

기존 `CaseEvent`(`pipeline_m2.py`)는 앞의 6개 필드를 이미 가지고 있다. "서사 연결과 근거 수준"만 신규 필드다.

### 6.2 시퀀스 절단

- 시퀀스는 이벤트 경계를 유지하여 자른다.
- max sequence length 초과 시 첫 구간은 앞에서부터 제한에 최대한 가깝게 자른다.
- 마지막 구간은 뒤에서부터 제한에 최대한 가깝게 자른다.
- 중간 구간이 필요한 경우 event overlap 정책을 별도 버전으로 기록한다.
- event 내부 토큰을 임의로 절단하지 않는다.

### 6.3 학습 전 편집

허용 action은 다음으로 제한한다.

```text
INCLUDE | EXCLUDE | MASK | REPLACE | SPLIT | MERGE | REORDER | REWINDOW
```

이 action 집합은 CF1S의 `ATOMIC`/`PAIR` 편집 transaction 개념(`core_candidates.py`)과 겹치지만, CF1S는 **미세조정 이후 인과검증 목적**의 편집(원본을 최소 변경)이고, 여기서는 **사전학습 입력 curation 목적**의 편집(잘못된/저품질 서사 시퀀스를 학습셋에서 제외)이라는 차이가 있다. 두 편집 체계를 하나의 `EditTransaction` 스키마로 통합하되, `edit_purpose: PRETRAIN_CURATION | CAUSAL_VERIFICATION | RL_POLICY` 필드로 구분해야 한다.

모든 편집은 원본을 덮어쓰지 않고 immutable transaction으로 저장한다. 편집 전후 sequence hash, 편집 이유, 적용 규칙, 영향받은 split, vocab 영향과 재학습 필요성을 기록한다.

---

## 7. 사전학습

### 7.1 모델 입력

- 정형 시계열 이벤트 토큰
- 자연어 서사 embedding 또는 narrative token
- 이미지 embedding
- hyperspectral Cube embedding
- 시간·공간·생육 단계 정보
- modality missingness mask

### 7.2 학습 목적

사전학습은 하나의 목적에 의존하지 않고 다음을 결합한다.

\[
L_{total}=\lambda_1L_{MLM}+\lambda_2L_{SOP}+\lambda_3L_{time}+\lambda_4L_{time-text}+\lambda_5L_{cross-modal}
\]

| 손실 | 목적 | life2vec 대응 |
|---|---|---|
| `MLM` | event/token 문맥 복원 | 기존 Stage-A 사전학습 objective 확장 |
| `SOP/temporal order` | 이벤트 순서와 시간 구조 학습 | 기존 `abspos_reference` 시간 정보 확장 |
| `time reconstruction` | 시계열 상태·변화 표현 보존 | 신규 |
| `time-text contrastive` | 시계열과 서사 latent 정렬 | 신규 (§5 grounding 데이터 필요) |
| `cross-modal matching` | 시계열·서사·이미지·Cube 관계 학습 | 신규, `ONLINE1` 문서의 Cube C1/C2 arm과 연계 |

손실 적용 여부와 가중치는 W&B sweep 대상에 포함하며, 해당 loss를 적용하지 않는 선택지도 포함한다.

### 7.3 대조학습 pair

pair 품질을 다음과 같이 구분한다.

```text
GOLD_EXPERT
SILVER_RULE_GROUNDED
SILVER_MODEL_GROUNDED
WEAK_TEMPORAL_MATCH
UNVERIFIED
CONTRADICTED
```

- 동일 의미가 여러 구간에 존재할 수 있으므로 multi-positive를 허용한다.
- random negative 외에 생육 단계, 시간 방향, 지속시간, farm/zone이 유사한 hard negative를 구성한다.
- farm ID, 날짜 또는 modality 존재 여부만으로 정답을 맞히는 shortcut을 검사한다.
- target 또는 전문가 정답에서 파생된 holdout pair는 금지한다.

### 7.4 W&B 기록

- dataset/vocab/ontology/narrative registry hash
- 모델 구조와 checkpoint
- 모든 loss와 modality별 loss
- retrieval 및 reconstruction 지표
- gradient norm, activation norm, dead modality
- 학습 재개 및 checkpoint parity
- sweep config와 selection rule
- best model 선정 시 development 지표만 사용했는지 여부

---

## 8. 미세조정

### 8.1 목적

사전학습 encoder를 기반으로 regression target을 예측하며, 입력 시계열·서사·이미지·Cube가 예측에 기여한 경로를 원시 데이터까지 역추적한다.

### 8.2 모델 출력

- point estimate
- calibrated uncertainty
- fold/checkpoint별 예측
- modality별 ablation
- event/token attribution
- prediction head 입력 activation

### 8.3 표현 추적

다음 위치를 혼동하지 않는다.

| 위치 | 의미 |
|---|---|
| Sequence position | 토큰의 시퀀스 내 위치 |
| Token embedding | vocab embedding table의 고정 벡터 |
| Contextual activation | checkpoint·layer·문맥에 따른 hidden state |
| SAE feature activation | contextual activation을 SAE로 분해한 희소 값 |

동일 토큰이라도 sequence, checkpoint, layer가 다르면 별도 activation으로 기록한다. **이 표현 구분이 사용자가 요청한 "미세조정 데이터 포인트가 사전학습 단어사전·가중치를 통한 위치와 연결"의 정확한 구현 위치다** — `Token embedding`은 이미 pretrain의 frozen `RegistryVocabulary`에서 고정돼 있고, finetune은 `Contextual activation`만 새로 만든다. 즉 "3차원 공간에서 어떻게 위치하는지 보는" 요구는 `Contextual activation`을 UMAP/PCA로 투영하는 §10 3D Explorer로 구현한다.

---

## 9. SAE 기반 기계적 해석

### 9.1 원칙

SAE latent는 자동으로 monosemantic feature로 확정하지 않는다. 다음 상태를 사용한다.

```text
SAE_LATENT
→ CANDIDATE_SEMANTIC_FEATURE
→ EVALUATED_SEMANTIC_FEATURE
→ INTERVENTION_SUPPORTED_FEATURE
→ CIRCUIT_SUPPORTED_FEATURE
```

### 9.2 SAE 학습 위치

초기 pilot은 다음 세 지점에서 수행한다.

1. 사전학습 encoder 중간층
2. 사전학습/미세조정 encoder 최종층 (기존 `event_mean`/`event_max` pooled output, `core_canonical.stage_a_output_sha`로 이미 해시 추적됨)
3. regression 또는 dynamics head 입력 직전

필요 시 다음 SAE를 별도 version으로 관리한다.

- `SAE_PRETRAIN`
- `SAE_FINETUNE`
- `SAE_WORLD`
- `SAE_POLICY`

### 9.3 SAE sweep

- dictionary expansion factor
- L1, Top-K 또는 BatchTopK sparsity
- learning rate
- activation normalization
- dead feature resampling/처리
- 대상 layer와 position pooling
- reconstruction–sparsity trade-off

### 9.4 SAE 평가

| 영역 | 지표 |
|---|---|
| 복원 | reconstruction error, explained variance, downstream fidelity |
| 희소성 | 평균 L0, activation frequency, dead feature ratio |
| 의미 | concept precision/recall, positive/negative consistency |
| 분해 | splitting, absorption, duplicate, composition |
| 안정성 | farm·생육 단계·계절·checkpoint 간 stability |
| 기능 | ablation, steering, activation patching, prediction delta |

SAE reconstruction과 sparsity가 좋아도 의미적 순도가 확보됐다고 판정하지 않는다.

### 9.5 개념 매핑

SAE feature와 서사/ontology concept는 1:1을 강제하지 않는다 (1:1, 1:N, N:1, 계층 관계, contextual relation).

### 9.6 사전학습–미세조정 특징 정렬

미세조정 이후 특징 상태를 다음으로 분류한다.

```text
PRESERVED | SPLIT | MERGED | SUPPRESSED | NEWLY_EMERGED | UNMATCHED
```

activation correlation, decoder similarity, 사례 중첩, concept score와 intervention effect를 근거로 정렬한다.

### 9.7 인과 및 회로 증거 등급

| 등급 | 의미 |
|---|---|
| `E0_OBSERVED` | feature activation 관측 |
| `E1_ASSOCIATED` | 상태·서사와 통계적으로 연관 |
| `E2_PREDICTIVE` | 출력 예측에 유용 |
| `E3_INTERVENTION` | ablation/steering으로 출력 변화 재현 |
| `E4_MEDIATED` | 편집 효과의 feature 매개 확인 |
| `E5_CIRCUIT` | upstream/downstream 계산 경로 확인 |

`E3` 미만을 인과적 특징이라고 부르지 않으며, SAE Top-K만으로 circuit tracing을 주장하지 않는다. 이 등급 체계는 CF1S의 `SELECTED_MATERIAL`/`SELECTED_CONTROL_NO_MATERIAL`/`EXCLUDED_PARTIAL_MATERIAL` disposition 체계와 같은 설계 원칙(자기선언 금지, 재계산 기반 판정)을 공유한다 — SAE feature 등급도 `classify_case_disposition()`과 동일하게 **저장된 라벨이 아니라 매번 재계산**해야 한다.

---

## 10. 3D 표현 탐색

UMAP/PCA 등의 3D projection은 다음 용도로만 사용한다.

- 데이터 분포 및 군집 탐색
- 이상치 탐지
- train/development/holdout 겹침 확인
- modality별 분포 비교
- 사전학습과 미세조정 전후의 거시적 변화 확인

UI에는 원본 차원, projection 방법·파라미터·seed, checkpoint/layer, 색상 label의 출처와 neighborhood preservation 지표를 표시한다. 3D 거리나 군집만으로 개념의 존재, 단일 의미성 또는 인과성을 확정하지 않는다.

---

## 11. 개념 분할·병합 및 재학습

### 11.1 변경 유형

| 유형 | 예 | 기본 대응 |
|---|---|---|
| 해석 특징 변경 | SAE 고온 특징 세분화 | SAE 재학습·probe |
| Ontology 변경 | HIGH_VPD를 생육 단계별 분리 | ontology version·재매핑 |
| Event 변경 | 이벤트 조건·window 변경 | 영향 시퀀스 재생성 |
| Vocab 변경 | token 분할·병합·bin 변경 | 재토큰화·embedding 재학습 |
| 데이터 규칙 변경 | 전체 시퀀스 생성 규칙 변경 | continual/full pretraining 판정 |

**운영상 중대 영향**: `Vocab 변경`이 발생하면 CF1S의 `code_tree_sha256`/`policy_tree_sha256`는 영향받지 않지만 `tokenizer_sha256`/`vocab_sha256`가 바뀌어 **기존 55건(Dev3 3 + Validation20 20 + Primary32 32) 전체의 stable lock이 무효화**된다. Vocab 변경을 승인하기 전에 재인증 비용(전체 재실행 시간)을 §14 규모 산정과 함께 명시적으로 보고해야 한다.

### 11.2 분할·병합 후보

- 동일 특징이 서로 반대 prediction effect를 가짐
- 생육 단계·시간대에 따라 의미가 분리됨
- 하나의 SAE feature에 여러 독립 개념이 혼합됨
- 하나의 개념이 다수 feature에 불안정하게 분산됨
- 전문가 불일치가 반복됨
- 두 개념이 원시 조건·활성·예측 효과에서 지속적으로 동일함

### 11.3 승인 절차

```text
후보 탐지
→ 근거 보고서
→ 전문가/개발 승인
→ ontology/vocab/SAE 새 버전
→ 영향 데이터 재생성
→ 필요한 범위 재학습
→ 이전 version과 회귀 비교
→ (vocab 변경 시) CF1S 55건 재인증
```

SAE dictionary 크기 증가만으로 분할·병합이 해결됐다고 판정하지 않는다.

---

## 12. 월드모델 준비

### 12.1 모델 역할 분리

| 모델 | 역할 | life2vec 대응 |
|---|---|---|
| Representation model | 시퀀스 공통 표현 | Stage-A reencoder (기존, frozen) |
| Outcome model | 회귀 target 예측 | finetune critic (기존, fold 0/1/2) |
| Plausibility model | 편집된 시퀀스의 분포상 가능성 | 신규 |
| Dynamics model | 현재 상태에서 다음 상태 예측 | 신규 |
| Action-conditioned world model | 행동에 따른 다중 시점 상태전이 | 신규, §12.2 행동 데이터 필요 |
| Uncertainty model | epistemic/aleatoric uncertainty | 신규 |
| **Constraint verifier** | 물리·시간·ontology·권한 검사 | **`disposition_profiles.py` + `core_verifier.py` G1–G12 (기존, 그대로 재사용)** |
| Policy | 이벤트 편집 행동 선택 | 신규 (§13) |

encoder 공유는 허용하지만 각 head의 loss, 검증 지표와 사용 권한은 분리한다.

### 12.2 행동 데이터 계약

월드모델 학습 전에 관측과 행동을 구분한다.

- 난방·냉방 설정
- 환기창 개폐
- 차광
- 관수 시작·종료·양
- 양액 EC/pH 설정
- 농작업
- 행동 시각과 반응 지연
- 행동 전후 상태

행동 데이터가 없으면 상태 예측 모델을 action-conditioned world model 또는 인과 개입 모델로 표현하지 않는다.

### 12.3 월드모델 승격 기준

- next-state 성능이 persistence/seasonal baseline보다 우수
- multi-step rollout의 horizon별 오류 보고
- action conditioning이 실제 반응 차이를 학습
- uncertainty calibration 통과
- OOD action 및 OOD state 탐지
- 실제 trajectory와 rollout의 물리·시간 일관성
- development에서 acceptance 고정 후 holdout(Validation20) blind 평가

기준 미충족 시 RL 환경으로 사용하지 않고 bounded edit의 보조 scorer로만 사용한다.

---

## 13. 이벤트 편집과 RL

### 13.1 단계적 도입

1. 단일 이벤트 perturbation — **이미 CF1S의 `ATOMIC` 편집으로 존재**
2. ontology 제한 bounded multi-event search — **이미 CF1S의 `PAIR` 편집(`TWO_EVENT_ONLY`)으로 존재**
3. outcome/plausibility/dynamics 비교
4. 전문가 preference 또는 ranking 수집
5. behavior cloning/imitation
6. offline policy evaluation
7. conservative model-based offline RL
8. holdout(Validation20) blind evaluation

1~2단계는 신규 구현이 아니라 **이미 55건 검증을 통과한 CF1S 파이프라인 자체**다. RL 도입은 3단계부터 시작되는 신규 작업이며, RL이 찾은 편집 후보는 반드시 기존 CF1S 검증(잠긴 임계값, fold 격리, holdout 재평가)을 통과해야 "실제 효과"로 인정한다 — RL을 CF1S의 대체가 아니라 **CF1S 이전 단계의 후보 생성기**로 위치시킨다.

### 13.2 State와 Action

State에는 sequence representation, 생육 단계, farm/zone, 예측과 불확실성, 편집 가능 event, raw inversion 정보와 이전 편집 이력을 포함한다.

Action은 다음으로 제한한다.

```text
NO_OP | SELECT_EVENT | MODIFY | INSERT | DELETE | SPLIT | MERGE | REWINDOW
```

`NO_OP`는 항상 후보에 포함한다.

### 13.3 보상과 제약

학습 가능한 soft score:

- 목표 예측 변화
- 개연성
- 서사 일관성
- 편집 최소성
- 정상 trajectory 유사성
- 전문가 선호
- epistemic uncertainty
- SAE feature transition의 자연스러움

강제 hard constraint (기존 `disposition_profiles.py`/`core_raw_transaction.py`의 `assert_change_set_closed` 등으로 이미 상당 부분 구현):

- 물리적 범위
- 시간 역전 방지
- 생육 단계 순서
- sensor/actuator 타입
- ontology dependency/conflict
- 원시값 복원 가능성
- split 접근 권한과 누수 방지

hard constraint 위반은 음수 reward가 아니라 즉시 `REJECT`로 처리한다 (`CoreContractError` 패턴 재사용).

### 13.4 OOD action 정책

```text
IN_SUPPORT → 허용
NEAR_SUPPORT_LOW_UNCERTAINTY → 제한 rollout 허용
NEAR_SUPPORT_HIGH_UNCERTAINTY → 보류/검토
OUT_OF_SUPPORT → 차단
PHYSICAL_INVALID → 즉시 차단
```

### 13.5 SAE 연결

편집 전후 다음을 기록한다.

- raw/event/token diff
- outcome 및 uncertainty delta
- SAE feature activation delta
- 새롭게 발생한 feature combination
- ablation/steering 결과
- policy가 반복적으로 악용하는 특징

새로운 feature 조합은 자동으로 비현실적이라고 확정하지 않고 `NOVEL_FEATURE_COMBINATION`으로 분류하여 uncertainty, raw constraint와 전문가 검토를 거친다.

---

## 14. 데이터 규모 및 RL 준비도

**Train35(Development3 3건 + Primary32 32건)와 Validation20(20건, holdout)은 이미 실제 GPU 실행으로 생성·검증 완료됐다** (`CF1S_TRAIN35_REFERENCE_AGGREGATE_V1`, 전 cohort G1–G12 FINAL PASS). RL 가능성은 이 55건에서 생성된 단순 window 수가 아니라 다음으로 판단한다.

- independent farm 수
- independent growth cycle 수
- effective trajectory count
- action–response pair 수
- action coverage
- state–action support
- 동일 case 내부 상관성
- rare action 빈도

**실측 참고**: 55건 검증에서 44건은 `CONSTRUCTIBLE_SELECTED`까지 도달했지만 전부 `CONTROL_WITHIN_LOCKED_THRESHOLD`(잠금 임계값 이내)로 판정됐다 — 즉 현재 2-event 편집 범위에서는 어떤 케이스에서도 임계값을 넘는 인과효과가 관측되지 않았다. 이는 RL 준비도 평가에 그대로 반영해야 할 사실이다: **action–response 신호가 이미 알려진 편집 범위 안에서는 희박하다는 것이 실측으로 확인된 상태**이므로, RL을 위한 action coverage 확장(예: 3+ event 편집, 다른 feature 조합)이 선행되지 않으면 RL이 학습할 신호 자체가 부족할 수 있다.

독립 trajectory와 action coverage가 부족하면 RL을 수행하지 않고 bounded search와 preference ranking 결과까지만 정식 산출물로 채택한다.

---

## 15. UI 구성

| UI | 주요 기능 | 생성 artifact | life2vec 내 위치(제안) |
|---|---|---|---|
| Data Grounding & Curation | 서사↔시계열, 근거·반례, 검토 큐 | grounding decision | `src/online2/v2/finetune_v03/counterfactual/grounding_ui/` (신규) |
| Sequence Composer | 이벤트 단위 include/exclude/mask/split/merge | sequence transaction | 기존 `core_raw_transaction.py` 위에 UI 계층 추가 |
| 3D Explorer | 거시 분포·이상치 탐색 | projection artifact | 신규 |
| SAE Feature Dashboard | feature heatmap, 사례, 품질, 개입 | feature evaluation | 신규 |
| Concept Governance | N:M mapping, split/merge, 승인 | ontology/mapping version | 기존 `core_locks.py` 잠금 워크플로우 위에 UI 계층 추가 |
| World Model Validator | rollout, uncertainty, OOD | readiness report | 신규 |
| Edit Policy Lab | 편집 diff, reward, gate, policy 비교 | edit transaction | 기존 CF1S quarantine/promotion 결과 뷰어 확장 |
| Run & Artifact Control | W&B run/sweep 및 lineage | run/selection manifest | 신규, 기존 `outputs/cf1s_core/` 아티팩트 구조 재사용 |

UI 화면 상태는 정본으로 사용하지 않는다. 모든 변경은 versioned artifact와 immutable transaction으로 저장한다.

---

## 16. 단계별 수행 일정

### Phase 0. 계약 및 기준선

- schema·ID·split·권한 고정
- 기존 Online1/Online2(life2vec) artifact inventory — **CF1S 55건 검증 결과가 이미 이 inventory의 핵심**
- baseline checkpoint 및 지표 snapshot
- W&B project/group/job_type 규칙 고정

### Phase 1. Grounding 및 큐레이션

- 양방향 검색, pair 품질 등급, hard negative·multi-positive, 큐레이션 UI, grounding acceptance

### Phase 2. 시퀀스 및 사전학습

- event/token/sequence 생성, 이벤트 단위 절단, 멀티목적 사전학습, modality 및 누수 검증, W&B sweep과 checkpoint lock

### Phase 3. 미세조정 및 표현 추적

- regression head, uncertainty, raw→prediction trace, attribution/ablation, 사전학습 대비 표현 변화

### Phase 4. SAE Pilot

- 선택 layer activation 수집, SAE sweep, feature 평가, concept N:M mapping, 편집 전후 activation delta

### Phase 5. Concept Governance

- split/merge 후보, ontology/vocab 영향 분석, 승인 및 새 version, 필요 범위 재토큰화·재학습 (+ CF1S 재인증)

### Phase 6. 월드모델 준비

- transition/action dataset, dynamics/plausibility/uncertainty, multi-step rollout, readiness 판정

### Phase 7. 편집 정책

- bounded search baseline (**기존 CF1S 재사용**), preference/imitation, 조건부 offline RL, policy exploitation 검사, holdout(Validation20) blind 평가

### Phase 8. 통합 및 잠금

- 최종 code/data/model lock, lock 이후 전체 재실행, artifact hash 및 selection manifest, 최종 acceptance report

---

## 17. Acceptance Criteria

### A. 계보와 누수

- `A1`: 모든 학습 token이 raw point/event까지 역추적 가능
- `A2`: dataset/vocab/ontology/model/SAE/policy hash 기록 — **기존 `core_locks.py` 확장**
- `A3`: TRAIN/DEVELOPMENT/HOLDOUT 접근 로그 존재 — **기존 `case_fold_provenance` 확장**
- `A4`: target-derived holdout narrative가 사전학습에 사용되지 않음

### B. Grounding

- `B1`: 모든 narrative가 supporting window 또는 `NO_SUPPORT`를 가짐
- `B2`: 관측·파생·해석·인과·추천이 구분됨
- `B3`: 양방향 retrieval과 cycle consistency 보고
- `B4`: 자동 승인 오류율 및 검토율 보고

### C. 사전학습·미세조정

- `C1`: 모든 loss와 적용 여부가 W&B에 기록됨
- `C2`: modality별 성능과 ablation 보고
- `C3`: checkpoint resume 재현성 확인
- `C4`: 미세조정 예측에서 원시 데이터까지 역추적 가능

### D. SAE

- `D1`: reconstruction과 downstream fidelity 보고
- `D2`: dead/splitting/absorption/duplicate 평가
- `D3`: 서사 개념과 N:M mapping 허용
- `D4`: activation만으로 인과를 주장하지 않음
- `D5`: `INTERVENTION_SUPPORTED`는 개입 실험을 통과해야 함
- `D6`: 사전학습–미세조정 feature alignment 보고

### E. 3D

- `E1`: 참고용 표기와 projection metadata 제공
- `E2`: 원공간 평가 없이 군집 의미를 확정하지 않음

### F. 월드모델

- `F1`: outcome과 dynamics 평가 분리
- `F2`: action-conditioned transition 검증
- `F3`: multi-step rollout과 uncertainty calibration 보고
- `F4`: acceptance 전 RL 환경 승격 금지

### G. 편집과 RL

- `G1`: NO_OP와 bounded search baseline 포함 (**기존 CF1S**)
- `G2`: hard constraint 우회 불가
- `G3`: OOD/uncertainty gate 기록
- `G4`: 편집 전후 raw/token/SAE/output diff 기록
- `G5`: development 확정 후 holdout(Validation20) blind 평가
- `G6`: 데이터 부족 시 RL 미적용 결정도 정상 결과로 인정

### H. 최종 잠금

- `H1`: 최종 code lock 이후 전체 pipeline 재실행
- `H2`: 모든 artifact hash와 parent lineage 기록
- `H3`: selection_manifest_chain_hash 기록
- `H4`: 최종 보고서가 실행 artifact와 일치

### I. 저장소 범위 (신규)

- `I1`: 모든 신규 구현이 `life2vec/src/online2` 트리 내에 위치
- `I2`: agrichallenge 등 외부 저장소 코드에 대한 직접 import 0건
- `I3`: 외부 저장소 코드를 참고했을 경우 커밋/문서에 "참고용, 미재사용" 명시

---

## 18. 중단·보류 조건

다음 조건에서는 다음 단계로 자동 진행하지 않는다.

- Grounding unsupported claim이 허용치를 초과
- vocab/event 변경으로 기존 의미 호환성이 붕괴
- SAE downstream fidelity가 부족
- SAE feature 대부분이 dead 또는 불안정
- dynamics가 단순 baseline보다 열위
- multi-step rollout 오류가 급격히 누적
- action coverage 부족 (§14 실측 근거로 이미 우려 확인)
- uncertainty가 OOD를 구분하지 못함
- 정책이 world model 오류를 악용
- holdout 또는 정답 정보 접근 흔적 발견

보류는 실패 은폐가 아니라 readiness 결과로 기록한다.

---

## 19. 위험 및 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| 서사 환각 | 잘못된 학습 pair | 근거·반례·NO_SUPPORT·검토 큐 |
| 대조학습 shortcut | 의미 대신 ID/시간 학습 | hard negative·group split·ablation |
| SAE 과신 | 잘못된 monosemantic 주장 | feature 품질·개입·증거 등급 |
| 개념 변경 비용 | 반복 재학습 + **CF1S 55건 재인증** | 변경 유형별 SAE/ontology/vocab 분리, 재인증 비용 사전 보고 |
| world model exploitation | 비현실적 편집 | uncertainty·OOD·hard verifier |
| 데이터 부족 (§14 실측 확인) | RL 과적합 | bounded search·preference 우선 |
| 3D 왜곡 | 잘못된 군집 해석 | 참고용 제한·원공간 지표 병행 |
| holdout 누수 | 평가 무효 | ABAC·artifact provenance·blind lock |
| **외부 저장소 코드 혼입** | 재현성·라이선스·품질 불일치 | `I1`–`I3` 저장소 범위 acceptance |

---

## 20. 최종 완료 정의

프로젝트는 다음 조건을 모두 충족할 때 완료로 판정한다.

1. 원시 데이터부터 예측과 편집 결과까지 전체 계보가 재현된다.
2. 시계열과 서사를 양방향으로 조회하고 근거·반례를 확인할 수 있다.
3. 사전학습과 미세조정이 W&B artifact로 재현된다.
4. SAE 특징은 의미·안정성·기능 기준으로 등급화된다.
5. 3D 탐색과 기계적 해석 결과가 명확히 분리된다.
6. 개념 분할·병합이 versioned governance 절차를 따른다.
7. 이벤트 편집 전후 raw/event/token/SAE/output 변화가 확인된다.
8. RL은 월드모델 readiness와 데이터 충분성을 통과한 경우에만 수행된다.
9. 물리·시간·ontology·누수 제약은 모든 단계에서 강제된다.
10. 최종 lock 이후 전체 재실행 결과가 acceptance를 통과한다.
11. **모든 신규 구현이 life2vec 저장소 안에 있고, 외부 저장소 코드 직접 재사용이 0건이다.**

---

## 21. 최종 의사결정

본 계획은 `CONDITIONAL APPROVE` 상태로 착수한다.

우선순위는 다음과 같이 고정한다.

```text
계보와 데이터 계약
→ 시계열–서사 Grounding
→ 시퀀스 및 사전학습
→ 미세조정과 표현 추적
→ SAE Pilot
→ bounded 이벤트 편집 (기존 CF1S 재사용)
→ 월드모델 readiness
→ 조건부 offline RL
→ 회로 분석
```

SAE, 월드모델 또는 RL의 적용 자체를 성공으로 간주하지 않는다. 각 모듈이 사전에 고정된 검증 기준을 통과하고, 원시 데이터 및 모델 출력과 재현 가능한 계보로 연결될 때만 정식 기능으로 승격한다. **모든 구현은 life2vec 저장소 안에서 이루어지며, agrichallenge 등 외부 저장소의 기존 코드는 어떤 단계에서도 참고 이상의 용도로 사용하지 않는다.**
