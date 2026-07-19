# Pre-LLM 의사결정 과정 (Process)

상태: 운영 설명서  
대상: online2 **55건** (`example_set` 35 + `problem_set` 20)  
경계: **LLM(열린 진단·Gemma 보고서) 호출 직전**까지

---

## 1. 목적

사전학습 Stage A 표현, 진단 finetune 모델, CF M1 의사결정 추천 모듈을 한 파이프라인으로 묶어,  
각 케이스에 대해 다음을 **결정론적으로** 산출한다.

1. 정상/비정상 및 Known·Unknown·Reject 라우팅  
2. 비정상 기여 이벤트·measurement group·actuator span (개입시점)  
3. Path A 목표 상태 후보 / Path B B0 조작 후보 (개입내용)  
4. Gate·적격성 (분석 보고서 가능 여부; 관리 자동실행은 차단)

LLM 서사·열린 진단 문장 생성은 **하지 않는다**.

---

## 2. 사용 모델 계층

```text
[사전학습] Event-grain Transformer
    → Stage A event_mean/max 캐시 (이미 산출된 parquet)
         ↓
[진단 학습] finetune_v03 5-task (본 run은 3-fold repeated CV best)
    → Binary abnormal + Fine diagnosis + routing
         ↓
[의사결정 추천] CF M1 counterfactual 패키지
    → attribution → locus → Path A/B B0 → Gate
         ↓
[여기까지] decision_package.json
         ↓
[미실행] P3 LLM / Gemma 보고서
```

| 계층 | Artifact | 역할 |
|------|----------|------|
| 사전학습 | `outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt` | 토큰→이벤트 표현 학습 (본 배치에서는 **캐시만** 사용) |
| Stage A 캐시 | `v2_finetune_v02/event_embeddings/*.parquet` | 케이스별 이벤트 임베딩 (55건 모두 존재) |
| 진단 | `v2_finetune_v03/runs/cv_repeated_20260715_161945/r0_fold{0,1,2}_best.pt` | 위험도·진단명·라우팅 |
| CF M1 | `src/online2/v2/finetune_v03/counterfactual/` | 개입시점·개입내용 후보 |

---

## 3. 케이스별 처리 순서

```text
1) Diagnosis ensemble
   - 3-fold 모델 각각 encode_events(fuse) → binary / fine
   - 평균 p_abnormal, fine softmax, energy, fold 합의
   - route_case → NORMAL | KNOWN | UNKNOWN_ABNORMAL | REJECT

2) Attribution (fuse-aware event IxG)
   - 동일 encode/fuse 경로
   - ENVIRONMENT 이벤트에 Path A probe feature 토큰 분배
   - MG → event 집계 + fold consensus

3) Temporal recovery
   - actuator ZOH span 생성
   - embedding ACTUATOR 이벤트와 zone+timestamp 정렬

4) Locus selection
   - Path A: OBSERVED_STATE + editable + fold 합의
   - Path B: whitelist actuator span + gap=0 + fold 합의

5) Content candidates
   - Path A: adjacent ABS bin → nearest_feasible_interior → Gate2/4
   - Path B: NO_OP / truncate_* / clear_span (B0, capacity unknown)

6) Dual gate (LLM 이전)
   - analysis_report: loci 또는 후보 존재 시 true
   - management_suggestion: false (B1/response 없음)
   - llm_narrative: false
```

---

## 4. 실행 방법

```bash
conda activate life2vec
export CUDA_VISIBLE_DEVICES=0

python scripts/online2_v2/v03/run_pre_llm_decision_batch_v03.py \
  --config conf/m1/cf_m1_decision_pre_llm_55.yaml
```

설정 요약:

- labels: `labels_example35_problem20.csv` (55행)  
- GPU memory fraction: **0.4**  
- 출력 루트: `outputs/m1/decision_pre_llm/`

단건:

```bash
python scripts/online2_v2/v03/run_pre_llm_decision_batch_v03.py \
  --config conf/m1/cf_m1_decision_pre_llm_55.yaml \
  --case-id F385790_2025-03-20_2025-04-02
```

---

## 5. LLM 경계 (명시)

| 수행함 | 수행하지 않음 |
|--------|----------------|
| Binary/Fine 진단 | Unknown 시 retrieval+LLM 문장 |
| Known/Unknown/Reject 라우팅 | Gemma/로컬 LLM 보고서 |
| CF M1 loci·후보·Gate | PLC/자동 제어 |
| EXPLANATORY / EXPERT_REVIEW | OPERATIONAL_CANDIDATE |

다음 단계(별도 작업)에서 `decision_package.json`을 LLM 프롬프트 근거 패키지로 넘기면 된다.

---

## 6. 관련 문서

- 구조: [`PRE_LLM_DECISION_STRUCTURE.md`](./PRE_LLM_DECISION_STRUCTURE.md)  
- 결과: [`PRE_LLM_DECISION_RESULTS.md`](./PRE_LLM_DECISION_RESULTS.md)  
- CF M1 정본: `CF_M1_ATTRIBUTION_LOCUS_ARCHITECTURE_REVIEW.md`  
- 진단 상위: `DIAGNOSIS_FINETUNE_V03.md` §P3/P4
