# Pre-LLM 의사결정 결과 (Results)

상태: 55건 배치 결과 보고  
실행 시각: `2026-07-17T00:18:59+00:00`  
산출 루트: `outputs/m1/decision_pre_llm/`  
경계: 전 케이스 `llm_boundary = STOPPED_BEFORE_LLM`

관련 문서: [과정](PRE_LLM_DECISION_PROCESS.md) · [구조](PRE_LLM_DECISION_STRUCTURE.md)

---

## 1. 요약

| 항목 | 값 |
|------|-----|
| 케이스 | **55 / 55 성공**, 오류 0 |
| 라벨 소스 | `labels_example35_problem20.csv` (example 35 + problem 20) |
| 사전학습 | Stage A event embedding 캐시 (`v2_finetune_v02/event_embeddings`) |
| 진단 | `cv_repeated_20260715_161945` 3-fold ensemble |
| CF M1 | attribution → locus → Path A/B B0 → dual gate |
| LLM | **미실행** (전 건 `llm_narrative=false`) |

라우팅 분포: **KNOWN 24 · REJECT 28 · NORMAL 3 · UNKNOWN_ABNORMAL 0**

---

## 2. 사용 모델·데이터

| 계층 | 경로 | 배치에서의 역할 |
|------|------|----------------|
| 사전학습 ckpt (참조) | `outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt` | 본 배치에서 인코더 재로드 없음 |
| Stage A 캐시 | `outputs/online2/v2_finetune_v02/event_embeddings/` | 55건 전부 존재·사용 |
| 진단 | `…/v03/runs/cv_repeated_20260715_161945/r0_fold{0,1,2}_best.pt` | p_abnormal·fine·routing |
| CF 패키지 | `src/online2/v2/finetune_v03/counterfactual/` | 개입시점·개입내용 후보 |
| 설정 | `conf/m1/cf_m1_decision_pre_llm_55.yaml` | |
| 실행기 | `scripts/online2_v2/v03/run_pre_llm_decision_batch_v03.py` | |

---

## 3. 라우팅 결과

### 3.1 전체

| route | n | 평균 p_abnormal | 범위 |
|-------|---:|----------------:|------|
| KNOWN | 24 | 0.934 | 0.769–0.990 |
| REJECT | 28 | 0.899 | 0.464–0.991 |
| NORMAL | 3 | 0.288 | 0.209–0.357 |
| UNKNOWN_ABNORMAL | 0 | — | — |

### 3.2 set × route

| set | KNOWN | NORMAL | REJECT |
|-----|------:|-------:|-------:|
| example_set (35) | 20 | 3 | 12 |
| problem_set (20) | 4 | 0 | 16 |

### 3.3 route_reasons (건수, 중복 가능)

| reason | n |
|--------|--:|
| `passed_known_gates` | 24 |
| `low_fold_agreement` | 24 |
| `low_fine_confidence` | 15 |
| `p_abnormal_below_tau` | 3 |
| `binary_near_threshold` | 2 |

해석: REJECT의 주원인은 **fold 합의 부족**(3-fold에서 합의 임계 반영)과 **fine confidence 부족**이다. argmax 진단명이 GT와 같아도 게이트를 통과하지 못하면 REJECT로 남는다.

---

## 4. 진단 라벨 (argmax) 결과

### 4.1 example_set (GT 있음)

- **Top-1 argmax = GT: 35/35 (100%)**
- 그중 `route=KNOWN`: 20건 (라우팅까지 통과)
- `route=REJECT`이지만 라벨 일치는 12건 → **라벨은 맞지만 신뢰/합의 게이트 미통과**
- `route=NORMAL`: 3건 (GT도 `정상_운영`)

즉, 본 배치에서 “진단명 맞음”과 “KNOWN으로 확정”은 별개 지표다.

### 4.2 problem_set (GT 없음, 예측만)

| fine_pred_label | n |
|-----------------|--:|
| 기형과_생리장해 | 5 |
| 영양생장_부족 | 5 |
| 동해_난방실패 | 3 |
| 칼슘부족_잎끝마름 | 2 |
| 탄저병_발생_위험 | 2 |
| 정상_운영 | 1 |
| 영양생장_과다 | 1 |
| 뿌리_발육_부진 | 1 |

KNOWN으로 확정된 problem 케이스는 **4건**뿐이며, 나머지는 대부분 REJECT로 LLM·열린 진단 전 단계에 보류된다.

---

## 5. CF M1 개입시점·개입내용

전 케이스 공통 패턴(평균):

| 지표 | 값 |
|------|-----|
| Path A loci | 8.0 (전 건 8) |
| Path B loci | 8.0 (전 건 8) |
| Path A candidates | 16.0 |
| Path B candidates | ≈63.5 (케이스별 상이) |
| span 정렬률 | ≈0.984 |

- Path A: OBSERVED_STATE 목표 상태 후보 (adjacent bin 기반)
- Path B: B0만 (`NO_OP` / `truncate_*` / `clear_span`), **OPERATIONAL_CANDIDATE·관리 자동실행 없음**
- Dual gate (전 55건 동일):
  - `analysis_report = true`
  - `management_suggestion = false`
  - `llm_narrative = false`

---

## 6. 산출물 위치

| 산출 | 경로 |
|------|------|
| 코호트 요약 | `outputs/m1/decision_pre_llm/cohort_summary.csv` / `.json` |
| set×route | `outputs/m1/decision_pre_llm/route_by_set.csv` |
| 실행 로그 | `outputs/m1/decision_pre_llm/batch_run.log` |
| 케이스 패키지 | `outputs/m1/decision_pre_llm/cases/{case_id}/decision_package.json` (+ attribution/locus/candidates) |
| 예시 (KNOWN) | `cases/F385790_2025-03-20_2025-04-02/` — GT=탄저병_발생_위험, p_abn≈0.986 |

각 `decision_package.json`은 LLM 입력 직전 스냅샷이며, `stages_not_run`에 열린 진단·Gemma 보고서가 명시된다.

---

## 7. 한계·주의

1. **Stage A 재인코딩**: 배치 모드는 캐시 임베딩 섭동 경로를 쓰며, 사전학습 인코더 풀 리로드는 하지 않는다.
2. **3-fold vs 설계상 5-fold**: 합의 임계는 fold 수에 맞게 clamp되어 동작한다. REJECT가 많은 주된 요인으로 fold 합의가 잡힌다.
3. **KNOWN ≠ 관리 추천**: 분석 보고 게이트만 열리고, 관리 제안·LLM 서사는 의도적으로 닫혀 있다.
4. **problem_set**: GT가 없어 진단 정확도는 평가하지 않았고, 라우팅·후보 생성만 보고한다.
5. **후보 개수 상한**: Path A loci/후보가 케이스마다 동일한 상한에 가까운 값은 top-k·게이트 정책의 인한 것이며, “모든 가능 개입”을 의미하지 않는다.

---

## 8. 다음 단계 (본 문서 범위 밖)

1. REJECT/UNKNOWN 케이스에 한해 LLM 열린 진단·검색 파이프라인 연결  
2. KNOWN + analysis_report 케이스에 Gemma(또는 동등) 분석 보고서 생성  
3. Path B B1(용량·응답 모델) 이후 `management_suggestion` 게이트 재평가  

현재 55건 산출은 위 1–2 단계의 **입력 계약**까지 완료된 상태다.
