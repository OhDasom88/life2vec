# Counterfactual 추론 의사결정 구조 검토 (v0.3 적용)

상태: **설계 검토 반영**  
작성일: 2026-07-15  
상위 계획: [`DIAGNOSIS_FINETUNE_V03.md`](./DIAGNOSIS_FINETUNE_V03.md)  
파이프라인 사실: [`pipeline_trace_v02/`](./pipeline_trace_v02/)  
**토큰 편집 → 원시 수치·용량·지속시간:** [`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md) (갭 전용; 본 문서는 모델 공간 편집까지)

본 문서는 “v0.2에서 정상 맥락 counterfactual 실험 준비” 권고안을 **현재 소스·산출물·v0.3 우선순위**에 대조한 검토이다.

---

## 1. 종합 판정

| 항목 | 판정 |
|------|------|
| **A/B 단계 분리** (진단 CF → 관리 개입 CF) | **채택.** v0.3 P4를 A-first로 고정 |
| 선행 5개 (margin risk, token attr, constrained MLM, Stage A 재인코딩, beam) | **채택.** smoke에 충분·필요 |
| DINO fuse를 saliency에 맞추기 | **필수 버그픽스** (진단과 saliency 입력 불일치, 코드 확인됨) |
| `R=1-P(정상)` / margin / 5-fold | **채택.** v0.3 binary head 전에도 v0.2로 실험 가능 |
| Deterministic inference masker ≠ `GroupedMLMMasker` | **채택.** 학습용 80/10/10은 CF에 부적합 |
| Bundle 후보 = 관측 등장 묶음(방법 1) | **채택.** registry 생성보다 안전 |
| 편집 후 Stage A 전 window 재인코딩 | **채택(정확 우선).** 이후 부분 갱신 최적화 |
| Path A (센서 토큰 직접 편집) | **실험 가능.** 단 MLM은 **언어적 개연성**이지 물리 시뮬이 아님 |
| Path B (actuator 편집 + 미래 ENV/ROOT MLM 복원) | **A 성공 후.** 현행 MLM을 future dynamics로 과대해석하면 안 됨 |
| v0.2만으로 관리제안 완성 | **비권장.** P0(raw·token)·재인코딩 경로 없이 보고서/개입안 품질 불가 |

**한 줄:** 권고안의 준비 체크리스트와 A→B 순서는 v0.3 P4에 그대로 넣을 만하다.  
다만 Path A의 성공은 “정상 확률 상승”이지 “실제 농장이 정상화된다”는 증거가 아니며, Path B는 **별도 제약·도메인 그래프** 없이는 reward hacking에 취약하다.

---

## 2. 코드 대조 — 맞음 / 구멍

### 2.1 확인된 사실 (권고안과 일치)

1. **일상 추론은 cache only**  
   `event_mean/max`만 바꾸지 않으면 진단이 안 바뀜 → Stage A 재인코딩 경로 필수.

2. **MLM은 finetune ckpt에 없음**  
   사전학습 `outputs/online2/v2_runs/full_event_grain_earlystop/best.ckpt`의 `model` dict에  
   `mlm_decoder.*` + `transformer.*` 존재. fold `*_best.pt`만으로는 후보 토큰 생성 불가.

3. **Saliency ↔ 진단 입력 불일치**  
   `finetune_v02/saliency.event_input_x_gradient`는 `meta→pad→head`만 호출.  
   `EventPoolingDiagnosisModelV02.encode_events`의 `fuse_image_into_events`를 **우회**.  
   → CF/설명 전에 fuse를 saliency·risk 경로에 넣는 것은 **최우선 수정**.

4. **Token saliency deferred**  
   v0.2 eval에 `layer_analysis/` 없음 → 인용 토큰 0.  
   `v01_layer_saliency.token_ixg_for_event` 패턴으로 연결 가능.

5. **raw는 tokenized parquet에 없음**  
   `cell_occurrences.raw_value` 재연결 필요 (P0).

6. **`construct_target_window` / `encode_batch_pool` / `window_to_tensors`**  
   `cache_stage_a_event_embeddings.py`에 이미 있음 → `stage_a_reencoder`는 새 발명보다 래핑.

7. **Actuator = ZERO/POSITIVE**  
   Path B의 “3시간→1시간”은 timestamp 구간 병합으로는 표현 가능하나, **세기/중간 상태는 약함**.  
   → 지속시간·용량·℃ 등 **실행 단위로의 변환은 본 CF 문서 범위 밖**이며 [`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md)에서 계층으로 정의한다.  
   → Path B 탐색의 1급 객체는 단일 이벤트 `token_edits`가 아니라 **segment `edit_scope`** (truncate_end / clear_span 등). `token_edits`는 scope 적용의 파생 결과.

### 2.2 권고안에서 보강할 점

| 권고 | 보강 |
|------|------|
| Encoder+MLM 로드 | Lightning이 아닌 custom ckpt: `torch.load(...)['model']` + `hparams` 키로 복원. `load_frozen_encoder` (Stage A 스크립트) 패턴 재사용 |
| IMAGE fusion saliency | `encode_events`를 saliency/risk 단일 진입점으로 사용 (meta 직행 금지) |
| margin risk | v0.3에 `H_binary`가 생기면 **binary logit을 주 risk**로 이전; v0.2 실험은 10-class proxy 허용 |
| Cross-fitting critic | v0.3 문서와 동일하게 **탐색 fold ≠ holdout fold**를 Path A smoke 이후부터 권장 (1-fold smoke만 예외) |
| 정상 4건 calibration | OOF/leave-in으로 τ 잡을 때 **farm 암기** 가능 → threshold는 보수적으로, 과편집률을 별도 지표로 |
| 미래 IMAGE 제외 | Path B에서 **필수**. DINO를 개입 이후에도 쓰면 과거 병징이 남음 |
| MLM top-5 필터 | family/schema 필터 + **관측 bundle 목록** 둘 다 필요. vocab top-k만 쓰면 문법 붕괴 |

### 2.3 과대 기대를 줄일 문장 (문서·실험 로그에 고정)

```text
Path A: “모델이 정상으로 보도록 토큰을 어떻게 바꿀 수 있는가?” (classifier-oriented)
Path B: “실행 가능한 actuator 변경 + 후속 관측 가설” (intervention-oriented; MLM≠물리엔진)
```

---

## 3. v0.3에 넣는 위치

```text
P0  설명 복구 + saliency fuse + token attr + raw 연결     ← CF 게이트
P1  multi-task (binary 등)                              ← 이후 risk를 binary로 교체
P2  open-set
P3  열린 진단
P4a Path A: 진단 counterfactual (본 문서 최소~두 번째 버전)
P4b Path B: 관리 개입 (A 통과 + actuator whitelist + 미래 mask)
```

**v0.2로 smoke를 먼저 돌리는 것**과 **v0.3 본체 학습**은 병행 가능하나:

- CF 코드는 `src/online2/v2/counterfactual/` (또는 `finetune_v03/counterfactual/`)에 **신규 isolation**
- v0.2 `saliency.py`를 in-place로 크게 바꾸지 말고, v0.3에서 fuse-aware risk/saliency를 구현한 뒤 v0.2는 읽기용

권장 패키지 트리(권고안 §16 채택):

```text
src/online2/v2/counterfactual/   # 또는 finetune_v03/counterfactual/
  risk.py
  event_selector.py
  token_attribution.py
  inference_masker.py
  constrained_mlm.py
  stage_a_reencoder.py
  search.py
  validator.py
  diff.py
  intervention.py              # Path B만
  action_grounding.py          # 토큰 diff → raw/용량/지속시간 (CF_ACTION_GROUNDING_V03)
```

스크립트: `scripts/online2_v2/v03/run_counterfactual_smoke_v03.py`

---

## 4. 선행 작업 5개 — v0.3 체크리스트화

| # | 작업 | 상태 | 비고 |
|---|------|------|------|
| 1 | 비정상–정상 **margin risk** + (fuse 경유) saliency | 미구현 | objective = logsumexp(abn)−z_normal |
| 2 | 선택 event **token / measurement-group attribution** + raw | 미구현 | `token_ixg_for_event` 포크 |
| 3 | **Deterministic** group mask + **constrained** MLM bundles | 미구현 | 학습 `GroupedMLMMasker` 금지 |
| 4 | 편집 후 **Stage A window 재인코딩** → cache 패치 | 미구현 | 초기: 영향 window 전부 |
| 5 | 소형 **beam/greedy** + 5-fold risk 비교 | 미구현 | smoke는 fold 1 허용 |

이 다섯이 끝나기 전 Path B·관리제안 HTML 출력은 하지 않는다.

---

## 5. Path A 최소 smoke (권고 §17 채택)

- 예시 비정상 **1건**, IMAGE 없음 또는 fuse 수정 완료건  
- fold **1** → 해석은 나중에 5-fold  
- top positive event **3**, group **1**/event, bundle **3**, beam **4**, max edits **2**  
- 출력: 원본 risk, 선택 근거, 후보 3, ΔR, token/raw diff  

성공 기준(기술적):

1. 원본 진단 재현 (fuse 포함)  
2. mask→MLM이 **동일 group 복원** top-k에서 비랜덤 수준  
3. 최소 1개 후보에서 ensemble(또는 단 fold) risk **유의 감소**  
4. 편집이 feature family·스키마를 깨지 않음  

성공 기준이 **아닌** 것: “실제 권장 영농법” 또는 “물리적으로 유일한 처방”.

---

## 6. Path B로 넘어가기 전 게이트

Path A에서:

- 예시 비정상 여러 건에서 ΔR·편집 패턴이 진단과 **대략 정렬**  
- 정상 4건 **과편집률** 낮음  
- MLM bundle exact-match 복원이 feature별로 보고됨  

그다음:

- actuator whitelist  
- 영향 변수 mask 규칙(초안 도메인 그래프)  
- 개입 이후 IMAGE 제거  
- cross-fold holdout critic  
- backward elimination / Importance  

---

## 7. 위험도 정의 — v0.2 실험 vs v0.3 본선

### v0.2로 즉시 실험 (proxy)

\[
R_{\mathrm{margin}}=\mathrm{logsumexp}(z_{\setminus\mathrm{normal}})-z_{\mathrm{normal}}
\]

또는 \(R=1-P(\text{정상_운영})\).

한계: “영양생장 부족 → 탄저”로만 옮기면 margin이 안 줄 수 있음 /  
정상 softmax 질량이 작아 포화·불안정.

### v0.3 P1 이후 (본선)

\[
R = \sigma(H_{\mathrm{binary}}(h))
\]

(+ 선택) prototype 거리. CF objective와 학습 objective가 일치.

---

## 8. 평가 단계 (권고 §15) — 수용 + 한 줄 가드

| 단계 | 수용 | 가드 |
|------|------|------|
| 0 원본 재현 | ✅ | fuse ON/OFF 둘 다 로그 |
| 1 MLM 복원 | ✅ | actuator literal / ABS bin 분리 리포트 |
| 2 정상 보존 | ✅ | 원본 유지 후보가 beam 상위 |
| 3 비정상 1건 | ✅ | Path A only |
| 4 예시 35 | ✅ | 정상 과편집률을 주 지표에 포함 |
| 5 문제 20 | ✅ | **정답률이 아니라** EC/고온저습/actuator 패턴 정합 |

---

## 9. 최종 권고 (의사결정)

1. **이 의사결정 구조를 v0.3 P4a의 공식 스펙으로 채택**한다.  
2. **A 완성 전 B·관리제안 UI 금지.**  
3. **saliency/risk의 DINO fuse 정렬 + token/raw 연결을 P0/CF 공통 게이트**로 둔다.  
4. MLM은 **constrained infill 엔진**으로만 문서화한다 (물리 WM 아님).  
5. v0.3 multi-task binary가 생기면 risk를 binary로 승격하고, v0.2 10-class proxy 실험 로그는 baseline으로만 남긴다.  
6. 구현 isolation: `counterfactual/` 신규 패키지 + `v03` 스크립트; v0.2 eval 엔트리포인트는 유지.  
7. 운영/보고서용 개입 문장은 토큰 diff만으로 끝내지 말고 [`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md)의 `grounded_actions`(또는 명시적 `ungroundable`)를 동봉한다.

---

## 10. 산출물 스키마 (최소)

권고 §12 JSON/diff 스키마를 그대로 쓰되, 메타에 고정:

```json
{
  "cf_path": "A_diagnosis",
  "risk_definition": "margin_v02_proxy | binary_v03",
  "image_fuse_in_saliency": true,
  "mlm_role": "constrained_infill_not_dynamics",
  "holdout_fold": null
}
```
