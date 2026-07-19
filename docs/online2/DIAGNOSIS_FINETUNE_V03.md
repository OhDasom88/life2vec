# Online2 진단 Finetune v0.3 구조안

상태: **설계 문서 (implementation pending)**  
버전: `0.3.0`  
작성일: 2026-07-15  

**진행 순서:** [`ROADMAP_V03.md`](./ROADMAP_V03.md)  
버전 맵: [`FINETUNE_VERSIONS.md`](./FINETUNE_VERSIONS.md)  
파이프라인 추적(v0.2 기준): [`pipeline_trace_v02/`](./pipeline_trace_v02/)  
현실성 전제: 라벨 n=35, Stage A cache + PAD 경로, 닫힌 10-class의 일반화 한계(미공개 20 약 35% rulebase 일치).

> **Isolation:** v0.1·v0.2 코드·산출물을 in-place로 “업그레이드”하지 않는다.  
> v0.3는 `src/online2/v2/finetune_v03/`, `scripts/online2_v2/v03/`, `outputs/online2/v2_finetune_v03/` 에만 쓴다.  
> v0.1/v0.2 Stage A parquet·체크포인트는 **읽기 전용** 재사용 가능.

---

## 1. 목표

온라인2는 예시 10개 진단명 문자열 일치가 아니라 **상태·원인·관리 제안의 타당성**을 본다.  
예시에 없는 진단도 감점 대상이 아니며, 미공개에는 염분장해·응애·야간고온 등이 실제로 등장한다.

따라서 v0.3의 최종 추론은 다음 순서를 따른다.

```text
1. 정상인가, 비정상인가?
2. 비정상이라면 기존 10종으로 설명 가능한가? (Known)
3. 아니면 Unknown abnormal → 검색·근거 기반 열린 진단 (판단 불가 허용)
4. (이후) 정상으로 돌리려면 언제 무엇을 바꿀 것인가? — P4, 축소 범위
```

v0.2 대비 핵심 변경:

| 항목 | v0.2 | v0.3 |
|------|------|------|
| 모델 수 | 5-fold × (분류+semantic) | **5-fold × 1 multi-task** (기본) |
| 출력 | 닫힌 10-class 강제 | binary → Known/Unknown → 10종 또는 열림 |
| Embedding | state/cause align | + **metric/prototype** (retrieval·거리) |
| 보고서 | saliency 중심 | **P0:** raw·token attribution 복구 후 근거 패키지 |
| Counterfactual | 없음 | **P4 선택 트랙** (현행 MLM을 WM으로 오용하지 않음) |

---

## 2. 데이터·표본 수 (필수 전제)

```text
이벤트·window 수 ≫ 0  ≠  독립 supervised 라벨 수
독립 진단 라벨 = example 35 (farm당 1 case)
정상_운영 = 4, 비정상 = 31
```

금지:

- 같은 case/farm의 window가 train과 val에 동시 포함
- 모든 구간에 사례 진단명 강제 부여 후 “표본 증가” 주장
- 이미지·farm ID로 사례 암기하는 split

검증: **farm/case Group K-fold** (기존 `make_stratified_farm_folds` 계열 유지).  
fold별 class metric은 참고만, **주 지표는 35건 OOF 합산**.

---

## 3. 권장 모델 구성 (기본안)

```text
Frozen Stage A event cache (mean‖max, optional DINO fuse)
  → EventMetadataEncoder + PAD → h_case
       ├─ H_binary          → p_abnormal          (weighted BCE)
       ├─ H_fine            → 10-class logits     (CE, v0.2 유지)
       ├─ H_coarse          → optional            (전문가 taxonomy 합의 전 OFF)
       └─ H_projection      → z (L2)              (supcon / prototype)
```

체크포인트:

[
5 \text{ folds} \times 1 \text{ multi-task model} = 5 \text{ models}
]

**비권장:** task마다 전체 모델 → \(5\times3=15\) models (표현 불일치·통합 추론 곤란).

**조건부 분리 (최대 10 models):** binary↔fine에 **실증된** negative transfer가 있을 때만

```text
계열 A: binary + normal prototype
계열 B: fine diagnosis + disease prototype
```

같은 pretrained/Stage A에서 시작하되 embedding 공간은 별도.

MLM loss는 downstream에 **상시 결합하지 않음**.  
필요 시 `continual pretrain → 이후 multi-task FT`로 단계 분리.

---

## 4. Task head 정의

### 4.1 Binary normality

- 라벨: `정상_운영→0`, 나머지→1  
- Loss: **weighted BCE** (첫 구현; focal은 n 작아 불안정 가능)  
- 불균형 대응: case-level weight, 정상 oversample(제한적), balanced acc / AUROC / **AUPRC**, calibration  
- 정상 4건 OOF이므로 fold별 정상 metric만으로 성공 판정 금지

### 4.2 Fine diagnosis (10종)

- v0.2 CE head 유지 → known 재현·retrieval 정렬·기존 성능 보존  
- binary 추가 후에도 **제거하지 않음**

### 4.3 Coarse (선택)

- 예: 정상 / 생육균형 / 근권·양분 / 병해 / 온도·환경  
- **전문가 합의 전 기본 OFF**

### 4.4 Projection / prototype

- \(z=\mathrm{normalize}(H_{\mathrm{proj}}(h))\)  
- 용도: same-diagnosis retrieval, 정상·진단 prototype 거리, Known/Unknown, CF 후 정상 접근  
- v0.2 `z_state`/`z_cause`와 병행 가능하나, v0.3 metric head는 **거리 학습**이 주 목적 (진단명 문자열 fusion 금지)

별도 학습 없이 쓰는 양:

- energy score ← fine logits  
- prototype ← OOF embedding 통계  
- Known/Unknown calibrator ← OOF + leave-one-diagnosis-out (아래)  
- counterfactual critic ← **동일 5-fold cross-fitting**

---

## 5. Loss (초기안)

\[
L=\lambda_b L_{\mathrm{binary}}+\lambda_f L_{\mathrm{fine}}+\lambda_c L_{\mathrm{coarse}}+\lambda_s L_{\mathrm{supcon}}+\lambda_p L_{\mathrm{proto}}+\lambda_r L_{\mathrm{consist}}
\]

| 항 | 초기 λ | 비고 |
|----|--------|------|
| binary | 1.0 | weighted BCE |
| fine | 1.0 | CE |
| coarse | 0 (또는 0.2–0.3) | OFF 기본 |
| supcon | 0.1 | |
| prototype | 0.1 | |
| consistency | 0.1 | event dropout 등 **약한** aug만 |

가중치는 OOF로 조정. GradNorm/PCGrad는 weight·LR 조절 실패 후에만.

---

## 6. 학습 순서

### Stage 1 — heads only

- Stage A cache / (선택) PAD 일부 고정  
- task heads 학습  
- 확인: binary 분리 가능? fine 유지? embedding 클러스터? 정상 prototype?

### Stage 2 — light unfreeze

- PAD attention 또는 LoRA/adapter만 낮은 lr  
- **전체 token encoder end-to-end 풀기 금지** (35건 과적합).  
  현재 스택은 cache 경로이므로 encoder LoRA는 **파이프라인 확장 시에만**.

### Stage 3 — negative transfer 대응

의심 신호: binary↑·fine macro-F1↓, gradient cosine 지속 음수, fold마다 head 방향 상충.  
순서: λ 조정 → head별 lr → GradNorm/PCGrad → **최후** 2계열 분리.

---

## 7. Known / Unknown / 열린 진단

### 7.1 결정 트리

```text
p_abnormal < τ_abn     → 정상_운영 (또는 정상 후보)
else:
  is_known(score)      → argmax fine (또는 prototype NN)
  else                 → UNKNOWN_ABNORMAL → 열린 진단 파이프
```

Known score 예 (합의 필요, OOF로 캘리브):

- max fine prob, min prototype distance, energy, fold agreement, embedding 분산, (선택) image·시계열 불일치

통과 실패 시 **max-prob class를 최종 진단으로 쓰지 않음**.

### 7.2 Open-set (라벨에 unknown 없음)

- **Leave-one-diagnosis-out**은 주 학습이 아니라 **threshold/calibration 검증**  
- class n=3 제외 시 변동 큼 → episode 결과의 중앙값·안정 구간만 사용

### 7.3 Unknown 시 LLM

자유 병명 생성 금지. 최소 패키지:

1. p_abnormal  
2. top events + **token + raw 값** (P0)  
3. 정상 대비 편차  
4. 유사 example 3–5 (cross-farm)  
5. 차이점·반증  
6. 이미지 + DINO 유사  
7. 후보 사전(염분·응애·야간고온 등 rule/온톨로지)  
8. **판단 불가** 옵션  

출력: 1차 후보 / 감별 / 유사·차이 / 현장 확인 / 확신도.

---

## 8. Embedding 검증 (UMAP 불충분)

필수:

- cross-farm retrieval / Precision@k  
- prototype accuracy  
- farm-ID probe (높으면 농가 암기 의심)  
- \(d(\text{같은진단,다른농가}) < d(\text{다른진단,같은농가})\)

---

## 9. 정상 구간·MIL

- **1차:** case-level만 (정상 4 / 비정상 31)  
- 비정상 case 앞구간 자동 정상 금지  
- 고정밀 구간 확장은 전문가 rule·생육·센서·actuator·이미지 조건 충족 시에만  
- TopKMean event MIL은 **이후 확장**; 초기은 case PAD 유지

---

## 10. 이미지

- 별도 late-fusion 이미지 모델 **필수 아님** (slot/DINO residual 경로 유지)  
- 열린 진단에서는 image 기여·DINO ablation \(\Delta R_{\mathrm{image}}\) 분리 보고  
- **기본 Stage A IMAGE 제외**이므로, ablation은 IMAGE/DINO가 켜진 cache·모델 기준

Retrieval: sequence embedding + DINO + (선택) 생육·환경 조건.

---

## 11. Counterfactual / 관리 제안 (P4)

상세 의사결정·코드 대조: [`COUNTERFACTUAL_INFERENCE_V03.md`](./COUNTERFACTUAL_INFERENCE_V03.md)  
토큰 최소 편집 → ℃/용량/지속시간: [`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md) (**현재 부재 계층; 신규 설계**)

### 11.1 두 경로 (반드시 분리)

| 경로 | 질문 | 편집 대상 | v0.3 단계 |
|------|------|-----------|-----------|
| **A. 진단 CF** | 무엇을 바꾸면 모델이 정상으로 보는가? | ENV/ROOT/ACT 등 (목표 상태) | **P4a 우선** |
| **B. 관리 개입 CF** | 어떤 actuator 변경 + 후속 복원으로 위험이 주는가? | ACTUATOR만 직접 편집, 이후 ENV/ROOT는 mask→infill | P4b (A 통과 후) |

Path A 성공 = “classifier-oriented 정상화”이지 물리 보장 아님.  
Path B에서 현행 MLM은 **물리 future model이 아님** (constrained infill만).

### 11.2 필수 선행 (CF 게이트, P0과 공유)

1. Risk/saliency objective = 비정상–정상 **margin** (v0.2 proxy) → 이후 **binary head**  
2. Saliency·risk가 **`fuse_image_into_events` / `encode_events`와 동일 입력**  
3. Token·measurement-group attribution + **raw value** 재연결  
4. Deterministic inference masker + constrained MLM bundles (학습 `GroupedMLMMasker` 금지)  
5. 편집 후 **Stage A window 재인코딩** (cache만 고치면 결과 불변)

사전학습 `best.ckpt`에 `mlm_decoder` 존재; fold finetune ckpt만으로는 후보 생성 불가.

### 11.3 Risk critic · 탐색

- v0.2 실험: \(R_{\mathrm{margin}}=\mathrm{logsumexp}(z_{\setminus n})-z_n\) 또는 \(1-P(\text{정상})\)  
- v0.3 P1+ : \(R=P(\mathrm{abnormal})\) (+ prototype)  
- Cross-fitting holdout critic (탐색≠검증 fold)  
- Smoke: greedy / beam 소형 → 이후 beam·최소 편집(Importance)  
- Path B: actuator whitelist, 영향 변수 그래프, **개입 이후 IMAGE 제거**

보상: \(\Delta R - \lambda(\mathrm{edit}+\mathrm{OOD}+\mathrm{uncertainty}+\mathrm{safety})\).  
초기: RL보다 constrained beam.

### 11.4 금지

- 현행 양방향 랜덤/그룹 MLM을 “환경 시뮬레이터”로 취급  
- P0·재인코딩 없이 관리제안 보고서 양산  
- A 미검증 상태에서 B 시간구간 개입안을 제품 스펙으로 고지

Actuator ZERO/POSITIVE 한계 → 구간 병합은 가능, 세기 표현은 P0 토큰 개선 전제.

---

## 12. 구현 우선순위

### P0 — 설명·근거 복구 (게이트)

- (가능하면) saliency 경로에 DINO fusion 일관 적용  
- **token attribution** (`layer_analysis` 생성·보고서 연결)  
- evidence에 **raw 값** 연결 (cell_occurrences / registry)  
- 근거 없는 LLM 문장 차단, `[상태][원인][관리]` 형식  
- actuator literal 한계 문서화 또는 토큰 세분화 착수  

**P0 미완이면 P3 열린 LLM 진단 금지.**

### P1 — Multi-task downstream

- `finetune_v03` 모델: binary + fine + projection  
- 5-fold OOF, weighted BCE, 약한 consistency  
- 평가: balanced/AUROC/AUPRC (binary), macro-F1 (fine), retrieval@k  

### P2 — Open-set calibration

- prototypes, energy, fold disagreement  
- leave-one-diagnosis-out으로 τ 조정  
- UNKNOWN_ABNORMAL 라우팅  

### P3 — 열린 진단

- 유사 사례·DINO retrieval, 근거 패키지, 제한 LLM, 판단 불가  

### P4 — 의사결정 (선택)

- **P4a:** 진단 counterfactual ([`COUNTERFACTUAL_INFERENCE_V03.md`](./COUNTERFACTUAL_INFERENCE_V03.md)) + Path A **action grounding**  
- **P4b:** actuator-only 관리 개입 + 미래 mask/infill (A 통과 후) + **duration/capacity grounding**  
- holdout critic · 최소 편집 · (선택) 별도 future-span 복원 모듈  
- 접지 계약: [`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md)  

---

## 13. 평가표 (요약)

| 영역 | 지표 |
|------|------|
| Binary | balanced acc, AUROC, AUPRC, sens/spec, Brier, ECE, 정상 FPR |
| Fine | macro-F1, balanced acc, top-2, OOF confusion |
| Embedding | cross-farm P@k, prototype acc, farm-ID probe |
| Open-set | LODO AUROC, FPR@95TPR, unknown 강제매핑 비율 |
| CF (P4) | ΔR, holdout ΔR, edit 수, OOD, fold 일치, 규칙 위반 |

비교 기준(제출/내부): 전문가 최종 rulebase (`submission_online2_v2` 계열)와의 일치율 — **닫힌 10-class 강제 매핑 비율 감소**가 v0.3 성공 신호 중 하나.

---

## 14. 산출물·코드 배치 (예정)

```text
docs/online2/DIAGNOSIS_FINETUNE_V03.md          # 본 문서
scripts/online2_v2/v03/                         # 진입점 (scaffold)
src/online2/v2/finetune_v03/                    # model/dataset/loss/open_set
outputs/online2/v2_finetune_v03/
  event_embeddings/     # symlink or copy policy from v02 (read-only OK)
  runs/cv_*/
  evaluation/
  prototypes/
  reports/
```

재사용:

- Stage A: `v2_finetune` / `v2_finetune_v02` caches (정책 명문화)  
- PAD 블록: v0.1 `event_pooling_finetune.py` import only  
- 라벨: `labels_example_score90.csv` + `label_map.json`  
- Eval glue: v02 pipeline 패턴을 v03로 포크  

---

## 15. 최종 권고 (문서화 요약)

> **5개 fold별 공유 PAD multi-task 모델**에 binary · 10-class · projection head를 붙인다.  
> Open-set은 추가 전체 모델 없이 거리·energy·합의로 한다.  
> 열린 진단은 normal→known/unknown→retrieval/근거 LLM.  
> Counterfactual은 동일 5-fold cross-fitting critic으로 시작하되, **현행 MLM을 future WM으로 쓰지 않는다.**  
> **P0→P1→P2**를 완료 게이트로 두고, P3·P4는 그 위에 올린다.

negative transfer가 반복 확인될 때만 binary/fine을 2계열(최대 10 ckpt)로 분리한다.
