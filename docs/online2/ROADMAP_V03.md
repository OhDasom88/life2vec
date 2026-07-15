# v0.3 진행 순서

상위: [`DIAGNOSIS_FINETUNE_V03.md`](./DIAGNOSIS_FINETUNE_V03.md) · CF: [`COUNTERFACTUAL_INFERENCE_V03.md`](./COUNTERFACTUAL_INFERENCE_V03.md)  
CV 결과: [`DIAGNOSIS_FINETUNE_V03_CV_REPORT.md`](./DIAGNOSIS_FINETUNE_V03_CV_REPORT.md) · **개선 정본:** [`DIAGNOSIS_FINETUNE_V03_PLAN.md`](./DIAGNOSIS_FINETUNE_V03_PLAN.md) · 초기 분석·튜닝: [`DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md`](./DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md)

원칙: **v0.1/v0.2 in-place 수정 금지** → `finetune_v03/` · `scripts/.../v03/` · `outputs/.../v2_finetune_v03/`.

---

## 한눈에

```text
0  스캐폴드·캐시 정책
→ P0  설명/근거 게이트 (fuse·token·raw·보고서 형식)
→ P1  5× multi-task (binary + 10종 + projection) + OOF
→ P2  Known/Unknown calibration
→ P3  열린 진단 (retrieval + 제한 LLM)     ※ P0 게이트
→ P4a 진단 counterfactual (Path A)         ※ P0 + StageA 재인코딩
→ P4b 관리 개입 counterfactual (Path B)    ※ P4a 통과 후
```

병행 가능: **P0 일부**와 **P1 모델 골격**은 동시에 가능.  
절대 병행 금지: P3 without P0 · P4b without P4a · MLM을 물리 시뮬로 취급.

---

## Phase 0 — 레포 골격 (수일)

1. `src/online2/v2/finetune_v03/` 패키지 뼈대 (config, version string `0.3.0`)
2. `scripts/online2_v2/v03/` 진입점 stub
3. Stage A cache **read-only** 정책 (v02/v01 경로 config)
4. 산출 루트 `outputs/online2/v2_finetune_v03/` 생성

**완료:** import만으로 v0.2 fold forward 재현 스모크.

---

## P0 — 설명·근거 복구 (게이트)

순서:

1. **Saliency/risk를 `encode_events`·DINO fuse와 동일 경로로**
2. **Token attribution** 켜기 → `layer_analysis/` 산출
3. **event → measurement group → cell → raw** join을 evidence에 부착
4. Gemma 프롬프트: `[상태][원인][관리]` + 수치 인용 필수 / saliency 남용 금지
5. Actuator ZERO/POSITIVE 한계 문서화 (또는 bin/ordinal 개선 착수)

**완료 게이트:** 사례 1건에서 `인용 토큰 > 0` + raw ℃/%/dS/m가 보고서 또는 evidence에 존재.  
→ 미통과 시 P3·관리제안 양산 금지.

---

## P1 — Multi-task 학습 (핵심)

순서:

1. 모델: `H_binary` + `H_fine`(10) + `H_proj` (coarse OFF)
2. Loss: weighted BCE + CE + (약) supcon/proto/consistency
3. 학습: heads-first → (필요 시) PAD만 light unfreeze
4. **5-fold farm CV**, 주 지표 = **OOF 합산**
5. OOF artifact: probs, binary p, z, fold agree

**완료 게이트:**

- fine: example OOF가 v0.2 재현 수준에서 크게 붕괴하지 않음  
- binary: balanced / AUPRC / 정상 FPR 리포트  
- projection: cross-farm P@k 베이스라인  

Negative transfer 심하면 λ·lr 조정 후, 최후 수단으로 2계열(≤10 ckpt).

---

## P2 — Open-set

1. normal / class prototypes (OOF embedding)
2. energy + fold disagreement
3. Leave-one-diagnosis-out으로 **τ만** calibration (주학습 X)
4. 라우팅: normal → known → `UNKNOWN_ABNORMAL`

**완료 게이트:** unknown 강제매핑 비율·LODO AUROC 기록, τ freeze.

---

## P3 — 열린 진단

전제: **P0 통과**.

1. sequence + DINO retrieval
2. 근거 패키지 (events · tokens · raw · 유사/차이 · 반증)
3. 제한 LLM + **판단 불가**
4. (선택) rulebase 후보 사전(염분·응애·야간고온…)

**완료 게이트:** 미공개 20에서 “10종 밖 진단 서술 가능 + 근거 수치” 샘플 세트.

---

## P4a — 진단 Counterfactual (Path A)

전제: P0 (특히 fuse·token·raw·재인코딩). P1 binary가 있으면 risk를 binary로 승격(없으면 v0.2 margin proxy).

1. `counterfactual/` 패키지: risk · selector · attr · inference_masker · constrained_mlm · stage_a_reencoder · search  
2. Smoke: 비정상 1건 · fold1 · event3 · beam4 · edit≤2  
3. MLM 복원 테스트 → 정상 4건 과편집 검사 → 예시 확장  
4. 5-fold + (이후) holdout critic · 최소 편집

**완료 게이트:** “후보 bundle 적용 시 risk가 재현 가능하게 감소” 로그 + schema 유효.

## P4b — 관리 개입 (Path B)

전제: **P4a 게이트 통과**.

1. Actuator whitelist · 시간 구간 병합  
2. 영향 ENV/ROOT mask 규칙 · **개입 이후 IMAGE 제외**  
3. Beam + holdout critic + Importance  
4. 직접개입 / 기대상태 / 검증지표 분리 출력  

MLM = constrained infill only (물리 WM 아님).

---

## 권장 캘린더 감각 (참고)

| 구간 | 내용 |
|------|------|
| W1 | Phase0 + P0 fuse/token/raw |
| W2–W3 | P1 학습·OOF |
| W4 | P2 + P0 보고서 형식 마감 |
| W5 | P3 샘플 |
| W6+ | P4a smoke → 확장 → (선택) P4b |

실제 일정은 GPU·스윕과 P0 난이도에 따라 조정.

---

## 지금 바로 할 일 (다음 한 스텝)

1. `finetune_v03` 모델 클래스에 binary/fine/proj head 골격  
2. 동시에 saliency를 `encode_events` 경로로 맞추는 유틸 (P0-1)  
3. 1 case evidence에 raw join prototype  

문서만으로 멈춘 상태의 구현 착수점이면 **이 세 가지부터**.
