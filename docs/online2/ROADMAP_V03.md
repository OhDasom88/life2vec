# v0.3 진행 순서

상위: [`DIAGNOSIS_FINETUNE_V03.md`](./DIAGNOSIS_FINETUNE_V03.md) · CF: [`COUNTERFACTUAL_INFERENCE_V03.md`](./COUNTERFACTUAL_INFERENCE_V03.md) · Grounding: [`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md)  
CV 결과: [`DIAGNOSIS_FINETUNE_V03_CV_REPORT.md`](./DIAGNOSIS_FINETUNE_V03_CV_REPORT.md) · **개선 정본:** [`DIAGNOSIS_FINETUNE_V03_PLAN.md`](./DIAGNOSIS_FINETUNE_V03_PLAN.md) · 초기 분석·튜닝: [`DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md`](./DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md)  
**후속 구축·게이트:** [`DIAGNOSIS_FINETUNE_V03_FOLLOWUP.md`](./DIAGNOSIS_FINETUNE_V03_FOLLOWUP.md) (Sweep1→final5→evidence→열린진단→CF)

원칙: **v0.1/v0.2 in-place 수정 금지** → `finetune_v03/` · `scripts/.../v03/` · `outputs/.../v2_finetune_v03/`.

---

## 변경이력

- **2026-07-25**: Open-set(P2)·열린 진단 LLM(P3)·관리개입 CF(P4b)의 **파이프라인 순서**를 사용자 확정 방향으로 갱신. 기존엔 P2(규칙기반 open-set) → P3(LLM) → P4a → P4b(RL 행위편집)가 서로 독립적인 순서였으나, 새 방향은 **RL을 이벤트 편집(행위 추천)에만 쓰고, "상황인식 → RL 행위추천" 파이프라인의 결과로부터 open-set을 추론한 뒤, 그 결과를 LLM 보고서 입력으로 넘긴다**. 상세: "진행 개괄 (2026-07-25 기준)" · "P2/P3/P4b 갱신 방향 (2026-07-25)" 절. **P0/P1/P4a의 개별 구현 항목·게이트는 폐기되지 않는다** — 바뀌는 건 P2/P3/P4b의 순서와 입력 소스뿐이다. 아래 "한눈에"와 P2/P3/P4b 각 절의 원문은 v1 설계 기록으로 그대로 남겨두고 갱신 방향 절에서 대체 관계를 명시한다.

---

## 진행 개괄 (2026-07-25 기준)

| Phase | 상태 | 비고 |
|---|---|---|
| Phase 0 (스캐폴드) | 완료 | |
| P0 (설명/근거 게이트) | 완료 | saliency·token attribution·raw join 배선됨 |
| P1 (멀티태스크 학습) | 완료, 지속 튜닝 중 | 핵심 목표는 **binary(정상/비정상) + fine(10종)**. `Projection`(z_proj)은 별도 진단 목표가 아니라 사례 간 거리·prototype용 **보조 임베딩 head**(open-set·유사사례 근거 검색에 사용). 현재 pos_weight×lr×weight_decay 하이퍼파라미터 sweep(sweep1c, 480 조합) 진행 중 |
| P2 (Open-set) | 구현됨(v1, 규칙기반) — **순서 갱신 예정** | prototype 거리 + energy + fold 불일치 + LODO τ calibration. 아래 갱신 방향에서 RL 행위추천 파이프라인 결과 기반으로 대체 예정, 규칙기반은 안전판으로 존속 |
| P3 (열린 진단, LLM) | 배선됨(saliency + 유사사례 검색 기반) — **입력 소스 갱신 예정** | 추후 RL 행위추천 파이프라인의 결과를 입력으로 받도록 변경 |
| P4a (진단 Counterfactual, CF1S) | 진행 중 | `counterfactual/cf1s/core_*.py` 다수 커밋 전 상태 |
| P4b (관리개입 Counterfactual, RL) | 미착수 | 이번 방향 전환의 핵심 — 상황인식→행위추천 RL 파이프라인이 여기서 만들어지고, 그 결과가 P2/P3 자리로 흘러감 |

---

## P2 / P3 / P4b 갱신 방향 (2026-07-25, 설계 단계 — 미구현)

기존 순서(P2 규칙기반 open-set → P3 LLM → P4a → P4b RL, 아래 "한눈에" 참조)를 다음으로 대체한다:

```text
상황인식 (진단 임베딩, P1 산출물)
→ RL 기반 행위 추천 파이프라인 (이벤트 편집 정책 — 기존 P4b 범위)
→ 그 파이프라인 결과로부터 Open-set(정상/known/unknown) 추론 (기존 P2를 대체)
→ (진단 + open-set + 추천 행위) 결과를 LLM에 입력 → 보고서 (기존 P3 자리, 입력 소스 변경)
```

- **RL의 용도는 이벤트 편집(행위 추천)으로 한정** — open-set 판정 자체를 위한 별도 RL 분류기를 새로 만드는 게 아니다.
- Open-set은 이제 **RL 행위-추천 파이프라인의 산출물을 1차 근거**로 판정하고, 기존 규칙기반(prototype 거리/energy/LODO τ)은 **불가피한 최소 수준의 검증(안전판)**으로만 남긴다.
- 이 방향대로면 **LLM 보고서(P3) 단계가 RL 행위추천 파이프라인(P4b) 완성을 전제**로 하게 된다 — 기존 "P3는 P0만 전제"였던 의존관계가 "P3는 P0 + (신규) RL 행위추천 파이프라인"으로 바뀐다.
- P0/P1/P4a의 개별 구현 항목·게이트는 그대로 유효하다. P2의 규칙기반 로직도 폐기되지 않고 안전판으로 재사용된다. **바뀌는 것은 순서와 무엇이 무엇의 입력이 되는지뿐이다.**
- 아직 코드 미착수 — 설계 단계 기록. 착수 시 이 절 아래에 진행상황을 추가할 것.

---

## 한눈에 (v1 순서 — 구현·게이트 참조용, 위 "갱신 방향"으로 대체 예정)

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

> **v1 설계 기록.** 2026-07-25 방향 갱신으로 이 규칙기반 판정은 RL 행위추천 파이프라인 결과 기반 판정의 **안전판**으로 역할이 바뀔 예정 — "P2 / P3 / P4b 갱신 방향" 절 참조. 아래 게이트·구현 항목은 그대로 유효.

1. normal / class prototypes (OOF embedding)
2. energy + fold disagreement
3. Leave-one-diagnosis-out으로 **τ만** calibration (주학습 X)
4. 라우팅: normal → known → `UNKNOWN_ABNORMAL`

**완료 게이트:** unknown 강제매핑 비율·LODO AUROC 기록, τ freeze.

---

## P3 — 열린 진단

> **v1 설계 기록.** 2026-07-25 방향 갱신으로 이 단계의 입력이 saliency+유사사례 검색에서 **RL 행위추천 파이프라인 결과**로 바뀔 예정, 전제도 "P0 통과"에서 "P0 + RL 행위추천 파이프라인 완성"으로 바뀔 예정 — "P2 / P3 / P4b 갱신 방향" 절 참조. 아래 게이트·구현 항목(근거 패키지 구성 등)은 그대로 유효.

전제: **P0 통과**.

1. sequence + DINO retrieval
2. 근거 패키지 (events · tokens · raw · 유사/차이 · 반증)
3. 제한 LLM + **판단 불가**
4. (선택) rulebase 후보 사전(염분·응애·야간고온…)

**완료 게이트:** 미공개 20에서 “10종 밖 진단 서술 가능 + 근거 수치” 샘플 세트.

---

## P4a — 진단 Counterfactual (Path A)

전제: P0 (특히 fuse·token·raw·재인코딩). P1 binary가 있으면 risk를 binary로 승격(없으면 v0.2 margin proxy).

1. `counterfactual/` 패키지: risk · selector · attr · inference_masker · constrained_mlm · stage_a_reencoder · search · **action_grounding**  
2. Smoke: 비정상 1건 · fold1 · event3 · beam4 · edit≤2  
3. MLM 복원 테스트 → 정상 4건 과편집 검사 → 예시 확장  
4. 5-fold + (이후) holdout critic · 최소 편집  
5. Path A 후보에 **bin→원시 구간 grounding** dry-run ([`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md))

**완료 게이트:** “후보 bundle 적용 시 risk가 재현 가능하게 감소” 로그 + schema 유효 (+ grounding 또는 `ungroundable` 필드).

## P4b — 관리 개입 (Path B)

> **핵심 변경 지점.** 2026-07-25 방향 갱신에서 "상황인식→RL 행위추천" 파이프라인이 바로 이 단계에서 만들어지고, 그 결과가 P2(open-set)·P3(LLM 보고서)의 입력으로 흘러 들어간다 — "P2 / P3 / P4b 갱신 방향" 절 참조. 전제(P4a 게이트 통과)는 유지.

전제: **P4a 게이트 통과**.

1. Actuator whitelist · **지속시간(run-length) API** · capacity B0(참고 raw)  
2. 영향 ENV/ROOT mask 규칙 · **개입 이후 IMAGE 제외**  
3. Beam + holdout critic + Importance  
4. 직접개입 / 기대상태 / 검증지표 분리 출력 (`grounded_actions` 동봉)  

MLM = constrained infill only (물리 WM 아님). 용량 정밀도는 토큰 세분화(B2) 전제 — 상세는 grounding 문서.

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
