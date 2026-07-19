# Finetune v0.3 후속 과정 구축 계획

| 항목 | 값 |
|------|-----|
| 버전 | `0.3.0` |
| 작성일 | 2026-07-16 |
| 상태 | **운용 계획 (Sweep1 진행 중)** |
| 정본 설계 | [`DIAGNOSIS_FINETUNE_V03_PLAN.md`](./DIAGNOSIS_FINETUNE_V03_PLAN.md) |
| 로드맵 | [`ROADMAP_V03.md`](./ROADMAP_V03.md) |
| CF 스펙 | [`COUNTERFACTUAL_INFERENCE_V03.md`](./COUNTERFACTUAL_INFERENCE_V03.md) |
| Isolation | `finetune_v03/` · `scripts/.../v03/` · `outputs/.../v2_finetune_v03/` only |

> 본 문서는 PLAN의 “무엇을 만들 것인가”를 바탕으로, **지금 돌아가는 Sweep부터 최종 추론·근거·CF까지를 어떤 순서로 구축·검증할지**를 운용 단위로 고정한다.  
> HP·라우팅 수식의 상세 스펙은 PLAN이 우선한다.

---

## 1. 현재 위치 (As-of)

```text
기준선 CV 완료 …… cv_20260715_104037_wandb (OOF fine≈0.78, bin AUROC≈0.91)
코드 반영 …… multi-task + pos_weight + consistency/proto + route_case + repeated-CV
▶ 지금 …… Sweep 1 (Binary/Loss)  Parallel agents
다음 …… Sweep 2 → Sweep 3 → cv_final5 → decision eval → P0 강화 → 열린진단/CF
```

| 항목 | 값 |
|------|-----|
| W&B Group | `finetune_v03_improvement_r1` |
| Sweep 1 | [Berry2Vec/8uhwp5w8](https://wandb.ai/dasom-oh/Berry2Vec/sweeps/8uhwp5w8) · `job_type=sweep_1_loss_binary` |
| Config | `conf/sweep/finetune_v03_s1_loss_binary.yaml` |
| Mode | `cv_repeated` (3-fold × 3 repeats, `monitor=val_loss`) |
| 그리드 규모 | ≈216 (lr × λ_binary × pos_weight × λ_consistency × thr-eval) |
| 산출 루트 | `outputs/online2/v2_finetune_v03/` |

의사결정 핵심 코드 (현재 학습 모델이 먹일 경로):

| 역할 | 경로 |
|------|------|
| 라우팅 | `src/online2/v2/finetune_v03/open_set.py` (`route_case`) |
| 임계·open-set cfg | `src/online2/v2/finetune_v03/config.py` |
| 학습·OOF·route 로그 | `scripts/online2_v2/v03/run_diagnosis_finetune_v03.py` |
| thr / 헤드 정합 지표 | `src/online2/v2/finetune_v03/metrics.py` |
| risk · IxG | `src/online2/v2/finetune_v03/risk_saliency.py` |
| raw evidence | `src/online2/v2/finetune_v03/evidence_raw.py` |

목표 추론 골격 (PLAN과 동일):

```text
Binary 정상·비정상
  ├─ NORMAL
  └─ 비정상
       → Fine + proto/energy/agree
            ├─ KNOWN
            ├─ UNKNOWN_ABNORMAL
            └─ REJECT / REVIEW
                 → 근거(event/token/raw)
                 → 검색·열린 진단
                 → Counterfactual 개입 (이후)
```

---

## 2. 구축 원칙

1. **선정은 OOF / repeated-CV 요약으로만** — fold별 `val_macro_f1`로 Sweep 승자를 고르지 않는다.
2. **train cfg와 infer cfg 분리** — open-set 캘리브레이션 누수 방지(`OpenSetTrainCfg` vs `OpenSetInferCfg`).
3. **이미지 adapter는 고정 ON** — Sweep에서 ablation하지 않는다 (PLAN).
4. **게이트 미확인 시 다음 phase 양산 금지** — 특히 UNKNOWN 보고서·CF Path B.
5. **v0.1/v0.2 in-place 수정 금지** — 신규는 `v03` 경로만.

---

## 3. Phase 맵 (후속만)

```text
F0  Sweep1 모니터·승자 동결          ◀ 진행 중
F1  Sweep2 Metric + Open-set 라우팅
F2  Sweep3 Architecture (query/attn)
F3  cv_final5 + 의사결정 평가 패키지
F4  Evidence/Routing 운영 게이트 (P0+)
F5  열린 진단 (retrieval + 제한 LLM)
F6  Counterfactual Path A → B
```

ROADMAP의 P1~P4와 대응:

| 본 문서 | ROADMAP | 요지 |
|---------|---------|------|
| F0–F2 | P1 강화 | Loss·metric·구조 Sweep |
| F3 | P1 완료 + P2 캘리브 | 5-fold freeze + route 지표 |
| F4 | P0 게이트 재확인 | 우승 ckpt로 evidence |
| F5 | P3 | UNKNOWN → 열린 진단 |
| F6 | P4a/b | CF 의사결정 |

---

## 4. Phase별 구축 계획

### F0 — Sweep1 완료 및 승자 동결 (지금)

**목적:** binary imbalance · fine–binary consistency · operating threshold 확정.

**할 일**

1. Sweep `8uhwp5w8` finished/failed 모니터링; 에이전트 고갈 시 재가동.
2. Run별 artifact 확인: `*_oof_*.csv`, `*_oof_binary_thresholds.csv`, route 컬럼.
3. 승자 선정표 작성 (동일 group 필터):
   - Fine macro-F1 (OOF / repeat 평균)
   - Binary AUROC / AUPRC / thr별 balanced acc · normal recall
   - Fine–Binary contradiction rate, Brier
   - `val_loss` early-stop과 모순 없는지 점검
4. Frozen config를 JSON으로 저장  
   `outputs/online2/v2_finetune_v03/selection/sweep1_winner.json`

**산출**

- `selection/sweep1_winner.json` (`lr`, `pos_weight`, `λ_*`, 권장 `tau_binary`)
- 짧은 노트: `docs/online2/` 또는 run README 한 절

**게이트:** 상위 설정이 baseline(`dqrr69z6`) 대비 binary 정합이 악화되지 않고, contradiction이 감소하거나 동등할 것.

**스크립트 (추가·권장):**

```text
scripts/online2_v2/v03/summarize_sweep_oof.py   # wandb/API 또는 로컬 runs 집계
```

---

### F1 — Sweep2 Metric learning & Open-set

**목적:** projection/proto로 Known 게이트를 만들고, reject/open-set on·off를 비교.

**선행:** F0 winner freeze → `conf/sweep/finetune_v03_s2_metric_openset.yaml`의 고정란에 주입 (또는 ungent 시 CLI default 갱신).

**할 일**

1. 동일 `WANDB_GROUP=finetune_v03_improvement_r1`, `job_type=sweep_2_metric_openset`로 sweep 생성·에이전트 기동.
2. 스윕 후보 (PLAN §12.2): `λ_supcon`, `λ_prototype`, `supcon_temperature`, `use_reject`, `use_open_set`, `tau_fine`, `prototype_percentile`, `min_fold_agreement`.
3. OOF에서 route 분포 집계: NORMAL / KNOWN / UNKNOWN / REJECT 비율.
4. LODO(leave-one-diagnosis-out) τ 캘리브 **초안** 스크립트 착수 (학습 재실행 없이 τ만).

**산출**

- `selection/sweep2_winner.json`
- `selection/routing_thresholds_oof.json` (τ_binary, τ_fine, energy/proto 분위수)
- (옵션) `outputs/.../analysis/route_confusion_oof.json`

**게이트**

- `use_open_set=true`일 때 강제 Known 비율이 유의미하게 감소
- reject 과다(전 표본 REJECT)면 τ 완화 후보를 winner에서 제외

**신규·보강 코드**

| 항목 | 경로 |
|------|------|
| LODO calibrator | `src/online2/v2/finetune_v03/lodo.py` + `scripts/.../v03/run_lodo_calibrate_v03.py` |
| Route eval | `src/online2/v2/finetune_v03/route_eval.py` (risk–coverage, AURC) |

---

### F2 — Sweep3 Architecture

**목적:** task-specific query · attention residual · (가능 시) token→event attention pool 효과 확인.

**선행:** F1 winner를 architecture yaml 고정란에 반영.

**할 일**

1. `finetune_v03_s3_architecture.yaml` sweep 실행.
2. `token_attention_pool=true`가 **실질 동작**하려면 Stage A **token hidden 재캐시**가 필요 — 없으면 no-op이므로 이 phase에서 병행 착수 여부 결정.
3. 구조 변경 시 saliency를 pad `attn` vs IxG **병행 로그**로 비교 (의사결정 근거 품질).

**산출**

- `selection/sweep3_winner.json` (= **final HP freeze** 후보)
- (선택) token cache 스펙: `docs` 짧은 절 + `scripts/.../cache_stage_a_token_hiddens_v03.py`

**게이트:** Fine/Binary가 F1 대비 붕괴하지 않고, (구조 ON 시) event attn과 위험도 정렬이 개선되거나 동등.

**토큰 캐시 (구조 게이트)**

```text
미완 → 완료 조건:
  event별 token hidden + mask parquet/pt
  Stage A encoder read-only reuse
  AttentionEventPooler가 residual로 mean‖max에 합류
```

토큰 캐시가 늦으면: Sweep3에서 `token_attention_pool=false`만 채택하고 pooler는 F4 뒤로 미룬다.

---

### F3 — Final 5-fold & 의사결정 평가 패키지

**목적:** 튜닝이 끝난 config로 운영용 ckpt·라우팅 통계를 고정.

**할 일**

1. `--mode cv_final5 --n-folds 5`로 재학습 (image ON, freeze HP).
2. Infer cfg로 전환: `OpenSetInferCfg` (`oof_only_calibration=false` 등 PLAN §19).
3. 표준 리포트 생성:
   - OOF fine / binary
   - route 분포·오분류 표
   - threshold risk–coverage 곡선
   - fold disagreement 히스토그램
4. Ensemble 추론 엔트리 정리 (`run_ensemble_infer_v03.py` 신설 또는 train 스크립트 모드).

**산출**

```text
outputs/online2/v2_finetune_v03/final_<stamp>/
  fold{i}_best.pt
  oof_predictions.parquet
  oof_routes.csv
  open_set_metrics.json
  risk_coverage.json
  FINAL_REPORT.md
```

**게이트 (운영 채택)**

- NORMAL: binary+fine 합의가 문서화된 τ에서 안정
- KNOWN: high-confidence 오답 급증 없음
- UNKNOWN/REJECT: 비율·리콜 목표를 `FINAL_REPORT`에 숫자로 기록

---

### F4 — Evidence / Routing 운영 게이트 (P0+)

**목적:** 의사결정 레이블이 “설명 가능”해야 P3·보고서 양산.

**할 일**

1. F3 ckpt로 `run_p0_evidence_demo_v03.py`를 **전체 example case** 배치화.
2. Saliency: IxG 유지 + PAD/`AttentionDecoder` α(attn) 병행 저장.
3. Gemma/리포트 프롬프트: route 결과를 필드로 강제  
   `[NORMAL|KNOWN|UNKNOWN|REJECT]` + 수치 인용 + saliency 남용 금지.
4. Actuator/bin 한계를 evidence 메타에 명시.

**산출**

```text
outputs/.../v2_finetune_v03/evidence_batch_<stamp>/
  case_id/evidence_p0.json
  case_id/top_events_ixg.json
  case_id/top_events_attn.json
```

**게이트:** ROADMAP P0와 동일 — 인용 토큰>0 및 raw 단위(℃/%/dS/m) 부착. 미통과 시 F5 금지.

**신규 스크립트 (권장)**

```text
scripts/online2_v2/v03/run_evidence_batch_v03.py
scripts/online2_v2/v03/run_route_report_v03.py
```

---

### F5 — 열린 진단 (UNKNOWN 경로)

**목적:** `UNKNOWN_ABNORMAL`을 닫힌 10-class로 억지 매핑하지 않고 retrieval(+제한 LLM)로 서술.

**선행:** F3 routing freeze + F4 게이트 통과.

**할 일**

1. fold-local / OOF prototype 인덱스 구축 (임베딩 merge 금지 규칙 준수).
2. UNKNOWN 케이스에 대해 top-k 유사 사례·센서 패턴 검색.
3. LLM은 **검색·수치 evidence만** 입력; Fine argmax 진단명 사용 금지.
4. rulebase/전문가 일치율에서 “강제 10종 매핑 비율” 감소를 성공 신호로 기록.

**산출**

- `src/online2/v2/finetune_v03/open_diagnosis.py`
- `scripts/online2_v2/v03/run_open_diagnosis_v03.py`
- 배치 리포트 마크다운

**게이트:** UNKNOWN 샘플에서 Fine class 문자열을 최종 진단으로 쓰지 않음이 자동 검증됨.

---

### F6 — Counterfactual 의사결정

**목적:** 위험도 \(R\) 감소 방향의 개입 후보를 제안 (PLAN §17, CF 문서).

**순서 (고정)**

```text
Path A  진단 맥락 CF (증상/원인 가설 편집)  → smoke
        + action grounding (bin→원시 구간)   → CF_ACTION_GROUNDING_V03
Path B  관리 개입 CF                         → Path A 통과 후
        + duration / capacity grounding
```

**할 일**

1. Risk 정의를 binary \(P(\mathrm{abnormal})\) (+ prototype)로 본선화; v0.2 10-class margin은 baseline만.
2. Stage A **재인코딩**으로 편집 시퀀스 평가 (MLM을 물리 시뮬로 쓰지 않음).
3. Cross-fitting: 탐색 fold ≠ holdout fold.
4. 패키지 isolation: `src/online2/v2/counterfactual/` 또는 `finetune_v03/counterfactual/` (+ `action_grounding.py`).
5. 토큰 최소 편집을 ℃·actuator 지속시간·용량으로 번역 ([`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md)).

**산출**

- CF smoke 스크립트 + JSON 스키마 (`risk_definition`, `path`, `delta_R`, `grounded_actions`)
- 사례 소수에 대한 A/B 결과표

**게이트:** CF 문서 권고안 체크리스트 충족; Path B는 A 없이 착수하지 않음; 보고서용 개입은 grounding 또는 `ungroundable` 명시.

---

## 5. 의사결정 스택 — 구축 체크리스트

라우팅이 운영 가능하려면 아래가 모두 “선택·저장·재현” 가능해야 한다.

| # | 구성 요소 | 상태 (문서 작성 시점) | F-phase |
|---|-----------|----------------------|---------|
| 1 | Binary \(p_{\mathrm{abn}}\) + `pos_weight` 튜닝 | Sweep1 진행 | F0 |
| 2 | Fine–Binary consistency loss/로그 | 코드 있음 · Sweep1 | F0 |
| 3 | `route_case` NORMAL/KNOWN/UNKNOWN/REJECT | 코드 있음 | F1 캘리브 |
| 4 | Prototype / energy / fold agree τ | 골격 · 캘리브 미완 | F1–F3 |
| 5 | Train vs Infer open-set cfg | config 반영 | F3 |
| 6 | LODO / risk–coverage 전용 eval | 부분 | F1–F3 |
| 7 | Evidence IxG + raw (+ attn) | P0 데모만 | F4 |
| 8 | UNKNOWN → open diagnosis | 미착수 | F5 |
| 9 | CF Path A/B + action grounding | 문서만 (`CF_ACTION_GROUNDING_V03`) | F6 |
| 10 | Token Stage A cache + pooler | stub | F2 병행 또는 후순위 |

---

## 6. 일정 가이드 (용량 기준, 일자 가변)

| Phase | 의존 | 대략 작업량 |
|-------|------|-------------|
| F0 | GPU 소진까지 | Sweep1 잔여 + 승자표 0.5일 |
| F1 | F0 | Sweep2 + LODO 초안 1–2일 |
| F2 | F1 | Sweep3 (+ token cache면 +2–4일) |
| F3 | F2 freeze | final5 + 리포트 1일 |
| F4 | F3 | 배치 evidence 1–2일 |
| F5 | F4 게이트 | retrieval 파이프 수일 |
| F6 | F5 또는 F3+문서 합의 | Path A smoke → B |

병행 허용: F2의 token cache 구현과 F1 LODO 스크립트.  
병행 금지: F5 without F4 · F6b without F6a · Sweep 승자 없이 final5.

---

## 7. 산출물·네이밍 규약

```text
outputs/online2/v2_finetune_v03/
  logs/                          # PROGRESS.jsonl, sweep agent logs
  selection/
    sweep1_winner.json
    sweep2_winner.json
    sweep3_winner.json
    routing_thresholds_oof.json
  final_<stamp>/                 # cv_final5
  evidence_batch_<stamp>/
  analysis/                      # route, LODO, risk-coverage
```

W&B:

```text
group = finetune_v03_improvement_r1   # r2…로 revision
job_type ∈ {sweep_1_loss_binary, sweep_2_metric_openset, sweep_3_architecture, cv_final5}
```

---

## 8. 즉시 액션 (운영자)

1. Sweep1 UI에서 완료율·실패 traceback 확인 (`outputs/.../logs/sweep1_agent_*.log`).
2. 완료 후 `summarize`로 top-k 뽑고 `sweep1_winner.json` 기록.
3. winner를 s2 yaml에 박고 Sweep2 생성 — **같은 group**.
4. F3 전에는 token cache scope(이번 revision에 넣을지)만 결정하면 됨.

---

## 9. 문서 관계

```text
DIAGNOSIS_FINETUNE_V03.md          초기 설계
DIAGNOSIS_FINETUNE_V03_PLAN.md     상세 스펙 (정본)
DIAGNOSIS_FINETUNE_V03_FOLLOWUP.md 본 문서 — 후속 구축·게이트 운용
ROADMAP_V03.md                     phase 한눈에
COUNTERFACTUAL_INFERENCE_V03.md    F6 상세
DIAGNOSIS_FINETUNE_V03_CV_REPORT.md 기준선 근거
```

PLAN과 충돌 시 **PLAN 스펙을 따르고**, 본 문서의 phase 순서·게이트만 갱신한다.
