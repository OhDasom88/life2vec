# Finetune v0.3 CV 학습결과 상세 보고서

| 항목 | 값 |
|------|-----|
| 버전 | `0.3.0` (multitask: binary + fine 10-class + projection) |
| Run | `cv_20260715_104037_wandb` |
| 경로 | `outputs/online2/v2_finetune_v03/runs/cv_20260715_104037_wandb/` |
| 원본 사본 | 동일 내용이 run 디렉터리 `REPORT.md`에도 있음 |
| Wandb | [Berry2Vec / dqrr69z6](https://wandb.ai/dasom-oh/Berry2Vec/runs/dqrr69z6) |
| 작성일 | 2026-07-15 |
| 관련 설계 | [`DIAGNOSIS_FINETUNE_V03.md`](./DIAGNOSIS_FINETUNE_V03.md), [`ROADMAP_V03.md`](./ROADMAP_V03.md) |
| 개선·튜닝 계획 | [`DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md`](./DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md) |

---

## 1. 요약 (Executive)

**한 줄:** fold별 val macro-F1(~0.54)만 보면 실패로 보이지만, **공정 재측정(OOF / ensemble)에서는 example 닫힌 10-class가 v0.2 베스트와 동급**이다.

| 평가 프로토콜 | Acc | macro-F1 | 오류 수 | 해석 |
|---------------|-----|----------|--------|------|
| **True OOF** (주 지표 권장) | **0.800** | **0.776** | 7/35 | fold val에만 쓰인 케이스로 예측 |
| **5-fold soft ensemble** | **0.943** | **0.940** | 2/35 | v0.2 보고 방식과 동형 |
| fold mean val macro-F1 (학습 로그) | — | ~**0.54** | — | **비교·조기종료 기준으로 부적절** |

| Binary (OOF) | 값 |
|--------------|-----|
| AUPRC | **0.989** |
| AUROC | **0.911** |
| balanced Acc @ thr=0.5 | 0.871 |
| FP (정상→비정상) | **0** |
| FN (비정상→정상) @0.5 | **8** |

**결론 방향**

1. 대규모 아키텍처 교체는 불필요 (ensemble 기준 negative transfer “붕괴” 아님).
2. 먼저 **평가 파이프를 OOF 중심으로 고정**.
3. 그다음 **짧은 λ 스윕 + binary 임계 캘리브**.
4. 미공개/신규 진단 목표는 **P2 open-set → P3 열린 진단** (분류 튜닝만으로는 불가).

---

## 2. 실험 설정

### 2.1 데이터

| 항목 | 값 |
|------|-----|
| 라벨 | `outputs/online2/v2_finetune/labels_example_score90.csv` (n=**35**) |
| 라벨맵 | 10 class (`정상_운영` = id 6) |
| Stage A cache | `outputs/online2/v2_finetune_v02/event_embeddings` (read-only, DINO 포함) |
| Split | farm-stratified **5-fold** (`seed=2023`) |
| fold당 | train ≈28 / val ≈7 |

클래스 분포 (전수 35):

| 진단 | n |
|------|---|
| 칼슘부족_잎끝마름 | 5 |
| 정상_운영 | 4 |
| 탄저병_발생_위험 | 4 |
| 기형과_생리장해 | 4 |
| 영양생장_과다 / 부족 / 동해 / 뿌리 / 잿빛 / 흰가루 | 각 3 |

### 2.2 모델·손실

```text
Frozen Stage A event_mean‖max (+ optional DINO fuse)
  → EventMetadataEncoder + PAD → h_case
       ├─ H_binary     → p(abnormal)     BCE (pos_weight)
       ├─ H_fine       → 10-class logits  CE
       └─ H_proj       → z (L2)           SupCon (λ=0.1)
```

| 하이퍼파라미터 | 값 |
|----------------|-----|
| lr / weight_decay | 3e-4 / 1e-2 |
| batch_size | 8 |
| hidden_dim / num_heads / proj_dim | 192 / 4 / 128 |
| dropout | 0.2 |
| λ_binary : λ_fine : λ_supcon | **1.0 : 1.0 : 0.1** |
| epochs (상한) | 150 |
| early-stop | min 30 + patience 20 |
| **monitor** | `val_macro_f1` (fold val, n=7) ← 한계 있음 |
| device | cuda |
| wandb sweep | **없음** (단일 run) |

### 2.3 산출물

| 파일 | 설명 |
|------|------|
| `fold{0..4}_best.pt` | fold별 best 체크포인트 |
| `fold*_history.json` | epoch 곡선 |
| `summary.json` | fold best 요약 |
| `run_meta.json` | 전체 args |
| `analysis_oof_ensemble.json` | OOF/ensemble 재분석 |
| `oof_predictions.csv` | 케이스별 OOF 예측 |
| `wandb/` | 로컬 wandb run 디렉터리 |

---

## 3. Fold별 학습 경과

학습 중 early-stop으로 선택한 **fold val** 지표 (n_val=7).  
macro-F1은 클래스 결측으로 과소평가되므로 **참고만**.

| fold | stop epoch | best epoch | best Acc | best macro-F1 | best val_loss | best bin AUPRC | final train_loss |
|------|------------|------------|----------|---------------|---------------|----------------|------------------|
| 0 | 69 | 49 | 0.571 | 0.400 | 2.075 | 0.976 | 0.509 |
| 1 | 85 | 65 | **1.000** | 0.700 | 0.639 | n/a (val에 정상 0) | 0.440 |
| 2 | 71 | 51 | 0.571 | 0.333 | 2.029 | 0.976 | 0.571 |
| 3 | 90 | 70 | **1.000** | 0.700 | 0.785 | 1.000 | 0.186 |
| 4 | 71 | 51 | 0.857 | 0.567 | 1.845 | 0.976 | 0.525 |
| **평균** | — | — | **0.800** | **0.540** | — | — | — |

관찰:

- train_loss는 fold마다 ~0.2–0.6까지 하락 → 소표본에서 과적합 여지.
- fold1/3은 val Acc=1.0이지만, 전역 OOF는 0.80 → **fold 운·지표 불일치**.
- monitor를 fold macro-F1에 두면 early-stop이 불안정할 수 있음.

---

## 4. 공정 평가: True OOF

각 케이스를 **그 케이스가 val에 들어간 fold 모델**로만 예측 (학습에 미사용).

### 4.1 전체 지표

| 지표 | 값 |
|------|-----|
| n | 35 |
| Accuracy | **0.800** (28/35 정답) |
| Balanced Acc | 0.775 |
| macro-F1 (10 class) | **0.776** |
| weighted-F1 | 0.793 |
| 정상_운영 recall | 0.75 (3/4) |
| 고신뢰 오분류 (p_pred ≥ 0.8) | **0** |
| conf 평균 (정답 / 오답) | 0.531 / 0.474 |

### 4.2 클래스별 (OOF)

| 클래스 | support | precision | recall | F1 | 평가 |
|--------|---------|-----------|--------|----|------|
| 기형과_생리장해 | 4 | 1.00 | 1.00 | **1.00** | 강함 |
| 동해_난방실패 | 3 | 1.00 | 1.00 | **1.00** | 강함 |
| 영양생장_과다 | 3 | 1.00 | 1.00 | **1.00** | 강함 |
| 칼슘부족_잎끝마름 | 5 | 1.00 | 1.00 | **1.00** | 강함 |
| 탄저병_발생_위험 | 4 | 0.80 | 1.00 | **0.89** | 양호 |
| 뿌리_발육_부진 | 3 | 1.00 | 0.67 | **0.80** | 양호 |
| 정상_운영 | 4 | 0.50 | 0.75 | **0.60** | 경계 |
| 잿빛곰팡이병_발생_위험 | 3 | 0.50 | 0.67 | **0.57** | 약함 |
| 흰가루병_발생_위험 | 3 | 1.00 | 0.33 | **0.50** | 약함 |
| 영양생장_부족 | 3 | 0.50 | 0.33 | **0.40** | 약함 |

### 4.3 Confusion matrix (OOF, rows=true / cols=pred)

행·열 순서 = label_map 순서:

`기형과, 동해, 뿌리, 영양과다, 영양부족, 잿빛, 정상, 칼슘, 탄저, 흰가루`

| true \ pred | 기형 | 동해 | 뿌리 | 과다 | 부족 | 잿빛 | 정상 | 칼슘 | 탄저 | 흰가루 |
|-------------|------|------|------|------|------|------|------|------|------|--------|
| 기형과 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 동해 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 뿌리 | 0 | 0 | 2 | 0 | 0 | 0 | **1** | 0 | 0 | 0 |
| 영양과다 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 영양부족 | 0 | 0 | 0 | 0 | 1 | **1** | **1** | 0 | 0 | 0 |
| 잿빛 | 0 | 0 | 0 | 0 | **1** | 2 | 0 | 0 | 0 | 0 |
| 정상 | 0 | 0 | 0 | 0 | 0 | **1** | 3 | 0 | 0 | 0 |
| 칼슘 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 |
| 탄저 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| 흰가루 | 0 | 0 | 0 | 0 | 0 | 0 | **1** | 0 | **1** | 1 |

### 4.4 OOF 오분류 7건 상세

| case_id | fold | 정답 | 예측 | p_pred | p_abn | 해석 |
|---------|------|------|------|--------|-------|------|
| F393975_2024-10-21_2024-11-03 | 2 | 뿌리_발육_부진 | 정상_운영 | 0.709 | 0.154 | 비정상→정상 (fine+binary 모두 실패형) |
| F295998_2024-12-08_2024-12-21 | 4 | 영양생장_부족 | 정상_운영 | 0.546 | 0.104 | 동일 패턴 |
| F605134_2025-02-18_2025-03-03 | 2 | 잿빛곰팡이병 | 영양생장_부족 | 0.489 | 0.387 | 병↔영양 혼동 |
| F882016_2024-09-16_2024-09-29 | 2 | 흰가루병 | 탄저병 | 0.440 | **0.966** | 비정상은 맞음, 병종 혼동 |
| F760592_2025-01-15_2025-01-28 | 0 | 영양생장_부족 | 잿빛곰팡이병 | 0.403 | 0.595 | 저신뢰 혼동 |
| F420458_2025-02-16_2025-03-01 | 0 | 정상_운영 | 잿빛곰팡이병 | 0.402 | 0.314 | **fine↔binary 불일치** (binary는 정상 쪽) |
| F302129_2024-10-15_2024-10-28 | 0 | 흰가루병 | 정상_운영 | 0.328 | 0.162 | 저신뢰 + binary FN |

**오류 패턴 요약**

1. **정상으로의 붕괴 (3건):** 뿌리적진·영양부족·흰가루 → 정상.  
2. **병명끼리 혼동 (흰가루↔탄저, 잿빛↔영양).**  
3. **고신뢰 오답 없음** → 불확실 샘플을 UNKNOWN으로 보내는 open-set이 적합.  
4. **헤드 불일치:** fine이 병명인데 binary p_abn이 낮은 케이스 / 그 반대.

---

## 5. Soft ensemble (참고, v0.2 동형)

5개 fold softmax 평균으로 **전 케이스** 예측.

| 지표 | 값 |
|------|-----|
| Acc | **0.943** (33/35) |
| macro-F1 | **0.940** |
| 정상 recall | **1.00** |
| Binary AUPRC | ~1.0 |
| fold 예측 불일치 케이스 | **10 / 35** |

Ensemble에서도 틀린 2건 (모두 → `정상_운영`, 저~중신뢰):

| case_id | 정답 | 예측 | 비고 |
|---------|------|------|------|
| F393975_… | 뿌리_발육_부진 | 정상_운영 | OOF와 동일 실패 |
| F302129_… | 흰가루병 | 정상_운영 | OOF와 동일 실패 |

→ v0.2 베스트 보고(~Acc 0.94 / F1 0.93)와 **동급**. Multitask 추가로 example 닫힌 분류가 무너지지 않았다.

> 주의: ensemble-all35는 일부 fold가 해당 케이스를 train에서 본 상태라 **낙관적**. 공식 비교·게이트는 **OOF**를 쓸 것.

---

## 6. Binary head 심층

| 항목 | OOF |
|------|-----|
| 양성(비정상) / 음성(정상) | 31 / 4 |
| AUPRC / AUROC | **0.989 / 0.911** (랭킹 우수) |
| thr=0.5 FP | 0 |
| thr=0.5 FN | 8 |

의미:

- **스코어 랭킹은 잘됨** → τ를 OOF로 맞추면 운영 recall을 개선할 여지.
- 현재 thr=0.5는 음성(정상) 보호에 치우쳐 abnormal을 놓치는 편.
- fine 10-class와 binary를 **독립 argmax/0.5로 쓰면 라우팅이 어긋날 수 있음** → 합의 규칙 필요.

---

## 7. v0.2 대비 위치

| 관점 | v0.2 (베스트 계열) | v0.3 (본 run) |
|------|-------------------|---------------|
| Example ensemble Acc/F1 | ~0.94 / ~0.93 | **0.94 / 0.94** |
| True OOF | (당시 보고 체계와 다를 수 있음) | **0.80 / 0.78** |
| 헤드 | 분류 (+ semantic align) | **binary + fine + proj** |
| Open-set / 신규 진단 | 없음 (10종 강제) | 아직 미구현 (의도된 로드맵) |
| 하이퍼 스윕 | 있음 (예: 20-run) | **없음** (기본 λ) |

**해석:** v0.3의 “성능 문제”로 보였던 fold F1은 **평가 착시**에 가깝다. example 닫힌 분류 ceiling은 이미 v0.2와 비슷한 수준이며, 제품 목표(미공개·신규 진단) 돌파는 튜닝이 아니라 **설계(P2/P3)**.

---

## 8. 한계와 리스크

1. **독립 라벨 n=35** — farm/case 단위; window 수와 혼동 금지.  
2. **fold val macro-F1 monitor** — early-stop·모델 선택 왜곡.  
3. **닫힌 10-class** — 염분·응애·야간고온 등 신규 라벨은 구조상 불가.  
4. **binary FN @0.5** — 운영 전 τ 캘리브 필수.  
5. **헤드 불일치** — 추론 라우팅 미정의 시 보고서/액션 혼란.  
6. **ensemble 낙관** — 배포 게이트로 쓰면 OOF를 과대평가.

---

## 9. 이후 방안 (우선순위)

### P0 — 평가·파이프 고정 (즉시)

- train/평가 주 지표 = **OOF Acc / macro-F1 + binary AUPRC**  
- fold macro-F1은 wandb 참고용으로만  
- 본 보고서의 `analysis_oof_ensemble.json` / `oof_predictions.csv`를 스크립트로 편입  
- early-stop monitor 후보: `val_loss` (또는 OOF proxy)

### P1a — 짧은 튜닝 (설계 변경 없음)

목표: OOF를 소폭 끌어올리거나 안정화 (예: 0.80 → 0.85+).

| 축 | 후보 (소수 grid, ≤12 runs) |
|----|---------------------------|
| λ_supcon | 0, 0.05, 0.1 |
| λ_binary | 0.5, 1.0, 2.0 |
| monitor | val_loss vs (참고) val_macro_f1 |
| (선택) lr | 1e-4, 3e-4 |

**하지 말 것:** 긴 wandb sweep, coarse head 재도입, Stage A 전수 재캐시, 아키텍처 갈아엎기.

### P1b — Binary τ + 헤드 합의

- OOF에서 FPR/FNR 목표에 맞게 **τ 선택**  
- 규칙 예: `p_abn < τ → 정상 후보`, `fold disagreement 또는 low max-prob → UNKNOWN`  
- fine/binary 불일치 시 우선순위 문서화

### P2 — Open-set

- normal / class prototype, energy, fold disagreement  
- LODO로 τ만 calibration  
- 라우팅: 정상 → Known → `UNKNOWN_ABNORMAL`

### P3 — 열린 진단 (P0 게이트 후)

- retrieval + raw/token 근거 + 제한 LLM  
- 미공개 20 / 신규 진단 서술 — **분류 스윕으로 대체 금지**

### P4 — Counterfactual (이후)

- Path A 진단 CF → Path B 관리 개입 (로드맵 준수)

---

## 10. 권장 의사결정

| 질문 | 답 |
|------|----|
| 지금 학습이 실패인가? | **아니오** (OOF/ensemble 기준). fold F1만 보면 오판. |
| 튜닝이 필요한가? | **짧게만** (λ·monitor·binary τ). |
| 설계를 바꿔야 하는가? | **제품 목표(열림/미공개)를 위해서는 예** — P2/P3. example Acc만이면 대규모 재설계 불필요. |
| 다음 실행 한 가지 | OOF를 CI/스크립트에 고정한 뒤 λ grid ≤12 + τ 캘리브. |

---

## 11. 참고·재현

```bash
# Wandb
# https://wandb.ai/dasom-oh/Berry2Vec/runs/dqrr69z6

# 학습 (동일 설정 재현 예)
python3 scripts/online2_v2/v03/run_diagnosis_finetune_v03.py \
  --mode cv --n-folds 5 --epochs 150 --wandb \
  --wandb-project Berry2Vec --wandb-entity dasom-oh \
  --run-dir outputs/online2/v2_finetune_v03/runs/cv_20260715_104037_wandb

# 분석 산출물
# analysis_oof_ensemble.json
# oof_predictions.csv
```

시각 요약 캔버스: `~/.cursor/projects/home-dasom-life2vec/canvases/v03-cv-analysis.canvas.tsx`
