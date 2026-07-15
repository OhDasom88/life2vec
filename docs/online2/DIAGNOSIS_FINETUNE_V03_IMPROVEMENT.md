# Finetune v0.3 — 전반 개선 방향 + 튜닝 계획

| 항목 | 값 |
|------|-----|
| 기준 run | `cv_20260715_104037_wandb` |
| Wandb | [dqrr69z6](https://wandb.ai/dasom-oh/Berry2Vec/runs/dqrr69z6) |
| CV 상세 | [`DIAGNOSIS_FINETUNE_V03_CV_REPORT.md`](./DIAGNOSIS_FINETUNE_V03_CV_REPORT.md) · run `REPORT.md` |
| 로드맵 | [`ROADMAP_V03.md`](./ROADMAP_V03.md) |
| 작성일 | 2026-07-15 |
| Isolation | `finetune_v03/` · `scripts/.../v03/` · `outputs/.../v2_finetune_v03/` only |

> **Callout — 문서 역할**  
> v0.3 개선의 **정본(canonical) 상세 계획**은 [`DIAGNOSIS_FINETUNE_V03_PLAN.md`](./DIAGNOSIS_FINETUNE_V03_PLAN.md)입니다.  
> 본 문서(`DIAGNOSIS_FINETUNE_V03_IMPROVEMENT.md`)는 **초기 CV 분석 + 튜닝 스케치**로 유지합니다.

---

## 0. 현 위치 (왜 이 문서가 필요한가)

| 지표 (재측정) | 값 | 의미 |
|---------------|-----|------|
| **OOF** fine Acc / macro-F1 | **0.80 / ≈0.78** | 주 지표. example 닫힌 분류의 솔직한 수준 |
| **OOF** binary AUROC / AUPRC | **≈0.91 / ≈0.99** | 랭킹은 강함. thr=0.5에서 FN 다수 |
| **Ensemble** Acc / F1 | **≈0.94 / 0.94** | v0.2 베스트와 동급 (낙관적 참고) |
| Fold val macro-F1 평균 | ~**0.54** | **튜닝·조기종료에 쓰면 노이즈** (n≈7, 클래스 결측) |

**문제:** `monitor=val_macro_f1` + 소표본 fold val → HP 스윕 신호대잡음비 낮음.  
**원칙:** 모델 선택·스윕 winner는 **OOF** (`fine_macro_f1`, `bin_auprc` / `bin_balanced_acc`)로만. fold val는 학습 로그용.

```text
지금 할 일 우선순위
  A. 평가 고정 (OOF 스크립트)           ← 인프라
  B. 짧은 코드 패치 (pos_weight 등)     ← 튜닝 가능 공간 열기
  C. 1차 wandb 스윕 (소표본 grid)       ← 성능·안정
  D. P2/P3 설계 트랙                    ← 제품 ceiling (닫힌 10종 한계 돌파)
병행: P0 evidence (raw/token) 게이트 유지
하지 말 것: 긴 전수 sweep · 아키텍처 전면 교체 · Stage A 전수 재캐시
```

---

## 1. 개선 방향 한눈에 (레이어)

| 레이어 | 목표 | 수단 | 기대 효과 |
|--------|------|------|-----------|
| **L0 평가** | fold F1 착시 제거 | OOF + ensemble 리포트 파이프 | “망했다/안 망했다” 판정 정상화 |
| **L1 HP 튜닝** | OOF 소폭↑·분산↓ | CLI/wandb + `pos_weight` 노출 | example Acc/F1·binary 운영 thr |
| **L2 헤더 정합** | fine↔binary 불일치 | τ 캘리브 + 라우팅 규칙 | 추론/보고서 일관성 |
| **L3 Open-set** | Known / UNKNOWN | energy·disagree·prototype | 강제매핑 감소 |
| **L4 열린 진단** | 신규 진단·미공개 | retrieval + raw/token + LLM | 제품 목표 |
| **L5 CF** | 관리 개입 | Path A → B (로드맵) | 이후 트랙 |

튜닝(L1)만으로 L4는 도달 불가. L1은 **L2–L3의 재료를 안정화**하는 단계.

---

## 2. 튜닝 계획 — CLI / wandb로 이미 가능한 것

`scripts/online2_v2/v03/run_diagnosis_finetune_v03.py` + `apply_wandb_config`에 연결됨.

| 파라미터 | 기본 | 스윕 후보 | 우선도 | 비고 |
|----------|------|-----------|--------|------|
| `lr` | 3e-4 | 1e-4, 3e-4, 1e-3 | **높음** | |
| `weight_decay` | 1e-2 | 1e-3, 1e-2, 5e-2 | **높음** | n=35 과적합 |
| `dropout` | 0.2 | 0.05, 0.1, 0.2, 0.3 | **높음** | |
| `lambda_binary` | 1.0 | 0.5, 1, 2, 4 | **높음** | binary 강조 |
| `lambda_fine` | 1.0 | 0.5, 1, 2 | 중 | |
| `lambda_supcon` | 0.1 | **0**, 0.05, 0.1, 0.3 | **높음** | 0 = ablation |
| `hidden_dim` | 192 | 96, 128, 192 | 중 | capacity↓ 권장 |
| `num_heads` | 4 | 2, 4 | 중 | `hidden % heads == 0` |
| `proj_dim` | 128 | 64, 128 | 낮 | supcon 켤 때 |
| `batch_size` | 8 | 4, 8 (≤ train n) | 중 | |
| `monitor` | `val_macro_f1` | **`val_loss`**, (+후술 binary) | **높음** | fold val 불안정 |
| `early_stop_*` | 20 / 30 | patience 10–30 | 낮 | |
| `seed` | 2023 | 소수 | 낮 | 분산 확인용 |

참조(과거 Life2Vec binary): `conf/sweep/hackathon_finetune.yaml`은 **`pos_weight` dense grid**가 핵심. **v03에는 아직 없음 → §3 최상 패치.**

---

## 3. 튜닝 계획 — 짧은 소스 수정으로 노출해야 하는 것

| 파라미터 | 현재 상태 | 수정 위치 | 권장 노출 | 우선도 |
|----------|-----------|-----------|-----------|--------|
| **`pos_weight` / `pos_weight_scale`** | train `n_neg/n_pos` **자동 고정** | `binary_pos_weight()`, `parse_args`, `apply_wandb_config` | `--pos-weight` 또는 `--pos-weight-scale`×공식 | **최상** |
| **`supcon_temperature`** | `losses.py` **0.1 하드코딩** | loss 인자 + CLI + wandb map | 0.05–0.2 | 중 |
| **`ff_dim`** | `hidden_dim*2` 고정 | `train_one_split` cfg | `--ff-dim` / multiplier | 낮 |
| **`use_image_adapter` / `gate_init`** | adapter 항상 ON, gate≈0.01 | `model.py`, `ImageAdapterConfig` | `--no-image-adapter`, `--gate-init` | **중–높** |
| **`use_mean_max`** | True 고정 | cfg + CLI | on/off | 중 |
| **fine `class_weight` / label_smoothing** | CE plain | `losses.py` | CE weight, smooth 0–0.1 | 중 |
| **binary target / threshold** | abnormal=1, thr=0.5 | dataset + metrics / 평가 | `y_normal` 옵션, **thr 스윕(평가)** | 중 |
| **`monitor`에 binary 지표** | fine F1 / val_loss만 | train loop + argparse | `val_bin_auprc`, `val_bin_balanced_acc` | **높음** |
| **LR scheduler** | 없음 (constant AdamW) | train loop | cosine / warmup | 중 |
| **consistency λ** | config만, loss 미연결 | `losses.py` | enable 후 스윕 | 낮(나중) |

### 3.1 최소 패치 체크리스트 (`pos_weight`)

```text
1) argparse: --pos-weight (None=auto) | --pos-weight-scale (default 1.0)
2) binary_pos_weight(): return scale * n_neg/n_pos  OR 절대값
3) apply_wandb_config mapping에 pos_weight, pos_weight_scale 추가
4) conf/sweep/finetune_v03.yaml (+ optional finetune_v03_binary.yaml) 신설
5) train 종료 시 OOF 자동 dump → wandb.summary (winner 선정용)
```

### 3.2 monitor 확장 (패치 후)

| monitor | 언제 쓰면 |
|---------|-----------|
| `val_loss` | 1차 기본 (즉시, 코드 거의 불필요) |
| `val_bin_auprc` | binary·라우팅이 우선일 때 (코드 열기) |
| `val_bin_balanced_acc` | FPR/FNR 균형 |
| `val_macro_f1` | **비권장** (현 run 노이즈원) |

Winner는 여전히 **OOF**로 고른다. monitor는 early-stop용 proxy일 뿐.

---

## 4. 추천 1차 스윕 (소표본 · 바로)

**목표 metric (OOF):** `bin_auprc`, `bin_balanced_acc`, `fine_macro_f1`  
**금지:** fold val만으로 run 채택.

**순서**

1. 코드 1회: `pos_weight_scale` (+ 선택 `monitor=val_bin_auprc` / `val_loss`)
2. OOF 평가를 train 끝난 뒤 자동 기록
3. 아래 grid / random (총 budget 감: 12–24 agents 권장, hackathon식 dense는 `pos_weight` 축에만)

```text
pos_weight_scale ∈ {0.25, 0.5, 1, 2, 4}   # 또는 absolute pos_weight ∈ {0.1 … 2.0}
lambda_binary    ∈ {1, 2, 4}
lambda_supcon    ∈ {0, 0.1}
lr               ∈ {1e-4, 3e-4}
dropout          ∈ {0.1, 0.2}
weight_decay     ∈ {1e-3, 1e-2}
monitor          ∈ {val_loss}               # 1차는 단일 권장
use_image_adapter ∈ {true, false}           # 노출 후
```

이미 CLI만으로 가능한 **예열 run** (패치 전):

```bash
# monitor를 val_loss로, λ_supcon=0 ablation, wd↑ — OOF는 별도/종료후 평가
python3 scripts/online2_v2/v03/run_diagnosis_finetune_v03.py --mode cv --wandb \
  --monitor val_loss --lambda-supcon 0 --weight-decay 1e-2 --dropout 0.2 \
  --wandb-run-name v03_cv_warmup_supcon0
```

---

## 5. 구조 변경 실험 (스윕이 아닌 별 트랙 A/B)

짧은 HP가 아니라 **실험 축**. 1차 스윕 이후에만.

| 축 | 내용 | 비용 |
|----|------|------|
| Pooler | PAD(MHA) vs additive `AttentionDecoder` | 모델 포크 |
| Binary head | shared `h_case` Linear vs binary-only pooled decoder | 중 |
| Saliency | IxG-only vs decoder `attn`/`α` 병행 | evidence면 소 |
| Task schedule | binary-only warm-up → fine 추가 | 학습 루프 |
| CE | class_weight / label_smoothing | loss (중비용) |

Gate: A/B는 **동일 OOF 프로토콜**로만 비교. fold F1로 우열 가리지 않음.

---

## 6. 설계 트랙 (튜닝으로 대체 불가)

### 6.1 L2 — Binary τ + fine 합의

- OOF에서 thr 스윕 → 목표 FPR/FNR에 τ freeze  
- 규칙 예:
  - `p_abn < τ` → 정상 후보
  - `max softmax` 낮음 또는 **fold disagreement** → `UNKNOWN`
  - fine 병명 vs binary 정상 불일치 → UNKNOWN 또는 binary 우선(문서화)

현 OOF: thr=0.5에서 FP=0, FN=8 → **recall 쪽 thr 완화** 후보.

### 6.2 L3 — P2 Open-set

- normal / class prototypes (OOF embedding)  
- energy + fold disagreement (본 run에서 disagree **10/35**)  
- LODO로 τ만 calibration  
- Known/Unknown 라우팅

### 6.3 L4 — P3 열린 진단 (※ P0 게이트)

- sequence + DINO retrieval  
- raw/token evidence 패키지  
- 제한 LLM + 판단 불가  
- 미공개 20·신규 진단(염분·응애 등) — **10-class CE 스윕 금지**

### 6.4 L0/P0 — Evidence (병행)

- fuse-aware saliency 유지  
- token attribution 실배선 (현재 hook만)  
- Gemma `[상태][원인][관리]` + 수치 인용

### 6.5 L5 — P4 CF (이후)

- Path A → Path B. MLM ≠ 물리 WM.

---

## 7. 실행 로드맵 (스프린트 단위)

| Sprint | 산출 | 완료 기준 |
|--------|------|-----------|
| **S0** | OOF/ensemble 스크립트 → train 후 자동 · REPORT 갱신 | CV마다 `oof_metrics.json` + wandb summary |
| **S1** | `pos_weight(_scale)` + `monitor` binary 선택 + sweep yaml | CLI로 pos_weight 스윕 가능 |
| **S2** | 1차 sweep (12–24) + image on/off | OOF F1 또는 bin_auprc가 baseline 대비 개선 또는 동등+안정 |
| **S3** | τ 캘리브 + 라우팅 규칙 문서/코드 | OOF 기준 FNR/FPR 목표 표 |
| **S4** | P2 UNKNOWN 프로토타입 | 강제매핑 비율·disagree 로그 |
| **S5** | P0 token 실배선 + P3 샘플 | 미공개 샘플에서 열린 진단+수치 근거 |

Baseline 고정: **본 문서의 `cv_20260715_104037_wandb` OOF** (Acc 0.80 / F1 0.78 / bin AUPRC 0.989).

---

## 8. Sweep YAML 초안 (신설 예정)

경로 제안: `conf/sweep/finetune_v03.yaml`

```yaml
# DRAFT — S1 패치 후 agent에 연결
program: scripts/online2_v2/v03/run_diagnosis_finetune_v03.py
method: grid   # 또는 bayes; pos_weight는 grid 권장
project: Berry2Vec
entity: dasom-oh
parameters:
  mode: { value: cv }
  n_folds: { value: 5 }
  epochs: { value: 150 }
  monitor: { value: val_loss }
  wandb: { value: true }
  lr: { values: [0.0001, 0.0003] }
  weight_decay: { values: [0.001, 0.01] }
  dropout: { values: [0.1, 0.2] }
  lambda_binary: { values: [1.0, 2.0, 4.0] }
  lambda_supcon: { values: [0.0, 0.1] }
  # S1 이후:
  # pos_weight_scale: { values: [0.25, 0.5, 1.0, 2.0, 4.0] }
  # use_image_adapter: { values: [true, false] }
```

실행 패턴은 v02와 동일: `wandb sweep` → `wandb agent … --count N`.  
**각 agent 종료 시 OOF를 summary에 올려** UI에서 fold val가 아니라 OOF로 정렬.

---

## 9. 의사결정 표 (튜닝 vs 설계)

| 목표 | 튜닝(L1)? | 설계(L2+)? | 비고 |
|------|-----------|------------|------|
| example OOF 0.78→0.85+ | **예** | 보조 | pos_weight·λ·wd·dropout |
| binary 운영 recall | **예** (pos_w + τ) | 라우팅 | |
| fold 분산·early-stop 안정 | monitor=`val_loss` | OOF 게이트 | |
| 미공개/신규 진단 | **아니오** | **P2/P3 필수** | |
| 보고서 수치 근거 | 아니오 | P0 | |
| v0.2 ensemble 동급 유지 | 동급 이미 달성 | 회귀만 감시 | |

---

## 10. 한 줄 요약

| 상태 | 내용 |
|------|------|
| **즉시 (코드 없이)** | `monitor=val_loss`, `λ_supcon=0` ablation, `lr/wd/dropout/λ_binary` CLI·wandb 스윕 — **선정은 OOF** |
| **짧은 수정 후 필수** | **`pos_weight(_scale)`**, binary monitor, image on/off, OOF 자동 기록, `conf/sweep/finetune_v03.yaml` |
| **별 실험** | additive Attention decoder, binary-only warm-up, decoder-α saliency |
| **제품 ceiling** | P2 UNKNOWN → P3 열린 진단 (닫힌 CE 스윕으로 대체 금지) |

---

## 11. 관련 파일

| 경로 | 역할 |
|------|------|
| `scripts/online2_v2/v03/run_diagnosis_finetune_v03.py` | train + wandb CLI |
| `src/online2/v2/finetune_v03/losses.py` | CE / BCE / SupCon |
| `src/online2/v2/finetune_v03/model.py` | PAD + heads + image adapter |
| `conf/sweep/hackathon_finetune.yaml` | 과거 pos_weight 참고 |
| `conf/sweep/diagnosis_finetune_v02.yaml` | v02 sweep 패턴 |
| `outputs/.../cv_20260715_104037_wandb/analysis_oof_ensemble.json` | baseline OOF |

다음 구현 권장 순서: **(1) OOF 자동 평가 훅 → (2) pos_weight_scale 패치 → (3) sweep yaml → (4) agent 가동**.
