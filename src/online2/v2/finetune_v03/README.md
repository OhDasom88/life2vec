# finetune_v03

**근거**: `docs/online2/DIAGNOSIS_FINETUNE_V03_PLAN.md` (canonical) — 사전학습 encoder를 frozen으로 두고 정상/비정상(binary) + 세부 진단(fine, 10-class) 두 head를 함께 학습하는 v0.3 진단 미세조정 패키지.
**출력 위치**: `outputs/online2/v2_finetune_v03/` (isolated — v0.1/v0.2 출력과 분리)

## 핵심 확인 사항 (2026-07-23 감사)

- **vocab/frozen encoder는 사전학습과 완전히 공유된다.** 이 패키지는 vocab을 직접 다루지 않는다 — `scripts/online2_v2/cache_stage_a_event_embeddings.py`가 사전학습과 동일한 `RegistryVocabulary`/`life2vec_token_registry_v2.json`으로 이벤트를 토큰화하고, `load_frozen_encoder()`(`strict=False`, `requires_grad_(False)`, `.eval()`)로 사전학습 encoder를 얼린 채 `event_mean`/`event_max`를 미리 캐싱해 parquet으로 저장한다. `dataset.py`는 이 캐시만 읽으므로 별도 tokenizer가 존재하지 않는다 — 사전학습 사전·가중치가 그대로 반영된다는 뜻이다.
- **실측 결과 존재**: `docs/online2/DIAGNOSIS_FINETUNE_V03_CV_REPORT.md`, run `cv_20260715_104037_wandb` — True OOF **Acc 0.800, macro-F1 0.776, binary AUROC 0.911 / AUPRC 0.989**.
- **saliency는 구현돼 있으나 LLM으로 연결되지 않았다.** `risk_saliency.py::event_ixg_abnormal_margin`(이벤트별 signed input×gradient)이 실제로 동작하고 `counterfactual/evaluation/diagnosis_critic.py`의 risk_before/after 계산에 쓰이지만, 소비처가 데모/attribution 스크립트뿐이다 — repo 전체에 실제 LLM API 호출은 없다(`finetune_v03/llm_evidence/`에서 신규 구현 중, 아래 참조).
- **UNK 비율을 정기적으로 감사하는 스크립트는 없다.** 과거 한 번(FARM_LOCAL vocab 슬롯 8→24) UNK 위험을 사후 패치한 이력만 있음.

## 파일 맵

| 파일 | 역할 |
|---|---|
| `model.py` | `EventPoolingDiagnosisModelV03` — task-specific query 기반 event pooling, binary/fine/projection head, DINO 이미지 fusion(`fuse_image_into_events`) |
| `dataset.py` | `DiagnosisEventDatasetV03` — 캐시된 `event_mean`/`event_max`만 읽음(토큰화는 안 함), repeated/stratified fold 분할 |
| `losses.py` | fine CE + binary BCE + fine-binary consistency + SupCon/prototype 결합 loss |
| `metrics.py` | fine/binary metrics, head-consistency, score-distribution, per-class F1·confusion matrix·calibration reliability bins |
| `open_set.py` | NORMAL / KNOWN / UNKNOWN_ABNORMAL / REJECT 라우팅 (`route_case`) — 임계값 캘리브레이션은 아직 미완 |
| `risk_saliency.py` | `event_ixg_abnormal_margin` — abnormal-risk margin에 대한 이벤트별 input×gradient saliency |
| `checkpoint.py`, `config.py`, `cv.py`, `pos_weight.py`, `sample_weights.py`, `evidence_raw.py`, `version.py` | 체크포인트 저장/설정/CV 분할/loss pos_weight/`WeightedRandomSampler` 표본 가중치/원시 근거 조회/버전 상수 |

## 학습 실행

```
scripts/online2_v2/v03/run_diagnosis_finetune_v03.py --mode cv_repeated ...
```

`Canonical plan`: `docs/online2/DIAGNOSIS_FINETUNE_V03_PLAN.md`. 로그는 `outputs/online2/v2_finetune_v03/logs/PROGRESS.jsonl` + (옵션) W&B에 per-epoch scalar(loss/AUROC/AUPRC/ECE/Brier/NLL) + per-class F1 + best-epoch 시점 confusion matrix/calibration reliability 테이블로 기록된다.

## 클래스 불균형 처리 — pos_weight(loss) + WeightedRandomSampler(샘플링) (2026-07-24)

원본 life2vec(`survival`/`eos` 태스크)의 클래스 불균형 처리 방식을 확인하고 online2에 이식했다 — `CLSDataModule.get_train_weights()`(`src/data_new/datamodule.py`)가 `1/count(class)` 정규화 가중치로 `WeightedRandomSampler(replacement=True)`를 만들어 학습 배치에서 소수 클래스를 오버샘플링하는 방식.

- **`pos_weight`(loss 재가중, 기존)**: `losses.py`의 BCE에 실제로 적용됨(실측 fold별 0.095~0.150). `apply_wandb_config()`에 이미 스윕 가능하게 배선돼 있고 `conf/sweep/finetune_v03_s1_loss_binary.yaml`에 `["auto","0.25","0.5","1.0"]` 그리드까지 설계돼 있지만, **스윕이 끝까지 실행된 적은 없다**(`sweep1_winner.json` 없음) — 지금까지의 모든 결과는 스윕 승자가 아니라 그냥 기본값 `"auto"`.
- **`WeightedRandomSampler`(샘플링, 신규)**: `sample_weights.py::class_balanced_sample_weights` + `run_diagnosis_finetune_v03.py --weighted-sampler`(기본 `true`). **fine(10-class) 라벨이 아니라 binary(정상/비정상) 라벨로 가중치를 계산한다** — 처음엔 life2vec처럼 fine(주 target)로 짰다가, 실측해보니 이 데이터는 fine class 10개가 이미 3~5건씩 고르게 분포돼 있어서 fine 기준 오버샘플링은 원본과 거의 같은 정상/비정상 비율(10.2% vs 원본 11.4%)만 나왔다 — 진짜 불균형(정상 4건 vs 비정상 31건)은 9개 fine class를 "비정상"으로 뭉치면서 생기는 binary-level 현상이라, binary로 가중치를 바꾸니 실측 48.8%/51.2%로 실제 교정됨을 확인했다. life2vec은 TARGET 자체가 이진이라 이 구분이 없었던 것 — online2는 fine/binary가 분리돼 있어서 life2vec을 문자 그대로 베끼면 안 통했다.
- val/test는 두 기법 다 적용 안 함(life2vec의 `FixedSampler` 철학과 동일 — 평가는 재샘플링 없이 고정).

### life2vec 논문(arXiv:2306.03009) 실측 확인 (2026-07-24)

WebFetch로 논문 본문을 직접 확인했다:
- **§4.2.5 Data Split**: "무작위로 70/15/15 split, 시퀀스 어떤 feature와도 무관하게 완전히 무작위"(원문 인용) — stratification도 grouping도 없다. online2가 이미 쓰는 `make_repeated_group_folds`(farm-그룹, stratified)가 논문 방식보다 더 보수적이다 — 바꿀 필요 없음.
- **§4.4.2**: 사망예측을 Positive-Unlabeled 문제로 놓고 **Asymmetrical Cross-Entropy Loss**를 쓴다(원문 인용: "right-censored outcomes"로 인한 label 불확실성 때문). online2의 `pos_weight`(단순 BCE 재가중)는 이것과 다른 기법 — 논문에 이 값을 얼마로 써야 하는지 선례가 없다.
- **`WeightedRandomSampler`는 논문 본문에 언급이 없다** — `CLSDataModule.get_train_weights()`는 코드에만 있는 구현 디테일이다. 즉 online2의 `--weighted-sampler`는 life2vec **논문**이 아니라 life2vec **코드**를 따른 것.
- 하이퍼파라미터 탐색 방법(grid/random/bayesian)은 논문 본문에 구체적으로 안 나온다("SI: Implementation Details" 참조라고만 되어 있고 그 부분은 미확인) — 탐색 방법 선택에 논문 선례를 못 썼다는 뜻, `conf/sweep/finetune_v03_s1b_pos_weight_sampler.yaml`의 grid 설계는 순수 경험적 선택.

### Sweep 1b — pos_weight 0.05 단위 + weighted_sampler grid (2026-07-24, 진행 중)

`conf/sweep/finetune_v03_s1b_pos_weight_sampler.yaml` — 기존 sweep 1(`["auto","0.25","0.5","1.0"]`, 완주 안 됨)을 이어, pos_weight를 0.05~1.00까지 0.05 단위(20개) × `weighted_sampler`(true/false) × `lr`(2개) = 80 트라이얼 grid. 탐색 단계라 `n_repeats=1`(최종 확정은 승자 조합만 `n_repeats=3`로 재실행 필요). wandb sweep `u38ay314`(project Berry2Vec) — 첫 트라이얼 실측 확인(pos_weight=0.05 적용됨, 정상 학습 진행).

## 전체 하이퍼파라미터 정리 (2026-07-24, 비대칭 데이터 탐색용)

`run_diagnosis_finetune_v03.py`의 CLI 인자 전체 + `config.py`의 dataclass 필드를 근거로 정리. "탐색됨"은 `conf/sweep/finetune_v03_s1*.yaml`에 실제로 grid로 들어간 것만 표시(설계만 되고 실행 안 된 것은 "미탐색"으로 표시).

### 1) 최적화 (모델 전체에 적용 — "layer별"이 아니라 옵티마이저 단위)
| 파라미터 | 기본값 | 설명 | 상태 |
|---|---|---|---|
| `--lr` | 3e-4 | AdamW 학습률. **스케줄러 없음**(warmup/decay 전혀 없음, 고정값 그대로 전 epoch 사용) | sweep1: {1e-4,3e-4} → sweep1c: 더 넓게 확장 |
| `--weight-decay` | 1e-2 | AdamW weight decay — 지금까지 항상 적용은 됐지만(고정값) **스윕된 적은 없음**. Anthropic "Superposition, Memorization, and Double Descent"가 이 값이 반복 샘플 암기(overfitting)를 줄인다고 보고 — 비대칭 데이터 탐색에 직접 관련 | sweep1c에서 신규 스윕 |
| `--batch-size` | 8 | 35개 train fold 샘플 기준 — 사실상 이미 매우 stochastic | 미탐색 |
| `--epochs` / `--early-stop-patience` / `--early-stop-min-epochs` | 150 / 20 / 30 | monitor(`val_loss`) 기준 조기종료 | 미탐색 |
| `--seed` | 2023 | | 미탐색 |

### 2) 클래스 불균형 처리 (이번 세션의 핵심 관심사)
| 파라미터 | 기본값 | 설명 | 상태 |
|---|---|---|---|
| `--pos-weight` | "auto" | BCE loss 재가중, `N_정상/N_비정상` 또는 절대값 | sweep1: {auto,0.25,0.5,1.0} → sweep1b/1c: **0.05~1.00, 0.05 단위(20개)** |
| `--weighted-sampler` | true | `WeightedRandomSampler` 오버샘플링(2026-07-24 신규 추가) | sweep1b 실측: **True가 평균 F1을 0.640→0.130으로 악화시킴(5배)** — sweep1c부터 **항상 False로 고정** |

### 3) Loss 항별 가중치 (`LossWeightsV03`)
| 파라미터 | 기본값 | 설명 | 상태 |
|---|---|---|---|
| `--lambda-fine` | 1.0 | fine(10-class) CE 가중치 | 미탐색(고정) |
| `--lambda-binary` | 1.0 | binary BCE 가중치 | sweep1: {0.5,1.0,2.0} 설계만 됨, 실행 안 됨(sweep1 winner 없음) |
| `--lambda-consistency` | 0.05 | fine-binary 일관성 loss(`--consistency-mode`: stopgrad_bce/js) | sweep1 설계만, 미실행 |
| `--lambda-supcon` | 0.0 | SupCon(z_proj 기반) — 기본 꺼짐 | 미탐색 |
| `--lambda-prototype` | 0.05 | prototype/center loss(z_proj 기반) | 미탐색 |
| `--supcon-temperature` | 0.1 | | 미탐색 |

### 4) 아키텍처 (`ArchitectureV03` + `EventPoolingConfig`) — 사실상 이 코드베이스에서 "layer"에 가장 가까운 축
| 파라미터 | 기본값 | 설명 | 상태 |
|---|---|---|---|
| `--hidden-dim` | 192 | task head의 hidden size | 미탐색 |
| `--proj-dim` | 128 | SupCon/prototype projection head 차원 | 미탐색 |
| `--num-heads` | 4 | attention pooling head 수 | 미탐색 |
| `--dropout` | 0.2 | task head 전역 dropout — 유일하게 이미 걸려 있는 정규화. **이번 세션에 실측**: dropout만으로는 `weighted_sampler=True`의 반복 입력 암기를 못 막았음(그래도 F1 0.13까지 무너짐) | 미탐색(값 자체는 안 바꿔봄) |
| `--task-specific-query` | true | fine/binary/proj 헤드별 독립 attention query 사용 여부 | 미탐색(고정) |
| `--attention-residual` | true | pooling residual 연결 | 미탐색(고정) |
| `--token-attention-pool` | false | 토큰 단위 attention pooling — **stub**(token cache 없으면 사실상 no-op, `finetune_v03/README.md` 기존 gap 목록 참조) | 미탐색 |
| `--use-image-adapter` | true | DINO 이미지 임베딩 융합 on/off | 미탐색(고정) |
| `--max-events` | 4096 | 이벤트 시퀀스 길이 cap — 실측: 전체 55케이스가 정확히 3928개 이벤트라 이 cap은 한 번도 실제로 발동 안 함(패딩/truncation 없음) | 해당 없음(고정 데이터 특성) |

### 5) Open-set / 라우팅 (`RoutingThresholds`, eval 단계 — 그래디언트에 직접 영향 없음)
| 파라미터 | 기본값 | 설명 | 상태 |
|---|---|---|---|
| `--use-reject` / `--use-open-set` | true / true | REJECT/UNKNOWN_ABNORMAL 라우팅 on/off | 미탐색 |
| `--tau-fine` | 0.5 | fine 판단 임계값 | 미탐색 |
| `--min-fold-agreement` | 4 | fold 간 합의 임계값(3-fold CV에서는 자동으로 min(3,4)=3으로 낮춤, `run_pre_llm_decision_batch_v03.py` 참조) | 미탐색 |
| `--binary-threshold` | 0.5 | 단일 평가 임계값 | sweep1 설계만, 미실행 |
| `--binary-thresholds` | "0.25,0.5,0.75" | threshold table(항상 다 계산, 재학습 불필요 — eval-only 그리드) | 해당 없음(재학습 안 필요) |

### 6) CV/데이터 분할
| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--n-folds` / `--n-repeats` / `--cv-seeds` | 3 / 3 / "2023,2024,2025" | `make_repeated_group_folds`(farm-그룹, stratified) — life2vec 논문의 단순 무작위 split보다 이미 보수적(위 "life2vec 논문 실측 확인" 절 참조) |

## Sweep 1c — pos_weight(0.05단위) × lr(확장) × weight_decay, weighted_sampler=False 고정 (2026-07-24)

사용자 요청("weighted sampler를 false로 유지, pos_weight은 0.05 단위로 확인, lr도 좀더 많은 영역을 탐색, weight decay도 학습에 적용")에 따라 `conf/sweep/finetune_v03_s1c_pos_weight_lr_wd.yaml` 신설:
- `weighted_sampler`: **항상 False**(sweep1b 실측 근거)
- `pos_weight`: 0.05~1.00, 0.05 단위 20개(동일)
- `lr`: sweep1b의 {1e-4,3e-4} 2개 → **{1e-5,3e-5,1e-4,3e-4,1e-3,3e-3} 6개로 확장**
- `weight_decay`: **신규 스윕 축**, {0.0, 1e-3, 1e-2, 1e-1} 4개
- 총 grid = 20×6×4 = **480 트라이얼**(n_repeats=1, 탐색 단계 — 승자 확정 시 n_repeats=3 재실행 필요)

## Stage A에 ZONE_LOCAL 접두 추가 + 새 코퍼스로 재학습 (2026-07-26)

사용자 지적("raw 데이터에는 farm id와 zone id가 있는데 토큰화된 이벤트에는 그 정보가 없음 / zone은 이벤트별로 달라서 포함이 되어야 할것 같음")에 따라 조사한 결과, 사전학습 시퀀스는 `expand_sequence_rows()`가 이벤트마다 `NARRATIVE`/`FARM_LOCAL`/`ZONE_LOCAL` 접두 토큰을 즉석에서 붙이는데, finetune Stage A 캐싱(`cache_stage_a_event_embeddings.py`)은 이 로직을 타지 않고 `events_tokenized_v2.parquet`의 원본 SENTENCE를 그대로 이어붙여서 **zone 정보가 finetune 모델 입력에 전혀 들어가지 않고 있었다**(케이스 하나가 실제로는 여러 zone에 걸치는데도). `zone_present()`/`CaseEvent.token_count`(+1 예약)/`window_to_tensors`(윈도우별로 그 안에 실제 등장하는 zone만 로컬 인덱스 재부여, 사전학습과 동일 컨벤션)를 추가해 해결 — 새로 만드는 게 아니라 이미 이 run의 vocab에 있던 `ZONE_LOCAL|0`~`7` 토큰을 재사용한다.

이 fix를 반영해 Stage A 캐시를 55케이스 전부 재생성(`outputs/online2/v2_finetune_v03_simplified/event_embeddings`, `full_event_grain_simplified_2026-07-26` run + `v2_build_expanded_full1517` vocab 기준)하고, sweep1c 승자 하이퍼파라미터(wandb에서 직접 재조회 — `lr=3e-4, weight_decay=0.01, pos_weight=0.55, weighted_sampler=false`, `cv/mean_best_macro_f1=0.7711`, 480 트라이얼 중 1위, 단 n_repeats=1)로 재학습(`runs/cv_repeated_zone_local_simplified_2026-07-26`, 기본값인 n_repeats=3로 — sweep1c 코멘트가 스스로 요구한 "승자 구간 n_repeats=3 재확인"을 겸함).

**결과를 있는 그대로 보고**: 새 재학습의 `val_fine_macro_f1` 평균(9개 fold-repeat) = **0.343(stdev=0.205, range 0.044~0.633)** — sweep1c가 보고한 0.7711보다 뚜렷이 낮다. 다만 두 숫자는 **조건이 다르다**: sweep1c의 0.7711은 n_repeats=1(fold 3개뿐인 단일 샘플, 노이즈에 취약 — 실제로 이번 9개 fold-repeat의 fold별 값도 0.044~0.633으로 매우 넓게 흩어짐), 이번 값은 새 인코더(다른 토큰화·vocab)로 만든 임베딩 기준. **즉 "새 코퍼스가 더 나쁘다"고 결론 내릴 근거가 아직 없다** — (1) 승자 hyperparameter가 애초에 노이즈였을 가능성과 (2) 새 인코더/임베딩이 실제로 이 downstream task에 덜 맞을 가능성이 뒤섞여 있어 분리가 안 됐다. 공정 비교를 하려면 **OLD 임베딩(`v2_finetune_v02`)으로도 동일하게 n_repeats=3을 돌려 노이즈 수준부터 확인**해야 한다 — 아직 안 함.

## 토큰 단위 IxG를 v0.3 모델에 연결 (2026-07-26)

`src/online2/v2/v01_layer_saliency.py::token_ixg_for_event`(v0.1, 동결 — 직접 수정 안 함)가 이미 정확히 이 문제(이벤트 pooling 이후엔 토큰별 gradient를 못 구하는 한계)를 푸는 방법을 구현해뒀던 걸 발견 — 관심 이벤트 하나만 grad 활성화 상태로 다시 인코딩해서 케이스 텐서에 이어붙이고, 진단 objective에서 그 이벤트의 입력 토큰 임베딩까지 역전파한다(근사가 아니라 진짜 objective→토큰 gradient). 다만 v0.1 시절 모델 인터페이스와 옛 토큰 스킴(FEATURE|/OBSERVED_VALUE| 분리) 기준이라 지금 그대로는 못 쓴다 — `run_token_attr_hook_v03.py`가 이 gap을 `next_wiring`으로 미리 적어두기만 하고 실제 연결은 안 돼 있던 상태.

새 `token_saliency.py`에 v0.3 모델(`EventPoolingDiagnosisModelV03.forward`, `risk_margin_from_logits`/`risk_from_binary_logit` 재사용) 인터페이스로 이식 + `classify_token_type`을 2026-07-26 결합 토큰(`FEATURE_NAME|value`) 스킴에 맞게 재작성(오히려 더 단순해짐 — `"|" in t` 하나면 측정값 토큰 전부 잡힘). 이식 도중 실제 버그 하나 발견·수정: 타겟 이벤트의 표시용 토큰 목록이 그 이벤트 자신의 `ZONE_LOCAL|i` 접두 토큰을 빼먹어서 모든 점수가 한 자리씩 밀려 있었음(`sentence_tokens`만 쓰고 실제 임베딩된 접두 토큰은 반영 안 함) — window_to_tensors와 동일하게 접두를 재구성하도록 수정.

방금 재학습한 새 체크포인트(`r0_fold0_best.pt`)로 실검증: 5개 이벤트에 대해 재인코딩한 풀링 벡터와 캐시된 값의 L2 오차(`pool_l2_err`)가 평균 **1.6e-7**(부동소수점 오차 수준, 사실상 0) — 재인코딩 경로가 캐싱 경로와 완전히 일치함을 확인. 토큰별 IxG 점수도 0이 아니고(1e-7~1e-6대) 이벤트마다 다른 순위로 나옴 — degenerate하지 않음.

## 남은 gap (2026-07-23 감사 기준)

- open-set 임계값 캘리브레이션 미완 (`DIAGNOSIS_FINETUNE_V03_FOLLOWUP.md` 체크리스트 #4)
- token-attention pooling은 stub (`token_attention_pool` 옵션 — token cache 없으면 no-op)
- §16 "열린 진단"(자유서술 감별진단 + DINO 유사사례 검색) 미착수
- §17 counterfactual 보상 기반 편집 루프는 문서만 있고 코드 없음 (`diagnosis_critic.py`가 risk_before/after 좁은 범위만 구현)
