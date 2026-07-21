# multimodal_pretrain

**근거**: 계획서 §7 (사전학습)
**Phase**: Phase 2 · **범위**: 현재 필수 (§3.1) · **상태**: 신규였던 3개 손실(time reconstruction/time-text contrastive/cross-modal matching) + 결합 로직 + pair 품질 등급 구현 완료(33개 테스트 통과). 실제 GPU 학습 루프 연결은 아직

## ⚠ 착수 전 점검 결과 — README의 원래 전제(`pipeline_m2.py`/`stage_a_reencoder.py` 확장)가 틀렸다

착수 전 조사 결과 두 가지가 밝혀졌다.

1. **`counterfactual/pipeline_m2.py`/`stage_a_reencoder.py`는 애초에 사전학습 코드가 아니었다.** 둘 다 CF1S 반사실적 설명(counterfactual explanation) 파이프라인 소속이다 — `pipeline_m2.py`의 "MLM"은 반사실 편집 후보를 생성하는 제약 디코더(`candidates/mlm_path_a.py`)이지 학습 목적함수가 아니고, `stage_a_reencoder.py`는 **frozen** 인코더로 이미 학습된 표현을 재인코딩만 한다(gradient 없음). 이 문서 최초 작성 시의 파일 경로 추정이 틀렸다.
2. **`L_MLM`과 `L_SOP`는 이미 구현되어 실제로 학습되고 있었다.** `src/tasks/grouped_mlm.py`의 `GroupedMLM`(online2 v2용, measurement-group-aware masking + SameTimeGroup 기반 SOP)과 `scripts/online2_v2/run_v2_pretrain_loop.py`가 `model.cls_w * sop_loss + model.mlm_w * mlm_loss`로 실제 학습 루프를 돌리고 있다(`conf/task/online2_v2_grouped_mlm.yaml`로 연결됨). 이걸 다시 만들지 않는다.

즉 §7.2의 5개 손실 중 실제로 신규인 건 3개뿐이었다:

| 손실 | 상태 | 실제 근거 |
|---|---|---|
| `MLM` | **이미 구현·학습 중** | `src/tasks/grouped_mlm.py` + `scripts/online2_v2/run_v2_pretrain_loop.py` — 재구현 안 함 |
| `SOP` | **이미 구현·학습 중** | 위와 동일 |
| `time reconstruction` | 신규, 구현 완료 | 저장소 어디에도 없었음(`losses.py`) |
| `time-text contrastive` | 신규, 구현 완료 | InfoNCE류 코드가 전혀 없었음(`losses.py`) |
| `cross-modal matching` | 신규, 구현 완료 | online1에 recall@K **평가** 스텁만 있었고 학습 손실은 없었음(`losses.py`) |

online1 쪽(`src/online1/pretrain_smoke.py`, `reports/ONLINE1_서사_CUBE_사전학습_및_회귀_수행계획서_20260720.md`)에 Cube latent 코사인 복원 손실과 alignment recall@K 프로토타입이 있었지만, online1/online2 격리 원칙(두 문서가 이미 "서로의 산출물은 참고용으로만 쓴다"고 합의)에 따라 코드는 재사용하지 않고 손실의 수학적 형태만 참고했다.

## 구현한 것

- [`losses.py`](losses.py) — 신규 3개 손실 + 공용 `multi_positive_info_nce`(time-text와 cross-modal 둘 다 이걸로 구현, 계획서 §7.3의 multi-positive 허용을 그대로 반영). 순수 torch 함수, CPU에서 테스트됨(14개 테스트).
- [`loss_composition.py`](loss_composition.py) — `PretrainLossWeights`(λ1..λ5) + `compose_total_loss(...)`. **MLM/SOP 손실 값은 여기서 재계산하지 않고 인자로만 받는다**(`run_v2_pretrain_loop.py`가 계산한 실제 값을 그대로 넣는다는 뜻). 가중치가 0보다 큰데 해당 손실 값을 안 주면 즉시 에러(fail-closed) — "이번 스텝에 깜빡했다"와 "의도적으로 껐다"를 구분한다. 새로 추가된 3개 손실은 기본 가중치 0.0(꺼짐)으로 시작 — 검증 안 된 손실을 기본으로 켜두지 않는다(8개 테스트).
- [`contrastive_pairs.py`](contrastive_pairs.py) — pair 품질 6단계(`GOLD_EXPERT`~`CONTRADICTED`)를 **새 신호 없이** 이번 세션에서 이미 만든 것들로 채운다: §5.2 규칙 기반 매칭(기본 `SILVER_RULE_GROUNDED`) → §5.1 검색 4분기 판정으로 승격/강등 → §5.3 사람 결정(`decisions.jsonl`)이 최종적으로 덮어씀(`GOLD_EXPERT`/`CONTRADICTED`). `find_hard_negatives`는 farm은 겹치지만 템플릿(의미)이 다른 후보를 골라 "farm ID만으로 구분되는 shortcut"을 막는다(11개 테스트).

## 아직 없는 것

- **실제 GPU 학습 루프 연결**: `loss_composition.compose_total_loss`를 `run_v2_pretrain_loop.py`에 실제로 꽂아 넣는 작업은 하지 않았다. 지금은 손실 계산 자체의 정확성만 순수 함수 단위로 보장된 상태다.
- **`sweep_config.py`/`wandb_logging.py`**: 학습 루프에 연결되기 전까지는 만들 이유가 없어서(스윕할 실행 자체가 없음) 보류했다. 연결 이후가 자연스러운 다음 단계.
- **`input_modality.py`(modality missingness mask)**: 실제 멀티모달 배치 구성(이미지/Cube embedding을 어떻게 배치에 태울지)이 `run_v2_pretrain_loop.py` 통합과 함께 결정돼야 해서 미룸.
- **shortcut 검증의 실측 부분**: `strip_identifying_tokens`는 전처리 유틸일 뿐이다. "farm ID를 지워도 여전히 정답을 맞히는가"를 실제로 검증하려면 학습된 time-text 인코더가 있어야 하는데, 아직 아무 손실도 실제로 학습된 적이 없다(순수 함수 검증만 함) — 닭과 달걀 문제, 학습 루프 연결 이후에나 가능.

## 리스크

원시 데이터 전체 27MB, 케이스 35건(train)/20건(holdout) 규모. 새로 추가하는 3개 손실이 이 규모에서 통계적으로 유의미한지는 착수 전 별도 검증 필요(§4.3 transductive 활용 범위 포함) — `loss_composition`이 기본 가중치를 0으로 둔 것도 이 리스크에 대한 방어다. 데이터가 부족하면 손실 일부(특히 cross-modal matching)를 꺼둔 채로 두는 것도 정상 결과다(§7.2 마지막 문장, `compose_total_loss`가 이미 이 선택을 1급으로 지원).

## 의존성

- 기존(재사용, 확장 아님): `src/tasks/grouped_mlm.py`, `scripts/online2_v2/run_v2_pretrain_loop.py`(MLM/SOP 값의 소스)
- 신규: `../narrative_grounding/`(pair 품질 신호 전부의 소스)
- 참고만 함(재사용 안 함): `src/online1/pretrain_smoke.py`, `reports/ONLINE1_서사_CUBE_사전학습_및_회귀_수행계획서_20260720.md`

## Acceptance 연결

C1–C4 (계획서 §17-C)
