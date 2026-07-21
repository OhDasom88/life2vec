# representation_tracking

**근거**: 계획서 §8.3 (표현 추적), `ActivationRef` 객체 (§4.1)
**Phase**: Phase 3 · **범위**: 현재 필수 (§3.1) · **상태**: 구현 완료(스키마·캡처 어댑터·modality ablation), 22개 테스트 통과 + 실제 캐시 데이터로 검증

## 목적

미세조정 과정에서 4가지 표현 위치를 혼동하지 않고 분리 기록한다.

| 위치 | 의미 |
|---|---|
| Sequence position | 토큰의 시퀀스 내 위치 |
| Token embedding | vocab embedding table의 고정 벡터 (frozen `RegistryVocabulary`) |
| Contextual activation | checkpoint·layer·문맥에 따른 hidden state |
| SAE feature activation | contextual activation을 SAE로 분해한 희소 값 (`../sae/`가 생성) |

동일 토큰이라도 sequence, checkpoint, layer가 다르면 별도 activation으로 기록한다.

## ⚠ 착수 전 점검 결과 — 하위 재료가 이미 상당 부분 있었다

1. **`get_sequence_embedding()` vs `forward_finetuning()` 구분이 이미 있다.** `src/transformer/transformer.py`가 고정 vocab lookup(`get_sequence_embedding`, Token embedding)과 문맥별 hidden state(`forward_finetuning`/`forward_finetuning_with_embeddings`, Contextual activation)를 이미 분리해서 계산한다 — 이 모듈은 그 구분을 재발명하지 않고 타입(`RepresentationKind`)으로만 강제한다.
2. **Contextual activation 캡처 자체가 이미 있다.** `scripts/online2_v2/cache_stage_a_event_embeddings.py`가 Stage-A pooled activation(`event_mean`/`event_max`, 384차원)을 이미 parquet으로 캐시하고 있다. 실측 확인: `outputs/online2/v2_finetune_v02/event_embeddings/`에 55개 case 전부(예: `F354848_2025-03-25_2025-04-07.parquet` 3,928행) 캐시돼 있음. 다만 **checkpoint_id가 데이터 컬럼이 아니라 `--ckpt` 실행 인자에만 있어서 기록되지 않는다** — 이게 실제로 메꾼 구멍이다.
3. **event/token attribution도 이미 있다.** `counterfactual/attribution/token_ixg_v03.py`가 gradient 기반 IxG로, `aggregation.py`/`fold_consensus.py`가 역할 가중 집계·fold 간 consensus를 이미 계산한다. diagnosis 목적(binary abnormal logit)에 한정되지만 재구현하지 않는다.
4. **원래 README가 언급한 `core_canonical.stage_a_output_sha`는 "연결점"이 아니라 해시일 뿐이었다.** `event_mean`/`event_max` 값을 저장/조회하지 않고 lineage 검증용 해시만 계산한다 — 실제 activation 값을 가져오려면 parquet을 직접 읽어야 한다(`capture.py`가 하는 일).
5. **정말 없었던 건 modality 단위 ablation뿐이었다**(grep으로 확인). event/token attribution과 헷갈리지 않게 이것만 새로 만들었다.

## 구현한 것

- [`schemas.py`](schemas.py) — `RepresentationKind`(`SEQUENCE_POSITION`/`TOKEN_EMBEDDING`/`CONTEXTUAL_ACTIVATION`/`SAE_FEATURE_ACTIVATION`) + `ActivationRef`. `TOKEN_EMBEDDING`/`SEQUENCE_POSITION`은 `pooling="none"`을 `__post_init__`에서 강제한다 — "pooled activation을 고정 embedding으로 착각"하는 버그를 타입 레벨에서 막는다. `identity_key`는 (kind, checkpoint, layer, sequence, position, pooling)의 해시라 이 중 하나만 달라도 다른 activation으로 취급된다(§8.3 원문 요구사항). `artifact_ref`(저장 위치)는 정체성에 안 들어간다 — 같은 activation을 다른 곳에 다시 저장해도 논리적으로는 같은 것이라서다.
- [`capture.py`](capture.py) — `activation_refs_from_stage_a_parquet(path, checkpoint_id=...)`: 기존 캐시 parquet을 읽어 행마다 (mean, max) `ActivationRef` 쌍을 만든다. `load_activation_vector(ref)`로 실제 384차원 벡터를 다시 읽어올 수 있다. 실제 캐시 파일(`v2_finetune_v02/event_embeddings/*.parquet`)로 end-to-end 확인함(커밋에는 합성 fixture 테스트만 포함, 실측은 대화형으로 검증).
- [`attribution.py`](attribution.py) — `ModalityAblationRecord` + `summarize_modality_ablation`(modality별 n/mean_delta/mean_absolute_delta 집계) + `find_dead_modalities`(§7.4 "dead modality" 로깅 항목과 연결) + `rank_modalities_by_contribution`. **모델을 modality 제외하고 다시 순전파하는 건 여기서 안 한다** — 그건 실제 학습된 멀티모달 모델이 있어야 하는데(`../multimodal_pretrain/`가 아직 손실 함수만 있고 학습 루프에 안 붙었음) 아직 없다. 이미 계산된 두 예측값(포함/제외)을 받아 delta만 계산·집계하는 순수 함수다.

## 아직 없는 것

- `checkpoint_id`를 실행 시점에 자동으로 알아내는 코드(지금은 호출자가 직접 넘겨야 함) — `cache_stage_a_event_embeddings.py`가 컬럼에 안 남기므로, 그 스크립트 자체를 고치거나 실행 로그에서 매핑을 만드는 작업이 필요.
- modality ablation을 실제로 "돌리는" 코드(모델 두 번 순전파) — `../multimodal_pretrain/`의 GPU 학습 루프 연결 이후에나 가능.
- SAE_FEATURE_ACTIVATION을 실제로 채우는 생성처 — `../sae/`가 아직 미착수.

## 의존성

- 기존(재사용): `src/transformer/transformer.py`(embedding/activation 구분 primitive), `scripts/online2_v2/cache_stage_a_event_embeddings.py`(activation 캡처 산출물), `counterfactual/attribution/`(event/token attribution, 재구현 안 함)
- 참고만 함(해시 용도로만 겹침): `cf1s/core_canonical.py`의 `stage_a_output_sha`
- 하위 소비자: `../sae/`(activation 소스), `../representation_explorer/`(projection 입력)

## Acceptance 연결

C4 (계획서 §17-C)
