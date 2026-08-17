# concept_tcav

**근거**: 사용자 요청("UI에서는 concept space, tcav, entity summary를 한눈에 확인") — 계획서에는 TCAV가 명시돼 있지 않지만, 감사 결과 `analysis/tcav/`(원본 life2vec, 이 저장소 내부, 외부 코드 아님)에 이미 구현돼 있었음을 확인하고 이식했다.
**Phase**: 신규 (Phase 4 SAE / Phase 5 concept_governance와 인접) · **상태**: 핵심 모듈 구현 완료 + 실제 체크포인트로 end-to-end 검증 완료

## 이식 내역 — 무엇을 재사용했고 무엇을 새로 짰는지

`analysis/tcav/utils.py::TCAV`에서 **프레임워크-독립적인 핵심 수식만** 그대로 가져왔다: concept/random activation을 bootstrap-ensembled linear classifier(`BaggingClassifier` + 기본 `LogisticRegression`)로 분리해 CAV(`coef_`)를 얻고, 예측 민감도 gradient와 CAV의 부호 일치율로 TCAV score를 낸다(`calculate_cavs`/`calculate_score`와 동일한 산술 — [cav.py](cav.py)에서 합성 데이터로 방향 정렬/역정렬/무작위 케이스 3가지를 실측 검증했다).

captum(`LayerActivation`, `LayerGradientXActivation`) 기반 activation 추출부는 원본 `TransformerEncoder.forward_with_embeddings` API 전용이라 online2의 `EventPoolingDiagnosisModelV03`에는 그대로 안 맞는다 — 대신 [event_gradients.py](event_gradients.py)에서 `risk_saliency.py::event_ixg_abnormal_margin`과 같은 forward/backward 경로를 재사용하되, IxG로 곱해서 반환하는 대신 활성값 `z`와 gradient `z.grad`를 분리해서 반환하도록 변형했다.

## 파일 맵

| 파일 | 역할 |
|---|---|
| [`cav.py`](cav.py) | `train_cavs`/`tcav_score` — 프레임워크 독립적인 CAV 학습 + TCAV score 산술. `ConceptActivationVectors`/`TCAVScore`는 `representation_explorer/metadata.py`와 같은 원칙으로 최소 표본 수(`n_concept`/`n_random` >= 2)를 `__post_init__`에서 강제한다. |
| [`event_gradients.py`](event_gradients.py) | `event_activation_and_gradient` — 이벤트별 `z`(활성값)와 `z.grad`(binary/fine abnormal-risk objective에 대한 gradient)를 분리 반환. |
| [`narrative_concept_probe.py`](narrative_concept_probe.py) | `run_narrative_tcav` — narrative catalog의 `narrative_id` 매치를 concept 정의로 삼아(매치=concept, 비매치=random) 55개 finetune 케이스 전체에서 CAV→score를 계산하는 end-to-end 함수. |

## concept 정의: narrative catalog를 그대로 쓴다

새 concept 라벨링 체계를 만들지 않고, 이미 있는 `training_events_v2.parquet`의 `narrative_id` 매치를 concept 정의로 재사용했다 — matcher가 이미 "이 이벤트가 이 패턴에 해당하는가"의 진리조건 역할을 하고 있으므로(§2 평가방법론 문헌조사에서 이미 정리한 것과 같은 논리), 별도 사람 라벨링 없이 바로 TCAV를 돌릴 수 있다.

## 실측 검증 (2026-07-23)

`probe_oof_3fold` fold0 체크포인트(정상 학습된 실제 binary head) + fold0 val 12개 케이스 + narrative `A05`(가장 흔한 서사, 매치 이벤트 2.5M+)로 end-to-end 실행:

```
CAV: n_concept=30630 n_random=16506, coefs shape (50, 192)
TCAV score sign_mean=0.267 sign_std=0.072, n_examples=30630
```

즉 A05 매치 이벤트에서 abnormal-risk gradient가 A05 concept 방향과 반대로 정렬되는 경우가 더 많다(0.267 < 0.5) — A05 패턴이 나타난 이벤트일수록 모델이 그 지점을 "정상 쪽으로 미는" 신호로 쓴다는 뜻일 수 있다. **이 해석은 `E1_ASSOCIATED`/`E2_PREDICTIVE` 등급까지만이다** — `narrative_sae_worldmodel/README.md`의 §9.7 증거 등급 체계를 그대로 따라, ablation/steering으로 재현하기 전까지(`E3_INTERVENTION`) 인과적 주장을 하지 않는다.

## 의존성 / 필요했던 선행 변경

- `src/online2/v2/diagnosis_dataset.py::DiagnosisEventDataset.__getitem__`에 `event_ids` 필드를 추가했다(기존엔 캐시 parquet에 `event_id` 컬럼이 있는데도 sample dict에서 빠져 있었음) — TCAV concept 분리에 이벤트별 id가 필수라 이 변경 없이는 narrative 매치와 activation을 이어붙일 수 없었다. `collate_diagnosis_batch*`는 명시적 키만 골라 쓰는 구조라 이 추가로 인한 회귀는 없음(smoke 테스트로 확인).
- 기존: `risk_saliency.py`(forward/backward 경로), `finetune_v03/checkpoint.py::load_fold_model_v03`, `training_events_v2.parquet`의 `narrative_id`.

## 아직 없는 것

- SAE feature를 concept으로 쓰는 경로(현재는 narrative_id만) — `../sae/`가 이벤트별 feature activation을 노출하면 바로 연결 가능.
- UI 연결(`../ui/`) — 이 모듈은 계산 계층만 구현했다. `pipeline_explorer`에 탭으로 붙이는 작업은 별도([`../ui/pipeline_explorer/README.md`](../ui/pipeline_explorer/README.md) 참조).
