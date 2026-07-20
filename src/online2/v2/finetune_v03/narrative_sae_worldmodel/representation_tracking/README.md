# representation_tracking

**근거**: 계획서 §8.3 (표현 추적), `ActivationRef` 객체 (§4.1)
**Phase**: Phase 3 · **범위**: 현재 필수 (§3.1) · **상태**: 미착수

## 목적

미세조정 과정에서 4가지 표현 위치를 혼동하지 않고 분리 기록한다.

| 위치 | 의미 |
|---|---|
| Sequence position | 토큰의 시퀀스 내 위치 |
| Token embedding | vocab embedding table의 고정 벡터 (frozen `RegistryVocabulary`) |
| Contextual activation | checkpoint·layer·문맥에 따른 hidden state |
| SAE feature activation | contextual activation을 SAE로 분해한 희소 값 (`../sae/`가 생성) |

동일 토큰이라도 sequence, checkpoint, layer가 다르면 별도 activation으로 기록한다. life2vec 범위판이 명시한 대로, "미세조정 데이터 포인트가 사전학습 단어사전·가중치를 통한 위치와 어떻게 연결되는가"라는 요구는 이 모듈의 `Contextual activation` 기록 + `../representation_explorer/`의 3D projection으로 구현한다.

## 구현 계획

1. `schemas.py` — `ActivationRef`(checkpoint, layer, position, pooling, activation artifact 경로).
2. `capture.py` — finetune 실행 중 Stage-A/critic activation을 캡처해 `ActivationRef`로 저장. 기존 `event_mean`/`event_max` pooled output과 `core_canonical.stage_a_output_sha` 해시 체계를 그대로 연결점으로 사용.
3. `attribution.py` — §8.2 요구사항(modality별 ablation, event/token attribution, prediction head 입력 activation)을 이 캡처 결과 위에 구현.

## 의존성

- 기존: 기존 finetune critic(fold 0/1/2 출력), `cf1s/core_canonical.py`
- 하위 소비자: `../sae/`(activation 소스), `../representation_explorer/`(projection 입력)

## Acceptance 연결

C4 (계획서 §17-C)
