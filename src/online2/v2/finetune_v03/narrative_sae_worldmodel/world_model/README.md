# world_model

**근거**: 계획서 §12 (월드모델 준비)
**Phase**: Phase 6 · **범위**: 조건부 확장 (§3.2, acceptance 통과 시에만) · **상태**: 미착수 — **착수 전 리스크 검토 필수**

## ⚠ 착수 전 필독 — §14 실측 신호 부재

CF1S 55건 검증에서 **44건이 `CONSTRUCTIBLE_SELECTED`까지 도달했으나 전부 `CONTROL_WITHIN_LOCKED_THRESHOLD`로 판정됐다** — 현재 2-event 편집 범위 안에서 임계값을 넘는 인과효과가 관측된 케이스가 0건이라는 뜻이다. 즉 action–response 학습 신호가 이미 알려진 편집 범위 안에서는 희박하다는 것이 실측으로 확인된 상태다. action coverage 확장(3+ event 편집, 다른 feature 조합)이 선행되지 않으면 이 모듈이 학습할 신호 자체가 부족할 수 있다. §12.3 승격 기준을 충족하지 못하면 **RL 환경으로 사용하지 않고 bounded edit의 보조 scorer로만 사용**한다 — 이것이 정상 결과다.

## 모델 역할 분리 (§12.1)

| 모델 | 역할 | life2vec 대응 |
|---|---|---|
| Representation model | 시퀀스 공통 표현 | Stage-A reencoder (기존, frozen) |
| Outcome model | 회귀 target 예측 | finetune critic (기존, fold 0/1/2) |
| Plausibility model | 편집된 시퀀스의 분포상 가능성 | 신규 |
| Dynamics model | 현재 상태→다음 상태 예측 | 신규 |
| Action-conditioned world model | 행동에 따른 다중 시점 상태전이 | 신규, §12.2 행동 데이터 필요 |
| Uncertainty model | epistemic/aleatoric uncertainty | 신규 |
| Constraint verifier | 물리·시간·ontology·권한 검사 | `disposition_profiles.py` + `core_verifier.py` G1–G12 (기존, 그대로 재사용) |
| Policy | 이벤트 편집 행동 선택 | `../edit_policy_rl/`에서 구현 |

encoder 공유는 허용하되 각 head의 loss·검증지표·사용권한은 분리한다.

## 행동 데이터 (§12.2) — 원시 소스 확인됨

`/data/datasets/agrichallenge/online2/data/A_actuator/*.csv` (존/농장별 파일)가 존재한다. 단 반응 지연·행동 전후 상태까지 정제된 형태인지는 미확인 — 이 모듈 1단계에서 가공이 필요하다.

## 구현 계획

1. `action_dataset.py` — A_actuator 원시 CSV에서 난방/냉방/환기창/차광/관수(시작·종료·양)/양액 EC·pH/농작업 이벤트 추출, 행동 시각과 반응 지연 페어링, 행동 전후 상태 기록.
2. `plausibility_model.py`, `dynamics_model.py`, `action_conditioned_model.py`, `uncertainty_model.py` — 각 head 별도 파일, loss·검증지표·사용권한 분리된 config.
3. `promotion_gate.py` — §12.3 승격 기준 7개 항목 자동 체크리스트: next-state가 persistence/seasonal baseline보다 우수 / multi-step rollout horizon별 오류 보고 / action conditioning이 실제 반응 차이 학습 / uncertainty calibration 통과 / OOD action·state 탐지 / 실제 trajectory와 rollout의 물리·시간 일관성 / development acceptance 고정 후 holdout(Validation20) blind 평가. 미충족 시 RL 환경 승격을 코드 레벨에서 차단.

## 의존성

- 기존: `cf1s/disposition_profiles.py`, `cf1s/core_verifier.py`(그대로 재사용), Stage-A reencoder, finetune critic
- 신규: `../representation_tracking/`, A_actuator 원시 데이터 가공

## Acceptance 연결

F1–F4 (계획서 §17-F)
