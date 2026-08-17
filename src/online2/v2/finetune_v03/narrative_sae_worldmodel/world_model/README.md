# world_model

**근거**: 계획서 §12 (월드모델 준비)
**Phase**: Phase 6 · **범위**: 조건부 확장 (§3.2, acceptance 통과 시에만) · **상태**: **부분 구현(2026-07-24)** — bounded-edit 보조 scorer 역할만, RL 승격은 `promotion_gate.py`로 코드 레벨 차단됨

## 범위 결정 (2026-07-24)

사용자에게 세 옵션(① 지금 항목만 진행 ② action coverage 확장까지 포함 ③ world_model 전체 보류)을 물었고, **①(권장)을 선택**했다 — dynamics/plausibility/action-conditioned/uncertainty 모델과 3+ event 편집(action coverage 확장)은 **이번 범위에서 명시적으로 제외**한다. 대신 아래 최소 구현만 한다:

1. [`bounded_edit_scorer.py`](bounded_edit_scorer.py) — 이미 완성된 `counterfactual/evaluation/diagnosis_critic.py`(fold-ensembled risk_before/after)를 이 트리에서도 찾을 수 있게 얇게 재노출. 새 계산 없음 — §12.1 role table의 "Outcome model" 역할이 사실 이미 구현돼 있었다는 걸 명시적으로 드러낸 것뿐이다.
2. [`promotion_gate.py`](promotion_gate.py) — §12.3 7개 승격 기준을 실제 코드로 강제. `PromotionDecision.__post_init__`가 기준 미충족 상태에서 `promoted=True`를 만들 수 없게 막고(생성 자체가 실패), `current_known_status()`는 **지금 실제 상태**(7개 전부 미통과, 근거: 아래 §14 실측)를 반환한다. dynamics/plausibility/uncertainty 모델을 나중에 만들더라도, 이 게이트에 진짜 증거를 넣기 전까지는 promoted=True를 구성할 수 없다.

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

1. `action_dataset.py` — A_actuator 원시 CSV에서 난방/냉방/환기창/차광/관수(시작·종료·양)/양액 EC·pH/농작업 이벤트 추출, 행동 시각과 반응 지연 페어링, 행동 전후 상태 기록. **이번 범위에서 제외** — action coverage 확장(3+ event 편집) 작업의 일부라 2026-07-24 범위 결정으로 다음 세션으로 미뤘다.
2. `plausibility_model.py`, `dynamics_model.py`, `action_conditioned_model.py`, `uncertainty_model.py` — **이번 범위에서 제외**, 같은 이유. 이 4개 없이는 `promotion_gate.py`의 7개 기준 중 5개(next_state_beats_baseline, multistep_rollout_error_reported, action_conditioning_learns_real_effect, uncertainty_calibration_passes, ood_detection_works, rollout_physically_consistent)에 진짜 증거를 채울 수 없다 — 게이트가 왜 지금 전부 FALSE인지의 직접적 이유다.
3. `promotion_gate.py` — **완료**. §12.3 승격 기준 7개 항목을 `evaluate_promotion_gate()`로 구현, `current_known_status()`가 실제 현재 상태(전부 미통과)를 반환. 합성 데이터로 fail-closed 강제(모든 기준 통과 없이는 `promoted=True` 구성 자체가 `ValueError`)와 all-pass 정상 경로 둘 다 실측 검증했다.

## 의존성

- 기존: `cf1s/disposition_profiles.py`, `cf1s/core_verifier.py`(그대로 재사용), Stage-A reencoder, finetune critic
- 신규: `../representation_tracking/`, A_actuator 원시 데이터 가공

## Acceptance 연결

F1–F4 (계획서 §17-F)
