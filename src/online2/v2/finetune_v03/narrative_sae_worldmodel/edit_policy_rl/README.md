# edit_policy_rl

**근거**: 계획서 §13 (이벤트 편집과 RL)
**Phase**: Phase 7 · **범위**: 조건부 확장 (§3.2, acceptance 통과 시에만) · **상태**: 미착수 — 1~2단계는 신규 구현이 **아님**

## 1~2단계는 기존 CF1S를 그대로 쓴다

- 단일 이벤트 perturbation → 이미 CF1S `ATOMIC` 편집 (`cf1s/core_candidates.py`, `candidates/cf1s_composer.py`)
- ontology 제한 bounded multi-event search → 이미 CF1S `PAIR` 편집(`TWO_EVENT_ONLY`)

이 디렉토리는 **3단계(outcome/plausibility/dynamics 비교)부터** 신규 구현이다. RL이 찾은 편집 후보는 반드시 CF1S 기존 검증(잠긴 임계값, fold 격리, holdout 재평가)을 통과해야 "실제 효과"로 인정한다 — RL은 CF1S를 대체하지 않고 **CF1S 이전 단계의 후보 생성기**로 위치시킨다.

## ⚠ world_model과 동일한 신호 부재 리스크 상속

`../world_model/README.md`의 §14 실측(55건 중 임계값 초과 인과효과 0건)이 그대로 적용된다. `world_model/`의 §12.3 승격 기준을 통과하지 못하면 이 모듈은 offline RL까지 가지 않고 bounded search + preference ranking 단계에서 멈춘다 — G6("데이터 부족 시 RL 미적용 결정도 정상 결과")을 실제 판정 기준으로 실행할 것.

## State / Action (§13.2)

- State: sequence representation, 생육단계, farm/zone, 예측+불확실성, 편집 가능 event, raw inversion 정보, 이전 편집 이력
- Action: `NO_OP | SELECT_EVENT | MODIFY | INSERT | DELETE | SPLIT | MERGE | REWINDOW` — `NO_OP`는 항상 후보에 포함

## 구현 계획

1. `state_action_schema.py` — 위 State/Action 스키마.
2. `soft_score.py` — 학습 가능 soft score 8종: 목표 예측 변화 / 개연성 / 서사 일관성 / 편집 최소성 / 정상 trajectory 유사성 / 전문가 선호 / epistemic uncertainty / SAE feature transition의 자연스러움.
3. `hard_constraint.py` — 7종 hard constraint(물리적 범위/시간 역전 방지/생육단계 순서/sensor·actuator 타입/ontology dependency·conflict/원시값 복원 가능성/split 접근 권한과 누수 방지)는 `CoreContractError` 패턴을 재사용해 **음수 reward가 아니라 즉시 REJECT**로 처리(`cf1s/disposition_profiles.py`, `core_raw_transaction.py`의 `assert_change_set_closed` 확장).
4. `ood_gate.py` — 5단계 OOD 정책: `IN_SUPPORT`(허용) / `NEAR_SUPPORT_LOW_UNCERTAINTY`(제한 rollout) / `NEAR_SUPPORT_HIGH_UNCERTAINTY`(보류·검토) / `OUT_OF_SUPPORT`(차단) / `PHYSICAL_INVALID`(즉시 차단).
5. `preference_collection.py` → `imitation.py` → `offline_policy_eval.py` → `conservative_offline_rl.py` — §13.1의 4~7단계를 순서대로 구현.
6. `sae_linkage.py` — 편집 전후 raw/event/token diff, outcome·uncertainty delta, SAE feature activation delta, 새 feature combination, ablation/steering 결과, policy가 반복 악용하는 특징 기록. 새 feature 조합은 자동으로 비현실적이라 확정하지 않고 `NOVEL_FEATURE_COMBINATION`으로 분류해 uncertainty·raw constraint·전문가 검토를 거친다.

## 의존성

- 기존: `cf1s/core_candidates.py`, `candidates/cf1s_composer.py`, `cf1s/disposition_profiles.py`, `cf1s/core_raw_transaction.py`
- 신규: `../world_model/`(dynamics/plausibility), `../sae/`(§13.5 연결)

## Acceptance 연결

G1–G6 (계획서 §17-G)
