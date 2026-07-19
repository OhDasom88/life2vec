# CF-1S Development3 재실행 및 다중 이벤트 편집 모듈 검증 수행계획서(초안)

## 1. 목적

본 계획의 목적은 CF-1S 2-event 다중 이벤트 편집 경로가 실제 GPU production 실행에서 계약대로 작동하는지 다시 검증하고, 검증 완료 후에만 Train35 및 Validation20 결과 추출로 확장할 수 있는 상태를 만드는 것이다.

이번 작업은 기존 `20260719T064717Z`의 수치 결과를 재승인하는 작업이 아니다. 기존 결과는 디버깅 비교용 잠정 evidence로만 보존하고, 수정된 코드와 새 run ID에서 생성된 evidence를 기준으로 판정한다.

## 2. 현재 상태와 공식 판정

```text
Development3 numeric result: 잠정적으로 재현됨
Development3 final contract verification: FAIL
35+20 full expansion: HOLD

development_execution_readiness: NOT_READY
development_evaluation_coverage: UNVERIFIED
scientific_result: PROVISIONAL_NOT_SUPPORTED × 3
highest_claim_level: NONE
```

기존 run의 GPU forward와 risk/delta 생성은 확인됐다. 그러나 completion과 verifier verdict 충돌, PRE lock 누락, 실제 load/IxG trace 부재, bundle-parent SHA 미연결, Fold2 logit 누락 때문에 최종 evidence로 승인하지 않는다.

## 3. 범위

### 3.1 이번 실제 실행 범위

```text
cohort: Development3
execution_scope: TWO_EVENT_ONLY
actual family: A / B / AB
three_event_execution_status: OUT_OF_SCOPE
```

3-event 규칙과 6개 순열은 unit/contract test만 유지한다. Development3 coverage와 scientific result에는 포함하지 않는다.

### 3.2 비범위

- Train35 및 Validation20 전체 GPU 실행
- 3-event ABC 실제 효과 검증
- 독립 holdout 주장
- 인과적 개입 효과 주장
- 농작업 추천 또는 실행 가능한 edit 출력
- threshold, candidate policy 또는 scientific rule의 성능 개선

이번 작업에서는 scientific 로직과 후보 정책을 변경하지 않는다. B1–B5 계약 결함만 교정한다.

## 4. 기존 evidence 보존

1. 기존 evidence 디렉터리, ZIP, receipt, completion 및 verdict를 수정·삭제·덮어쓰기하지 않는다.
2. `EVIDENCE_LINEAGE.json`에 다음 상태를 유지한다.

```json
{
  "artifact_status": "INVALIDATED_FOR_FINAL_CLAIM",
  "preservation_purpose": "DEBUGGING_AND_DELTA_COMPARISON",
  "superseded_by": null
}
```

3. 새 실행은 충돌 불가능한 run ID와 새 quarantine 경로를 사용한다.
4. 동일 run ID 또는 final 경로가 이미 존재하면 삭제·대체하지 않고 fail-closed한다.
5. 새 run이 최종 승격된 후에만 기존 lineage의 `superseded_by`를 외부 lineage update artifact로 갱신한다.

## 5. 수정 작업

## B1. Completion과 verifier verdict 단일화

### 대상

- `core_promotion.py`
- `core_orchestrator.py`
- completion/verdict 생성 스크립트

### 변경

1. G1–G12의 권위 있는 원천을 public-key-only verifier verdict 하나로 고정한다.
2. completion은 verifier verdict의 gate 결과를 복사만 하며 자체 PASS 계산을 금지한다.
3. `NOT_RUN`, `BLOCKED_BY_Gx`, `FAIL`을 PASS로 변환하지 못하게 한다.
4. 다음 조건을 모두 만족하지 않으면 `final_report_allowed=false`로 고정한다.

```text
G1–G12 == PASS
public_key_only_verification == PASS
final_package_immutable_readback == PASS
external_promotion_receipt == VALID
attested_evaluation_coverage != UNVERIFIED
```

5. completion과 verdict의 gate map, evidence root 및 run ID가 다르면 completion 발급을 거부한다.

### Acceptance B1

- verifier에서 `G3=NOT_RUN`인 fixture로 completion 생성 시 실패
- 모든 gate PASS fixture에서만 completion 생성
- completion gate map과 verdict gate map의 byte-equivalent canonical SHA 일치

## B2. PRE stable lock과 qualification artifact 완결

### 대상

- `core_locks.py`
- `core_authorization.py`
- `authorize_cf1s_development_smoke.py`
- `run_cf1s_core_development_production_v03.py`

### 변경

1. `FINAL_CODE_LOCK` 후 qualification test와 실제 preflight를 실행한다.
2. stable lock manifest에 다음 필드를 필수로 포함한다.

```text
code_tree_sha256
test_tree_sha256
policy_tree_sha256
qualification_test_log_sha
qualification_preflight_sha
qualification_test_node_id_manifest_sha
development_manifest_sha256
case_input_manifest_root_sha256
fold_routing_sha256
stage_a_checkpoint_manifest_sha256
critic_checkpoint_manifest_sha256
tokenizer_sha256
vocab_sha256
feature_schema_sha256
binning_registry_sha256
label_map_sha256
deterministic_runtime_policy_sha256
risk_head_contract_sha256
execution_scope
```

3. 필수 필드가 null, 빈 문자열 또는 placeholder이면 PRE authorization 발급을 거부한다.
4. PRE authorization은 개별 일부 lock이 아니라 `stable_lock_sha256` 전체를 서명한다.
5. 실행 직전 실제 observed stable lock을 새로 계산해 PRE의 SHA와 비교한다. authorization artifact의 값을 expected 값으로 재사용하지 않는다.
6. 실행 후 동일 stable lock을 재계산하고 PRE/POST exact equality를 검증한다.
7. runtime timestamp, PID, GPU free memory, duration, run ID 등은 별도 runtime observation으로 분리한다.

### 실제 preflight 필수 증거

- cold rebuild equivalence
- official batch field parity
- Stage-A–critic pairing
- fold isolation
- deterministic runtime
- risk–logit contract

단순 boolean 선언이나 파일명만 있는 preflight는 거부한다. 각 항목은 검사 대상 SHA, 관측값, tolerance, PASS/FAIL 및 생성 코드 SHA를 포함한다.

### Acceptance B2

- qualification 필드 null fixture의 authorization 발급 실패
- preflight placeholder fixture 거부
- observed code/policy/checkpoint/input drift 주입 시 PRE 소비 실패
- post lock drift 시 attestation 및 promotion 차단

## B3. 실제 IxG·Stage-A·critic load/forward trace 연결

### 대상

- `core_trace.py`
- `core_manifest.py`
- `core_runtime.py`
- `core_candidates.py`
- Stage-A/critic loader adapter
- IxG preselect adapter

### 변경

다음 실제 함수 경계에 global trace writer를 주입한다.

- Search fold별 IxG model load 및 forward
- Stage-A checkpoint resolve/load
- critic checkpoint resolve/load
- baseline/identity/candidate checkpoint forward
- pipeline invocation

각 operation은 다음 상태기계를 따라야 한다.

```text
REQUESTED
  → ALLOWED → STARTED → COMPLETED | FAILED
  → DENIED
```

DENIED는 side effect 전에 기록하고, DENIED 이후 STARTED/COMPLETED가 없어야 한다. 각 event는 global monotonic sequence, previous-event SHA, event SHA, operation ID, parent invocation ID, phase, scope, case, candidate, transaction SHA, fold, checkpoint SHA, Stage-A pairing SHA, closure SHA/state와 failure code를 포함한다.

### 실제 count 규칙

```text
checkpoint_forward_count = CHECKPOINT_FORWARD + COMPLETED
pipeline_invocation_count = PIPELINE_INVOCATION + COMPLETED
attribution_forward_count = ATTRIBUTION_FORWARD + COMPLETED
stage_a_load_count = STAGE_A_LOAD + COMPLETED
critic_load_count = CRITIC_LOAD + COMPLETED
attribution_load_count = ATTRIBUTION_MODEL_LOAD + COMPLETED
```

FAILED와 DENIED는 실제 성공 count에 포함하지 않는다.

### Development3 필수 관측

```text
attribution_forward_count > 0
attribution_load_count > 0
stage_a_load_count > 0
critic_load_count > 0
fold2_attribution_load_before_closure_count == 0
fold2_critic_load_before_closure_count == 0
nonrequested_fold_forward_count == 0
broken_trace_chain_count == 0
```

실제로 수행된 operation의 count가 0이면 G3는 PASS할 수 없다.

### Acceptance B3

- 실제 단일 case GPU smoke에서 위 load/forward count nonzero
- fold2 pre-closure load 요청을 side effect 전에 DENIED
- missing, duplicate, terminal-after-terminal 및 broken hash-chain fixture 거부
- summary count와 trace 재계산 count 불일치 시 package 폐기

## B4. Bundle과 exact parent transaction 연결

### 대상

- `core_raw_transaction.py`
- `core_selection.py`
- `core_orchestrator.py`
- candidate ledger builder

### 변경

1. 선택된 AB bundle ledger에 정확한 A/B canonical transaction SHA를 기록한다.

```json
{
  "parent_transaction_shas": ["<A_SHA>", "<B_SHA>"]
}
```

2. 다음 불변식을 검증한다.

```text
AB atomic set == exact union(A atomic set, B atomic set)
AB parent SHA set == frozen closure parent SHA set
incremental 계산 parent SHA set == ledger parent SHA set
```

3. A/B 순서는 canonical atomic key로 정렬하며 제출 순서는 별도 provenance에만 둔다.
4. parent SHA 누락·중복·다른 feature/event parent 연결은 G7 FAIL 및 해당 bundle `NOT_EVALUABLE`로 처리한다.

### Acceptance B4

- Development3 모든 selected AB의 parent SHA가 정확히 2개
- parent union/closure/incremental reference exact match
- 빈 parent list fixture와 wrong-parent fixture 거부

## B5. Search/Fold2 risk·logit ledger 완결

### 대상

- `core_output.py`
- candidate ledger builder
- `core_scientific.py`

### 변경

각 candidate/parent/bundle과 fold에 다음 값을 저장한다.

```text
search baseline risk/logit
search candidate risk/logit
selection-blind Fold2 baseline risk/logit
selection-blind Fold2 candidate risk/logit
delta risk/logit
checkpoint SHA
semantic critic input SHA
trace reference
```

ledger에서 다음 계약을 재검증한다.

```text
abs(risk - sigmoid(abnormal_logit)) <= 1e-6
delta == candidate - matching-fold baseline
summary delta == ledger-derived delta
```

logit 누락, cross-fold baseline 비교, risk–logit mismatch는 해당 scope `NOT_EVALUABLE`로 처리한다.

### Acceptance B5

- Search 0/1 및 Fold2 2의 baseline/candidate risk+logit 모두 존재
- ledger-only 재계산 결과가 case summary와 exact 일치
- Fold2 logit 누락 fixture 거부

## 6. Development3 재실행 전 검증

## 6.1 Unit/contract/integration

최소 테스트:

- completion이 verifier verdict를 덮어쓰지 못함
- stable lock 필수 SHA null/placeholder 거부
- observed PRE/POST drift 차단
- 실제 load/IxG trace와 trace-derived count
- typed transaction 및 bare dictionary 차단
- AB exact parent SHA 연결
- Search/Fold2 risk·logit ledger 재계산
- frozen closure 이후 candidate/parent 변경 차단
- unsigned/invalid attestation promotion 차단
- 기존 evidence byte 불변 및 run ID 충돌 차단

## 6.2 FINAL_CODE_LOCK lifecycle

```text
B1–B5 구현 완료
→ FINAL_CODE_LOCK
→ qualification tests/preflight
→ stable PRE lock 계산
→ PRE authorization 발급
→ observed PRE lock 검증
→ 동일 node-ID 전체 test rerun
→ GPU stage smoke
→ Development3 실행
→ post stable lock 검증
→ immutable evidence root
→ POST attestation
→ public-key-only verifier
→ same-filesystem atomic no-replace promotion
→ final readback
→ external receipt
→ verifier-derived final verdict/completion
```

Final lock 이후 source/test/policy 변경 또는 qualification/final node-ID manifest 불일치가 발생하면 모든 새 test/GPU evidence를 폐기하고 FINAL_CODE_LOCK부터 다시 실행한다.

## 7. Development3 실행 및 승인 게이트

### 7.1 새 run 원칙

- 새 Development3 manifest와 새 run ID 사용
- 기존 run과 동일한 3 case 및 동일 policy를 사용해 비교 가능성 유지
- threshold와 candidate generation/scientific policy는 변경하지 않음
- 실행은 RTX 3090 단일 deterministic writer
- final 경로가 존재하면 fail-closed

### 7.2 G1–G12

| Gate | 승인 조건 |
|---|---|
| G1 Transaction | typed transaction, Gate0/4, change-set, atomic evidence 및 SHA chain |
| G2 Runtime | official Stage-A/batch parity, deterministic runtime, fold pairing |
| G3 Trace | 실제 IxG/Stage-A/critic load 및 checkpoint forward trace |
| G4 Evidence | ledger에서 summary/delta/status 완전 재계산 |
| G5 Authorization | non-null full stable PRE lock 및 observed lock 일치 |
| G6 Selection | Search 2/2 material/control/mixed 판정 및 deterministic manifest |
| G7 Closure | selected AB와 exact A/B parent SHA 동결 |
| G8 Re-evaluation | Fold2 frozen replay만 수행, pre-closure load/forward 0 |
| G9 Scientific | material/control/incremental 분리 및 ledger-derived 판정 |
| G10 Attestation | post lock, evidence root, detached signature, public-key verifier |
| G11 Promotion | atomic no-replace rename, immutable readback, 외부 receipt |
| G12 Final rerun | final lock 이후 qualification·동일 tests·GPU·Dev3 연속 실행 |

gate 상태는 `PASS|FAIL|BLOCKED_BY_Gx|NOT_RUN`만 허용한다. completion은 verifier verdict의 상태를 그대로 사용한다.

### 7.3 최종 승인 조건

```text
G1–G12 == PASS
completion gate map == verifier gate map
qualification SHA fields != null
PRE signed stable_lock_sha == observed pre/post stable_lock_sha
IxG/Stage-A/critic actual load count > 0
selected AB parent_transaction_shas count == 2
Search/Fold2 baseline·candidate risk/logit complete
immutable readback == PASS
external receipt == VALID
```

모두 통과한 경우에만 다음 값을 새 evidence에서 계산한다.

```text
development_execution_readiness
attested_evaluation_coverage
development_scientific_result
highest_claim_level
```

결과를 `NOT_SUPPORTED×3`으로 미리 확정하지 않는다.

하나라도 실패하면:

```text
development_execution_readiness: NOT_READY
development_evaluation_coverage: UNVERIFIED
highest_claim_level: NONE
numeric result: PROVISIONAL_* only
35+20 expansion: HOLD
```

## 8. 기존 잠정 결과 비교

새 run이 G1–G12 PASS한 후에만 기존 run과 비교한다.

candidate identity를 먼저 다음으로 분류한다.

```text
EXACT_CANONICAL_TRANSACTION_MATCH
EVENT_FEATURE_TARGET_MATCH
EVENT_FEATURE_ONLY_MATCH
NO_COMPARABLE_CANDIDATE
```

그 후 delta 비교를 분류한다.

```text
EXACT_MATCH
NUMERICALLY_CLOSE
DIRECTION_ONLY_MATCH
MATERIALITY_CLASS_MATCH
DIVERGED_EXPECTED_DUE_TO_PIPELINE_CORRECTION
UNEXPLAINED_DIVERGENCE
```

기존 bare transaction과 신규 canonical transaction SHA가 다르다는 이유만으로 unexplained divergence로 판정하지 않는다.

## 9. 35+20 확장 전환 조건

Development3의 실제 verifier verdict가 G1–G12 PASS일 때만 다음 단계로 전환한다.

1. `VALIDATION20`을 우선 실행한다.
2. Validation20 final package가 승격된 후 `TRAIN35_DIAGNOSTIC`을 실행한다.
3. 두 cohort는 manifest, run ID, threshold provenance, evidence root, receipt 및 보고서를 분리한다.

### Validation20 provenance

```text
finetune_training_seen: false
pretraining_seen: 실제 transductive 사용 여부에 따라 명시
scientifically_independent_holdout: false
selection_blind_to_fold2: true
```

### Train35 provenance

```text
finetune_training_seen: true
scientifically_independent_holdout: false
interpretation: training-seen diagnostic model-response evidence
```

Threshold는 사전 지정된 Train35 calibration case의 Search baseline repeat/identity noise로 한 번만 고정하고 Validation20 결과로 변경하지 않는다.

## 10. 시간 제한 운영

### 권장 우선순위

```text
1. B1–B5 수정 및 테스트
2. Development3 계약 재실행
3. Validation20 전체 실행
4. 남는 시간에 Train35 실행
```

### 예상 시간

| 단계 | 예상 |
|---|---:|
| B1–B5 수정·테스트 | 30–90분 |
| Development3 재실행 | 15–25분 |
| Validation20 | 1.5–2.5시간 |
| Train35 | 2.5–4시간 |

### 중단 기준

- B1–B5 테스트 실패: Development3 GPU 실행 금지
- Development3 G1–G12 중 하나라도 미통과: 55 case 확대 금지
- Validation20 package 미승격: Train35 실행은 가능하더라도 최종 우선순위를 낮추고 별도 diagnostic으로만 수행
- deadline까지 Validation20 완료가 불가능하면 완료 case만 provisional package로 보존하고 coverage를 PARTIAL 또는 UNVERIFIED로 보고

## 11. 최종 산출물

- Development3 verifier verdict
- Development3 completion(단, verifier PASS 시에만)
- candidate ledger 및 global trace
- PRE/POST stable lock과 authorization/attestation
- immutable final package 및 외부 receipt
- 기존/신규 evidence comparison
- 35+20 확장 GO/HOLD 결정서

## 12. 최종 선언

본 계획은 기존 수치 결과를 승인하기 위한 것이 아니라, 다중 이벤트 편집 모듈의 production transaction·trace·evidence·authorization 계약을 먼저 복구하고 Development3에서 검증하기 위한 계획이다. G1–G12가 실제 verifier에서 모두 PASS하기 전에는 READY, COMPLETE 또는 확정 scientific result를 주장하지 않으며 35+20 전체 실행을 시작하지 않는다.
