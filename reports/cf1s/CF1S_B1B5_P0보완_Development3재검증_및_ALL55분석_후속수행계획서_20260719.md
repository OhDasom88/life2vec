# CF1S B1–B5 P0 보완, Development3 재검증 및 ALL55 분석 후속 수행계획서

| 항목 | 값 |
|---|---|
| 문서 성격 | Delta remediation / execution control plan |
| 문서 상태 | `APPROVED FOR IMPLEMENTATION` (본 개정 반영 완료 기준) |
| 기준일 | 2026-07-19 |
| 기준 정본 | `CF1S_Development3_재실행_및_다중이벤트_편집모듈_검증_수행계획서.md` |
| 구현 스냅샷 | `CF1S_B1B5_IMPL_PACKAGE_20260719T094800Z` |
| 범위 | P0-1~P0-6, Development3, Validation20, Primary32, ALL55 층화 분석 |

## 1. 목적과 권위 범위

본 문서는 기존 정본을 대체하지 않는다. B1–B5 구현 스냅샷에서 추가로 확인된 verifier 독립성 결함을 폐쇄하고, 실제 GPU 재검증부터 55개 case의 다중 이벤트 편집 결과 추출·층화 분석까지 연결하는 후속 통제 문서다.

기존 정본의 다음 정책은 변경하지 않는다.

- `TWO_EVENT_ONLY`
- Development3에서 사전 잠금한 risk-scale materiality threshold
- candidate 생성·selection policy
- control-only `NOT_EVALUATED` 의미
- execution / material-effect / control coverage 분리
- 인과적 효과·추천·독립 holdout 과대 주장 금지

## 2. 현재 공식 상태

```text
document_status = APPROVED FOR IMPLEMENTATION
P0 completion = PARTIAL
Development3 = HOLD
Validation20 = HOLD
Primary32 = HOLD
ALL55 aggregate = NOT_ISSUED
scientific result = 기존 잠정값 유지
highest_claim_level = NONE
```

기존 `119 passed in 3.53s, exit_code=0`은 구현 스냅샷의 회귀 기록인 `PASS_REPORTED_NOT_QUALIFICATION`이다. cohort별 D-LOCK/E-LOCK에 결합된 qualification evidence, G1–G12 PASS 또는 실행 READY의 근거로 사용하지 않는다.

P0-6과 machine-readable G12 구현·검증이 끝나기 전에는 `DEVELOPMENT3_FINAL_CODE_LOCK`으로 진행하지 않는다. 문서 승인은 실행 성공이나 P0 완료를 의미하지 않는다.

## 3. 집합과 명칭

```text
Development3 = 3
Primary32 = 32
Train35 reporting set = Development3 ∪ Primary32
Validation20 = 20
ALL55_STRATIFIED_ANALYSIS_SET
  = Development3 ∪ Primary32 ∪ Validation20
```

- `Train35`는 35건 reporting set이며 ALL55가 아니다.
- `Train35(55-case)` 표현을 금지한다.
- Train35와 ALL55는 실행 package가 아니다.
- 두 집계는 source cohort package를 SHA로 참조하는 외부 reference manifest만 생성한다.
- case evidence 복사, package pooling, pooled evidence root 생성을 금지한다.

## 4. P0 보완 계약

### 4.1 P0-1 — G10 final destination 독립 계산

verifier는 POST artifact의 `intended_final_destination`을 expected 값으로 재사용하지 않는다.

필수 계약:

1. trusted output root와 `run_id`, allowlisted cohort로 expected final destination을 결정적으로 계산한다.
2. POST intended destination, receipt final path, 실제 `package_dir.resolve()`가 모두 expected destination과 같아야 한다.
3. 상대 경로, symlink 우회, root 이탈, 다른 cohort namespace를 거부한다.
4. 불일치 시 G10 또는 G11을 fail-closed한다.

Failure injection:

```text
POST destination 변조 후 재서명 → G10 FAIL
```

### 4.2 P0-2 — FINAL의 PRE_PROMOTION 재현

FINAL은 supplied PRE verdict의 self-hash와 `promotion_eligible=true`만 신뢰하지 않는다.

```text
recomputed_pre_verdict =
  current immutable inputs를 PRE_PROMOTION phase로 재검증한 결과
```

다음은 exact match여야 한다.

- canonical bytes / `verdict_sha256`
- `run_id`
- `evidence_root_sha256`
- `verifier_code_sha256`
- gate map
- `promotion_eligible`
- schema, verdict kind, phase

비교 성공 이후에만 FINAL의 `source_pre_promotion_verdict_sha256`을 기록한다.

Failure injection:

```text
다른 run/root의 eligible PRE verdict 주입 → FINAL FAIL
```

### 4.3 P0-3 — G11 promotion outcome/receipt 결합

G11 verifier는 문자열과 `final_readback.ok=true`를 판정 근거로 사용하지 않는다.

직접 검증 항목:

- actual package path == expected final destination
- receipt final path == actual package path
- receipt destination == POST intended destination
- receipt POST attestation SHA == 실제 canonical POST payload SHA
- actual package evidence root == POST/receipt evidence root
- readback manifest root/count == 실제 package 재계산 값
- source/destination filesystem 조건 일관성
- allowed promotion primitive 및 receipt schema
- receipt의 run/cohort 결합

기존 receipt 멱등 재사용은 `created_at` 등 최초 생성 전용 필드를 제외한 의미 필드가 모두 같을 때만 허용한다.

G11의 승인 문구는 다음으로 제한한다.

```text
atomic promotion outcome and receipt consistency verified
```

신뢰된 promotion service 서명, OS audit record 또는 별도 promotion key가 없는 한 다음 주장을 금지한다.

```text
renameat2 syscall independently proven
```

Failure injection:

```text
final_readback.ok=true인 forged receipt → G11 FAIL
```

### 4.4 P0-4 — immutable final-rerun snapshot

mutable 외부 실행 경로의 파일은 G12 판정 근거로 사용하지 않는다.

순서:

1. pytest JUnit XML 또는 고정 JSON schema의 machine-readable test result artifact를 생성한다.
2. canonical node-ID manifest와 machine-readable runtime observation artifact를 생성한다.
3. 사람이 읽는 final-rerun text log는 보조 자료로 생성하되 판정 원천으로 사용하지 않는다.
4. result, node-ID, runtime observation, text log를 quarantine package 또는 immutable artifact store에 byte snapshot한다.
5. snapshot을 evidence root에 포함한다.
6. observation manifest에 canonical relative path와 SHA를 기록한다.
7. POST attestation을 evidence root와 observation manifest SHA에 결합한다.
8. G12가 snapshot 파일을 직접 읽어 SHA와 내용을 재계산한다.

machine-readable result parser는 schema/version과 parser code SHA를 stable lock에 고정한다. G12는 result artifact에서 다음 값을 직접 파생한다.

```text
exit_code
passed_count
failed_count
deselected_count
canonical_test_selection
execution_timestamp
```

observation manifest의 선언값은 파싱 결과와 exact match여야 한다. node/host identity는 별도의 machine-readable runtime observation snapshot에서 읽으며, observation 선언값과 exact match한다.

qualification과 final-rerun의 node-ID 및 test selection이 exact match여야 한다.

Failure injection:

```text
존재하지 않는 snapshot에 임의 64-hex SHA 선언 → G12 FAIL
실제 failed_count=1인 result snapshot
  + observation failed_count=0
  → G12 FAIL
parser schema/version 또는 parser code SHA 불일치 → G12 FAIL
runtime observation의 node/host와 선언값 불일치 → G12 FAIL
```

### 4.5 P0-5 — authorization self-reference 차단

`consume_pre_execution_authorization()`은 `expected_locks` fallback을 허용하지 않는다.

```text
expected_locks is not None
→ AUTH_SELF_REFERENCE_FORBIDDEN
```

성공 경로에는 filesystem에서 fresh recomputation한 non-empty `observed_stable_lock`이 필수다. 서명 payload의 lock을 observed 값으로 재주입하는 모든 호출을 금지한다.

Failure injection:

```text
expected_locks=authorization.stable_lock_manifest
→ authorization 거부
```

### 4.6 P0-6 — case-disposition expected-trace verifier

G3는 package 전체의 operation count가 nonzero라는 이유만으로 PASS하지 않는다. 각 case의 canonical disposition 원장과 case trace-root를 결합하고, disposition별 expected trace profile과 observed trace를 비교한다.

필수 구현:

- case별 disposition 결정 원장
- disposition별 expected trace profile
- case trace-root와 disposition exact 결합
- case별 expected/observed trace 비교
- 모든 case profile PASS일 때만 cohort G3 PASS
- package 전체 nonzero count만으로 G3 PASS하는 기존 경로 제거
- 다른 case의 trace로 누락 case의 count를 충족하는 경로 차단

Failure injection:

```text
selected case의 Fold2 trace 삭제           → G3 FAIL
한 case trace 전체 삭제                    → G3 FAIL
NOT_CONSTRUCTIBLE candidate forward=0       → G3 PASS
NOT_CONSTRUCTIBLE 실패 종결 trace 없음       → G3 FAIL
NOT_EVALUATED DENIED 후 side effect 발생     → G3 FAIL
다른 case trace로 global count 충족          → G3 FAIL
```

각 failure injection의 observed 결과가 해당 case profile의 expected 결과와 일치해야 P0-6 GO다.

### 4.7 인접 방어

- stable-lock 비교 helper는 양쪽 manifest의 canonical SHA를 각각 재검증한다.
- receipt validity는 쓰기 함수 성공 여부가 아니라 독립 validator 결과로 결정한다.
- receipt 생성 실패는 `PROMOTED_BUT_UNATTESTED_BY_RECEIPT` 또는 동등한 fail-closed 상태로 보존한다.

## 5. P0 합격 게이트

| Gate | GO | NO-GO |
|---|---|---|
| P0 구현 | P0-1~P0-6과 인접 방어 완료 | 하나라도 미완료 |
| P0-1~P0-5 targeted tests | 기존 필수 5종이 예상 gate에서 fail-closed | 공격 통과 또는 다른 gate의 허위 PASS |
| P0-6 G3 targeted tests | disposition failure injection 6종이 예상 case profile과 일치 | 누락 case가 global count로 은폐되거나 정상 disposition이 허위 FAIL |
| Full regression | 전체 CF1S suite exit 0, canonical selection/log SHA 보존 | 실패, 누락, 예상 밖 deselection |
| DEVELOPMENT3_FINAL_CODE_LOCK | P0-1~P0-6, machine-readable G12, full regression 후 code/test/policy/node selection SHA 고정 | 선행 blocker 미완료 또는 이후 변경 |

`DEVELOPMENT3_FINAL_CODE_LOCK` 이후 code, test, policy 또는 node selection이 바뀌면 기존 Development3 qualification·pilot·실행 evidence를 폐기하고 targeted/full regression부터 다시 수행한다.

Development3 FINAL 이후 cohort-aware verifier/lock/auth/runner를 구현하면 D-LOCK은 확대 실행에 재사용할 수 없다. cross-cohort failure injection과 full regression을 거쳐 `EXPANSION_FINAL_CODE_LOCK`을 새로 생성한다. Primary32 전에 code 또는 policy가 다시 변경되면 `PRIMARY32_FINAL_CODE_LOCK`을 생성하거나 E-LOCK을 재생성한다.

각 cohort stable lock은 해당 시점의 다음 값을 포함한다.

- code/test/policy tree SHA
- verifier code SHA
- cohort schema SHA
- runner/routing SHA
- canonical test selection SHA
- qualification result SHA

## 6. Case disposition별 G3 trace 계약

본 절은 §4.6 P0-6의 disposition별 expected trace profile 상세이며, 참고 설명이 아니라 P0 합격을 차단하는 필수 구현 계약이다.

Cohort G3는 모든 case에 candidate forward nonzero를 일괄 요구하지 않는다.

```text
observed trace == case disposition별 expected trace
```

| Case disposition | 필수 trace | candidate forward |
|---|---|---|
| `CONSTRUCTIBLE_SELECTED` | attribution, baseline/identity, A/B/AB, Fold2 closure replay | nonzero 필수 |
| `CONSTRUCTIBLE_ALL_GATE_FAILED` | attribution, baseline/identity, gate trace, 실패 종결 | side effect 전 차단이면 0 허용 |
| `NOT_CONSTRUCTIBLE` | constructibility/grounding 시도, 정확한 실패 종결 | 0이 정상 |
| `NOT_EVALUATED` | 선행 차단 사유, `DENIED` | 0이 정상 |
| `NOT_EVALUABLE` | 실행된 범위의 trace, 정확한 실패 종결 | 완료 범위에 따라 부분 허용 |

- trace 없이 disposition만 선언한 `NOT_CONSTRUCTIBLE`은 G3 FAIL이다.
- 정상적인 `NOT_CONSTRUCTIBLE`·`NOT_EVALUATED`는 cohort completeness 실패가 아니다.
- disposition은 canonical ledger/manifest와 case trace-root에 결합한다.

## 7. Cohort-aware 실행 계약

현재 Development3 하드코딩을 임의 generic mode로 바꾸지 않는다.

허용 cohort:

```text
DEVELOPMENT3 | VALIDATION20 | PRIMARY32
```

각 schema는 다음을 고정한다.

- authorization type과 허용 execution kind
- stable-lock cohort ID
- expected case count와 ordered case-ID root
- trusted final destination
- runner 및 fold-routing
- run ID, evidence root, verdict, receipt, completion namespace
- 동일한 G1–G12 의미

차단 조건:

- Dev3 authorization으로 Validation20/Primary32 실행
- Validation20 authorization으로 Primary32 실행
- cohort ID만 변경한 기존 서명 재사용
- unknown cohort
- expected count보다 적거나 많은 package
- 다른 cohort routing 사용

추가 failure injection:

```text
Dev3 auth + Validation20 package       → G5 FAIL
Validation20 auth + Primary32 manifest → G5 FAIL
20-case lock + 19-case package         → completeness FAIL
Primary32 routing + Validation20 run   → runtime/routing FAIL
unknown cohort                         → authorization FAIL
```

## 8. 단계별 실행

### Phase D0 — Development3

```text
P0-1~P0-6 remediation + targeted failure injection
→ full regression
→ DEVELOPMENT3_FINAL_CODE_LOCK (D-LOCK)
→ qualification + immutable machine-readable same-node final-rerun snapshot
→ Development3 single-case traced GPU pilot
→ Development3 full 3-case run
→ PRE_PROMOTION
→ no-replace promotion
→ FINAL G1–G12
→ completion
```

Development3 FINAL PASS 전에는 Validation20 authorization을 발급하지 않는다.

### Phase E0 — 확대 입력 잠금

고정 항목:

- Validation20 / Primary32 case-input root
- cohort 간 case ID disjointness
- case별 raw-event source SHA
- fold routing / checkpoint pairing
- tokenizer / schema / binning / label-map
- Development3 locked threshold와 threshold SHA
- case별 training-seen provenance
- cohort별 authorization
- expected constructibility inventory
- pilot case ID

constructibility inventory 허용 범위:

- 입력 존재 여부
- raw grounding 가능성
- schema/tokenizer 호환성
- 최소 이벤트 수
- 정적 Gate0 조건

금지:

- Fold2 forward
- candidate risk/delta
- effect 방향
- 결과 기반 candidate/threshold 변경

pilot case ID는 manifest에서 결정적으로 사전 고정한다. pilot 결과를 보고 case를 교체하지 않는다. pilot에서 허용되는 변경은 계약 결함 수정뿐이며, 변경 시 해당 cohort의 code lock과 pilot을 다시 수행한다.

Development3 FINAL 뒤에는 cohort-aware verifier/lock/auth/runner를 구현하고 cross-cohort failure injection과 full regression을 수행한다. 그 결과를 `EXPANSION_FINAL_CODE_LOCK`(E-LOCK)으로 고정한 뒤에만 Validation20 qualification/pilot을 시작한다.

### Phase E1 — Validation20

계약:

- Problem20→Validation20 변환 lineage
- `finetune_training_seen=false`
- `pretraining_seen` 사실 기록
- selection-blind Fold2 closure
- Development3 threshold 재사용
- threshold/candidate policy 변경 금지
- 별도 run ID / stable lock / authorization / evidence root / verdict / receipt
- 기본 과학 표현: `finetune-unseen, selection-blind-to-Fold2 evaluation`
- `independent holdout`은 데이터 provenance뿐 아니라 candidate 생성, selection, threshold 결정의 독립성까지 별도 evidence로 모두 입증된 경우에만 예외적으로 허용

순서:

```text
사전 고정 1-case pilot
→ PILOT_PRECHECK_PASS
→ 새 run ID로 pilot case를 포함한 full 20-case 실행
→ cohort PRE_PROMOTION
→ promotion
→ cohort FINAL
→ Validation20 cohort report
```

### Phase E2 — Primary32

Primary32는 training-seen diagnostic으로 해석한다.

- case별 `finetune_training_seen` 실제 상태 기록
- 외부 일반화·replication 주장 금지
- Validation20과 별도 package
- 앞선 결과를 보고 candidate/threshold 변경 금지
- Primary32 전에 code 또는 policy가 E-LOCK 이후 변경됐다면 `PRIMARY32_FINAL_CODE_LOCK`을 새로 생성하거나 E-LOCK을 재생성한 뒤 qualification을 다시 수행

순서:

```text
사전 고정 1-case pilot
→ PILOT_PRECHECK_PASS
→ 새 run ID로 pilot case를 포함한 full 32-case 실행
→ cohort PRE_PROMOTION
→ promotion
→ cohort FINAL
→ Primary32 cohort report
```

### Phase E3 — ALL55 reference aggregate

세 cohort가 모두 FINAL PASS한 경우에만 발급한다.

```json
{
  "reporting_set": "ALL55_STRATIFIED_ANALYSIS_SET",
  "source_packages": [
    {"cohort": "Development3", "case_count": 3, "evidence_root": "..."},
    {"cohort": "Validation20", "case_count": 20, "evidence_root": "..."},
    {"cohort": "Primary32", "case_count": 32, "evidence_root": "..."}
  ]
}
```

필수 결합값:

- canonical manifest SHA
- aggregate builder code SHA
- source cohort별 FINAL verdict/completion SHA
- source case count와 ordered case-ID root
- analysis schema/version
- threshold SHA

한 cohort가 실패하면:

```text
ALL55 aggregate = NOT_ISSUED
통과한 cohort report = 유지
실패 cohort = HOLD / UNVERIFIED
```

부분 cohort 결과를 ALL55로 표현하지 않는다.

## 9. Pilot 계약

pilot은 cohort FINAL이 아니다.

```text
artifact status = PILOT_PRECHECK_PASS | PILOT_PRECHECK_FAIL
```

- 실제 GPU transaction/trace/ledger 생성
- pilot 전용 verifier로 production contract 검사
- final namespace 승격 없음
- receipt/completion 없음
- 과학 결과 보고 금지
- PASS 후 전체 cohort를 새 run ID로 실행
- pilot case도 전체 cohort에서 다시 실행
- pilot artifact를 cohort FINAL evidence로 재사용 금지

## 10. Resumable execution과 trace 결합

case별 독립 trace chain 방식을 사용한다.

- 각 case는 독립 `trace_events.json`과 trace-root SHA를 갖는다.
- cohort manifest는 canonical case 순서로 case trace-root를 참조한다.
- cohort verifier는 각 chain과 전체 case 완전성을 검증한다.
- 서로 다른 process의 trace를 가짜 global chain으로 병합하지 않는다.
- 완료 case artifact는 immutable하게 닫는다.
- 실패 case부터 재개할 수 있다.
- 재개 시 input/code/checkpoint/policy lock과 trace-root를 재검증한다.
- 하나라도 다르면 새 run ID로 cohort 전체를 다시 실행한다.
- 부분 실행은 FINAL 승격·receipt·completion을 생성하지 않는다.

## 11. 분석 기본 행과 exact-parent 결합

분석 기본 키:

```text
cohort_id
case_id
candidate_bundle_id
candidate_family_id
A_transaction_sha
B_transaction_sha
AB_transaction_sha
fold_id
```

한 case에 selected bundle이 여러 개일 수 있다. A/B/AB를 case당 하나로 가정하거나 덮어쓰지 않는다.

금지 결합:

- 서로 다른 AB bundle의 A와 B
- 다른 fold의 baseline
- Search A/B와 Fold2 AB
- control bundle과 material bundle residual
- 여러 bundle을 case당 하나로 암묵 축약

case material status는 evaluable selected material bundle 중 사전 잠긴 canonical rank 1 bundle을 사용한다. descriptive distribution은 모든 evaluable bundle을 포함하고 bundle 수를 명시한다. best-effect bundle 사후 선택을 금지한다.

## 12. 수치 분석 schema

동일 bundle·fold에서:

```text
delta_A  = A_risk  - baseline_risk
delta_B  = B_risk  - baseline_risk
delta_AB = AB_risk - baseline_risk

incremental_over_A = delta_AB - delta_A
incremental_over_B = delta_AB - delta_B
interaction_residual = delta_AB - delta_A - delta_B
```

동일한 원시값과 파생값을 abnormal-logit 척도에도 보존한다.

- risk delta: locked materiality threshold 적용
- abnormal-logit delta: risk–logit 계약 및 기술통계
- logit materiality: 별도 사전 잠금이 없으면 판정 금지
- interaction residual: 기술적 모델 출력 상호작용이며 인과적 interaction이 아님

각 분포는 다음을 저장한다.

```text
N_total
N_evaluable
N_excluded_by_reason
median
Q1
Q3
min
max
raw_row_manifest_sha256
```

## 13. 분석축과 분모

cohort별 분석:

- constructibility 비율
- material/control 선택 비율
- `delta_A`, `delta_B`, `delta_AB` 분포
- interaction residual 분포
- threshold 초과 비율
- Search 0/1 방향 일치율
- Search aggregate와 Fold2 방향 일치율
- material effect support rate
- control threshold 초과율
- parent/ledger/trace failure rate
- execution/material/control coverage
- case별 training-seen provenance

분모:

```text
constructibility_rate
  = constructible / all locked cases

material_support_rate
  = supported material bundles
    / evaluable selected material bundles

control_exceedance_rate
  = threshold-exceeded controls
    / evaluable selected controls

case_material_coverage
  = cases with at least one evaluable selected material
    / all locked cases
```

`NOT_EVALUABLE`, `NOT_EVALUATED`, `NOT_CONSTRUCTIBLE`을 특정 분모에서 제외하면 제외 수와 이유를 함께 표시한다.

항상 다음 네 열을 함께 보고한다.

```text
Development3 | Validation20 | Primary32 | All55 descriptive total
```

All55는 기술적 합계이며 서로 다른 provenance의 cohort를 동질 표본으로 해석하지 않는다.

## 14. 분석 금지사항

- Primary32와 Validation20을 합친 일반화 성능 주장
- control-only case를 material-effect failure 분모에 포함
- `NOT_EVALUABLE`을 `NOT_SUPPORTED`에 포함
- constructible case만 제시하고 전체 분모 은닉
- Search-selected effect를 독립 발견으로 표현
- risk-scale residual을 인과적 interaction으로 표현
- Primary32를 외부 replication 근거로 사용
- 미완료 cohort를 ALL55로 표시

## 15. 최종 GO/HOLD

```text
P0 targeted + full regression PASS
  → Development3 revalidation GO

Development3 FINAL G1–G12 PASS
  → Development3 report 발급
  → Validation20 GO

Validation20 FINAL G1–G12 PASS
  → Validation20 report 발급
  → Primary32 GO

Primary32 FINAL G1–G12 PASS
  → Primary32 report 발급

세 cohort FINAL PASS
  → ALL55_STRATIFIED_ANALYSIS_SET 발급
```

어느 단계에서든 FAIL/NOT_RUN/BLOCKED이면 해당 단계 이후를 HOLD하고 `highest_claim_level=NONE`을 유지한다. 계획 승인은 실행 성공을 의미하지 않는다.

## 16. 구현 순서

P0-1~P0-5는 현재 구현을 다시 설계하지 않는다. 기존 필수 failure injection으로 회귀 여부만 확인한다.

최단 critical path:

1. P0-6 case-disposition expected-trace verifier 구현
2. machine-readable G12 result/runtime snapshot 및 parser lock 구현
3. P0-1~P0-5 필수 5종과 P0-6 G3 필수 6종, 총 targeted 11종 실행
4. 전체 CF1S 회귀
5. `DEVELOPMENT3_FINAL_CODE_LOCK`(D-LOCK)
6. Development3 qualification 및 사전 고정 single-case pilot
7. Development3 full 3-case FINAL
8. cohort-aware verifier/lock/auth/runner 구현과 cross-cohort failure injection
9. 전체 회귀 후 `EXPANSION_FINAL_CODE_LOCK`(E-LOCK)
10. Validation20 qualification / pilot / full 20-case FINAL
11. Primary32 전 변경 유무 검증; 변경 시 `PRIMARY32_FINAL_CODE_LOCK` 또는 E-LOCK 재생성
12. Primary32 qualification / pilot / full 32-case FINAL
13. ALL55 reference aggregate 및 층화 분석

본 문서는 위 개정 범위로 `APPROVED FOR IMPLEMENTATION`이다. 현재 P0 completion은 `PARTIAL`이며, P0-6·machine-readable G12·targeted/full regression이 끝나기 전 D-LOCK 생성과 Development3 실행 전환을 금지한다.

## 17. 최종 계획 검토 승인과 병행 준비

### 17.1 최종 승인 판정

```text
PLAN REVIEW = COMPLETE
document = APPROVED FOR IMPLEMENTATION
P0 completion = PARTIAL
Development3 = HOLD
Validation20 = HOLD
Primary32 = HOLD
ALL55 aggregate = NOT_ISSUED
highest_claim_level = NONE

IMPLEMENTATION = GO
D-LOCK = BLOCKED UNTIL P0-6/G12/TARGETED/FULL REGRESSION PASS
ALL55 EXECUTION = SEQUENTIALLY GATED
```

추가 계획 확장은 중단한다. 남은 위험은 문서가 아니라 P0-6, machine-readable G12, 실제 verifier verdict와 immutable evidence 생성에 있다.

### 17.2 P0와 병행 가능한 비효과 준비

GPU 실행을 앞당기기 위해 모델 효과를 보지 않는 다음 작업은 P0 수행과 병행할 수 있다.

- Validation20·Primary32 case-input root 계산
- Development3/Validation20/Primary32의 3/20/32 case ID 완전성 및 상호 disjointness 확인
- raw-event source SHA manifest 작성
- pilot case ID의 결정적 사전 고정
- fold routing/checkpoint manifest 준비
- training-seen provenance 정리
- ALL55 aggregate schema와 출력 template 준비

### 17.3 Development3 FINAL 전 금지

- Validation20/Primary32 authorization 발급
- candidate risk/delta 계산
- Fold2 forward
- threshold 또는 candidate policy 변경
- 부분 결과를 ALL55로 보고

### 17.4 판정 권위

이후 단계 전환은 테스트 개수, 실행기 summary 또는 수기 기록으로 승인하지 않는다. 각 단계의 canonical verifier verdict와 evidence root에 포함된 immutable machine-readable artifact만 판정 근거로 사용한다.
