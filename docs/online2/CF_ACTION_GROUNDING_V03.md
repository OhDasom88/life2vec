# Counterfactual 토큰 편집 → 실행 가능 개입 접지 (Action Grounding)

상태: **설계 개정 (invertibility · edit_scope 반영)**  
작성일: 2026-07-16 · 개정: 2026-07-16  
상위: [`COUNTERFACTUAL_INFERENCE_V03.md`](./COUNTERFACTUAL_INFERENCE_V03.md) · [`DIAGNOSIS_FINETUNE_V03.md`](./DIAGNOSIS_FINETUNE_V03.md) §11  
관련 사실: [`pipeline_trace_v02/02_tokenizer.md`](./pipeline_trace_v02/02_tokenizer.md) · [`pipeline_trace_v02/SOURCE_EXCERPTS/README.md`](./pipeline_trace_v02/SOURCE_EXCERPTS/README.md) §6 · corpus [`contracts/online2_corpus_build_plan_v7.md`](./contracts/online2_corpus_build_plan_v7.md) §2

---

## 0. 한 줄 결론

**연속값(ABS bin)은 사전·registry에 구간 의미를 남기면 구간 역산이 가능하다. Actuator 지속시간(3h→1h의 “어떻게”)은 단일 이벤트 토큰 편집만으로는 역산 불가하며, segment-level `edit_scope`가 필수다.**

```text
[상태 파악] binary / Known-Unknown / 진단명
    ↓
[CF search]  Path A: token/MG 편집 · Path B: **edit_scope** (+ 파생 token_edits)
    ↓
[본 계층]    Action Grounding
    ↓
[운영 출력]  원시 setpoint 구간 · 용량 · 지속시간 · 검증 지표
```

---

## 0.1 검토: “의미 남기면 역산 가능하지 않나?”

### 결론 표

| 신호 | vocab/registry에 의미 보존? | 역산 가능 수준 | 비고 |
|------|------------------------------|----------------|------|
| 연속 ENV/ROOT (`VALUE_ABS\|ABS_Bk`) | ✅ edges registry | **구간** `[lo,hi)` 확정 · 점값은 정책 | 이미 설계됨; exact float 복원은 원리상 불가(다대일) |
| GLOBAL/FARM_REL bin | ✅ | 상대 구간만 | setpoint 1차 소스로 부적합 |
| Actuator literal `POSITIVE`/`ZERO` | △ on/off만 | **상태**만 · 용량·지속시간 ❌ | `>0` 전부 POSITIVE로 붕괴 |
| 파생 `RH90_CONTINUOUS\|DUR_H04` 류 | corpus 계획에 sidecar | 창 길이는 sidecar면 가능 | **CF 편집 연산(truncate_end 등)은 여전히 별도** |
| 시계열 구간 편집 (3h→1h) | ❌ 토큰 문자열이 아님 | **edit_scope 필수** | 아래 §0.2 · §5 |

핵심 구분:

1. **어휘(토큰 타입)에 의미 남기기** → 타입·채널·bin 인덱스 ↔ 물리 구간 **조회** (Path A에 해당).  
2. **인스턴스 sidecar** (`cell_occurrences.raw_value`, `source_cell_id`) → 관측 점값·출처 join.  
3. **편집 연산 의미** (어디를 어떻게 줄일지) → 토큰 문자열에 넣을 대상이 아니라 **편집 스키마(`edit_scope`)** 에 둔다.

`TokenizerV2.MeasurementTokens`는 이미 `raw_value`·`source_cell_id`·`roles`를 갖는다. 다만 학습용 `events_tokenized`는 `SENTENCE`/`token_ids`/`token_roles`/`measurement_group_ids` 중심이라, **역산은 vocab만으로 끝내지 말고 registry + cell sidecar(+ edit_scope)로 닫는다.**

corpus v7도 “원시값은 sidecar에 보존 · bin registry 버전 고정”을 원칙으로 둔다 — 이는 **구간/점값 접지**를 위한 것이지, actuator span 연산의 대체재가 아니다.

### Path A에 대한 함의

“의미를 남기면 역산”은 **ABS(및 명시적 duration 파생 토큰)에 대해 성립**한다.  
구현은 새 마법이 아니라 `decode_interval` + (선택) observed raw anchoring이다.

### Path B에 대한 함의

Actuator를 bin/ordinal로 세분화해도 **시점별 용량**만 좋아질 뿐,  
“3시간 POSITIVE 구간을 끝에서 2시간 자를지 / 통째로 1시간으로 줄일지”는 **여러 이벤트에 걸친 연산**이라 토큰 타입만으로는 표현되지 않는다.

---

## 0.2 검토: 단일 `token_edits`로 3h→1h를 알 수 없음 — **채택**

다음 스키마만으로는 연산이 구분되지 않는다.

```json
{
  "event_id": "...",
  "from_tokens": ["OBSERVED_VALUE|POSITIVE"],
  "to_tokens": ["OBSERVED_VALUE|ZERO"]
}
```

동일하게 보일 수 있는 연산:

- 3시간 구간 **전체** 제거  
- **마지막** 2시간만 제거 (`truncate_end`)  
- 시작 시점 이동 (`shift_start`)  
- 중간 1시간만 제거 (`punch_hole`)  
- 전체 구간을 1시간으로 **축소** (`shrink_to` / rescale)

**채택:** actuator CF의 1급 입력은 `edit_scope`이고, `token_edits`는 그 scope를 시퀀스에 적용한 **파생 결과**(모델 forward용)로 둔다.

```json
{
  "edit_scope": {
    "type": "time_span",
    "feature": "circulation_fan",
    "farm_id": "F…",
    "zone_id": "z…",
    "source_event_ids": ["e1", "e2", "e3"],
    "operation": "truncate_end",
    "from_start": "2025-02-16T10:00:00Z",
    "from_end": "2025-02-16T13:00:00Z",
    "to_start": "2025-02-16T10:00:00Z",
    "to_end": "2025-02-16T11:00:00Z"
  }
}
```

`operation` 권장 enum: `set_span` · `truncate_end` · `truncate_start` · `shift` · `punch_hole` · `clear_span` · `extend`.

**`ground_actuator_span(edit_scope) → duration/capacity`** — 입력은 token diff가 아니라 **이 `edit_scope`**.

---

## 1. 갭 정의

| 계층 | 있음? | 하는 일 |
|------|-------|---------|
| Token / measurement-group 최소 편집 | ✅ 설계 (Path A) | `x → x'` 토큰 치환, beam, ΔR − edit cost |
| **Segment `edit_scope`** | ❌ → 본 개정에서 계약 | Path B 시간 구간 연산의 1급 표현 |
| Evidence용 raw join | ✅ 부분 (`evidence_raw.py`, P0) | **관측 설명**용 원시값 부착 |
| Bin edges → 구간 복원 | ✅ registry | Path A 구간 역산의 본체 |
| Actuator ZERO/POSITIVE | ✅ 토큰화 | on/off만. 용량·**연산 종류** 손실 |
| **Action Grounding** | ❌ | Path A: token(+registry) → 구간 · Path B: **edit_scope** → 지속/용량 |

과대해석 금지:

- MLM infill ≠ 물리 시뮬 (기존 CF 가드 유지).
- vocab에 의미를 더 넣어도 **다대일 bin·literal**은 점값·연산을 복원하지 못한다.
- Grounding은 처방 후보의 단위·구간 번역이지 성공 보장이 아니다.

---

## 2. 이미 있는 조각 (재사용)

### 2.1 연속 관측 (ENV / ROOT 등)

1. 토큰: `FEATURE|inside_temp_c` + `VALUE_ABS|ABS_Bk` (+ GLOBAL/FARM_REL).
2. 역구간: `binning_registry_v2_transductive.json`의 `rules[].edges`  
   → `edges[k] ≤ x < edges[k+1]` ([`02_tokenizer.md`](./pipeline_trace_v02/02_tokenizer.md)).
3. 원본 관측: `cell_occurrences.raw_value` / `raw_display`.
4. **없음:** bin → 대표값(mid/quantiles), “목표 원시값” 선택 정책, 단위·안전 클램프 테이블.

### 2.2 Actuator

1. `literal_state`: `0→ZERO`, `>0→POSITIVE`, `<0→NEGATIVE`  
   ([SOURCE_EXCERPTS README §6](./pipeline_trace_v02/SOURCE_EXCERPTS/README.md)).
2. Path B 의도: ACTUATOR만 직접 편집, 이후 ENV/ROOT는 mask→infill.
3. **없음:** POSITIVE의 용량(유량·개도·출력), 지속시간(분/시간), ramp, 허용 범위 whitelist의 수치 스키마.

### 2.3 CF 패키지 자리 (예정)

```text
src/online2/v2/counterfactual/
  … search.py / diff.py / intervention.py …
  action_grounding.py   ← 본 문서가 채우는 모듈 (신규)
```

`intervention.py`(Path B)와 역할을 분리한다:

| 모듈 | 책임 |
|------|------|
| `search` / `diff` | Path A: 토큰 편집. Path B: **`edit_scope` 탐색** 후 시퀀스에 적용해 token_edits 파생 |
| `intervention` | actuator whitelist · 미래 mask · IMAGE 제외 |
| **`action_grounding`** | Path A: token+registry → 구간. Path B: **`edit_scope` → 지속/용량** |

---

## 3. 목표 출력 스키마 (최소)

CF 후보 1건당 grounding 블록을 붙인다.

```json
{
  "cf_path": "A_diagnosis | B_management",
  "edit_scope": {
    "type": "time_span",
    "feature": "circulation_fan",
    "source_event_ids": ["e1", "e2", "e3"],
    "operation": "truncate_end",
    "from_start": "2025-02-16T10:00:00Z",
    "from_end": "2025-02-16T13:00:00Z",
    "to_start": "2025-02-16T10:00:00Z",
    "to_end": "2025-02-16T11:00:00Z"
  },
  "token_edits": [
    {
      "event_id": "e2",
      "measurement_group_id": "…",
      "feature": "circulation_fan",
      "from_tokens": ["OBSERVED_VALUE|POSITIVE"],
      "to_tokens": ["OBSERVED_VALUE|ZERO"],
      "derived_from_edit_scope": true
    }
  ],
  "grounded_actions": [
    {
      "kind": "setpoint_range",
      "feature": "inside_temp_c",
      "unit": "C",
      "observed_raw": 29.57,
      "from_bin": {"channel": "ABS", "index": 28, "lo": 28.1, "hi": 30.0},
      "to_bin": {"channel": "ABS", "index": 18, "lo": 22.0, "hi": 23.5},
      "suggested_target": {"policy": "bin_mid", "value": 22.75},
      "delta_raw": -6.82,
      "confidence": "interval_only"
    },
    {
      "kind": "actuator_span",
      "feature": "circulation_fan",
      "edit_scope_ref": true,
      "operation": "truncate_end",
      "from_hours": 3.0,
      "to_hours": 1.0,
      "capacity": {
        "policy": "unspecified_positive",
        "observed_raw": 1.0,
        "unit": "device_native",
        "note": "토큰에 세기 없음; CSV raw 또는 장비 스펙 테이블 필요"
      },
      "confidence": "duration_ok_capacity_weak"
    }
  ],
  "ungroundable": []
}
```

Path A만 있는 후보에서는 `edit_scope`를 생략하거나 `type: "token_mg"`로 둔다.  
Path B에서는 **`edit_scope` 필수**, `token_edits`는 적용 로그.

`confidence` 권장 enum:

- `interval_only` — bin 구간만 확실, setpoint는 정책 의존  
- `raw_anchored` — 관측 raw + 목표 bin으로 Δ 계산  
- `duration_ok_capacity_weak` — edit_scope로 시간은 확실, 용량 미상  
- `ungroundable` — 수치화 불가 (보고서 차단 또는 경고)

---

## 4. Path A — 연속 피처 접지

### 4.1 알고리즘 (초안)

입력: 토큰 편집 `(feature, from_suffix, to_suffix)`, 선택적으로 `observed_raw`.

1. `channel, k = parse(VALUE_ABS|ABS_Bk)` (GLOBAL/FARM_REL 동일).
2. `edges = registry.rule(feature, channel[, farm]).edges`.
3. `from_interval = [edges[k], edges[k+1])`, `to_interval` 동일.
4. **suggested_target** (설정 가능, 기본 `bin_mid`):
   - `bin_mid`: `(lo+hi)/2` (open hi면 `hi - ε`)
   - `bin_quantile_q`: 구간 내 학습분포 q (registry occupancy 있으면)
   - `nearest_edge_toward_normal`: 관측 raw에서 목표 구간 쪽 경계
5. `delta_raw = suggested_target - observed_raw` (raw 없으면 null).
6. 안전: feature별 `clamp[lo_safe, hi_safe]` (도메인 테이블; 없으면 warn).

### 4.2 다중 채널 충돌

같은 feature에 ABS·GLOBAL_REL·FARM_REL이 함께 바뀌면:

1. **ABS를 1차 setpoint 소스**로 한다 (물리 단위에 가장 가깝다).
2. REL 채널은 “농장/전역 상대 위치” 검증용으로만 보고.
3. ABS와 REL이 모순이면 `ungroundable` 또는 `confidence=conflict`.

### 4.3 한계 (문서에 고정)

- Bin은 등빈도 등이라 **구간 폭이 피처·위치마다 다름** → “ABS_B−10”이 항상 −2℃가 아님.
- Midpoint는 **편의 대표값**이지 최적 영농값이 아님.
- Path A grounding 결과는 “모델이 정상으로 보려면 이 관측 구간으로 옮겨야 한다”이지, 그 수단(난방/환기)은 Path B.

---

## 5. Path B — Actuator 용량·지속시간

### 5.1 지속시간 — `edit_scope` 입력 (필수)

**금지:** `ground_actuator_span(token_edits)` — POSITIVE→ZERO 한 줄로는 연산이 비식별.

**계약:**

```text
ground_actuator_span(edit_scope) -> {
  from_hours, to_hours, operation,
  affected_event_ids, capacity_hint, confidence
}
```

1. `edit_scope.source_event_ids` + timestamps로 **원본 span** `[from_start, from_end)` 확정.  
2. `operation` + `to_start`/`to_end`로 **목표 span** 확정 → `from_hours`/`to_hours`.  
3. 시퀀스 적용기는 scope를 이벤트별 `POSITIVE`/`ZERO`(또는 삭제)로 펼쳐 `token_edits`를 **파생**.  
4. Risk 평가는 파생된 시퀀스로 Stage A 재인코딩; grounding 리포트는 scope 기준.

Run-length는 scope를 **검증**할 때(원본 POSITIVE 연속 구간과 `source_event_ids` 일치 여부)만 쓴다.  
검색 공간의 1급 후보는 “어느 이벤트를 ZERO로”가 아니라 **어느 span에 어떤 operation**.

### 5.2 용량 (구조적 약함 · 단계적)

현재 토큰: `OBSERVED_VALUE|POSITIVE`만으로는 **용량 불가**.  
vocab에 ABS bin을 병행해도(B2) 시점별 세기만 개선되고, **span operation** 문제는 그대로 `edit_scope`가 담당한다.

| 단계 | 내용 |
|------|------|
| B0 | `capacity.policy = unspecified_positive`; span 내 CSV `observed_raw` 통계(예: max)만 참고 |
| B1 | feature별 `actuator_capacity_table` |
| B2 | `ordinal_actuator`/`flow`에 ABS bin 병행 → 시점 capacity interval + 여전히 edit_scope로 duration |

보고서 분리 표기:

```text
직접개입: edit_scope (feature, operation, from/to timestamps, hours)
기대상태: ENV/ROOT mask→infill 가설 (모델 공간)
검증지표: ΔR, OOD, 과편집률 — 물리 검증 아님
```

### 5.3 Whitelist

`actuator_whitelist.yaml` 예시 필드:

```yaml
circulation_fan:
  editable: true
  capacity_unit: device_native
  capacity_source: csv_raw
  duration_editable: true
  allowed_operations: [truncate_end, truncate_start, clear_span, set_span]
  max_span_hours: 24
  min_span_hours: 0
```

---

## 6. 파이프라인 위치

```text
P4a smoke (토큰 편집 + ΔR)
  → Path A grounding: registry decode_interval
P4b
  → search는 edit_scope 후보 생성
  → apply_scope → token_edits → 재인코딩 → ΔR
  → ground_actuator_span(edit_scope) + capacity B0
  → (병행) actuator 토큰 세분화 B2는 vocab 트랙
```

**게이트:** 보고서용 Path B 개입은 `edit_scope` + `grounded_actions` 필수.  
이벤트 단위 POSITIVE→ZERO 로그만 있는 후보는 smoke 내부용.

---

## 7. 구현 체크리스트

### 코드

- [ ] `BinningRegistryV2.decode_interval(feature, channel, suffix) -> (lo, hi)`
- [ ] `ground_continuous_edit(...)` → `setpoint_range` (Path A; token+registry)
- [ ] `EditScope` 스키마 · `apply_edit_scope(sequence) -> token_edits`
- [ ] `ground_actuator_span(edit_scope)` — **token_edits 입력 금지**
- [ ] `ground_actuator_capacity(...)` → B0/B1
- [ ] CF JSON: `edit_scope` · `grounded_actions` · `ungroundable`
- [ ] 테스트: ABS 왕복; 동일 POSITIVE→ZERO 토큰 diff에 대해 truncate_end vs clear_span이 **다른** grounded duration

### 산출물

- [ ] `actuator_whitelist.yaml` (`allowed_operations` 포함)
- [ ] feature `clamp` / unit 테이블 (ENV·ROOT 우선)
- [ ] smoke: Path A interval + Path B edit_scope 각 1건

### 문서 연동

- [x] 본 문서 (invertibility · edit_scope 개정)
- [x] 상위 CF / ROADMAP / V03 / FOLLOWUP 링크 (기존)
- [x] `COUNTERFACTUAL_INFERENCE_V03.md` Path B에 edit_scope 한 줄

---

## 8. 비목표

- 물리 온실 시뮬레이터 구축
- bin midpoint를 “권장 영농 최적값”으로 판매
- actuator 세기 없는 상태에서 유량 정밀 처방
- v0.1/v0.2 tokenizer in-place 대수술 (세분화는 v0.3+ 별 트랙)

---

## 9. 관련 코드·데이터 인덱스

| 자산 | 경로 | Grounding에서의 역할 |
|------|------|----------------------|
| Bin edges | `outputs/online2/v2_build/binning_registry_v2_transductive.json` | 구간 복원 |
| Encode | `src/online2/v2/binning.py` `encode_suffix` | 대칭 decode API 추가 지점 |
| Tokenizer literal | `src/online2/v2/tokenizer.py` `literal_state` | actuator on/off |
| Raw cells | `cell_occurrences.parquet` | observed_raw · span 시계열 |
| Evidence join | `src/online2/v2/finetune_v03/evidence_raw.py` | 설명용; grounding과 분리 유지 |
| Actuator CSV | `datasets/agrichallenge/online2/data/A_actuator/` | duration/capacity B0 |

---

## 10. 최종 권고

1. **역산은 두 갈래:** Path A는 registry(의미 보존된 bin)로 **구간** 역산; Path B는 **`edit_scope`로 연산·지속시간** 역산.  
2. vocab에 의미를 더 넣는 것(actuator ABS 등)은 용량 접지에 도움되나, **3h→1h 연산 모호성은 해소하지 못한다.**  
3. `ground_actuator_span`은 token diff가 아니라 **`edit_scope`만** 받는다.  
4. 보고서 게이트: Path B는 `edit_scope` + grounded hours; Path A는 bin interval(+ optional raw).  
5. exact float·물리 시뮬을 vocab 역산으로 약속하지 않는다.
