# 대화 정리: 상태 파악 · Action Grounding · edit_scope

작성일: 2026-07-16  
출처: Cursor 대화 (진단 finetune 이후 “이벤트·컨셉 조정” 및 역산·span 논의)  
관련 정본: [`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md) · [`COUNTERFACTUAL_INFERENCE_V03.md`](./COUNTERFACTUAL_INFERENCE_V03.md)

---

## 1. 맥락 구분

| 모듈 | 역할 |
|------|------|
| **현재 진행 학습 (상태 파악)** | 진단 맥락에서 이벤트·개념이 정상/비정상인지, 학습셋 진단명인지, 어떤 진단명인지 확인 |
| **그 다음 (조정·개입)** | 현재 상태를 바탕으로 **어떤 이벤트의 어떤 컨셉을 어느 정도 조정할지** |

상태 파악과 조정은 파이프라인상 분리한다. 조정 쪽 설계 정본은 Action Grounding / Counterfactual 문서다.

---

## 2. 문서 번들 (당시 요청)

상태 파악 이후의 **이벤트·컨셉 조정** 관련 문서를 모아 압축했다.

- 경로: `outputs/online2/docs_bundles/event_concept_adjustment_cf_docs.zip`
- 복사본: `outputs/online2/docs_bundles/event_concept_adjustment_cf_20260716/`
- 포함: CF 정본, V03·PLAN·FOLLOWUP·ROADMAP, saliency, WORLD_MODEL_REUSE, 번들 README

---

## 3. 갭 인식 → 신규 문서

문제 제기:

> 토큰 수준의 최소 편집은 설계되어 있지만, 그것을 **원시 수치**와 **actuator 용량·지속시간**으로 변환하는 계층이 없다.

검토 결과: 전용 계층 문서는 없었고, 조각만 존재했다.

| 있던 것 | 없던 것 |
|---------|---------|
| bin edges → `ABS_Bk` 구간 조회 | bin → setpoint 정책·클램프 |
| evidence raw join (설명용) | 편집 → 처방 변환 |
| actuator ZERO/POSITIVE | 용량·**편집 연산** 스키마 |
| “3h→1h는 구간 병합” 언급 | 지속시간 API |

→ 신규: [`CF_ACTION_GROUNDING_V03.md`](./CF_ACTION_GROUNDING_V03.md)  
→ CF / ROADMAP / V03 / FOLLOWUP / FINETUNE_VERSIONS에 교차 반영.

---

## 4. “의미를 남기면 역산 가능하지 않나?”

### 결론

**부분적으로 맞다. 대상에 따라 갈린다.**

| 신호 | 역산 |
|------|------|
| 연속값 `VALUE_ABS\|ABS_Bk` + edges registry | **구간** `[lo,hi)` 가능 · exact float는 다대일로 불가 |
| cell sidecar `raw_value` | 관측 점값 join |
| actuator `POSITIVE`/`ZERO` | on/off만 · 용량 불가 |
| 파생 duration 토큰 + sidecar | 창 길이는 가능해도 **편집 연산 종류**는 별개 |

세 가지를 섞지 말 것:

1. **어휘/registry에 의미 남기기** → 타입·bin ↔ 물리 구간 조회 (Path A)
2. **인스턴스 sidecar** → 관측 raw·cell 출처
3. **편집 연산 의미** → 토큰 문자열이 아니라 **`edit_scope`**

`TokenizerV2.MeasurementTokens`는 이미 `raw_value`·`roles`·`source_cell_id`를 갖는다. 학습용 event parquet는 SENTENCE/token 중심이라, 역산은 vocab만으로 닫지 말고 **registry + sidecar (+ Path B는 edit_scope)** 로 닫는다.

---

## 5. 단일 `token_edits`로 3h→1h를 알 수 없음 — 채택

```json
{ "event_id": "...", "from_tokens": ["POSITIVE"], "to_tokens": ["ZERO"] }
```

이것만으로는 다음이 구분되지 않는다.

- 3시간 구간 전체 제거  
- 마지막 2시간 제거 (`truncate_end`)  
- 시작 시점 이동  
- 중간 1시간만 제거  
- 전체 구간을 1시간으로 축소  

**채택:** actuator CF의 1급 입력은 segment-level `edit_scope`.

```json
{
  "edit_scope": {
    "type": "time_span",
    "source_event_ids": ["e1", "e2", "e3"],
    "operation": "truncate_end",
    "from_start": "10:00",
    "from_end": "13:00",
    "to_start": "10:00",
    "to_end": "11:00"
  }
}
```

- `token_edits` = scope를 시퀀스에 적용한 **파생 결과** (모델 forward용)
- `ground_actuator_span(**edit_scope**)` — token diff 입력 금지

---

## 6. 이벤트 묶음은 이미 의미적인가?

**예. 한 시점·한 장면 안에서는 이미 의미적으로 묶인다.**

```text
Cell → Measurement group [MEAS_SEP] → Event (ts+farm+zone+view)
    → SameTimeGroup → Sequence
```

| 묶음 | 역할 |
|------|------|
| measurement_group | 한 피처의 토큰(ABS/REL/QUALITY)이 같이 움직임 |
| event | 그 시각·구역·뷰의 관측 장면 |
| same_time_group | 같은 시각 E/A/R 동시성 |

이 구조는 Path A(MG 단위 편집·raw join)의 기반이다.  
부족했던 것은 “묶음 방식”이 아니라, **여러 이벤트를 하나의 시간 span 편집으로 다루는 상위 표현**이다.

---

## 7. 연속 POSITIVE = 3시간 유지?

예시: `10:00 / 11:00 / 12:00` fan `POSITIVE` → 이벤트 3개.

| 질문 | 답 |
|------|----|
| “시퀀스상 얼마나 켜져 있었나?” (상태 읽기) | 연속 POSITIVE → **약 3시간 유지로 해석 OK** (1h 해상도 run-length) |
| “그 3시간을 어떻게 1시간으로 바꿨나?” (CF 쓰기) | 어디를 ZERO로 바꿨는지에 따라 연산이 갈림 → **`edit_scope` 필요** |

**상태 해석(읽기)** 과 **구간 편집(쓰기)** 를 혼동하지 말 것.

주의:

- 해상도 1시간 → “정확히 180분”이 아니라 샘플이 연속 ON인 구간
- 중간 timestamp 결측 시 길이 해석·OFF 여부 정책 필요

---

## 8. 합의된 파이프라인 스케치

```text
[상태 파악] binary / Known-Unknown / 진단명
    ↓
[CF search]
    Path A: token / measurement-group 편집 + ΔR
    Path B: edit_scope 탐색 → 적용 → 파생 token_edits → 재인코딩 → ΔR
    ↓
[Action Grounding]
    Path A: registry decode_interval (+ optional raw anchor)
    Path B: ground_actuator_span(edit_scope) + capacity B0…
    ↓
[운영 출력] setpoint 구간 · 용량 · 지속시간 · 검증 지표
```

---

## 9. 관련 파일

| 파일 | 역할 |
|------|------|
| `docs/online2/CF_ACTION_GROUNDING_V03.md` | Grounding · invertibility · edit_scope 정본 |
| `docs/online2/COUNTERFACTUAL_INFERENCE_V03.md` | Path A/B CF 의사결정 |
| `docs/online2/ROADMAP_V03.md` | P4a/P4b |
| `docs/online2/DIAGNOSIS_FINETUNE_V03.md` | V03 상위 (§11 CF) |
| `docs/online2/pipeline_trace_v02/01_sequence.md` | event / STG 생성 |
| `docs/online2/pipeline_trace_v02/02_tokenizer.md` | bin ↔ 구간 |
| `src/online2/v2/tokenizer.py` | MeasurementTokens (raw·roles·MG) |

---

## 10. 한 줄 요약

이벤트·measurement group으로 **시점 스냅샷은 의미적으로 잘 묶여 있고**, 연속 POSITIVE는 **유지 시간으로 읽어도 된다.**  
다만 토큰 최소 편집만으로 **원시 setpoint·용량·“어떻게 줄였는지”** 까지는 안 되므로, Path A는 registry 역산·Path B는 **`edit_scope` + Action Grounding** 이 필요하다.
