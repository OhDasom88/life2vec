# 02. 토크나이저 · 레지스트리

## 핵심 코드

| 역할 | 파일 |
|------|------|
| 값 → 토큰 문자열 | `src/online2/v2/tokenizer.py` `TokenizerV2.tokenize_value` |
| 문자열 ↔ ID | `src/online2/v2/vocab.py` `VocabV2` |
| bin edges / suffix | `src/online2/v2/binning.py` `BinRuleV2`, `BinningRegistryV2` |
| 스키마 | `src/online2/v2/feature_schema.py`, `feature_schema_v2.yaml` |

## token ID → 의미 디코딩

1. `token_id` → `vocab_v2.json`의 `id_to_token`로 문자열 복원.
2. 문자열 예:
   - `FEATURE|inside_temp_c`
   - `ABS_B03` (절대값 bin 접미사)
   - `GLOBAL_REL_B12` / `FARM_REL_B05`
   - `VIEW|ENVIRONMENT`
   - `OBSERVED_VALUE|POSITIVE` (actuator literal)
3. bin 접미사 `ABS_B{k}`의 **수치 구간**은 vocab만으로 알 수 없고,  
   `binning_registry_v2_transductive.json`의 `rules[].edges`로 복원  
   (`edges[k] ≤ x < edges[k+1]`).

산출물:

- `outputs/online2/v2_build/vocab_v2.json`
- `outputs/online2/v2_build/binning_registry_v2_transductive.json`

샘플(ABS edges):  
`07_case_artifacts/.../02_binning_registry_feature_sample.json`

## bin 범위 · 대표값

- 범위: `BinRuleV2.edges` (equal-frequency + anchors 등으로 fit).
- **명시적 mid-point / 대표값 토큰은 없음.**  
  보고서·모델은 bin 인덱스/토큰만 보고, 실제 ℃·dS/m는 별도 raw에서 가져와야 함.

## raw value가 보존되는 위치

| 위치 | raw 보존? | 비고 |
|------|-----------|------|
| `cell_occurrences.parquet` | ✅ `raw_value`, `raw_display` | builder 단계 |
| SQLite / atomic_values (빌드 DB) | ✅ | 빌드 파이프라인 내부 |
| `events_tokenized_v2.parquet` | ❌ | SENTENCE / token_ids만 |
| Stage A `event_embeddings/*.parquet` | ❌ | mean/max 벡터 + 메타 |
| saliency / evidence JSON | △ | event_id·view·zone·saliency; 원시 수치 거의 없음 |
| Gemma rich payload | △ | saliency 중심; 재배 수치 인용 거의 0인 이유 |

정리: **학습·추론 경로로 갈수록 raw float는 사라지고 bin/latent만 남습니다.**
