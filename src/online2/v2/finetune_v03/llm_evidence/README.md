# llm_evidence

**근거**: 사용자 요청("saliency score를 증거로 제시, 이미지 및 학습 예시 사항과 함께 LLM에 입력하여 테스트 셋의 최종 증상을 판별")
**상태**: 파이프라인 전체 구현 완료 + 실제 체크포인트/데이터로 end-to-end 검증 완료 — **LLM 호출 자체만 미실행** (아래 이유 참조)

## 감사에서 확인된 사실 — 이 저장소에 LLM을 실제로 부르는 코드가 없었다

이번 세션 감사에서: `run_pre_llm_decision_batch_v03.py`는 이름과 달리 risk_before/after만 로컬 계산하고 LLM은 안 부른다(`"llm_boundary": "STOPPED_BEFORE_LLM"`을 스스로 명시). 저장소 전체에서 실제 LLM API 호출은 `generate_cf1s_case_analysis_v03.py` 단 하나뿐이고, 이것도 **로컬 서버**(`http://127.0.0.1:8080/chat/completions`, model `gemma-4-31b-it`)를 겨냥하며 클라우드 API 키는 이 환경에 없다(`ANTHROPIC_BASE_URL`은 기본 엔드포인트일 뿐 자격증명이 아님). 이 모듈은 그 로컬-서버 패턴을 그대로 재사용하도록 만들었다 — 새 프로토콜을 발명하지 않았다.

**지금 이 세션에서 `curl -sv http://127.0.0.1:8080/v1/models`로 실측 확인한 결과 로컬 서버가 떠 있지 않다** (`Connection refused`). 그래서 `--dry-run`으로 파이프라인 전체(증거 조립 → 프롬프트 생성 → 저장)를 실행하고 검증했고, 실제 네트워크 호출 한 줄만 로컬 서버가 뜨거나 `--api-base`/`--api-key`로 클라우드 엔드포인트를 넘기면 그대로 동작한다.

## 파일 맵

| 파일 | 역할 |
|---|---|
| [`evidence_bundle.py`](evidence_bundle.py) | `build_case_evidence` — `risk_saliency.py::event_ixg_abnormal_margin`(기존 함수, 새로 안 만듦)로 이벤트별 saliency 계산, 상위/하위 이벤트를 `events_tokenized_v2.parquet`에서 SENTENCE로 역참조, 모델 확률(p_abnormal/fine_probs), DINO 이미지 융합 여부 |
| [`nearest_examples.py`](nearest_examples.py) | `nearest_labeled_examples` — `representation_explorer/label_separation_probe.py`가 쓰는 것과 같은 OOF 표현(`h_binary`/`z_proj`)에서 코사인 유사도로 "학습 예시" 검색 |
| [`prompt.py`](prompt.py) | `build_diagnosis_prompt` — `generate_cf1s_case_analysis_v03.py::SYSTEM_PROMPT`와 같은 원칙(주어진 근거만 사용, 지어내지 않음)의 한국어 프롬프트 |
| [`llm_client.py`](llm_client.py) | `chat_completion` — `generate_cf1s_case_analysis_v03.py::_http_chat`를 일반화(재시도 루프 포함), OpenAI-호환 엔드포인트라면 로컬/클라우드 무관하게 동작 |

실행 진입점: `scripts/online2_v2/v03/run_llm_evidence_diagnosis_v03.py`

## 이미지 증거의 한계 — 정직하게 밝힌다

이 저장소 어디에도 vision 모델 연동이 없다(DINO는 임베딩만 만들고, 그 임베딩이 각주 없이 이벤트 표현에 融합될 뿐 원본 이미지 경로를 보존하지 않는다). 그래서 "이미지 증거"는 원시 이미지가 아니라 **"이 케이스에 DINO 이미지 임베딩이 융합되었는가"라는 텍스트 사실**로만 프롬프트에 들어간다 — 이미지 내용을 서술하는 척하지 않도록 시스템 프롬프트에 명시적으로 금지했다.

## 실측 검증 (2026-07-24, `--dry-run`)

`probe_oof_3fold/r0_fold0_best.pt` 체크포인트로 케이스 `F551358_2024-09-21_2024-10-04`(실제 정답: `뿌리_발육_부진`) 하나를 처리:

```
model_p_abnormal: 0.989
model_fine_probs top1: 뿌리_발육_부진=0.973   (실제 정답과 일치)
saliency 상위 이벤트: t=311.0h ... FEATURE|CIRCULATION_FAN ... (실제 SENTENCE 역참조 성공)
유사 학습 사례 3건: 코사인 유사도 0.594 / 0.421 / 0.374 (서로 다른 진단 — 억지로 맞춘 값 아님)
```

전체 파이프라인(증거 조립 → SENTENCE 역참조 → 유사사례 검색 → 프롬프트 조립 → 저장)이 실제 데이터로 끝까지 돌아간다는 걸 확인했다. `outputs/online2/v2_finetune_v03/llm_evidence/F551358_..._evidence.json`에 전체 결과가 남아있다.

## 다음에 실제로 LLM을 부르려면

1. 로컬 OpenAI-호환 서버를 `127.0.0.1:8080`에 띄우거나(`generate_cf1s_case_analysis_v03.py`가 이미 쓰는 방식),
2. `--api-base`/`--model`/`--api-key`로 클라우드 엔드포인트를 넘기고,
3. `--dry-run`을 빼고 실행한다.

## 의존성

- 기존: `risk_saliency.py`, `finetune_v03/checkpoint.py::load_fold_model_v03`, `representation_explorer/label_separation_probe.py`가 쓰는 것과 같은 `*_oof_representations.npz`, `events_tokenized_v2.parquet`
- 신규 코드는 없고, 전부 기존 함수 재사용 + 새 조립/프롬프트 계층
