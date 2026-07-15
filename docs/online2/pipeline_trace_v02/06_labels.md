# 06. 학습 · 라벨 정보

## 35건 label map

경로 (원본과 이 디렉토리 복사본):

- `outputs/online2/v2_finetune/label_map.json`
- `docs/online2/pipeline_trace_v02/06_labels/label_map.json`
- `outputs/online2/v2_finetune/labels_example_score90.csv`

10개 진단명:

1. 기형과_생리장해  
2. 동해_난방실패  
3. 뿌리_발육_부진  
4. 영양생장_과다  
5. 영양생장_부족  
6. 잿빛곰팡이병_발생_위험  
7. **정상_운영**  
8. 칼슘부족_잎끝마름  
9. 탄저병_발생_위험  
10. 흰가루병_발생_위험  

건수(예시 score90 기준): 칼슘 5, 정상/탄저/기형과 각 4, 나머지 각 3 → 합 35.

미공개 20건용 합본 CSV가 있으면 problem 쪽 라벨은 추론 placeholder(종종 `정상_운영`)로만 둘 수 있음 — **학습용 정답이 아님**.

## 정상 운영 사례 · “정상” 정의

예시 정상 farm (labels CSV):

- F420458, F194265, F209570, F396544

정의 출처: 공개 예시 답안 score90 문장 **「정상 운영입니다」** → 정규화 `정상_운영`.

주의:

- “센서 전부 이상 없음”이 아니라 **예시 답안의 진단 라벨**입니다.
- 정상 라벨 케이스에도 구역 편차·saliency peak가 있을 수 있습니다 (F420458 샘플 참고).

별도의 “정상 구간(시간 마스크)” 학습 타깃은 현재 분류 head에 없습니다.

## train / validation 분리

`src/online2/v2/diagnosis_dataset.py` `make_stratified_farm_folds`

- **Group K-fold by `farm_id`** (기본 5 fold)
- 진단별 farm 수를 greedy하게 균형
- seed 기본 2023

각 fold:

- `val`: 해당 fold farm들의 case
- `train`: 나머지 farm들의 case

## 농가 split 누수?

**동일 farm이 train과 val에 동시 포함되지 않습니다.**  
example set은 대체로 farm당 1 case라 case/farm leakage가 같게 동작합니다.

problem 20건은 CV 학습에 넣지 않고 별도 추론하는 구성이 일반적입니다.
