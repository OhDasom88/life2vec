# Label Studio 실험 연동

**용도**: 착과수 버킷 **오답 검수**, 라벨 품질 확인, 예측 결과 QA

## 프로젝트 명명

```
berry2vec-fruiting-EXP-NNN
```

예: `berry2vec-fruiting-EXP-001`

## 디렉터리

```
labelstudio/
├── projects/
│   └── fruiting/
│       ├── config.xml          # Label Studio import용 설정
│       └── README.md
data/labelstudio/
├── imports/                    # predict_fruiting.py CSV → LS JSON
│   └── EXP-001_predictions.json
└── exports/                    # 검수 완료 라벨 export
    └── EXP-001_reviewed.json
```

## 워크플로

```
predict_fruiting.py → predictions_all.csv
        ↓
scripts/export_labelstudio_fruiting.py  (EXP-NNN)
        ↓
data/labelstudio/imports/EXP-NNN_predictions.json
        ↓
Label Studio Import → 검수 (true/pred bucket, 오답 여부)
        ↓
Export JSON → data/labelstudio/exports/
        ↓
Notion EXP 페이지 + registry.yaml 업데이트
```

## Label Studio 실행 (로컬)

```bash
pip install label-studio
label-studio start --port 8080
```

1. **Create Project** → `berry2vec-fruiting-EXP-001`
2. **Settings → Labeling Interface** → `labelstudio/projects/fruiting/config.xml` 붙여넣기
3. **Import** → `data/labelstudio/imports/EXP-001_predictions.json`
4. 검수 후 **Export** → JSON

## 환경 변수 (선택)

```bash
export LABEL_STUDIO_URL=http://localhost:8080
export LABEL_STUDIO_API_KEY=<your-key>
```

Notion `Label Studio` 컬럼에는 프로젝트 URL을 기록:

```
http://localhost:8080/projects/<id>
```

## 검수 항목

| 필드 | 설명 |
|------|------|
| `true_bucket` | 실제 착과수 버킷 (0~5) |
| `pred_bucket` | 모델 예측 버킷 |
| `review_ok` | 예측이 타당한지 (yes/no) |
| `corrected_bucket` | 수정 버킷 (오답 시) |
| `notes` | 메모 |

## 실험별 Label Studio (현재)

| Exp | 프로젝트 | Import | 상태 |
|-----|----------|--------|------|
| EXP-001 | `berry2vec-fruiting-EXP-001` | `outputs/berry2vec_fruiting_predictions/predictions_all.csv` | **준비됨** (import 대기) |
| EXP-002 | — | — | — |
| EXP-003 | — | — | — |

## EXP-001 import 생성

```bash
python scripts/export_labelstudio_fruiting.py \
  --predictions outputs/berry2vec_fruiting_predictions/predictions_all.csv \
  --exp-id EXP-001 \
  --output data/labelstudio/imports/EXP-001_predictions.json
```

## Notion 연동

- DB 컬럼 **Label Studio**: 프로젝트 URL
- EXP 상세 페이지에 검수 완료 건수·수정 라벨 요약 기록
