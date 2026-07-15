# 월드모델 재사용 가능 범위 (1–5 기준)

질문: **현재 모델을 어느 정도까지 masked future world model로 재사용할 수 있는가?**

## 결론 (한 줄)

| 구성요소 | 재사용 |
|----------|--------|
| Frozen / finetune encoder weights (`Transformer` bidirectional) | **표현 백본으로 재사용 가능** |
| Stage A mean/max + PAD 분류기 | 진단용; 월드모델 목표와 **다름** (그대로 WM 아님) |
| MLM head (`MaskedLanguageModel`) | 토큰 복원 head로 재사용 가능하나 **미래 조건부 아님** |
| 현재 masking (`mlm_mask` / `GroupedMLMMasker`) | **랜덤(·그룹) MLM** — 시간 방향 span/future mask **없음** |
| DINO / IMAGE fuse | 선택적 multimodal 입력; WM과 직교, attach는 event summary 쪽 |

즉, **“encoder(±MLM head)를 초기화로 쓴 뒤, masking·attn을 미래 예측용으로 교체”**가 현실적 범위입니다.  
PAD+10-class 진단 스택을 월드모델이라고 부르기는 어렵습니다.

## 근거 요약

### 재사용하기 좋은 것

1. `Transformer.forward_finetuning` / `forward_finetuning_with_embeddings`  
   - 토큰(또는 주입된) 임베딩 → 전층 encoder hidden.
2. `get_sequence_embedding` + slot residual 주입 패턴  
   (`stage_a_image_encode.encode_batch_pool_with_dino_slots`).
3. Vocab / TokenizerV2 / event grain 시퀀스 포맷.
4. MLM decoder 구조 (pos gather → vocab).

### 부족한 것 (WM으로 가려면 필수 패치)

1. **Causal / future mask**: encoder layer가 양방향.  
2. **Time-direction span masking**: `mlm_mask`는 위치 랜덤; STG 블록 future span API 없음.  
3. **Horizon / rollout loss**: 다음 Δt·다음 STG·다음 값 분포 목표가 정의되어 있지 않음.  
4. **Raw 수치 디코딩**: bin만 복원; EC dS/m 등 continuous head는 별도.

### DINO / IMAGE (우선순위 1)

- 기본 Stage A cache: IMAGE 제외 → fusion 수식 미적용.  
- v0.2:  
  - `SharedImageAdapter`: \(r = g \cdot A(v/\lVert v\rVert)\)  
  - slot bake 또는 `event_mean/max += r`  
- WM 재사용과는 독립: multimodal 관측으로 넣는 경로만 확보되면 됨.

### PAD (우선순위 2–3)

- 입력이 이미 **pooled event summary**라 토큰 수준 future MLM과 층위가 다름.  
- WM을 토큰/event 시퀀스에 두려면 Stage A encoder 쪽에 붙이는 편이 맞음.

## 추천 작업 순서 (패치 설계)

1. `GroupedMLMMasker` 또는 `MLM.mlm_mask`에 **미래 STG span mask** 옵션 추가.  
2. (선택) encoder에 causal/prefix mask 실험.  
3. MLM head로 future tokens 예측 → 기존 ckpt로 warm-start.  
4. PAD 분류는 별 브랜치로 유지 (진단 평가와 분리).  
5. IMAGE는 attach-only + fold adapter 경로 유지 (`01_dino_fuse`).
