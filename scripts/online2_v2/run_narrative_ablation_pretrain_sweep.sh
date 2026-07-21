#!/usr/bin/env bash
# SOP 재조정 + 서사 다양성이 사전학습 표현에 미치는 영향을 검증하기 위한
# 6개 조건 pretraining sweep. 전부 5000 step, batch_size=40, max_length=1024로
# 고정해 조건 간 공정 비교가 가능하게 한다(early stop 비활성화 — 전부 같은
# step 수만큼 학습).
#
# E0 control            : 전체 코퍼스, sop=0.2/0.2 (실제 원본 체크포인트와 동일 설정)
# E1 sop_balanced        : 전체 코퍼스, sop=0.33/0.33 (더 균형 잡힌 SOP)
# E2 narrow_subset        : 최빈 8개 narrative_id만
# E3 single_table_only     : 테이블 1개짜리 서사(34개)만
# E4 multi_table_only      : 테이블 2개 이상 넘나드는 서사(46개)만
# E5 dedup_reduced         : (farm, target_event) 당 1개만 남긴 중복 제거 세트
set -uo pipefail
cd /home/dasom/life2vec
PY=/home/dasom/miniconda3/envs/life2vec/bin/python
BUILD_DIR=outputs/online2/v2_build
SUBSET_DIR=outputs/online2/v2_build/narrative_ablation
RUN_ROOT=outputs/online2/v2_runs/narrative_ablation
STEPS=5000
BATCH=40
MAXLEN=1024

mkdir -p "$RUN_ROOT"

# idempotent: skip arms that already finished (best.ckpt present), resume arms
# that were interrupted mid-run (last.ckpt present but no best.ckpt yet) — this
# script may be re-invoked after an environment interruption.
run_arm () {
  local name="$1"
  local parquet="$2"
  local sop_reverse="$3"
  local sop_shuffle="$4"

  if [ -f "$RUN_ROOT/$name/best.ckpt" ] && [ "$(tail -c 200 "$RUN_ROOT/${name}.log" 2>/dev/null | grep -c 'DONE PASS')" -gt 0 ]; then
    echo "=== [$(date -u +%FT%TZ)] skip arm (already done): $name ==="
    return 0
  fi

  local resume_args=()
  if [ -f "$RUN_ROOT/$name/last.ckpt" ]; then
    echo "=== [$(date -u +%FT%TZ)] resuming arm: $name from last.ckpt ==="
    resume_args=(--resume "$RUN_ROOT/$name/last.ckpt")
  else
    echo "=== [$(date -u +%FT%TZ)] starting arm: $name (parquet=$parquet sop=$sop_reverse/$sop_shuffle) ==="
  fi

  "$PY" scripts/online2_v2/run_v2_pretrain_loop.py \
    --mode full \
    --build-dir "$BUILD_DIR" \
    --parquet "$parquet" \
    --run-dir "$RUN_ROOT/$name" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --max-length "$MAXLEN" \
    --max-rows 0 \
    --sop-reverse "$sop_reverse" \
    --sop-shuffle "$sop_shuffle" \
    --val-frac 0.05 --val-every 100 --val-batches 32 \
    --no-early-stop \
    --skip-vram-calibrate \
    --no-wandb \
    "${resume_args[@]}" \
    >> "$RUN_ROOT/${name}.log" 2>&1
  local status=$?
  echo "=== [$(date -u +%FT%TZ)] finished arm: $name (exit=$status) ==="
  return $status
}

FULL_PARQUET="$BUILD_DIR/training_events_v2.parquet"

run_arm control            "$FULL_PARQUET"                          0.2  0.2  || echo "FAILED: control"
run_arm sop_balanced       "$FULL_PARQUET"                          0.33 0.33 || echo "FAILED: sop_balanced"
run_arm narrow_subset      "$SUBSET_DIR/narrow_subset.parquet"      0.2  0.2  || echo "FAILED: narrow_subset"
run_arm single_table_only  "$SUBSET_DIR/single_table_only.parquet"  0.2  0.2  || echo "FAILED: single_table_only"
run_arm multi_table_only   "$SUBSET_DIR/multi_table_only.parquet"   0.2  0.2  || echo "FAILED: multi_table_only"
run_arm dedup_reduced      "$SUBSET_DIR/dedup_reduced.parquet"      0.2  0.2  || echo "FAILED: dedup_reduced"

echo "=== all arms done ==="
