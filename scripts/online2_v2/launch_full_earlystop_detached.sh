#!/usr/bin/env bash
# Detached Online2 V2 full pretrain (survives SSH/Cursor disconnect).
# Usage:
#   bash scripts/online2_v2/launch_full_earlystop_detached.sh
#   bash scripts/online2_v2/launch_full_earlystop_detached.sh --resume outputs/.../best.ckpt

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_DIR="outputs/online2/v2_runs/full_event_grain_earlystop"
LOG="outputs/online2/v2_build/full_event_grain_earlystop_pretrain.log"
PID_FILE="outputs/online2/v2_build/full_event_grain_earlystop.pid"
BUILD="outputs/online2/v2_build"

source /home/dasom/miniconda3/etc/profile.d/conda.sh
conda activate life2vec

RESUME_ARGS=()
if [[ $# -gt 0 ]]; then
  RESUME_ARGS=("$@")
elif [[ -f "${RUN_DIR}/best.ckpt" ]]; then
  RESUME_ARGS=(--resume "${RUN_DIR}/best.ckpt")
elif [[ -f "${RUN_DIR}/last.ckpt" ]]; then
  RESUME_ARGS=(--resume "${RUN_DIR}/last.ckpt")
fi

# Kill any previous training on this run-dir (same script / same run name).
if pgrep -f "run_v2_pretrain_loop.py .*${RUN_DIR}" >/dev/null 2>&1; then
  echo "Stopping existing pretrain for ${RUN_DIR}..."
  pkill -TERM -f "run_v2_pretrain_loop.py .*full_event_grain_earlystop" || true
  sleep 3
  pkill -KILL -f "run_v2_pretrain_loop.py .*full_event_grain_earlystop" 2>/dev/null || true
fi

mkdir -p "$(dirname "$LOG")" "$RUN_DIR"
{
  echo ""
  echo "===== DETACHED LAUNCH $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "resume_args: ${RESUME_ARGS[*]:-none}"
} >>"$LOG"

# New session + nohup so Cursor/SSH hangup cannot kill the job.
setsid nohup env \
  PYTHONPATH=. \
  PYTHONUNBUFFERED=1 \
  python scripts/online2_v2/run_v2_pretrain_loop.py \
    --mode full \
    --max-rows 0 \
    --steps 100000 \
    --batch-size 56 \
    --max-length 1024 \
    --ckpt-every 200 \
    --val-every 100 \
    --val-batches 32 \
    --val-frac 0.05 \
    --early-stop-patience 15 \
    --early-stop-min-delta 1e-4 \
    --sop-reverse 0.20 \
    --sop-shuffle 0.20 \
    --target-vram-frac 0.70 \
    --skip-vram-calibrate \
    --wandb \
    --wandb-watch \
    --wandb-watch-log all \
    --wandb-watch-freq 100 \
    --wandb-project Berry2Vec \
    --wandb-entity dasom-oh \
    --wandb-run-name online2_v2_event_grain_earlystop_bg \
    --wandb-tags online2_v2,event_grain,global_abspos,full,early_stop,detached \
    --build-dir "$BUILD" \
    --run-dir "$RUN_DIR" \
    "${RESUME_ARGS[@]}" \
    >>"$LOG" 2>&1 < /dev/null &

echo $! >"$PID_FILE"
sleep 2
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Detached pretrain PID=$(cat "$PID_FILE")"
  echo "Log: $LOG"
  echo "PID file: $PID_FILE"
  echo "tail -f $LOG"
else
  echo "ERROR: process failed to start; see $LOG" >&2
  exit 1
fi
