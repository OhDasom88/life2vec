#!/usr/bin/env bash
# Sequential v0.3 pipeline — survives client disconnect (nohup / setsid).
# Order: import smoke → P1 smoke train → P0 evidence → P0 token hook → P1 5-fold CV
set -euo pipefail

ROOT="${ROOT:-/home/dasom/life2vec}"
cd "$ROOT"
PY="${PY:-python3}"
OUT="$ROOT/outputs/online2/v2_finetune_v03"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR" "$OUT/runs" "$OUT/p0"

STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/pipeline_${STAMP}.log"
PROGRESS="$LOG_DIR/PROGRESS.jsonl"

progress() {
  # shell JSONL line
  printf '{"ts":"%s","phase":"pipeline","msg":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")" \
    | tee -a "$PROGRESS"
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

log "=== v0.3 pipeline start stamp=$STAMP pid=$$ ==="
progress "start stamp=$STAMP"

# --- 0: import / package smoke ---
log "STEP0 import smoke"
"$PY" - <<'PY' 2>&1 | tee -a "$MAIN_LOG"
import torch
from src.online2.v2.finetune_v03 import EventPoolingDiagnosisModelV03
from src.online2.v2.finetune_v03.config import FinetuneV03Config
from src.online2.v2.finetune_v03.version import FINETUNE_VERSION
cfg = FinetuneV03Config(input_dim=384)
m = EventPoolingDiagnosisModelV03(cfg)
x = torch.randn(2, 8, 384)
pad = torch.ones(2, 8)
out = m(
    event_mean=x, event_max=x,
    case_age_hours=torch.zeros(2, 8),
    view_id=torch.zeros(2, 8, dtype=torch.long),
    zone_id=torch.zeros(2, 8, dtype=torch.long),
    local_hour=torch.zeros(2, 8, dtype=torch.long),
    padding_mask=pad,
)
assert "logits" in out and "abnormal_logit" in out and "z_proj" in out
print("OK", FINETUNE_VERSION, tuple(out["logits"].shape))
PY
progress "step0_import_ok"

# --- 1: P1 smoke train ---
SMOKE_DIR="$OUT/runs/smoke_${STAMP}"
log "STEP1 smoke train → $SMOKE_DIR"
DEVICE="cuda"
if ! "$PY" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  DEVICE="cpu"
fi
log "device=$DEVICE"
"$PY" scripts/online2_v2/v03/run_diagnosis_finetune_v03.py \
  --mode smoke \
  --epochs 5 \
  --no-early-stop \
  --batch-size 4 \
  --device "$DEVICE" \
  --run-dir "$SMOKE_DIR" \
  --no-wandb \
  2>&1 | tee -a "$MAIN_LOG"
progress "step1_smoke_done dir=$SMOKE_DIR"

CKPT="$SMOKE_DIR/fold0_best.pt"
if [[ ! -f "$CKPT" ]]; then
  # smoke saves fold0_best.pt when fold_index=0
  CKPT="$(ls -1 "$SMOKE_DIR"/*best.pt 2>/dev/null | head -1 || true)"
fi
log "smoke ckpt=$CKPT"

# --- 2: P0 evidence (fuse saliency + raw) ---
log "STEP2 P0 evidence demo"
P0_ARGS=(--device cpu --case-id F420458_2025-02-16_2025-03-01)
if [[ -n "${CKPT:-}" && -f "$CKPT" ]]; then
  P0_ARGS+=(--ckpt "$CKPT")
fi
"$PY" scripts/online2_v2/v03/run_p0_evidence_demo_v03.py "${P0_ARGS[@]}" \
  2>&1 | tee -a "$MAIN_LOG"
progress "step2_p0_evidence_done"

# --- 3: P0 token attr hook ---
log "STEP3 P0 token attribution hook"
"$PY" scripts/online2_v2/v03/run_token_attr_hook_v03.py --probe-only \
  --case-id F420458_2025-02-16_2025-03-01 \
  2>&1 | tee -a "$MAIN_LOG"
progress "step3_token_hook_done"

# --- 4: P1 5-fold CV ---
CV_DIR="$OUT/runs/cv_${STAMP}"
log "STEP4 5-fold CV → $CV_DIR"
"$PY" scripts/online2_v2/v03/run_diagnosis_finetune_v03.py \
  --mode cv \
  --n-folds 5 \
  --epochs 150 \
  --early-stop-patience 20 \
  --early-stop-min-epochs 30 \
  --batch-size 8 \
  --device "$DEVICE" \
  --run-dir "$CV_DIR" \
  --wandb \
  --wandb-project Berry2Vec \
  --wandb-entity dasom-oh \
  --wandb-run-name "v03_cv_${STAMP}" \
  --wandb-tags "online2_v2,diagnosis,finetune_v03,multitask,binary,event_pooling,cv" \
  2>&1 | tee -a "$MAIN_LOG"
progress "step4_cv_done dir=$CV_DIR"

log "=== v0.3 pipeline COMPLETE stamp=$STAMP ==="
progress "complete stamp=$STAMP"
echo "$CV_DIR" > "$LOG_DIR/LATEST_CV_DIR.txt"
echo "$SMOKE_DIR" > "$LOG_DIR/LATEST_SMOKE_DIR.txt"
