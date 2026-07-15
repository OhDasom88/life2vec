#!/usr/bin/env bash
# Launch Sweep-1 with parallel wandb agents targeting ~80% GPU memory.
# Usage: bash scripts/online2_v2/v03/launch_sweep1_parallel.sh [n_agents]
set -euo pipefail

ROOT="${ROOT:-/home/dasom/life2vec}"
cd "$ROOT"
# shellcheck disable=SC1091
source /home/dasom/miniconda3/etc/profile.d/conda.sh
conda activate life2vec

export WANDB_GROUP="${WANDB_GROUP:-finetune_v03_improvement_r1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Slight fragmentation tolerance for many concurrent processes
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="$ROOT/outputs/online2/v2_finetune_v03/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Estimate agents: free_mem * 0.80 / ~1800MiB per run (conservative incl. activations)
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
PER_RUN_MIB="${PER_RUN_MIB:-1800}"
TARGET_FRAC="${TARGET_FRAC:-0.80}"
CALC_N="$(python - <<PY
free=float("${FREE_MIB}")
per=float("${PER_RUN_MIB}")
frac=float("${TARGET_FRAC}")
n=int((free*frac)//per)
print(max(1, min(n, 16)))
PY
)"
N_AGENTS="${1:-$CALC_N}"

echo "[$(date '+%F %T')] free=${FREE_MIB}MiB per_run≈${PER_RUN_MIB}MiB → agents=${N_AGENTS} (target ${TARGET_FRAC})"

SWEEP_OUT="$(wandb sweep --project Berry2Vec --entity dasom-oh conf/sweep/finetune_v03_s1_loss_binary.yaml 2>&1 | tee "$LOG_DIR/sweep1_create_${STAMP}.log")"
echo "$SWEEP_OUT"
SWEEP_ID="$(echo "$SWEEP_OUT" | sed -n 's/.*wandb agent [^ ]*\/[^ ]*\/\([a-z0-9]*\).*/\1/p' | tail -1)"
if [[ -z "$SWEEP_ID" ]]; then
  SWEEP_ID="$(echo "$SWEEP_OUT" | sed -n 's|.*Berry2Vec/\([a-z0-9]*\).*|\1|p' | tail -1)"
fi
if [[ -z "$SWEEP_ID" ]]; then
  echo "Failed to parse SWEEP_ID" >&2
  exit 1
fi
SWEEP_PATH="dasom-oh/Berry2Vec/${SWEEP_ID}"
echo "$SWEEP_PATH" > "$LOG_DIR/LATEST_SWEEP1.txt"
echo "$N_AGENTS" > "$LOG_DIR/LATEST_SWEEP1_N_AGENTS.txt"
echo "SWEEP=$SWEEP_PATH N_AGENTS=$N_AGENTS"

PIDS=()
for i in $(seq 1 "$N_AGENTS"); do
  LOG="$LOG_DIR/sweep1_agent_${STAMP}_${i}.log"
  # Each agent pulls runs until queue empty (or count exhausted)
  nohup setsid wandb agent "$SWEEP_PATH" --count 30 \
    >"$LOG" 2>&1 &
  PIDS+=($!)
  echo "agent[$i] pid=${PIDS[-1]} log=$LOG"
  sleep 2  # stagger starts to avoid simultaneous model loads
done

printf '%s\n' "${PIDS[@]}" > "$LOG_DIR/sweep1_agent_pids_${STAMP}.txt"
echo "All agents launched. Monitor: nvidia-smi / wandb group=${WANDB_GROUP}"
echo "PIDs: ${PIDS[*]}"
