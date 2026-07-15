#!/usr/bin/env bash
# Launch Sweep-1 wandb agents inside a detached tmux session (survives SSH/network drop).
#
# Usage:
#   bash scripts/online2_v2/v03/launch_sweep1_tmux.sh [n_agents]
#   SWEEP_PATH=dasom-oh/Berry2Vec/8uhwp5w8 bash scripts/online2_v2/v03/launch_sweep1_tmux.sh 10
#
# Attach:  tmux attach -t v03_sweep1
# Detach:  Ctrl-b d
set -euo pipefail

ROOT="${ROOT:-/home/dasom/life2vec}"
cd "$ROOT"
# shellcheck disable=SC1091
source /home/dasom/miniconda3/etc/profile.d/conda.sh
conda activate life2vec

export WANDB_GROUP="${WANDB_GROUP:-finetune_v03_improvement_r1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SESSION="${TMUX_SESSION:-v03_sweep1}"
LOG_DIR="$ROOT/outputs/online2/v2_finetune_v03/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

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

# Reuse existing sweep by default (avoid creating a duplicate sweep queue).
SWEEP_PATH="${SWEEP_PATH:-}"
if [[ -z "$SWEEP_PATH" && -f "$LOG_DIR/LATEST_SWEEP1.txt" ]]; then
  SWEEP_PATH="$(tr -d '[:space:]' < "$LOG_DIR/LATEST_SWEEP1.txt")"
fi
if [[ -z "$SWEEP_PATH" ]]; then
  echo "SWEEP_PATH empty and no LATEST_SWEEP1.txt — creating a new sweep."
  SWEEP_OUT="$(wandb sweep --project Berry2Vec --entity dasom-oh conf/sweep/finetune_v03_s1_loss_binary.yaml 2>&1 | tee "$LOG_DIR/sweep1_create_${STAMP}.log")"
  echo "$SWEEP_OUT"
  SWEEP_ID="$(echo "$SWEEP_OUT" | sed -n 's/.*wandb agent [^ ]*\/[^ ]*\/\([a-z0-9]*\).*/\1/p' | tail -1)"
  if [[ -z "$SWEEP_ID" ]]; then
    SWEEP_ID="$(echo "$SWEEP_OUT" | sed -n 's|.*Berry2Vec/\([a-z0-9]*\).*|\1|p' | tail -1)"
  fi
  [[ -n "$SWEEP_ID" ]] || { echo "Failed to parse SWEEP_ID" >&2; exit 1; }
  SWEEP_PATH="dasom-oh/Berry2Vec/${SWEEP_ID}"
fi

echo "$SWEEP_PATH" > "$LOG_DIR/LATEST_SWEEP1.txt"
echo "$N_AGENTS" > "$LOG_DIR/LATEST_SWEEP1_N_AGENTS.txt"
echo "[$(date '+%F %T')] free=${FREE_MIB}MiB agents=${N_AGENTS} sweep=${SWEEP_PATH} session=${SESSION}"

# Stop any previous session with the same name
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Killing existing tmux session: $SESSION"
  tmux kill-session -t "$SESSION"
fi

INNER="$LOG_DIR/sweep1_tmux_inner_${STAMP}.sh"
cat > "$INNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
source /home/dasom/miniconda3/etc/profile.d/conda.sh
conda activate life2vec
export WANDB_GROUP="$WANDB_GROUP"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
SWEEP_PATH="$SWEEP_PATH"
N_AGENTS=$N_AGENTS
LOG_DIR="$LOG_DIR"
STAMP="$STAMP"
PIDS=()
echo "[$(date '+%F %T')] Starting \$N_AGENTS agents on \$SWEEP_PATH"
for i in \$(seq 1 "\$N_AGENTS"); do
  LOG="\$LOG_DIR/sweep1_agent_\${STAMP}_\${i}.log"
  wandb agent "\$SWEEP_PATH" --count 30 >"\$LOG" 2>&1 &
  PIDS+=(\$!)
  echo "agent[\$i] pid=\${PIDS[-1]} log=\$LOG"
  sleep 2
done
printf '%s\\n' "\${PIDS[@]}" > "\$LOG_DIR/sweep1_agent_pids_\${STAMP}.txt"
echo "Agents launched. Waiting (detach with Ctrl-b d)."
wait
echo "[$(date '+%F %T')] All agents finished."
exec bash
EOF
chmod +x "$INNER"

tmux new-session -d -s "$SESSION" -c "$ROOT" "bash '$INNER'"
echo "tmux session '$SESSION' started (detached)."
echo "  attach: tmux attach -t $SESSION"
echo "  logs:   $LOG_DIR/sweep1_agent_${STAMP}_*.log"
echo "  sweep:  https://wandb.ai/dasom-oh/Berry2Vec/sweeps/${SWEEP_PATH##*/}"
