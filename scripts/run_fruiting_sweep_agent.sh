#!/usr/bin/env bash
# Durable W&B sweep agent for fruiting finetune (survives terminal close).
#
# Usage:
#   bash scripts/run_fruiting_sweep_agent.sh <sweep_id> [count]
# Example:
#   bash scripts/run_fruiting_sweep_agent.sh abc123xy 50
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f setup.sh ]]; then
  # shellcheck source=/dev/null
  source setup.sh
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <fruiting_sweep_id> [agent_count]" >&2
  echo "Create a sweep first: bash scripts/run_fruiting_sweep.sh" >&2
  exit 1
fi

export WANDB_AGENT_DISABLE_FLAPPING=true
export WANDB_AGENT_MAX_INITIAL_FAILURES=100
export WANDB_START_METHOD=thread

SWEEP_ID="$1"
COUNT="${2:-50}"
LOG="logs/fruiting_sweep_agent.log"
mkdir -p logs

echo "[$(date)] Starting agent for sweep ${SWEEP_ID} (count=${COUNT})" >> "${LOG}"
nohup conda run -n life2vec wandb agent "dasom-oh/Berry2Vec/${SWEEP_ID}" --count "${COUNT}" \
  >> "${LOG}" 2>&1 &
echo $! > logs/fruiting_sweep_agent.pid
echo "Agent PID: $(cat logs/fruiting_sweep_agent.pid)"
echo "Log: ${LOG}"
