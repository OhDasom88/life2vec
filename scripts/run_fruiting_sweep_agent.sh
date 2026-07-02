#!/usr/bin/env bash
# Durable W&B sweep agent (survives terminal close).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f setup.sh ]]; then
  # shellcheck source=/dev/null
  source setup.sh
fi

export WANDB_AGENT_DISABLE_FLAPPING=true
export WANDB_AGENT_MAX_INITIAL_FAILURES=100
export WANDB_START_METHOD=thread

SWEEP_ID="${1:-4rp7i4by}"
COUNT="${2:-50}"
LOG="logs/fruiting_sweep_agent.log"
mkdir -p logs

echo "[$(date)] Starting agent for sweep ${SWEEP_ID} (count=${COUNT})" >> "${LOG}"
nohup conda run -n life2vec wandb agent "dasom-oh/Berry2Vec/${SWEEP_ID}" --count "${COUNT}" \
  >> "${LOG}" 2>&1 &
echo $! > logs/fruiting_sweep_agent.pid
echo "Agent PID: $(cat logs/fruiting_sweep_agent.pid)"
echo "Log: ${LOG}"
