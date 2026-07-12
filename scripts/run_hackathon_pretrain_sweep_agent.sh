#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f setup.sh ]]; then
  # shellcheck source=/dev/null
  source setup.sh
fi

SWEEP_ID="${1:?Usage: $0 <sweep_id> [count]}"
COUNT="${2:-}"

if [[ -n "${COUNT}" ]]; then
  conda run -n life2vec wandb agent "dasom-oh/Berry2Vec/${SWEEP_ID}" --count "${COUNT}"
else
  conda run -n life2vec wandb agent "dasom-oh/Berry2Vec/${SWEEP_ID}"
fi
